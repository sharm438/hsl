import argparse
import torch
import torch.multiprocessing as mp
import os
import time
import json
from copy import deepcopy
import utils
import aggregation
import train_node
import models
import math

# mp.set_start_method("spawn", force=True) # Only needed if you use multiprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment')
    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--aggregation", type=str, default='fedsgd', 
                        choices=['fedsgd', 'p2p', 'p2p_local', 'hsl'])
    parser.add_argument("--num_spokes", type=int, default=10)
    parser.add_argument("--num_hubs", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_time", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--spoke_budget", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sample_type", type=str, default="round_robin", choices=["round_robin","random"])

    # NEW #1: A single seed argument
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for data splitting + model init. 0 or negative => random")

    # HSL parameters
    parser.add_argument("--b_hs", type=int, default=2)
    parser.add_argument("--b_hh", type=int, default=1)
    parser.add_argument("--b_sh", type=int, default=2)

    ############################################################################
    # NEW FEATURES
    ############################################################################
    # (1) Monitor model drift
    parser.add_argument("--monitor_model_drift", action='store_true',
                        help="If set, compute average model drift among spokes before/after aggregation.")

    # (5) Monitor degrees
    parser.add_argument("--monitor_degree", action='store_true',
                        help="If set, compute and log node in/out degrees each round.")

    # (6) Graph simulation only
    parser.add_argument("--graph_simulation_only", action='store_true',
                        help="If set, skip actual training. Only simulate mixing matrices per round.")

    return parser.parse_args()

def main(args):
    ############################################################################
    # Initialize seed if provided
    ############################################################################
    if args.seed > 0:
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # If you want deterministic cuDNN:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Info] Using fixed seed={args.seed}")
    else:
        print("[Info] Using a random seed (non-reproducible).")

    aggregator_device = torch.device('cuda:'+str(args.gpu)
                                     if args.gpu>=0 and torch.cuda.is_available()
                                     else 'cpu')
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    ############################################################################
    # Load data (unless graph_simulation_only)
    ############################################################################
    if not args.graph_simulation_only:
        trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
        data, labels = trainObject.train_data, trainObject.train_labels
        test_data = trainObject.test_data
        inp_dim = trainObject.num_inputs
        out_dim = trainObject.num_outputs
        lr = args.lr
        batch_size = trainObject.batch_size
        net_name = trainObject.net_name

        # Distribute data among spokes
        distributedData = utils.distribute_data_noniid(
            (data, labels),
            args.num_spokes,
            args.bias,
            inp_dim,
            out_dim,
            net_name,
            torch.device("cpu"),  # keep data on CPU
            min_samples=(batch_size if batch_size else 1)  # ensure >= batch_size if possible
        )
        distributed_data = distributedData.distributed_input
        distributed_label = distributedData.distributed_output
        node_weights = distributedData.wts

        num_samples_per_spoke = [len(distributed_data[i]) for i in range(args.num_spokes)]
        min_size = min(num_samples_per_spoke)
        max_size = max(num_samples_per_spoke)
        mean_size = sum(num_samples_per_spoke)/len(num_samples_per_spoke)
        var_size = sum((s - mean_size)**2 for s in num_samples_per_spoke)/len(num_samples_per_spoke)
        sd_size = math.sqrt(var_size)
        print(f"[Data Dist] Min={min_size}, Max={max_size}, Mean={mean_size:.1f}, SD={sd_size:.1f}")

        # Initialize global model
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

        # Round-robin indices
        rr_indices = {}
        for node_id in range(args.num_spokes):
            ds_size = distributed_data[node_id].shape[0]
            if ds_size > 0:
                perm = torch.randperm(ds_size)
                distributed_data[node_id] = distributed_data[node_id][perm]
                distributed_label[node_id] = distributed_label[node_id][perm]
            rr_indices[node_id] = 0

    else:
        # If --graph_simulation_only, we just define placeholders
        test_data = None
        inp_dim, out_dim = 1, 1  # dummy
        net_name = 'lenet'      # dummy
        batch_size = 32         # dummy
        node_weights = torch.ones(args.num_spokes)/args.num_spokes
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    ############################################################################
    # Metrics and Additional Logging Structures
    ############################################################################
    metrics = {
        'round': [],
        'global_acc': [],
        'global_loss': [],
        'local_acc': [],
        'local_loss': [],
        'spoke_acc': [],
    }

    # Model drift metrics
    if args.monitor_model_drift:
        metrics['pre_drift'] = []   # drift among spokes before aggregator
        metrics['post_drift'] = []  # drift among spokes after aggregator

    # For p2p / p2p_local / hsl: store mixing matrices
    # We'll accumulate them each round, then average at the end.
    W_accumulator = None       # p2p or p2p_local single matrix
    count_W = 0

    # For HSL, we store each stage's matrix
    W_hs_acc = None
    W_hh_acc = None
    W_sh_acc = None
    count_hsl = 0

    # For node degrees
    # We'll store them in a separate dictionary, to be saved in a separate .json
    degree_dict = {
        'round': []
    }
    if args.monitor_degree:
        degree_dict['p2p_indegree'] = []     # or for p2p_local
        degree_dict['p2p_outdegree'] = []
        # HSL stage1: outdegree of spokes
        degree_dict['hsl_stage1_spoke_out'] = []
        # HSL stage2: indegree of hubs
        degree_dict['hsl_stage2_hub_in'] = []
        # HSL stage3: outdegree of hubs
        degree_dict['hsl_stage3_hub_out'] = []

    ############################################################################
    # Helper function: compute drift among node models
    ############################################################################
    def compute_model_drift(spoke_wts_list):
        """
        spoke_wts_list: list of shape [num_spokes, d] (CPU Tensors)
        Returns: average distance from mean model
        """
        stack = torch.stack(spoke_wts_list, dim=0)  # shape [num_spokes, d]
        mean_model = torch.mean(stack, dim=0)
        diffs = stack - mean_model
        # Euclidean norm or L2
        dists = torch.norm(diffs, dim=1)  # shape [num_spokes]
        return torch.mean(dists).item()

    ############################################################################
    # If graph_simulation_only, we skip actual local training steps
    # and only simulate aggregator steps.
    ############################################################################

    if not args.graph_simulation_only:
        # Actual training loop
        for rnd in range(args.num_rounds):
            global_wts = utils.model_to_vec(global_model).cpu()
            node_return = {}

            # -------------------------
            # Local training (sequential for demo)
            # -------------------------
            for node_id in range(args.num_spokes):
                updated_wts = train_node.local_train_worker_inline(
                    node_id,
                    global_wts,
                    distributed_data[node_id],
                    distributed_label[node_id],
                    inp_dim, out_dim, net_name,
                    args.num_local_iters,
                    batch_size, args.lr,
                    aggregator_device,
                    args.sample_type,
                    rr_indices
                )
                node_return[node_id] = updated_wts

            # --------------------------------------------------------
            # (1) Monitor drift among node models (PRE-aggregation)
            # --------------------------------------------------------
            if args.monitor_model_drift and (rnd+1) % args.eval_time == 0:
                # node_return[i] are the updated local spoke models
                # Convert to CPU if not already
                spoke_wts_list = [node_return[i].cpu() for i in range(args.num_spokes)]
                drift_val = compute_model_drift(spoke_wts_list)
                metrics['pre_drift'].append(drift_val)

            # -------------------------
            # Aggregation step
            # -------------------------
            if args.aggregation == 'fedsgd':
                local_models = [node_return[i] for i in range(args.num_spokes)]
                aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
                global_model = utils.vec_to_model(aggregated_wts.to(aggregator_device),
                                                  net_name, inp_dim, out_dim, aggregator_device)

                # Evaluate every eval_time
                if (rnd+1) % args.eval_time == 0:
                    traced_model = torch.jit.trace(
                        deepcopy(global_model),
                        torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                    )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    metrics['round'].append(rnd+1)
                    metrics['global_loss'].append(loss)
                    metrics['global_acc'].append(acc)
                    print(f"[Round {rnd+1}] FedSGD => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

                # Post-drift for fedsgd? 
                if args.monitor_model_drift and (rnd+1) % args.eval_time == 0:
                    # After aggregator, all spokes have the same model = global_model
                    # => drift is zero. But we can log it anyway:
                    post_wts_list = [aggregated_wts.cpu() for _ in range(args.num_spokes)]
                    drift_val = compute_model_drift(post_wts_list)
                    metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p':
                # (a) Generate random k-regular graph
                W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
                # (b) Do p2p aggregation
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                new_spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)

                # (c) If we want to accumulate W:
                if W_accumulator is None:
                    W_accumulator = W.clone().cpu()
                else:
                    W_accumulator += W.clone().cpu()
                count_W += 1

                # Degrees if monitoring
                if args.monitor_degree:
                    # For p2p, outdegree is k for all, indegree also k, but after permutation
                    # We can read off the indegree from the sum of each column
                    indegs = (W.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                # (d) We pick the "global model" as new_spoke_wts[0]
                global_model = utils.vec_to_model(new_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

                # Evaluate every eval_time
                if (rnd+1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(new_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                        traced_model = torch.jit.trace(
                            deepcopy(node_model),
                            torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                        )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        local_accs.append(acc)
                        local_losses.append(loss)
                    metrics['round'].append(rnd+1)
                    metrics['local_acc'].append(local_accs)
                    metrics['local_loss'].append(local_losses)
                    print(f"[Round {rnd+1}] P2P => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

                    # Pre/Post drift
                    if args.monitor_model_drift:
                        # Post aggregator drift
                        spoke_wts_list = [new_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p_local':
                # node_return => local trained wts
                # aggregator returns updated spoke wts + mixing matrix if requested
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(
                    spoke_wts, args.k, return_W=True
                )
                # Accumulate mixing matrix
                if W_accumulator is None:
                    W_accumulator = W_local.clone().cpu()
                else:
                    W_accumulator += W_local.clone().cpu()
                count_W += 1

                # Degrees if monitoring
                if args.monitor_degree:
                    # outdegree is always k for each node
                    # indegree is how many times each node was chosen
                    indegs = (W_local.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W_local.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                global_model = utils.vec_to_model(updated_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

                if (rnd+1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(updated_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                        traced_model = torch.jit.trace(
                            deepcopy(node_model),
                            torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                        )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        local_accs.append(acc)
                        local_losses.append(loss)
                    metrics['round'].append(rnd+1)
                    metrics['local_acc'].append(local_accs)
                    metrics['local_loss'].append(local_losses)
                    print(f"[Round {rnd+1}] P2P Local => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

                    # Post aggregator drift
                    if args.monitor_model_drift:
                        spoke_wts_list = [updated_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'hsl':
                # spoke_wts => local post-training
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                # Step 1: Hubs gather from b_hs spokes
                hub_wts = []
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,),
                                                         device=aggregator_device)
                    hub_agg = spoke_wts[chosen_spoke_ids].mean(dim=0)
                    hub_wts.append(hub_agg)
                    # Fill stage1_matrix: row=hub, columns=spokes
                    for s_id in chosen_spoke_ids:
                        # hub receives from these spokes => from the spoke's perspective, outdegree?
                        stage1_matrix[hub_id, s_id] = 1.0
                hub_wts = torch.stack(hub_wts)  # shape [num_hubs, dim]
                # Convert stage1_matrix to row-stochastic
                # each row might have 'b_hs' ones
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                # Step 2: "EL Local" among hubs
                # aggregator returns updated hub wts + mixing matrix
                updated_hub_wts, stage2_matrix = aggregation.p2p_local_aggregation(hub_wts, args.b_hh, return_W=True)
                # shape of stage2_matrix => [num_hubs, num_hubs]

                # Step 3: Spokes gather from b_sh hubs
                final_spoke_wts = []
                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,),
                                                       device=aggregator_device)
                    spoke_agg = updated_hub_wts[chosen_hub_ids].mean(dim=0)
                    final_spoke_wts.append(spoke_agg)
                    # fill stage3_matrix
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                final_spoke_wts = torch.stack(final_spoke_wts)
                # row-stochastic for stage3 => each row sums to 1
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                # Accumulate stage1, stage2, stage3
                if W_hs_acc is None:
                    W_hs_acc = stage1_matrix.clone().cpu()
                    W_hh_acc = stage2_matrix.clone().cpu()
                    W_sh_acc = stage3_matrix.clone().cpu()
                else:
                    W_hs_acc += stage1_matrix.clone().cpu()
                    W_hh_acc += stage2_matrix.clone().cpu()
                    W_sh_acc += stage3_matrix.clone().cpu()
                count_hsl += 1

                # Monitor degrees
                if args.monitor_degree:
                    # stage1: outdegree of spokes is #hubs that pick them? Actually code picks spokes from hubs
                    # but user wants "outdegree of every spoke in stage 1 mixing" 
                    # We'll interpret that as how many hubs each spoke is connected to. 
                    # Actually stage1_matrix is [num_hubs x num_spokes], so transpose => [num_spokes x num_hubs]
                    # outdegree_of_spoke[i] = #hubs that used spoke i? -> that's the column sum in stage1_matrix
                    stage1_T = stage1_matrix.t()  # shape [num_spokes, num_hubs]
                    spoke_outdeg_stage1 = (stage1_T > 0).sum(dim=1).tolist()

                    # stage2: indegree of every hub = #hubs that connected to it
                    # stage2_matrix is [num_hubs x num_hubs], so indegree is column sum
                    hub_indeg_stage2 = (stage2_matrix > 0).sum(dim=0).tolist()

                    # stage3: outdegree of every hub => #spokes that pick it?
                    # stage3_matrix is [num_spokes x num_hubs], so for each hub, how many spokes picked it => column sum
                    hub_outdeg_stage3 = (stage3_matrix > 0).sum(dim=0).tolist()

                    degree_dict['hsl_stage1_spoke_out'].append(spoke_outdeg_stage1)
                    degree_dict['hsl_stage2_hub_in'].append(hub_indeg_stage2)
                    degree_dict['hsl_stage3_hub_out'].append(hub_outdeg_stage3)

                # Final global model is mean across final_spoke_wts
                global_model = utils.vec_to_model(final_spoke_wts.mean(dim=0),
                                                  net_name, inp_dim, out_dim, aggregator_device)

                # Evaluate every eval_time
                if (rnd+1) % args.eval_time == 0:
                    spoke_accs=[]
                    for s_id in range(args.num_spokes):
                        s_model = utils.vec_to_model(final_spoke_wts[s_id], net_name, inp_dim, out_dim, aggregator_device)
                        traced_model = torch.jit.trace(
                            deepcopy(s_model),
                            torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                        )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        spoke_accs.append(acc)
                    metrics['round'].append(rnd+1)
                    metrics['spoke_acc'].append(spoke_accs)
                    print(f"[Round {rnd+1}] HSL => Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

                    # Post aggregator drift
                    if args.monitor_model_drift:
                        spoke_wts_list = [final_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

        # End of training loop

    else:
        # Graph simulation only: we do NOT do local training
        for rnd in range(args.num_rounds):
            # We'll just assume an initial spoke_wts or something
            # The dimension doesn't matter, we won't do any real training
            # We just want to simulate aggregator so it samples mixing matrices
            node_return = {}
            # create dummy local wts
            # shape [num_spokes, d], let d=10 for example
            d = 10
            spoke_wts = torch.randn(args.num_spokes, d, device=aggregator_device)

            if args.aggregation == 'p2p':
                W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
                new_spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)
                if W_accumulator is None:
                    W_accumulator = W.clone().cpu()
                else:
                    W_accumulator += W.clone().cpu()
                count_W += 1

            elif args.aggregation == 'p2p_local':
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(
                    spoke_wts, args.k, return_W=True
                )
                if W_accumulator is None:
                    W_accumulator = W_local.clone().cpu()
                else:
                    W_accumulator += W_local.clone().cpu()
                count_W += 1

            elif args.aggregation == 'hsl':
                # step 1: hub gather from spokes
                # We have no hubs => actually we do have args.num_hubs
                # We'll produce random data
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,),
                                                         device=aggregator_device)
                    for s_id in chosen_spoke_ids:
                        stage1_matrix[hub_id, s_id] = 1.0
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                # step 2: hub-hub mixing
                hub_wts = torch.randn(args.num_hubs, d, device=aggregator_device)  # dummy
                _, stage2_matrix = aggregation.p2p_local_aggregation(hub_wts, args.b_hh, return_W=True)

                # step 3: spokes gather from hubs
                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,),
                                                       device=aggregator_device)
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                # accumulate
                if W_hs_acc is None:
                    W_hs_acc = stage1_matrix.clone().cpu()
                    W_hh_acc = stage2_matrix.clone().cpu()
                    W_sh_acc = stage3_matrix.clone().cpu()
                else:
                    W_hs_acc += stage1_matrix.clone().cpu()
                    W_hh_acc += stage2_matrix.clone().cpu()
                    W_sh_acc += stage3_matrix.clone().cpu()
                count_hsl += 1

        # end of graph simulation loop

    ############################################################################
    # After training (or simulation), compute average mixing matrices, spectral gap
    ############################################################################
    if args.aggregation == 'p2p' or args.aggregation == 'p2p_local':
        if count_W > 0 and W_accumulator is not None:
            W_avg = W_accumulator / float(count_W)
            # compute spectral gap = 1 - second-largest eigenvalue magnitude
            e = torch.linalg.eigvals(W_avg)
            e_abs = torch.abs(e)
            e_sorted, _ = torch.sort(e_abs, descending=True)
            spectral_gap = (1.0 - e_sorted[1]).item()
            # store in metrics
            metrics['avg_mixing_matrix'] = W_avg.tolist()  # convert to list
            metrics['spectral_gap'] = spectral_gap
            print(f"[Info] {args.aggregation} average spectral gap: {spectral_gap:.4f}")

    elif args.aggregation == 'hsl':
        if count_hsl > 0 and W_hs_acc is not None:
            W_hs_avg = W_hs_acc / float(count_hsl)
            W_hh_avg = W_hh_acc / float(count_hsl)
            W_sh_avg = W_sh_acc / float(count_hsl)
            # store them
            metrics['W_hs_avg'] = W_hs_avg.tolist()
            metrics['W_hh_avg'] = W_hh_avg.tolist()
            metrics['W_sh_avg'] = W_sh_avg.tolist()

            # effective mixing = W_sh_avg x W_hh_avg x W_hs_avg
            # (dimension checks: W_hs_avg => hubs x spokes, W_hh_avg => hubs x hubs, W_sh_avg => spokes x hubs)
            # So the product for a "spoke->spoke" matrix might be: 
            #   W_sh_avg * W_hh_avg * W_hs_avg
            #   dimension: (spokes x hubs) * (hubs x hubs) * (hubs x spokes) => (spokes x spokes)
            W_eff = torch.matmul(W_sh_avg, torch.matmul(W_hh_avg, W_hs_avg))
            e = torch.linalg.eigvals(W_eff)
            e_abs = torch.abs(e)
            e_sorted, _ = torch.sort(e_abs, descending=True)
            spectral_gap = (1.0 - e_sorted[1]).item()
            metrics['spectral_gap'] = spectral_gap
            metrics['W_eff'] = W_eff.tolist()
            print(f"[Info] HSL average spectral gap: {spectral_gap:.4f}")

    # save metrics
    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training (or simulation) complete.")

    # If we monitored degrees, save them in a separate JSON
    if args.monitor_degree:
        with open(filename + '_degree.json', 'w') as f:
            json.dump(degree_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    main(args)
