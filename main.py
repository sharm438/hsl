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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment')
    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--aggregation", type=str, default='fedsgd', choices=['fedsgd', 'p2p', 'hsl'])
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
    parser.add_argument("--bidirectional", type=int, default=0, help="1 means spokes choose same hubs second time")
    parser.add_argument("--self_weights", type=int, default=0, help="1 means spokes include self weight in final aggregation")

    return parser.parse_args()

def main(args):
    aggregator_device = torch.device('cuda:'+str(args.gpu) if args.gpu>=0 and torch.cuda.is_available() else 'cpu')
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    trainObject = utils.load_data(args.dataset, args.batch_size, args.fraction)
    data, labels = trainObject.train_data, trainObject.train_labels
    test_data = trainObject.test_data
    inp_dim = trainObject.num_inputs
    out_dim = trainObject.num_outputs
    lr = args.lr
    batch_size = trainObject.batch_size
    net_name = trainObject.net_name

    distributedData = utils.distribute_data_noniid((data, labels), args.num_spokes, args.bias, inp_dim, out_dim, net_name, aggregator_device)
    distributed_data = distributedData.distributed_input
    distributed_label = distributedData.distributed_output
    node_weights = distributedData.wts

    # Global model on aggregator_device
    global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    manager = mp.Manager()
    rr_indices = manager.dict()
    for node_id in range(args.num_spokes):
        ds_size = distributed_data[node_id].shape[0]
        if ds_size > 0:
            distributed_data[node_id] = distributed_data[node_id].cpu()
            distributed_label[node_id] = distributed_label[node_id].cpu()
            perm = torch.randperm(ds_size)
            distributed_data[node_id] = distributed_data[node_id][perm]
            distributed_label[node_id] = distributed_label[node_id][perm]
        rr_indices[node_id] = 0

    metrics = {
        'round': [],
        'global_acc': [],
        'global_loss': [],
        'local_acc': [],
        'local_loss': [],
        'spoke_acc': []
    }

    return_dict = manager.dict()

    for rnd in range(args.num_rounds):
        global_wts = utils.model_to_vec(global_model).cpu()
        node_return = manager.dict()
        processes = []

        # Run local training on CPU
        for node_id in range(args.num_spokes):
            p = mp.Process(target=train_node.local_train_worker,
                           args=(node_id, global_wts, distributed_data[node_id], distributed_label[node_id],
                                 inp_dim, out_dim, net_name, args.num_local_iters, batch_size, lr, 'cpu',
                                 args.sample_type, rr_indices, node_return))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if args.aggregation == 'fedsgd':
            local_models = [node_return[i] for i in range(args.num_spokes)]
            aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
            global_model = utils.vec_to_model(aggregated_wts.to(aggregator_device), net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1) % args.eval_time == 0:
                traced_model = torch.jit.trace(deepcopy(global_model),
                                               torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                metrics['round'].append(rnd+1)
                metrics['global_loss'].append(loss)
                metrics['global_acc'].append(acc)
                print(f"Round {rnd+1}, Global Acc: {acc:.4f}, Global Loss: {loss:.4f}")

        elif args.aggregation == 'p2p':
            W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)])  # on CPU
            spoke_wts = spoke_wts.to(aggregator_device)  # move to aggregator_device
            spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)
            global_model = utils.vec_to_model(spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1)%args.eval_time==0:
                local_accs, local_losses = [], []
                for node_id in range(args.num_spokes):
                    node_model = utils.vec_to_model(spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(deepcopy(node_model),
                                                   torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    local_accs.append(acc)
                    local_losses.append(loss)
                metrics['round'].append(rnd+1)
                metrics['local_acc'].append(local_accs)
                metrics['local_loss'].append(local_losses)
                print(f"Round {rnd+1}, Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

        elif args.aggregation == 'hsl':
            # Step 1: W_hs & initial_hub_choices
            W_h, W_hs, initial_spoke_hub_choices = utils.generate_hsl_matrices_initial(args.num_hubs, args.num_spokes, args.k, args.spoke_budget, aggregator_device)

            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]) # CPU
            spoke_wts = spoke_wts.to(aggregator_device)

            # Hub aggregation:
            # hubs first aggregate from spokes: hub_wts = W_hs * spoke_wts
            # hubs aggregate among themselves: hub_wts = W_h * hub_wts
            hub_wts = torch.mm(W_hs, spoke_wts)
            hub_wts = torch.mm(W_h, hub_wts)

            b_s = args.spoke_budget

            if args.self_weights == 0:
                # If self_weights=0, we can use W_sh to aggregate to spokes
                W_sh = utils.generate_hsl_matrices_final(args.num_hubs, args.num_spokes, b_s, aggregator_device,
                                                         initial_spoke_hub_choices, args.bidirectional, args.self_weights)
                final_spoke_wts = torch.mm(W_sh, hub_wts)
                # No self addition needed
            else:
                # self_weights=1
                # We do NOT multiply W_sh with hub_wts here.
                # Instead, each spoke picks hubs (bidirectional logic) and averages:
                final_spoke_hub_choices = utils.get_final_spoke_hub_choices(args.num_hubs, args.num_spokes,
                                                                           b_s, aggregator_device,
                                                                           initial_spoke_hub_choices,
                                                                           args.bidirectional)
                # final_spoke_wts: for each spoke:
                # (spoke_wts[s] + sum_of_chosen_hub_wts)/ (b_s+1)
                final_spoke_wts = torch.zeros_like(spoke_wts)
                for s in range(args.num_spokes):
                    chosen_hubs = final_spoke_hub_choices[s]
                    chosen_hub_models = hub_wts[chosen_hubs].mean(dim=0) if len(chosen_hubs)>0 else 0
                    # Actually we want exact average including self: (self + sum_of chosen hubs)/ (b_s+1)
                    # If we must average chosen hubs exactly rather than their mean:
                    # sum_of_chosen = hub_wts[chosen_hubs].sum(dim=0)
                    # final = (spoke_wts[s] + sum_of_chosen)/(b_s+1)
                    # But chosen_hubs length = b_s, so mean = sum/b_s
                    # sum_of_chosen = chosen_hub_models * b_s
                    # final = (spoke_wts[s] + chosen_hub_models * b_s)/(b_s+1)
                    # This yields the same result as summation form.
                    sum_of_chosen = chosen_hub_models * b_s
                    final_spoke_wts[s] = (spoke_wts[s] + sum_of_chosen)/(b_s+1)

            global_model = utils.vec_to_model(torch.mean(final_spoke_wts, dim=0), net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1)%args.eval_time==0:
                spoke_accs=[]
                for s_id in range(args.num_spokes):
                    s_model = utils.vec_to_model(final_spoke_wts[s_id], net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(deepcopy(s_model),
                                                   torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    spoke_accs.append(acc)
                metrics['round'].append(rnd+1)
                metrics['spoke_acc'].append(spoke_accs)
                print(f"Round {rnd+1}, Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
