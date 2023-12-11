import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import sys
import pdb
from copy import deepcopy
import aggregation
import attack
import utils
import time

import json
           
def main(args):

    start_time = time.time()
    variables = {}
    num_spokes, num_hubs = args.num_spokes, args.num_hubs
    num_rounds, num_local_iters = args.num_rounds, args.num_local_iters

    if args.gpu == -1:  device = torch.device('cpu')
    else:  device = torch.device('cuda')
    filename = 'outputs/'+args.exp
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)

    # Load data.
    trainObject = utils.load_data(args.dataset)
    train_data = trainObject.train_data 
    test_data = trainObject.test_data
    inp_dim = trainObject.num_inputs
    out_dim = trainObject.num_outputs 
    net_name = trainObject.net_name
    lr_init = trainObject.lr
    batch_size = trainObject.batch_size 

    # Distribute data.
    distributedData = utils.distribute_data_by_label(device, batch_size, args.bias, train_data, num_spokes, inp_dim, out_dim, net_name)
    distributed_data = distributedData.distributed_input
    distributed_label = distributedData.distributed_output
    wts = distributedData.wts
    #wts = torch.ones(num_spokes).to(device) / num_spokes
    label_distr = utils.label_distr(distributed_label, out_dim)

    # Load models.
    net = utils.load_net(net_name, inp_dim, out_dim, device, seed=args.seed) 
    nets = {}
    for i in range(num_spokes): nets[i] = utils.create_model(net, net_name, inp_dim, out_dim, seed=args.seed)

    criterion = nn.CrossEntropyLoss()

    num_time_samples = int(num_rounds / args.eval_time)
    test_acc = np.empty(num_time_samples)
    local_test_acc = np.zeros((num_time_samples, num_spokes))
    predist = np.zeros(num_time_samples)
    postdist = np.zeros(num_time_samples)
    #Count the number of parameters
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()

    if (args.aggregation == 'fedsgd'): 
        fed_test_acc = np.zeros(num_time_samples)
        fed_test_acc_top5 = np.zeros(num_time_samples)
        fed_train_loss = np.zeros((num_rounds, num_spokes))
        net_fed = utils.create_model(net, net_name, inp_dim, out_dim, seed=args.seed)

    batch_idx = np.zeros(num_spokes)
   
    if (args.aggregation == 'p2p'):
      if (args.W == None):
          if (args.W_type == 'franke_wolfe'):
            W, num_connections, num_directed_edges = utils.franke_wolfe_p2p_graph(label_distr, 0.1, args.max_degree)
            variables['num_connections'] = str(num_connections)
            variables['num_directed_edges'] = str(num_directed_edges)
            print("Franke-Wolfe generated %f directional edges" %num_directed_edges)
          elif (args.W_type == 'erdos_renyi'):
            W = utils.erdos_renyi(num_spokes, args.num_edges, args.seed)
          elif (args.W_type == 'random_graph'):
            W = utils.random_graph(num_spokes, num_spokes, args.num_edges, args.agr) #choices: mean, random_static, adjacency_static
            #W = utils.fully_connect(num_spokes)
          elif (args.W_type == 'self_wt'):
            W = utils.self_wt(num_spokes, num_spokes, args.self_wt, device)
          elif (args.W_type == 'ring'):
            W = utils.connect_hubs(num_spokes)
          W = W.to(device)
          torch.save(W, filename+'_W.pt')
      else: 
          W = torch.load(args.W).to(device)

    elif (args.aggregation == 'hsl'):
      if (args.W_hs == None):
          #W_hs, W_sh = utils.random_graph(num_hubs, num_spokes, args.num_edges_hs, args.agr)
          W_hs, W_sh, hub_to_spokes, spoke_to_hubs = utils.greedy_hubs(num_hubs, num_spokes, num_spoke_connections=args.spoke_budget, label_distr=label_distr, map_type=args.map_type)
          num_spoke_connections = args.spoke_budget
          W_hs, W_sh = W_hs.to(device), W_sh.to(device)
          torch.save(W_hs, filename+'_W_hs.pt')
          torch.save(W_sh, filename+'_W_sh.pt')
      else: 
          W_hs = torch.load(args.W_hs).to(device)
          W_sh = torch.load(args.W_sh).to(device)
      if (args.W_h == None):
          W_h = utils.connect_hubs(num_hubs, args.k)
          W_h = W_h.to(device)
          torch.save(W_h, filename+'_W_h.pt')
      else: 
          W_h = torch.load(args.W_h).to(device)
      if (args.g > 1 and args.nbyz == 0):
          W_g = W_h.clone()
          for i in range(args.g - 1): W_g = torch.matmul(W_h, W_g)
          W_h = W_g
    
    attack_flag = 1
    nodes_attacked = [0]




    time_taken = time.time() - start_time
    mins = time_taken // 60
    secs = time_taken - mins*60
    print("Simulation setup in %d min %d sec." %(mins, secs))
    max_accuracy = -1
    start_time = time.time()
    for rnd in range(num_rounds):
        spoke_wts = []
        this_iter_minibatches = []
        if (args.dataset == 'cifar10'):
            lr = utils.get_lr(rnd, num_rounds, lr_init)
        else: lr = lr_init
        #lr = lr_init
        for worker in range(num_spokes):
            if (args.aggregation == 'fedsgd'):
                net_local = deepcopy(net_fed)
            else: 
                net_local = deepcopy(nets[worker]) 
            net_local.train()
            if (args.dataset == 'imagenet'):
                optimizer = torch.optim.Adam(net_local.parameters())
            elif (args.dataset == 'cifar10'):
                optimizer = torch.optim.SGD(net_local.parameters(), lr=lr) #, momentum=0.9, weight_decay=5e-4)
                #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15)
            else: 
                optimizer = torch.optim.SGD(net_local.parameters(), lr=lr)
            
            #sample local dataset in a round-robin manner
            avg_loss = 0
            for local_iter in range(num_local_iters):
                optimizer.zero_grad()
                if (batch_idx[worker]+batch_size < distributed_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                    batch_idx[worker] = batch_idx[worker] + batch_size
                else: 
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),distributed_data[worker].shape[0]))) 
                    batch_idx[worker] = batch_size - len(minibatch)
                    minibatch = np.concatenate((minibatch, np.asarray(list(range(int(batch_idx[worker]))))))
                this_iter_minibatches.append(minibatch)
                output = net_local(distributed_data[worker][minibatch].to(device))
                loss = criterion(output, distributed_label[worker][minibatch].to(device))
                if (not torch.isnan(loss)): #todo - check why loss can become nan
                    loss.backward()
                    optimizer.step()
                avg_loss += loss.item()
            avg_loss /= num_local_iters
            if (args.aggregation == 'fedsgd'):
                fed_train_loss[rnd][worker] = avg_loss

            ##append all gradients in a list
            spoke_wts.append([x.detach() for x in net_local.parameters() if x.requires_grad != 'null'])
            del net_local
            torch.cuda.empty_cache()
        ###Do the aggregation

        spoke_wts = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in spoke_wts])
        avg_wts = torch.mean(spoke_wts, 0)
        avg_model = utils.vec_to_model(avg_wts, net_name, inp_dim, out_dim, device)
        if (rnd % args.eval_time == args.eval_time - 1) and (rnd > 0):
            save_cdist = args.save_cdist
        else:
            save_cdist = 0
        if (args.aggregation == 'p2p'):
            spoke_wts, predist_val, postdist_val = aggregation.p2p(device, rnd, args.dataset, args.g, W, spoke_wts, save_cdist) 
            r = int(rnd / args.eval_time)
            predist[r], postdist[r] = predist_val, postdist_val
             

        elif (args.aggregation == 'hsl'):
            spoke_wts, predist_val, postdist_val = aggregation.hsl(device, rnd, args.dataset, args.g, W_hs, W_h, W_sh, spoke_wts, save_cdist)
            r = int(rnd / args.eval_time)
            predist[r], postdist[r] = predist_val, postdist_val

        if (args.aggregation == 'fedsgd'):
            if (attack_flag):
                recon_acc = attack.gradInv(rnd, net_fed, spoke_wts, nodes_attacked, distributed_data, distributed_label, this_iter_minibatches, torch.device('cuda'))
            net_fed = utils.vec_to_model(torch.matmul(wts, spoke_wts), net_name, inp_dim, out_dim, torch.device('cuda'))

        else:
            for i in range(num_spokes):
               nets[i] = utils.vec_to_model(spoke_wts[i], net_name, inp_dim, out_dim, torch.device('cuda'))

        ##Evaluate the learned model on test dataset
        if(args.dataset == 'cifar10'): dummy_data_shape = [128,3,32,32]
        elif(args.dataset == 'mnist'): dummy_data_shape = [32,1,28,28]
        elif (args.dataset == 'imagenet'): dummy_data_shape = [256, 3, 64, 64]
        else: pdb.set_trace()
        if (rnd%args.eval_time == args.eval_time-1 and rnd>0):
            if (args.aggregation == 'fedsgd'):
                var = num_spokes
                traced_model = torch.jit.trace(net_fed, torch.randn(dummy_data_shape).to(device))
                correct, total = 0, 0
                correct_top5 = 0
                with torch.no_grad():
                    for data in test_data:
                        images, labels = data
                        outputs = traced_model(images.to(device))
                        if (args.dataset == 'mnist' or args.dataset == 'cifar10'):
                            _, predicted = torch.max(outputs.data, 1)

                            correct += (predicted == labels.to(device)).sum().item()
                        elif (args.dataset == 'imagenet'):
                            top_indices = torch.topk(outputs.data, 5)[1].t()
                            matched = top_indices.eq(labels.to(device).view(1, -1).expand_as(top_indices))
                            correct_top5 += matched.float().sum().item()
                            _, predicted = torch.max(outputs.data, 1)
                            correct += (predicted == labels.to(device)).sum().item()
                            
                        total += labels.size(0)
                fed_test_acc_top5[int(rnd/args.eval_time)] = correct_top5/total 
                fed_test_acc[int(rnd/args.eval_time)] = correct/total
                if (correct/total > max_accuracy):
                    best_model = deepcopy(net_fed)
                    max_accuracy = correct / total
                print('Round: %d, test_acc: %.8f' %(rnd, fed_test_acc[int(rnd/args.eval_time)]))
            else:
                traced_models = []
                #Below code is optimized to avoid redundant calculation for spokes under the same hub, and general enough to accommodate p2p in the same block: see smart use of 'var'
                if (args.aggregation == 'p2p' or args.spoke_budget>1): var = num_spokes
                #TODO - fix the if else loop to cover all cases
                elif (args.aggregation == 'hsl' and args.spoke_budget==1): var = num_hubs
                for i in range(var + 1):
                    if i < var:
                      if (args.aggregation == 'hsl' and args.spoke_budget==1):
                          spoke_id = hub_to_spokes[i][0]
                      elif (args.aggregation == 'p2p' or args.spoke_budget>1):
                          spoke_id = i
                      traced_models.append(torch.jit.trace(nets[spoke_id], torch.randn(dummy_data_shape).to(device))) 
                    else:
                      traced_models.append(torch.jit.trace(avg_model, torch.randn(dummy_data_shape).to(device)))


                    acc = utils.compute_test_accuracy(traced_models[i], test_data, device)
                    if (i < var):
                      local_test_acc[int(rnd/args.eval_time)][spoke_id] = acc                
                      if (args.aggregation == 'hsl' and args.spoke_budget==1):
                        for spoke_id in hub_to_spokes[i][1:]:
                          local_test_acc[int(rnd/args.eval_time)][spoke_id] = acc                
                    else:
                      test_acc[int(rnd/args.eval_time)] = acc
                if (save_cdist):
                    print ('Round: %d, predist: %.8f, postdist: %.8f, test_acc:[%.3f, %.3f]->%.3f' %(rnd, predist[int(rnd/args.eval_time)], postdist[int(rnd/args.eval_time)], min(local_test_acc[int(rnd/args.eval_time)]), max(local_test_acc[int(rnd/args.eval_time)]), test_acc[int(rnd/args.eval_time)]))      
                else:
                    print ('Round: %d, test_acc:[%.3f, %.3f]->%.3f' %(rnd, min(local_test_acc[int(rnd/args.eval_time)]), max(local_test_acc[int(rnd/args.eval_time)]), test_acc[int(rnd/args.eval_time)]))      
            time_taken = time.time() - start_time
            hrs = int(time_taken/3600)
            mins = int(int(time_taken/60) - hrs*60)
            secs = int(time_taken - hrs*3600 - mins*60)
            print("Time taken %d hr %d min %d sec" %(hrs, mins, secs))
        if ((rnd%args.save_time == args.save_time-1 and rnd>0) or (rnd == num_rounds-1)):
            if (args.aggregation == 'fedsgd'): 
                np.save(filename+'_test_acc.npy', fed_test_acc)
                np.save(filename+'_train_loss.npy', fed_train_loss)
                torch.save(net_fed.state_dict(), filename+'_model.pth')
                #torch.save(best_model.state_dict(), filename+'_best_model.pth')
                if (args.dataset == 'imagenet'):
                    np.save(filename+'_top5_test_acc.npy', fed_test_acc_top5)
            else:
                np.save(filename+'_local_test_acc.npy', local_test_acc)
                np.save(filename+'_consensus_test_acc.npy', test_acc)
                if (save_cdist):
                    np.save(filename+'_postdist.npy', postdist)
                    np.save(filename+'_predist.npy', predist)
                    torch.save(avg_model.state_dict(), filename+'_model.pth')
        save_args = dict(args.__dict__).copy()
        save_args.update(variables)
        with open(filename + '_args.txt', 'w') as f:
            json.dump(save_args, f, indent=2)

if __name__ == "__main__":
    args = utils.parse_args()
    main(args)
