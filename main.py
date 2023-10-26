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

           
def main(args):

    start_time = time.time()
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
    net = utils.load_net(net_name, inp_dim, out_dim, device, seed=108) # Can pass seed here if desired.
    # TODO : spokes should use differential privacy in the first round
    nets = {}
    for i in range(num_spokes): nets[i] = utils.create_model(net, net_name, inp_dim, out_dim, seed=108)

    # TODO - compute avg weights if all models are different, else return net as the avg model
    past_avg_wts = utils.model_to_vec(net)
    byz = utils.load_attack(args.attack)
    if (args.attack != 'benign'): 
        if (args.mal_idx == None):
            mal_flag = utils.distribute_malicious(num_spokes, args.nbyz, 'random')
            mal_idx = np.where(mal_flag)[0]
            np.save(filename+'_mal_idx.npy', mal_idx)
        else:
            mal_idx = np.load(args.mal_idx)
            mal_flag = np.zeros(num_spokes)
            mal_flag[mal_idx] = 1
        avg_infc = torch.zeros(num_spokes).to(device)
        W_effective = torch.zeros((num_rounds, num_spokes, num_spokes))
    else: mal_flag = np.zeros(num_spokes)

    lamda = np.zeros(num_rounds)

    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(int(num_rounds/args.eval_time))
    local_test_acc = np.zeros((int(num_rounds/args.eval_time), num_spokes))
    predist = np.zeros(num_rounds)
    postdist = np.zeros(num_rounds)
    #Count the number of parameters
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()
    if (args.aggregation == 'fedsgd'): 
        fed_test_acc = np.zeros(int(num_rounds/args.eval_time))
        net_fed = utils.create_model(net, net_name, inp_dim, out_dim, seed=108)
    batch_idx = np.zeros(num_spokes)
   
    #if (args.self_wt != None): W = utils.self_wt(num_spokes, args.k, args.self_wt, device)
    #W = utils.k_regular_W(num_spokes, args.k, device)
    if (args.aggregation.find('p2p') != -1):
      if (args.W == None):
          if (args.W_type == 'franke_wolfe'):
            W = utils.franke_wolfe_p2p_graph(label_distr, 0.1, args.budget)
            nnbrs = [len(np.where(W[i])[0]) for i in range(len(W))]
            num_edges = (np.asarray(nnbrs).sum() - num_spokes)
            print("Franke-Wolfe generated %f directional edges" %num_edges)
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

    elif (args.aggregation.find('hsl') != -1):
      if (args.W_hs == None):
          #W_hs, W_sh = utils.random_graph(num_hubs, num_spokes, args.num_edges_hs, args.agr)
          W_hs, W_sh = utils.greedy_hubs(num_hubs, num_spokes, num_spoke_connections=1, label_distr=label_distr, map_type=args.map_type)
          W_hs, W_sh = W_hs.to(device), W_sh.to(device)
          torch.save(W_hs, filename+'_W_hs.pt')
          torch.save(W_sh, filename+'_W_sh.pt')
      else: 
          W_hs = torch.load(args.W_hs).to(device)
          W_sh = torch.load(args.W_sh).to(device)
      if (args.W_h == None):
          W_h = utils.connect_hubs(num_hubs)
          W_h = W_h.to(device)
          torch.save(W_h, filename+'_W_h.pt')
      else: 
          W_h = torch.load(args.W_h).to(device)
      if (args.g > 1 and args.nbyz == 0):
          W_g = W_h.clone()
          for i in range(args.g - 1): W_g = torch.matmul(W_h, W_g)
          W_h = W_g
    #W = utils.local_W(label_distr, device)
    #W = utils.k_regular_W(args.num_spokes, args.k, device)
    #W_hs, W_h, W_sh = utils.simulate_hsl_W(args.num_hubs, args.num_spokes, device)
    
    if (args.attack != 'benign'):
        if (args.aggregation == 'varsec_p2p' or args.aggregation == 'secure_p2p'):
            flag = 0
            real_cmax = np.zeros(num_spokes)
            capped = np.zeros(num_spokes)
            hist = np.zeros((num_spokes, num_spokes))
            local_cmax = np.ones(num_spokes)
            nnbrs = np.zeros(num_spokes)
            votes = np.zeros((num_rounds, num_spokes))
            infc = np.zeros((num_rounds, num_spokes))
            past_delta = np.zeros(num_spokes)
            c_lo = num_spokes * np.ones(num_spokes)
            c_hi = -1 * np.ones(num_spokes)
            metric_krum = np.zeros((num_rounds, num_spokes, 4))
            metric_vote = np.zeros((num_rounds, num_spokes, 4))
            for i in range(num_spokes):
                row_sum = len(torch.where(W[i]>0)[0])#int(W[i].sum().item())
                my_nbrs = torch.where(W[i])[0].cpu()
                nnbrs[i] = len(my_nbrs)

                if (args.aggregation == 'varsec_p2p'): local_cmax[i] = int(np.ceil((args.nbyz/num_spokes)*row_sum))
                if (args.aggregation == 'secure_p2p'): 
                    if (nnbrs[i]%2 == 0): local_cmax[i] = int((nnbrs[i]-2)/2)-1
                    else: local_cmax[i] = int((nnbrs[i]-2)/2)
                    if (local_cmax[i] == 0):
                        print("Node %d has %d neighbors, <=4 nbrs have 0 cmax" %(i, nnbrs[i]))
                        flag = 1
                    #local_cmax[i] = int(np.ceil((args.nbyz/num_spokes)*row_sum))
                if (i not in mal_idx): real_cmax[i] = len(np.intersect1d(my_nbrs, mal_idx))
                if (nnbrs[i] - 2 - 2*real_cmax[i] <= 1): #ideally zero but choosing 1
                    print("Some nodes do not have enough benign neighbors. Do you want to continue?")
                    flag = 1

            if (flag): pdb.set_trace()
            else: print("Graph topology passes security check")

    if (args.attack != 'benign' and args.aggregation == 'hsl'):
        flag = 0
        for i in range(len(W_hs)):
            nnbrs = len(torch.where(W_hs[i])[0].cpu())
            if (nnbrs <= 4):
                print("Hub %d has only %d spokes" %(i, nnbrs))
                flag = 1
        for i in range(len(W_h)):
            nnbrs = len(torch.where(W_h[i])[0].cpu())
            if (nnbrs <= 4):
                print("Hub %d has only %d neighbors" %(i, nnbrs))
                flag = 1
        for i in range(len(W_sh)):
            nnbrs = len(torch.where(W_sh[i])[0].cpu())
            if (nnbrs <= 4):
                print("Spoke %d has only %d hubs" %(i, nnbrs))
                flag = 1
        if (flag == 1):
            print("Do you want to continue?")
            pdb.set_trace()

    time_taken = time.time() - start_time
    mins = time_taken // 60
    secs = time_taken - mins*60
    print("Simulation setup in %d min %d sec." %(mins, secs))
    start_time = time.time()
    for rnd in range(num_rounds):
        spoke_wts = []
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
            optimizer = torch.optim.SGD(net_local.parameters(), lr=lr)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            #optimizer.zero_grad()
            
            #sample local dataset in a round-robin manner
            for local_iter in range(num_local_iters):
                optimizer.zero_grad()
                if (batch_idx[worker]+batch_size < distributed_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                    batch_idx[worker] = batch_idx[worker] + batch_size
                else: 
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),distributed_data[worker].shape[0]))) 
                    batch_idx[worker] = 0
                output = net_local(distributed_data[worker][minibatch].to(device))
                loss = criterion(output, distributed_label[worker][minibatch].to(device))
                if (not torch.isnan(loss)): #todo - check why loss can become nan
                    loss.backward()
                    optimizer.step()
                    #scheduler.step(loss)
            ##append all gradients in a list
            spoke_wts.append([x.detach() for x in net_local.parameters() if x.requires_grad != 'null'])
            #del net_local
            #torch.cuda.empty_cache()

        ###Do the aggregation

        spoke_wts = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in spoke_wts])
        avg_wts = torch.mean(spoke_wts, 0)
        avg_model = utils.vec_to_model(avg_wts, net_name, inp_dim, out_dim, device)
        if (rnd % args.eval_time == args.eval_time - 1) and (rnd > 0):
            save_cdist = args.save_cdist
        else:
            save_cdist = 0
        if (args.aggregation == 'p2p'):
            #W = wts.repeat((len(W), 1))
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd] = aggregation.p2p(device, rnd, args.dataset, args.g, W, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_flag, save_cdist) 
            #for i in range(num_spokes):
            #    nets[i] = utils.vec_to_model(global_wts[i], args.net, inp_dim, out_dim, torch.device('cuda'))
            #grad_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
            #net_fed = utils.vec_to_model(torch.matmul(wts, grad_list), args.net, inp_dim, num_outputs, torch.device('cuda'))

        elif (args.aggregation == 'hsl'):
            spoke_wts, predist[rnd], postdist[rnd] = aggregation.hsl(device, rnd, args.dataset, args.g, W_hs, W_h, W_sh, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_flag, save_cdist, args.threat_model)

        elif (args.aggregation == 'secure_p2p'):
            fmax = args.nbyz / args.num_spokes
            
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd], secure_W = aggregation.secure_p2p(device, rnd, args.dataset, args.g, W, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_flag, real_cmax, local_cmax, metric_krum, infc)
            W_effective[rnd] = secure_W.to('cpu')
            #if (rnd%args.eval_time ==0 and rnd > 0):
            #    r = rnd / args.eval_time
            #    avg_W = (avg_W*(r-1) + secure_W) / (r)
            #    contr = secure_W.sum(axis=0) - torch.diagonal(secure_W)
            #    avg_contr = avg_W.sum(axis=0) - torch.diagonal(avg_W)
            #    infc = secure_W[:][:,mal_idx].sum(axis=0)
            #    infc[mal_idx] = 0
            #    avg_infc = (avg_infc*(r-1) + infc) / (r)
            #    print("This round: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(contr[np.where(mal_flag==0)[0]].mean(), contr[np.where(mal_flag==1)[0]].mean(), infc.sum().item()))
            #    print("Running avg: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(avg_contr[np.where(mal_flag==0)[0]].mean(), avg_contr[np.where(mal_flag==1)[0]].mean(), avg_infc.sum().item()))
            

        elif (args.aggregation == 'varsec_p2p'):
            #fmax = args.nbyz / args.num_spokes
            
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd], secure_W, local_cmax, past_delta = aggregation.varsec_p2p(device, rnd, votes, args.g, W, past_avg_wts, spoke_wts, byz, lamda, mal_flag, local_cmax, real_cmax, capped, hist, args.th_lo, args.th_hi, infc, past_delta, c_lo, c_hi, args.p, metric_krum, metric_vote)
            #votes[rnd] = vote
            W_effective[rnd] = secure_W.to('cpu')
            #if (rnd%args.eval_time ==0 and rnd > 0):
            #    r = rnd / args.eval_time
            #    avg_W = (avg_W*(r-1) + secure_W) / (r)
            #    secure_W[mal_idx][:,mal_idx]=0
            #    avg_W[mal_idx][:,mal_idx]=0
            #    for i in range(num_spokes): secure_W[i][i] = 0
            #    pdb.set_trace()
            #    contr = secure_W.sum(axis=0)
            #    #avg_contr = avg_W.sum(axis=0)
            #    #pdb.set_trace()
            #    #contr = secure_W.sum(axis=0) - torch.diagonal(secure_W)
            #    avg_contr = avg_W.sum(axis=0) - torch.diagonal(avg_W)
            #    infc = secure_W[:][:,mal_idx].sum(axis=0)
            #    infc[mal_idx] = 0
            #    avg_infc = (avg_infc*(r-1) + infc) / (r)
            #    print("This round: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(contr[np.where(mal_flag==0)[0]].mean(), contr[np.where(mal_flag==1)[0]].mean(), infc.sum().item()))
            #    print("Running avg: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(avg_contr[np.where(mal_flag==0)[0]].mean(), avg_contr[np.where(mal_flag==1)[0]].mean(), avg_infc.sum().item()))

        elif (args.aggregation == 'secure_hsl'):
            spoke_wts, predist[rnd], postdist[rnd] = aggregation.secure_hsl(device, W_hs, W_h, W_sh, spoke_wts, byz, mal_flag)

        if (args.aggregation == 'fedsgd'):
            net_fed = utils.vec_to_model(torch.matmul(wts, spoke_wts), net_name, inp_dim, out_dim, torch.device('cuda'))
        else:
            for i in range(num_spokes):
               nets[i] = utils.vec_to_model(spoke_wts[i], net_name, inp_dim, out_dim, torch.device('cuda'))

        ##Evaluate the learned model on test dataset
        if(args.dataset == 'cifar10'): dummy_data_shape = [128,3,32,32]
        elif(args.dataset == 'mnist'): dummy_data_shape = [32,1,28,28]
        else: pdb.set_trace()
        if (rnd%args.eval_time == args.eval_time-1 and rnd>0):
            if (args.aggregation == 'fedsgd'):
                traced_model = torch.jit.trace(net_fed, torch.randn(dummy_data_shape).to(device))
                correct, total = 0, 0
                with torch.no_grad():
                    for data in test_data:
                        images, labels = data
                        outputs = traced_model(images.to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(device)).sum().item()
                fed_test_acc[int(rnd/args.eval_time)] = correct/total 
                print('Round: %d, test_acc: %.3f' %(rnd, fed_test_acc[int(rnd/args.eval_time)]))
            else:
                traced_models = []
                for i in range(num_spokes + 1):
                    if i < num_spokes:
                      traced_models.append(torch.jit.trace(nets[i], torch.randn(dummy_data_shape).to(device))) 
                    else:
                      traced_models.append(torch.jit.trace(avg_model, torch.randn(dummy_data_shape).to(device)))

                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data in test_data:
                            images, labels = data
                            outputs = traced_models[i](images.to(device))
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels.to(device)).sum().item()

                    if (i < num_spokes):
                      local_test_acc[int(rnd/args.eval_time)][i] = correct/total                
                    else:
                      test_acc[int(rnd/args.eval_time)] = correct/total
                if (save_cdist):
                    print ('Round: %d, predist: %.3f, postdist: %.6f, test_acc:[%.3f, %.3f]->%.3f' %(rnd, predist[rnd], postdist[rnd], min(local_test_acc[int(rnd/args.eval_time)]), max(local_test_acc[int(rnd/args.eval_time)]), test_acc[int(rnd/args.eval_time)]))      
                else:
                    print ('Round: %d, test_acc:[%.3f, %.3f]->%.3f' %(rnd, min(local_test_acc[int(rnd/args.eval_time)]), max(local_test_acc[int(rnd/args.eval_time)]), test_acc[int(rnd/args.eval_time)]))      
            if (args.attack != 'benign'): print("Avg ben acc: %.3f, avg mal acc: %.3f" %(local_test_acc[int(rnd/args.eval_time)][mal_flag==0].mean(), local_test_acc[int(rnd/args.eval_time)][mal_flag==1].mean()))
            time_taken = time.time() - start_time
            hrs = int(time_taken/3600)
            mins = int(int(time_taken/60) - hrs*60)
            secs = int(time_taken - hrs*3600 - mins*60)
            print("Time taken %d hr %d min %d sec" %(hrs, mins, secs))
        if ((rnd%args.save_time == args.save_time-1 and rnd>0) or (rnd == num_rounds-1)):
            if (args.aggregation == 'fedsgd'): np.save(filename+'_test_acc.npy', fed_test_acc)
            else:
                np.save(filename+'_local_test_acc.npy', local_test_acc)
                np.save(filename+'_consensus_test_acc.npy', test_acc)
                if (save_cdist):
                    np.save(filename+'_postdist.npy', postdist)
                    np.save(filename+'_predist.npy', predist)
                if (args.attack != 'benign' and (args.aggregation == 'secure_p2p' or args.aggregation == 'varsec_p2p')):
                    torch.save(W_effective, filename+'_W_eff.pt')
                    np.save(filename+'_lamda.npy', lamda)
                    np.save(filename+'_real_c.npy', real_cmax)
                    if (args.aggregation == 'varsec_p2p'): 
                        np.save(filename+'_votes.npy', votes)
                        np.save(filename+'_kmetric.npy', metric_krum)
                        np.save(filename+'_vmetric.npy', metric_vote)
                    if (args.aggregation == 'secure_p2p'):
                        np.save(filename+'_kmetric.npy', metric_krum)
                        
                    #torch.save(avg_infc, filename+'_avg_infc.pt')
                #torch.save(net.state_dict(), filename+'_model.pth')
if __name__ == "__main__":
    args = utils.parse_args()
    main(args)
