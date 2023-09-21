import argparse
import torch
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

#Learning rate scheduler used to train CIFAR-10

           
def main(args):
    
    nspokes = args.nspokes
    nhubs = args.nhubs
    nrounds = args.nrounds
    niters = args.niters

    if args.gpu == -1:  device = torch.device('cpu')
    else:  device = torch.device('cuda')

    filename = 'outputs/'+args.exp

    ###Load dataset
    train_data, test_data, inp_dim, out_dim, net_name, lr_init, batch_size = utils.load_data(args.dataset)
    each_worker_data, each_worker_label, wts = utils.distribute_data_fang(device, batch_size, args.bias, train_data, nspokes, inp_dim, out_dim, net_name)
    label_distr = utils.label_distr(each_worker_label, out_dim)

    ####Load models
    net = utils.load_net(net_name, inp_dim, out_dim, device)
    nets = {}
    for i in range(nspokes): nets[i] = utils.create_model(net, net_name, inp_dim, out_dim)

    #past_avg_wts = utils.model_to_vec(net)
    byz = utils.load_attack(args.attack)
    if (args.attack != 'benign'): 
        if (args.mal == None):
            mal_idx = utils.distribute_malicious(nspokes, args.nbyz, 'random')
            mal = np.where(mal_idx)[0]
            np.save(filename+'_mal.npy', mal)
        else:
            mal = np.load(args.mal)
            mal_idx = np.zeros(nspokes)
            mal_idx[mal] = 1
        avg_infc = torch.zeros(nspokes).to(device)
        W_effective = torch.zeros((nrounds, nspokes, nspokes))
    else: mal_idx = np.zeros(nspokes)

    #lamda = np.zeros(nrounds)

    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(int(nrounds/args.eval_time))
    local_test_acc = np.zeros((int(nrounds/args.eval_time), nspokes))
    predist = np.zeros(nrounds)
    postdist = np.zeros(nrounds)
    #Count the number of parameters
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()
    if (args.aggregation == 'fedsgd'): 
        fed_test_acc = np.zeros(int(nrounds/args.eval_time))
        net_fed = utils.create_model(net, net_name, inp_dim, out_dim)
    batch_idx = np.zeros(nspokes)
   
    #if (args.self_wt != None): W = utils.self_wt(nspokes, args.k, args.self_wt, device)
    #W = utils.k_regular_W(nspokes, args.k, device)
    if (args.aggregation.find('p2p') != -1):
      if (args.W == None):
          W = utils.random_graph(nspokes, nspokes, args.nedges, args.agr) #choices: mean, random_static, adjacency_static
          W = W.to(device)
          torch.save(W, filename+'_W.pt')
      else: 
          W = torch.load(args.W).to(device)

    elif (args.aggregation.find('hsl') != -1):
      if (args.W_hs == None):
          W_hs, W_sh = utils.random_graph(nhubs, nspokes, args.nedges_hs, args.agr)
          W_hs, W_sh = W_hs.to(device), W_sh.to(device)
          torch.save(W_hs, filename+'_W_hs.pt')
          torch.save(W_sh, filename+'_W_sh.pt')
      else: 
          W_hs = torch.load(args.W_hs).to(device)
          W_sh = torch.load(args.W_sh).to(device)
      if (args.W_h == None):
          W_h = utils.random_graph(nhubs, nhubs, args.nedges_h, args.agr)
          W_h = W_h.to(device)
          torch.save(W_h, filename+'_W_h.pt')
      else: 
          W_h = torch.load(args.W_h).to(device)
      if (args.g > 1 and args.nbyz == 0):
          W_g = W_h.clone()
          for i in range(args.g - 1): W_g = torch.matmul(W_h, W_g)
          W_h = W_g
    #W = utils.local_W(label_distr, device)
    #W = utils.k_regular_W(args.nspokes, args.k, device)
    #W_hs, W_h, W_sh = utils.simulate_hsl_W(args.nhubs, args.nspokes, device)
    
    if (args.attack != 'benign'):
        if (args.aggregation == 'varsec_p2p' or args.aggregation == 'secure_p2p'):
            flag = 0
            real_cmax = np.zeros(nspokes)
            capped = np.zeros(nspokes)
            hist = np.zeros((nspokes, nspokes))
            local_cmax = np.ones(nspokes)
            nnbrs = np.zeros(nspokes)
            votes = np.zeros((nrounds, nspokes))
            infc = np.zeros((nrounds, nspokes))
            past_delta = np.zeros(nspokes)
            c_lo = nspokes * np.ones(nspokes)
            c_hi = -1 * np.ones(nspokes)
            metric_krum = np.zeros((nrounds, nspokes, 4))
            metric_vote = np.zeros((nrounds, nspokes, 4))
            for i in range(nspokes):
                row_sum = len(torch.where(W[i]>0)[0])#int(W[i].sum().item())
                my_nbrs = torch.where(W[i])[0].cpu()
                nnbrs[i] = len(my_nbrs)

                if (args.aggregation == 'varsec_p2p'): local_cmax[i] = int(np.ceil((args.nbyz/nspokes)*row_sum))
                if (args.aggregation == 'secure_p2p'): 
                    if (nnbrs[i]%2 == 0): local_cmax[i] = int((nnbrs[i]-2)/2)-1
                    else: local_cmax[i] = int((nnbrs[i]-2)/2)
                    if (local_cmax[i] == 0):
                        print("Node %d has %d neighbors, <=4 nbrs have 0 cmax" %(i, nnbrs[i]))
                        flag = 1
                    #local_cmax[i] = int(np.ceil((args.nbyz/nspokes)*row_sum))
                if (i not in mal): real_cmax[i] = len(np.intersect1d(my_nbrs, mal))
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

    for rnd in range(nrounds):
        spoke_wts = []
        if (args.dataset == 'cifar10'):
            lr = utils.get_lr(rnd, nrounds, lr_init)
        for worker in range(nspokes):
            if (args.aggregation == 'fedsgd'):
                net_local = deepcopy(net_fed)
            else: 
                net_local = deepcopy(nets[worker]) 
            net_local.train()
            optimizer = torch.optim.SGD(net_local.parameters(), lr=lr)
            #optimizer.zero_grad()
            
            #sample local dataset in a round-robin manner
            for local_iter in range(niters):
                optimizer.zero_grad()
                if (batch_idx[worker]+batch_size < each_worker_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                    batch_idx[worker] = batch_idx[worker] + batch_size
                else: 
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),each_worker_data[worker].shape[0]))) 
                    batch_idx[worker] = 0
                output = net_local(each_worker_data[worker][minibatch].to(device))
                loss = criterion(output, each_worker_label[worker][minibatch].to(device))
                if (not torch.isnan(loss)): #todo - check why loss can become nan
                    loss.backward()
                    optimizer.step()
            ##append all gradients in a list
            spoke_wts.append([x.detach() for x in net_local.parameters() if x.requires_grad != 'null'])
            #del net_local
            #torch.cuda.empty_cache()

        ###Do the aggregation

        spoke_wts = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in spoke_wts])

        if (args.aggregation == 'p2p'):
            #W = wts.repeat((len(W), 1))
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd] = aggregation.p2p(device, rnd, args.dataset, args.g, W, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_idx, args.save_cdist) 
            #for i in range(nspokes):
            #    nets[i] = utils.vec_to_model(global_wts[i], args.net, inp_dim, out_dim, torch.device('cuda'))
            #grad_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
            #net_fed = utils.vec_to_model(torch.matmul(wts, grad_list), args.net, inp_dim, num_outputs, torch.device('cuda'))

        elif (args.aggregation == 'hsl'):
            spoke_wts, predist[rnd], postdist[rnd] = aggregation.hsl(device, rnd, args.dataset, args.g, W_hs, W_h, W_sh, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_idx, args.save_cdist, args.threat_model)

        elif (args.aggregation == 'secure_p2p'):
            fmax = args.nbyz / args.nspokes
            
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd], secure_W = aggregation.secure_p2p(device, rnd, args.dataset, args.g, W, past_avg_wts, spoke_wts, byz, args.attack_prob, lamda, mal_idx, real_cmax, local_cmax, metric_krum, infc)
            W_effective[rnd] = secure_W.to('cpu')
            #if (rnd%args.eval_time ==0 and rnd > 0):
            #    r = rnd / args.eval_time
            #    avg_W = (avg_W*(r-1) + secure_W) / (r)
            #    contr = secure_W.sum(axis=0) - torch.diagonal(secure_W)
            #    avg_contr = avg_W.sum(axis=0) - torch.diagonal(avg_W)
            #    infc = secure_W[:][:,mal].sum(axis=0)
            #    infc[mal] = 0
            #    avg_infc = (avg_infc*(r-1) + infc) / (r)
            #    print("This round: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(contr[np.where(mal_idx==0)[0]].mean(), contr[np.where(mal_idx==1)[0]].mean(), infc.sum().item()))
            #    print("Running avg: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(avg_contr[np.where(mal_idx==0)[0]].mean(), avg_contr[np.where(mal_idx==1)[0]].mean(), avg_infc.sum().item()))
            

        elif (args.aggregation == 'varsec_p2p'):
            #fmax = args.nbyz / args.nspokes
            
            past_avg_wts, spoke_wts, predist[rnd], postdist[rnd], secure_W, local_cmax, past_delta = aggregation.varsec_p2p(device, rnd, votes, args.g, W, past_avg_wts, spoke_wts, byz, lamda, mal_idx, local_cmax, real_cmax, capped, hist, args.th_lo, args.th_hi, infc, past_delta, c_lo, c_hi, args.p, metric_krum, metric_vote)
            #votes[rnd] = vote
            W_effective[rnd] = secure_W.to('cpu')
            #if (rnd%args.eval_time ==0 and rnd > 0):
            #    r = rnd / args.eval_time
            #    avg_W = (avg_W*(r-1) + secure_W) / (r)
            #    secure_W[mal][:,mal]=0
            #    avg_W[mal][:,mal]=0
            #    for i in range(nspokes): secure_W[i][i] = 0
            #    pdb.set_trace()
            #    contr = secure_W.sum(axis=0)
            #    #avg_contr = avg_W.sum(axis=0)
            #    #pdb.set_trace()
            #    #contr = secure_W.sum(axis=0) - torch.diagonal(secure_W)
            #    avg_contr = avg_W.sum(axis=0) - torch.diagonal(avg_W)
            #    infc = secure_W[:][:,mal].sum(axis=0)
            #    infc[mal] = 0
            #    avg_infc = (avg_infc*(r-1) + infc) / (r)
            #    print("This round: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(contr[np.where(mal_idx==0)[0]].mean(), contr[np.where(mal_idx==1)[0]].mean(), infc.sum().item()))
            #    print("Running avg: ben contr = %.3f, mal contr = %.3f, infc = %.3f" %(avg_contr[np.where(mal_idx==0)[0]].mean(), avg_contr[np.where(mal_idx==1)[0]].mean(), avg_infc.sum().item()))

        elif (args.aggregation == 'secure_hsl'):
            spoke_wts, predist[rnd], postdist[rnd] = aggregation.secure_hsl(device, W_hs, W_h, W_sh, spoke_wts, byz, mal_idx)

        if (args.aggregation == 'fedsgd'):
            net_fed = utils.vec_to_model(torch.matmul(wts, spoke_wts), net_name, inp_dim, out_dim, torch.device('cuda'))
        else:
            for i in range(nspokes):
               nets[i] = utils.vec_to_model(spoke_wts[i], net_name, inp_dim, out_dim, torch.device('cuda'))

        ##Evaluate the learned model on test dataset
        if(args.dataset == 'cifar10'): dummy_data_shape = [128,3,32,32]
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
                print('Iteration: %d, test_acc: %.3f\n' %(rnd, fed_test_acc[int(rnd/args.eval_time)]))
            else:
                traced_models = []
                for i in range(nspokes):
                    traced_models.append(torch.jit.trace(nets[i], torch.randn(dummy_data_shape).to(device))) 
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data in test_data:
                            images, labels = data
                            outputs = traced_models[i](images.to(device))
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels.to(device)).sum().item()

                    local_test_acc[int(rnd/args.eval_time)][i] = correct/total                
                print ('Iteration: %d, predist: %.3f, postdist: %.3f, test_acc:[%.3f, %.3f]\n' %(rnd, predist[rnd], postdist[rnd], min(local_test_acc[int(rnd/args.eval_time)]), max(local_test_acc[int(rnd/args.eval_time)])))      
            if (args.attack != 'benign'): print("Avg ben acc: %.3f, avg mal acc: %.3f" %(local_test_acc[int(rnd/args.eval_time)][mal_idx==0].mean(), local_test_acc[int(rnd/args.eval_time)][mal_idx==1].mean()))
        if ((rnd%args.save_time == args.save_time-1 and rnd>0) or (rnd == nrounds-1)):
            if (args.aggregation == 'fedsgd'): np.save(filename+'_test_acc.npy', fed_test_acc)
            else:
                np.save(filename+'_local_test_acc.npy', local_test_acc)
                if (args.save_cdist):
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
