import numpy as np
import torch
from copy import deepcopy
import math
import copy
import pdb

import utils
import torch.nn.functional as F


def gradInv(rnd, dataset_name, net_fed, spoke_wts, nodes_attacked, distributed_data, distributed_labels, minibatches, device):
  
  num_attack_iters = 24000
  criterion = torch.nn.CrossEntropyLoss()

  for node in nodes_attacked:
    minibatch = minibatches[node]
    data = distributed_data[node][minibatch]
    labels = distributed_labels[node][minibatch]# .float().requires_grad_(True)
    true_grad = spoke_wts[node] - utils.model_to_vec(net_fed) 
    dummy_data = torch.empty((minibatch.shape[0], data.shape[1], data.shape[2], data.shape[3]), device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([dummy_data], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_attack_iters // 2.667,
                                                     num_attack_iters // 1.6, num_attack_iters // 1.142], gamma=0.1)

    for i in range(num_attack_iters):
      def closure():
        optimizer.zero_grad()
        dummy_pred = net_fed(dummy_data)
        loss_model = criterion(dummy_pred, labels)#.requires_grad_(True) 
        recon_grads = torch.autograd.grad(loss_model, net_fed.parameters(), create_graph=True)
        recon_grad_tensor = torch.cat([grad.reshape(-1) for grad in recon_grads])
     
        loss_recon = 1 - torch.nn.functional.cosine_similarity(recon_grad_tensor, true_grad, 0, 1e-10)
        loss_recon += 0 #total variation
        loss_recon.backward()
    
        mse_recon = torch.norm(dummy_data - data) / len(minibatch)
        if (i%200 == 199):
          print(i, loss_model.item(), loss_recon.item(), mse_recon.item())
        return loss_recon

      optimizer.step(closure)
      scheduler.step()
    mse_recon = torch.norm(dummy_data - data) / len(minibatch)
    print("Round %d, recon_mse = %f" %(rnd, mse_recon))
    utils.plot(data, dummy_data)
    pdb.set_trace()
  return mse_recon



##No attack
def benign(device, rnd, attack_prob, past_avg_wts, spoke_wts, mal_idx, dev_type, capabilities, agr, dataset='cifar10'):

    return spoke_wts, past_avg_wts, 0


def shejwalkar(device, rnd, attack_prob, past_avg_wts, spoke_wts, mal_idx, dev_type='unit_vec', capabilities='ben', agr='p2p', dataset='cifar10'):
    nbyz = len(np.where(mal_idx==1)[0])
    if (nbyz == 0):
        print("Not attackers, proceeding with benign update")
        return spoke_wts, torch.mean(spoke_wts, 0), 0
    if (capabilities == 'nbrs'):
        present_avg_wts = torch.mean(spoke_wts[(is_mal_nbr == 1) & (mal_idx == 1)], 0)
    elif (capabilities == 'all'): 
        present_avg_wts = torch.mean(spoke_wts, 0)
    elif (capabilities == 'limited'):
        present_avg_wts = torch.mean(spoke_wts[(mal_idx == 1)], 0)
    elif (capabilities == 'ben'): ##Temporary change to median benign vector
        present_avg_wts = torch.median(spoke_wts[(mal_idx == 0)], 0)[0]

    attack_flag = 0
    if (attack_prob < 1):
        rd = np.random.randn()
        if (rd > attack_prob or rnd%2 == 0): #choose not to attack
            return spoke_wts, present_avg_wts, 0
        else: attack_flag = 1

    if (attack_prob == 1 or attack_flag == 1):
        ben_grads = present_avg_wts - past_avg_wts
        #print("Direction estimated, initiating attack")


        if dev_type == 'unit_vec':
            if (torch.norm(ben_grads) == 0):  deviation = torch.sign(ben_grads)
            else: deviation = ben_grads / torch.norm(ben_grads)  # unit vector, dir opp to good dir?
        elif dev_type == 'sign':
            deviation = torch.sign(ben_grads)
        elif dev_type == 'std':
            deviation = torch.std(ben_grads, 0) ##deviation is of the models - is it right? Comparing all models to nets[0]. a fixed model, std should be the nsame of grads and weights, right?

        #print(present_avg_wts)
        copy_wts = spoke_wts.clone()

        if dataset == 'mnist':
            lamda = torch.Tensor([10.0]).to(device) 
            threshold_diff = 0.1 #changed from 1e-5
        elif dataset == 'cifar10':
            lamda = torch.Tensor([5.0]).to(device) 
            threshold_diff = 0.1 #changed from 1e-5
        prev_loss = -1
        step = lamda/2 
        lamda_succ = 0
        iters = 0     
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            iters += 1
            mal_update = past_avg_wts + (ben_grads - lamda * deviation)
            copy_wts[mal_idx==1] = torch.stack([mal_update] * nbyz)

            #avg_wts = torch.mean(copy_wts, 0)
            loss = torch.norm(torch.mean(copy_wts, 0) - present_avg_wts)
            if prev_loss < loss: ##oracle outputs true ie if loss is increasing increase lamda even more for a stronger attack
                lamda_succ = lamda
                lamda = lamda + step / 2
            else:
                lamda = lamda - step / 2
            step = step / 2
            prev_loss = loss
            
            if (iters>10): 
                print("Attack has not converged in 10 iterations, debug.")
                pdb.set_trace()
        #print("lambda = ", lamda_succ)
        mal_update = (present_avg_wts - lamda_succ * deviation)
        spoke_wts[mal_idx==1] = torch.stack([mal_update]*nbyz)
        del copy_wts
        
        return spoke_wts, present_avg_wts, lamda_succ.item()
        #return model_re, lamda_succ, deviation 

    

