import torch
from copy import deepcopy
import math
import numpy as np
import copy
import pdb

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

    

