import torch
import numpy as np
import torch.nn as nn
import time
import pdb
import utils

import attack

def hsl(device, rnd, g, W_hs, W_h, W_sh, spoke_wts, save_cdist=0):
    pre_cdist, post_cdist = 0, 0
    if (save_cdist):
        pre_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(spoke_wts)

    if (g == 1):
        spoke_wts = torch.mm(torch.mm(W_sh, torch.mm(W_h, W_hs)), spoke_wts)
    else:
        spoke_wts = torch.mm(torch.mm(W_sh, torch.mm(torch.matrix_power(W_h, g), W_hs)), spoke_wts)   

    if (save_cdist):
        post_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(spoke_wts)
        print(pre_cdist, post_cdist)
    
    return spoke_wts, pre_cdist, post_cdist


def p2p(device, rnd, g, W, spoke_wts, net_name, num_inp, num_out, distributed_data, distributed_label, minibatch, attack_flag=0, node_attacked=[0], save_cdist=0):
    pre_cdist, post_cdist = 0, 0
    if (save_cdist): pre_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    pre_model_real = spoke_wts[node_attacked].reshape(-1)
    pre_model_est = torch.mean(spoke_wts, dim=0)
    for i in range(g):
        spoke_wts = torch.mm(W, spoke_wts)
    if (attack_flag):
        recon_acc_real = attack.gradInv(rnd, utils.vec_to_model(pre_model_real, net_name, num_inp, num_out, device), spoke_wts, node_attacked, distributed_data, distributed_label, minibatch, device)
        recon_acc_est = attack.gradInv(rnd, utils.vec_to_model(pre_model_est, net_name, num_inp, num_out, device), spoke_wts, node_attacked, distributed_data, distributed_label, minibatch, device)
    if (save_cdist): post_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    if (attack_flag): 
        print(rnd, post_cdist, recon_acc_real.item() , recon_acc_est.item())
        return spoke_wts, pre_cdist, post_cdist, recon_acc_real, recon_acc_est
    return spoke_wts, pre_cdist, post_cdist


