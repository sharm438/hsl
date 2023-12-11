import torch
import numpy as np
import torch.nn as nn
import time
import pdb
import utils


def hsl(device, rnd, dataset, g, W_hs, W_h, W_sh, spoke_wts, save_cdist=0):
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


def p2p(device, rnd, dataset, g, W, spoke_wts, save_cdist=0):
    pre_cdist, post_cdist = 0, 0
    if (save_cdist): pre_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    
    for i in range(g):
        spoke_wts = torch.mm(W, spoke_wts)
    if (save_cdist): post_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    return spoke_wts, pre_cdist, post_cdist


