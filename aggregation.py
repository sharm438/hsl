import torch
import numpy as np
import torch.nn as nn
import time
import pdb
import utils

def populate_dtable(spoke_wts, dtable, my_nbrs):

    for idx in my_nbrs:
        for jdx in my_nbrs:
            if ((idx != jdx) and (torch.isnan(dtable[idx][jdx]))):
                dtable[idx][jdx] = torch.norm(spoke_wts[idx] - spoke_wts[jdx], p=2)
                dtable[jdx][idx] = dtable[idx][jdx]

def varKrum(rnd, W, spoke_wts, gnd_truth_mal_idx, cmax, capped, hist, th_lo, th_hi, infc, metric_krum, metric_vote):

    nnodes = len(W)
    vote = np.zeros(nnodes)
    dtable = np.nan * torch.ones((nnodes, nnodes))
    secure_W = torch.zeros_like(W)
    nbrs = []
    ben_nbrs = []
    fil_nbrs = []
    mal = np.where(gnd_truth_mal_idx)[0]
    for i in range(nnodes): #in view of every node now
        my_nbrs = torch.where(W[i])[0].cpu()
        k = len(my_nbrs)
        if (cmax[i] >= k/2 - 1): #such high cmax cannot guarantee robustness in multi-krum 
            if (k%2 == 0): cmax[i] = int(k/2-2)
            else: cmax[i] = int(k/2)-1
            if (cmax[i] == 0): #if k<=4, then cmax = 0, then make a new connection
                _my_nbrs = np.delete(my_nbrs, np.where(my_nbrs==i)[0][0])
                not_my_nbrs = np.delete(np.arange(len(cmax)), my_nbrs)
                rev = np.sign(hist[_my_nbrs])[:, not_my_nbrs]
                node = not_my_nbrs[np.argmax(np.sum(rev, 0))].item()
                W[i][node] = W[node][i] = 0.1 #any nonzero value works, take care of it later
                my_nbrs = torch.cat((my_nbrs, torch.tensor([node])))
                #print("Rnd %d: Node %d made a new connection with %d, cmax will be incremented" %(rnd, i, node))
                if (len(my_nbrs) <= 4): 
                    _my_nbrs = np.delete(my_nbrs, np.where(my_nbrs==i)[0][0])
                    not_my_nbrs = np.delete(np.arange(len(cmax)), my_nbrs)
                    rev = np.sign(hist[_my_nbrs])[:, not_my_nbrs]
                    node = not_my_nbrs[np.argmax(np.sum(rev, 0))].item()
                    W[i][node] = W[node][i] = 0.1
                    my_nbrs = torch.cat((my_nbrs, torch.tensor([node])))
                    #print("Rnd %d: Node %d made a new connection with %d, cmax will be incremented" %(rnd, i, node))
                    if (len(my_nbrs) <= 4): pdb.set_trace()
                    else: cmax[i] = 1
                else: cmax[i] = 1
        nbrs.append(my_nbrs)
        populate_dtable(spoke_wts, dtable, my_nbrs)
        my_dtable = dtable[my_nbrs][:, my_nbrs]
        mk_sums = []
        for j in range(len(my_dtable)):
            sorted_vals = torch.sort(my_dtable[j])[0]
            mk_sums.append(sorted_vals[:int(len(my_nbrs)-cmax[i]-2)].sum().item())
        my_ben_nbrs = my_nbrs[np.argsort(mk_sums)[:-int(cmax[i])]]
        ben_nbrs.append(my_ben_nbrs)
        if (len(my_ben_nbrs) == 0): pdb.set_trace()
        infc[rnd][i] = len(np.intersect1d(my_ben_nbrs, mal))/len(my_ben_nbrs)
        #secure_W[i][i] = 1
        if len(my_ben_nbrs) == 0: 
            #print("Rnd %d, node %d: Multi-krum reduced beyond krum" %(rnd, i))
            exit(0)
        for nbr in my_ben_nbrs: 
            secure_W[i][nbr] = 1
        secure_W[i] = secure_W[i]/secure_W[i].sum().item()
        filtered = np.delete(my_nbrs, np.where(np.isin(my_nbrs, my_ben_nbrs)))
        tp = len(np.intersect1d(filtered, mal))
        fp = len(filtered)-tp
        fn = len(np.intersect1d(my_ben_nbrs, mal))
        tn = len(my_ben_nbrs) - fn
        metric_krum[rnd][i] = [tp, fp, fn, tn]
        fil_nbrs.append(filtered)
        for j in filtered:
            hist[i][j] -= (1 - cmax[i]/(len(nbrs[i])-1)) #maintains reputation
            if (j != i): 
                if (hist[i][j] <= th_lo): 
                    vote[j] -= 1
                    if j.item() in mal: metric_vote[rnd][i][0] += 1
                    else: metric_vote[rnd][i][1] += 1
        for j in my_ben_nbrs:
            hist[i][j] += cmax[i]/(len(nbrs[i])-1)
            if (j != i): 
                if (hist[i][j] >= th_hi): 
                    vote[j] += 1
                    if j.item() in mal: metric_vote[rnd][i][2] += 1
                    else: metric_vote[rnd][i][3] += 1
    spoke_wts = torch.mm(secure_W, spoke_wts)
    return secure_W, spoke_wts, vote, nbrs, ben_nbrs, fil_nbrs, dtable

def varsec_p2p(device, rnd, votes, g, W, past_avg_wts, spoke_wts, byz, lamda, mal_idx, cmax, real_cmax, capped, hist, th_lo, th_hi, infc, past_delta, c_lo, c_hi, p, metric_krum, metric_vote):

    pre_cdist, post_cdist = 0, 0
    avg_W = torch.zeros_like(W)
    avg_votes = np.zeros(len(W))
    delta = np.zeros(len(W))
    for i in range(g):
        spoke_wts, past_avg_wts, lamda[rnd] = byz(device, past_avg_wts, spoke_wts, mal_idx, 'unit_vec', 'ben', 'p2p')
        secure_W, spoke_wts, vote, nbrs, ben_nbrs, filtered, dtable = varKrum(rnd, W, spoke_wts, mal_idx, cmax, capped, hist, th_lo, th_hi, infc, metric_krum, metric_vote)
        avg_W += secure_W
        avg_votes += vote
    for i in range(len(W)):
        if (avg_votes[i] < 0): #Node downvoted by others
            if (cmax[i] < len(nbrs[i])/2 - 2): #if there is room for increment 
                cmax[i] += 1
                delta[i] = 1
                #print("Rnd %d: cmax[%d] incremented" %(rnd, i))
                #todo - if node was previously under attack and incrementing c redeemed it, ??? 
                if (past_delta[i] == -1):# cmax decremented previously, led to downvoting in current iteration
                    c_lo[i] = cmax[i]
            else: #cmax cannot be increased, remove a connection based on feedback
                my_nbrs = np.delete(nbrs[i], np.where(nbrs[i]==i)[0][0]) #removing self from the neighbor list
                rev = np.sign(hist[my_nbrs])[:, my_nbrs] #rating that my neighbors have given to each other - positive or negative
                node = my_nbrs[np.argmin(np.sum(rev, 0))].item() #negative rated by most of my neighbors
                W[i][node] = W[node][i] = 0 #breaking connection
                if (mal_idx[node] == 0 and len(np.intersect1d(my_nbrs, np.where(mal_idx)[0]))>0):
                    print("Wrong connection broken")
                    #pdb.set_trace()
                if (cmax[i] >= 2): #if there is room for decrement
                    cmax[i] -= 1
                    #print("Rnd %d: Node %d removed %d as a connection, cmax decremented" %(rnd, i, node))

        elif (avg_votes[i] > 0): #Node upvoted by others
            rd = np.random.randn() #Decrement cmax only a random number of times, not on every upvote
            if (rd > 1-p):
                if (cmax[i]>2): # room for decrement
                    if (cmax[i] > c_lo[i] and c_lo[i] < len(c_lo)): #if c_lo is set, dont decrement cmax lower than c_lo
                        cmax[i] -= 1
                        delta[i] = -1
                    elif (c_lo[i] == len(cmax)): #if c_lo is not set, decrement
                        cmax[i] -= 1
                        delta[i] = -1
                    #print("Rnd %d: cmax[%d] decremented" %(i, rnd))
                    #todo - if node was previously okay and decrementing c poisoned it, then increment c and set that as a hard threshold c_lo and then never decrement c
    print(rnd, real_cmax, cmax)
    votes[rnd] = avg_votes/g
    return past_avg_wts, spoke_wts, pre_cdist, post_cdist, avg_W/g, cmax, delta


def multiKrum(W, spoke_wts, gnd_truth_mal_idx, cmax, metric_krum, infc, rnd):

    nnodes = len(W)
    dtable = np.nan * torch.ones((nnodes, nnodes))
    #for i in range(nnodes): ##we assume W to be adjacency matrix and not the mixing weights
    #    row_sum = len(torch.where(W[i]>0)[0])
    #    cmax = int(np.ceil(fmax*row_sum))
    #    if (cmax >= int((row_sum-2)/2)): #robustness guarantee lose
    #        print("Warning: Node %d has % neighbors and suspects % of them. No robustness guaranntee")
    #        print("Suggestion: Decrease cmax or increase #connections")
    secure_W = torch.zeros_like(W)
    nbrs = []
    ben_nbrs = []
    fil_nbrs = []
    mal = np.where(gnd_truth_mal_idx)[0]
    for i in range(nnodes): #in view of every node now
        my_nbrs = torch.where(W[i])[0].cpu()
        nbrs.append(my_nbrs)
        populate_dtable(spoke_wts, dtable, my_nbrs) #only computing the necessary elements in the array to save computation
        my_dtable = dtable[my_nbrs][:, my_nbrs] #extracting the subset

        mk_sums = []
        for j in range(len(my_dtable)):
            sorted_vals = torch.sort(my_dtable[j])[0]
            mk_sums.append(sorted_vals[:int(len(my_nbrs)-cmax[i]-2)].sum().item())
                
        my_ben_nbrs = my_nbrs[np.argsort(mk_sums)[:-int(cmax[i])]]
        ben_nbrs.append(my_ben_nbrs)
        #secure_W[i][i] = 1
        for nbr in my_ben_nbrs: 
            secure_W[i][nbr] = 1

        secure_W[i] = secure_W[i]/secure_W[i].sum().item()
        infc[rnd][i] = len(np.intersect1d(my_ben_nbrs, mal))/len(my_ben_nbrs)
        filtered = np.delete(my_nbrs, np.where(np.isin(my_nbrs, my_ben_nbrs)))
        tp = len(np.intersect1d(filtered, mal))
        fp = len(filtered)-tp
        fn = len(np.intersect1d(my_ben_nbrs, mal))
        tn = len(my_ben_nbrs) - fn
        metric_krum[rnd][i] = [tp, fp, fn, tn]
        fil_nbrs.append(filtered)

        #if (i not in mal) and (len(np.intersect1d(my_ben_nbrs, mal))>0): pdb.set_trace()
    spoke_wts = torch.mm(secure_W, spoke_wts)
    return secure_W, spoke_wts

def secure_p2p(device, rnd, dataset, g, W, past_avg_wts, spoke_wts, byz, attack_prob, lamda, mal_idx, real_cmax, cmax, metric_krum, infc):
    pre_cdist, post_cdist = 0, 0
    avg_W = torch.zeros_like(W)
    for i in range(g):
        spoke_wts, past_avg_wts, lamda[rnd] = byz(device, rnd, attack_prob, past_avg_wts, spoke_wts, mal_idx, 'unit_vec', 'ben', 'p2p', dataset)
        secure_W, spoke_wts = multiKrum(W, spoke_wts, mal_idx, cmax, metric_krum, infc, rnd)
        avg_W += secure_W
    return past_avg_wts, spoke_wts, pre_cdist, post_cdist, avg_W/g
    
def aggr1(W_hs, spoke_wts, level):

    nnodes = len(W_hs[0])
    dtable = np.nan * torch.ones((nnodes, nnodes))
    secure_W = torch.zeros_like(W_hs)
    for i in range(len(W_hs)):
        my_nbrs = torch.where(W_hs[i])[0].cpu()
        nnbrs = len(my_nbrs)
        if (nnbrs%2 == 0): cmax = int((nnbrs-2)/2 -1)
        else: cmax = int((nnbrs-2)/2)
        if (cmax < 1):
            print("Not enough neighbors to have cmax >=1")
            pdb.set_trace()
            exit(0)
        populate_dtable(spoke_wts, dtable, my_nbrs)
        my_dtable = dtable[my_nbrs][:, my_nbrs]

        mk_sums = []
        for j in range(len(my_dtable)):
            sorted_vals = torch.sort(my_dtable[j])[0]
            mk_sums.append(sorted_vals[:int(len(my_nbrs)-cmax-2)].sum().item())
        my_ben_nbrs = my_nbrs[np.argsort(mk_sums)[:-cmax]]
        for nbr in my_ben_nbrs:
            secure_W[i][nbr] = 1
        secure_W[i] = secure_W[i]/secure_W[i].sum().item()
    return secure_W



def hsl(device, rnd, dataset, g, W_hs, W_h, W_sh, past_avg_wts, spoke_wts, byz, attack_prob, lamda, mal_idx, save_cdist=0, threat_model='benign'):
    pre_cdist, post_cdist = 0, 0
    if (save_cdist): pre_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(spoke_wts)

    if (threat_model == 'benign'):
        if (g == 1):
            spoke_wts = torch.mm(torch.mm(W_sh, torch.mm(W_h, W_hs)), spoke_wts)
        else:
            spoke_wts = torch.mm(torch.mm(W_sh, torch.mm(torch.matrix_power(W_h, g), W_hs)), spoke_wts)   
    else:
        if (threat_model == 'spokes'):
            spoke_wts, past_avg_wts, lamda[rnd] = byz(device, rnd, attack_prob, past_avg_wts, spoke_wts, mal_idx, 'unit_vec', 'ben', 'p2p', dataset)
            secure_W_hs = aggr1(W_hs, spoke_wts, 'first')
            spoke_wts = torch.mm(secure_W_hs, spoke_wts)
            for i in range(g):
                ### compute secure_weights 
                secure_W_h = aggr1(W_h, spoke_wts, 'second')
                spoke_wts = torch.mm(secure_W_h, spoke_wts)
            secure_W_sh = aggr1(W_sh, spoke_wts, 'third')
            spoke_wts = torch.mm(secure_W_sh, spoke_wts)
        elif (threat_model == 'hubs'):
            ## W_hs should still be secure
            spoke_wts = torch.mm(W_hs, spoke_wts)
            for i in range(g):
                ## attack goes on in every round
                spoke_wts = torch.mm(W_h, spoke_wts)
        elif (threat_model == 'hubs-and-spokes'):
            print("Do the needful")
        elif (threat_model == 'benign'):
            spoke_wts = torch.mm(W_h, spoke_wts)
            for i in range (g): spoke_wts = torch.mm(W_h, spoke_wts)
        else: print("Choose the threat model")
        #spoke_wts = torch.mm(W_sh, spoke_wts)
        
    #del hub_wts
    if (save_cdist): post_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(spoke_wts)
    return spoke_wts, pre_cdist, post_cdist


def p2p(device, rnd, dataset, g, W, spoke_wts, save_cdist=0):
    pre_cdist, post_cdist = 0, 0
    if (save_cdist): pre_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    
    #W = wts.repeat(len(wts), 1)
    #W = utils.simulate_W(len(wts), 3, device)
    for i in range(g):
        #pdb.set_trace()
        #spoke_wts, past_avg_weights, lamda[rnd] = byz(device, rnd, attack_prob, past_avg_wts, spoke_wts, mal_idx, 'unit_vec', 'ben', 'p2p', dataset)
        spoke_wts = torch.mm(W, spoke_wts)
    #del param_list
    if (save_cdist): post_cdist = torch.sum((spoke_wts-torch.mean(spoke_wts, dim=0))**2)/len(W)
    return spoke_wts, pre_cdist, post_cdist

def flair(device, byz, lr, grad_list, net, old_direction, susp, fs, cmax, weight):
    
    #reshaping the parameter list
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])

    #FS_min and FS_max used by an adversary in an adaptive attack
    fs_min = torch.sort(fs)[0][cmax-1]
    fs_max = torch.sort(fs)[0][-cmax]
    if 'adaptive_krum' in str(byz): #if the attack is adaptive
        param_list = byz(device, lr, param_list, old_direction, cmax, fs_min, fs_max)
    elif 'adaptive_trim' in str(byz):
        param_list = byz(device, lr, param_list, old_direction, cmax, fs_min, fs_max, weight)
    else: param_list = byz(device, lr, param_list, cmax) #non-adaptive attack
    flip_local = torch.zeros(len(param_list)).to(device) #flip-score vector
    penalty = 1.0 - 2*cmax/len(param_list) 
    reward = 1.0 - penalty

    ##Computing flip-score
    for i in range(len(param_list)):
        direction = torch.sign(param_list[i])
        flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
        flip_local[i] = torch.sum(flip*(param_list[i]**2))
        del direction, flip

    #updating suspicion-score
    argsorted = torch.argsort(flip_local).to(device)
    if (cmax > 0):
        susp[argsorted[cmax:-cmax]] = susp[argsorted[cmax:-cmax]] + reward
        susp[argsorted[:cmax]] = susp[argsorted[:cmax]] - penalty
        susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] - penalty  
    argsorted = torch.argsort(susp)

    #updating weights
    weights = torch.exp(susp)/torch.sum(torch.exp(susp))
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1,1))
    global_direction = torch.sign(global_params)

    #updating parameters
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params

    return net, global_direction, susp, flip_local, weights

##FEDSGD - weighted mean aggregation weighed by their data size



##FoolsGold
def foolsgold(device, byz, lr, grad_list, net, nbyz):
    start = time.time()    
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, lr, param_list, nbyz)
    num_workers = len(param_list)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(device)
    cs = torch.zeros((num_workers, num_workers)).to(device)
    for i in range (num_workers):
        for j in range (i):
            ## compute cosine similarity
            cs[i,j] = cos(param_list[i], param_list[j])
            cs[j,i] = cs[i,j]
    ###The foolsgold algorithm implemented below
    v = torch.zeros(num_workers).to(device)        
    for i in range (num_workers):
        v[i] = torch.max(cs[i])
      
    alpha = torch.zeros(num_workers).to(device)
    for i in range (num_workers):
        for j in range (num_workers):
            if (v[j] > v[i]):
                cs[i,j] = cs[i,j]*v[i]/v[j]
        alpha[i] = 1 - torch.max(cs[i])
    
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/(torch.max(alpha))
    alpha[alpha == 1] = 0.99
    alpha = torch.log(alpha/(1-alpha)) + 0.5
    alpha[(torch.isinf(alpha) + (alpha > 1))] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/torch.sum(alpha).item()
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), alpha.reshape(-1,1))
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params
    #print(time.time()-start)
    return net, alpha

#FABA 
def faba(device, byz, lr, grad_list, net, cmax):
    start = time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, lr, param_list, cmax)
    faba_client_list = np.ones(len(param_list)) #contains the current benign clients
    dist = np.zeros(len(param_list))
    G0 = torch.mean(param_list, dim=0)
    for i in range(cmax):
        for j in range(len(param_list)):
            if faba_client_list[j]:
                dist[j] = torch.norm(G0-param_list[j]).item()      
        outlier = int(np.argmax(dist))
        faba_client_list[outlier] = 0 #outlier removed as suspected 
        dist[outlier] = 0
        G0 = (G0*(len(param_list)-i) - param_list[outlier])/(len(param_list)-i-1) #mean recomputed

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
               param[1].data += G0[idx:(idx+param[1].nelement())].reshape(param[1].shape)
               idx += param[1].nelement()

    del param_list
    #print(time.time()-start)
    return net, faba_client_list  

#KRUM aggregation
def krum(device, byz, lr, grad_list, net, cmax):
    start = time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, lr, param_list, cmax) 
    k = len(param_list)-cmax-2
    #Computing distance between every pair of clients
    dist = torch.zeros((len(param_list), len(param_list))).to(device)
    for i in range(len(param_list)):
        for j in range(i):
            dist[i][j] = torch.norm(param_list[i]-param_list[j])
            dist[j][i] = dist[i][j]       
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += param_list[model_selected][idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list
    #print (time.time()-start)
    return net   

###FLTRUST aggregation
def fltrust(device, byz, lr, grad_list, net, nbyz):
    start = time.time()
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    #Client 1 acts as the root dataset holder
    server_params = param_list[0]
    server_norm = torch.norm(server_params)
    param_list = (param_list[1:])#[np.random.permutation(tau)]
    param_list = byz(device, lr, param_list, nbyz)
    
    #The FLTRUST algorithm
    ts = torch.zeros((len(param_list)))
    for i in range(len(param_list)):
        ts[i] = max(cos(server_params, param_list[i]), 0)
        param_list[i] = (server_norm/torch.norm(param_list[i])) * param_list[i] * ts[i]
    global_params = torch.sum(param_list, dim=0) / torch.sum(ts)
    del param_list
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del global_params
    #print(time.time()-start)
    return net, ts   

#TRIMMED MEAN
def trim(device, byz, lr, grad_list, net, cmax): 
    start=time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, lr, param_list, cmax)
    #Sorting every parameter
    sorted_array = torch.sort(param_list, axis=0)
    #Trimmin the ends
    trimmed = torch.mean(sorted_array[0][cmax:len(param_list)-cmax,:], axis=0)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += trimmed[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
                
    del param_list, sorted_array, trimmed
    #print(time.time()-start)
    return net  

#MEDIAN aggregation
def median(device, byz, lr, grad_list, net, cmax):
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, lr, param_list, cmax)
    sorted_array = torch.sort(param_list, axis=0)
    if (len(param_list)%2 == 1):
        med = sorted_array[0][int(len(param_list)/2),:]
    else:
        med = (sorted_array[0][int(len(param_list)/2)-1,:] + sorted_array[0][int(len(param_list)/2),:])/2

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += med[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
    del param_list, sorted_array
    return net
    
