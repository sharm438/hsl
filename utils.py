import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import nets
import attack
import pdb

import unidecode
import string
import torch.nn as nn

import random

import attrs

import resnet_cifar
import resnet_tinyim_net
import os

import networkx as nx

@attrs.define(frozen=False)
class TrainObject:
    dataset_name: str
    net_name: str
    train_data: torch.utils.data.dataloader.DataLoader
    test_data: torch.utils.data.dataloader.DataLoader
    num_inputs: int
    num_outputs: int
    lr: float
    batch_size: int

@attrs.define(frozen=False)
class DistributedData:
    distributed_input: list[torch.Tensor]
    distributed_output: list[torch.Tensor]
    wts: torch.Tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", help="aggregation rule", default='p2p', type=str)
    parser.add_argument("--num_spokes", help="# clients", default=20, type=int)
    parser.add_argument("--num_hubs", help="# servers", default=1, type=int)
    parser.add_argument("--num_rounds", help="# training rounds", default=500, type=int)
    parser.add_argument("--num_local_iters", help="# local iterations", default=1, type=int)

    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--exp", help="Experiment name", default='', type=str)

    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--seed", help="seed for model init", default=108, type=int)

    parser.add_argument("--save_time", help="array saving frequency", default=100, type=int)
    parser.add_argument("--eval_time", help="evaluation frequency", default=10, type=int)

    parser.add_argument("--num_edges", help="Out of the lower triangle", default=1, type=float)
    parser.add_argument("--W", help="Whether to load a saved W", default=None, type=str)
    parser.add_argument("--W_type", help="k-regular, erdos-renyi, franke-wolfe", default='k-regular', type=str)
    parser.add_argument("--max_degree", help="maximum degree of a node", default='10', type=int)
    parser.add_argument("--map_type", help="greedy or foolish", default='greedy', type=str)

    parser.add_argument("--spoke_budget", help="number of hubs a spoke connects to", default=1, type=int)


    parser.add_argument("--self_wt", help="Weight assigned to self in gossip averaging", default=None, type=float)


    parser.add_argument("--agr", help="aggregation", default="mean", choices=["mean", "random_static", "random_dynamic", "adjacency_static"])
    parser.add_argument("--W_hs", help="Whether to load a saved W_hs", default=None, type=str)
    parser.add_argument("--W_sh", help="Whether to load a saved W_sh", default=None, type=str)
    parser.add_argument("--W_h", help="Whether to load a saved W_h", default=None, type=str)
    parser.add_argument("--num_edges_hs", help="Out of nhubs*nspokes", default=1, type=float)
    parser.add_argument("--num_edges_h", help="Out of the lower triangle nhubs*(nhubs-1)/2", default=1, type=float)
    parser.add_argument("--save_cdist", help="whether to save predist and postdist", default=1, type=int)
    parser.add_argument("--g", help="number of gossip steps", default=1, type=int)
    return parser.parse_args()


def get_lr(rnd, nrounds, lr):

    rnd += 1
    mu = nrounds/4
    sigma = nrounds/4
    max_lr = lr
    if (rnd < nrounds/4):
        return max_lr*(1-np.exp(-25*(rnd/nrounds)))
    else:
        return max_lr*np.exp(-0.0125*(((rnd-mu)/sigma)**2))

def label_distr(labels, nlabels):

    nnodes = len(labels)
    distr = np.zeros((nnodes, nlabels))
    for i in range(nnodes):
        for j in labels[i]:
            distr[i][j.item()] += 1

    return distr

def plaw_graph(nnodes, exp):

    return W

def greedy_hubs(num_hubs, num_spokes, num_spoke_connections, label_distr, map_type):
    ## proposed algorithm to mimic global distribution locally
    W_hs = torch.zeros((num_hubs, num_spokes))
    W_sh = torch.zeros((num_spokes, num_hubs))
    num_hub_connections = int(num_spokes * num_spoke_connections / num_hubs)
    global_label_distr_norm = label_distr.sum(0) / label_distr.sum()
    for i in range(num_hub_connections+1):
        random_hub_sequence = random.sample(range(num_hubs), num_hubs)
        for hub in random_hub_sequence:
            available_spokes = np.where(W_hs.sum(0) < num_spoke_connections)[0]
            my_spokes = np.where(W_hs[hub])[0]
            duplicate_spokes = np.isin(available_spokes, my_spokes)
            available_spokes = available_spokes[~duplicate_spokes]
            hub_label_distr = label_distr[np.where(W_hs[hub])[0]].sum(0)
            ## TODO - undo changes
            if map_type == 'greedy': min_kl_div = np.inf
            elif map_type == 'foolish': min_kl_div = -np.inf
            closest_spoke = -1
            for spoke in available_spokes:
                updated_hub_label_distr = hub_label_distr + label_distr[spoke]
                updated_hub_label_distr_norm = updated_hub_label_distr / updated_hub_label_distr.sum()
                kl_div = compute_kl_div(global_label_distr_norm, updated_hub_label_distr_norm) 
                ## Undo changes
                if map_type == 'greedy': 
                    if kl_div < min_kl_div:
                        min_kl_div = kl_div
                        closest_spoke = spoke
                elif map_type == 'foolish': 
                    if kl_div > min_kl_div:
                        min_kl_div = kl_div
                        closest_spoke = spoke
            if closest_spoke != -1:
                #print("Hub %d chose spoke %d" %(hub, closest_spoke))
                W_hs[hub][closest_spoke] = 1
                W_sh[closest_spoke][hub] = 1
                hub_label_distr += label_distr[closest_spoke]

    kl_divs = []
    for hub in range(num_hubs): 
        hub_label_distr = label_distr[np.where(W_hs[hub])[0]].sum(0)
        hub_label_distr_norm = hub_label_distr / hub_label_distr.sum()
        kl_divs.append(compute_kl_div(global_label_distr_norm, hub_label_distr_norm))
        W_hs[hub] /= W_hs[hub].sum()
    print(kl_divs, np.mean(kl_divs))
    for spoke in range(num_spokes): 
        num_connections = len(np.where(W_sh[spoke])[0])
        if (num_connections != num_spoke_connections):
            print("Spoke %d has %d connections but needed %d connections" %(spoke, num_connections, num_spoke_connections))
        W_sh[spoke] /= W_sh[spoke].sum()
    return W_hs, W_sh

def compute_kl_div(p, q):
    p = np.where(p==0, 1e-10, p)
    q = np.where(q==0, 1e-10, q)
    kl_div = np.sum(p * np.log(p/q))
    return kl_div

def assign_hubs(num_hubs, num_spokes, label_distr):
    ## toy setup
    ## works when num_spokes is a multiple of num_labels and also num_hubs
    ## harcoded
    majority_label = np.argmax(label_distr, axis=1)
    W_hs = torch.zeros((num_hubs, num_spokes))
    W_sh = torch.zeros((num_spokes, num_hubs))
    for i in range(num_spokes):
        first_hub = majority_label[i]//2
        second_hub = (first_hub + 1) % num_hubs
        W_sh[i][first_hub] = W_sh[i][second_hub] = 0.5
        W_hs[first_hub][i] = W_hs[second_hub][i] = 1
    for i in range(num_hubs):
        W_hs[i] = W_hs[i] / W_hs[i].sum()
    return W_hs, W_sh

def connect_hubs(num_hubs):
    ## hardcoded to connect in a ring
    W = torch.zeros((num_hubs, num_hubs))
    for i in range(num_hubs):
        first_hub = (i+1) % num_hubs
        second_hub = (i-1) % num_hubs
        W[i][i] = W[i][first_hub] = W[i][second_hub] = 1/3
    return W

def erdos_renyi(num_nodes, num_edges, seed):

    W = torch.zeros((num_nodes, num_nodes))
    G = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
    for node in G.nodes:
        for nbr in G.adj[node]:
            W[node][nbr] = 1
        if (W[node].sum() < 1):
            print ("Some nodes are isolated, restart.")
            exit(-1)
        W[node][node] = 1
        W[node] /= W[node].sum()
    return W

def obj_function(W, PI, lamb):

    nn = PI.shape[0]
    val = (1/nn)*np.linalg.norm(W.dot(PI)-(1/nn)*np.ones((nn,nn)).dot(PI),'fro')**2
    val = val + (lamb/nn)*np.linalg.norm(W-(1/nn)*np.ones((nn,nn)),'fro')**2
    return val

from scipy.optimize import linear_sum_assignment

def franke_wolfe_p2p_graph(PI, lamb, budget, n_max=100):
    '''
    Baseline mixing matrix simulation code from ICML 2023.

    Args:
      PI: Class proportions of all s spokes as rows and class ID as cols.
      lamb: Hyperparameter quantifying the ratio variance / heterogeneity
      budget: Maximum number of edges a node can have.
      n_max: Number of iterations for numberical solution of the FW algorithm.

    Returns: 
      W_FW: s*s array as the mixing matrix.
    '''

    n = PI.shape[0]
    K = PI.shape[1]

    function_value = []
    W_FW_temp = np.identity(n)
    b = 0
    it = 0

    while ((b <= budget) and (it <= n_max)):
        W_FW = W_FW_temp
        function_value += [obj_function(W_FW,PI,lamb)]

        vv = W_FW.dot(PI)-np.repeat(np.mean(PI,0).reshape((1,-1)),n,0)
        vvv = np.sum([np.outer(vv[:,k],PI[:,k]) for k in range(K)],0)
        grad = (2/n)*vvv + (2/n)*lamb*(W_FW - (1/n)*np.ones((n,n)))

        ind = linear_sum_assignment(grad)
        P = np.zeros((n,n))
        P[ind] = 1

        uu = np.mean(PI,0)-W_FW.dot(PI)
        uuu = (P-W_FW).dot(PI)
        num = np.sum([np.dot(uu[:,k],uuu[:,k]) for k in range(K)]) - lamb*np.trace((W_FW - (1/n)*np.ones((n,n))).T.dot(P-W_FW))
        den = np.linalg.norm((P-W_FW).dot(PI),'fro')**2 + lamb*np.linalg.norm(P-W_FW,'fro')**2
        gamma = num / den
        if ((gamma < 0) or (gamma>1)):
            print('Check line-search')

        W_FW_temp = (1-gamma)*W_FW + gamma*P
        b = np.max(np.sum(W_FW_temp - np.identity(n) > 0,1))
        it += 1

    nnbrs = [len(np.where(W_FW[i])[0]) for i in range(len(W_FW))]
    num_directed_edges = (np.asarray(nnbrs).sum() - n)
    num_connections = 0
    for i in range(len(W_FW)):
        for j in range(i):
            if (W_FW[i][j] or W_FW[j][i]):
                num_connections += 1
    return torch.as_tensor(W_FW, dtype=torch.float32), num_connections, num_directed_edges

def compute_test_accuracy(net, test_data, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct/total
    

def fully_connect(num_spokes):
    ## harcoded
    W = torch.ones((num_spokes, num_spokes)) / num_spokes
    return W

def random_graph(nrows, ncols, nedges, agr="mean"):

    # If all nodes are peers
    if (nrows == ncols): 
        W = torch.zeros((nrows, ncols))
        nedges_rem = int(nedges)

        for i in range(nrows): W[i][i] = 1
        while (nedges_rem > 0):
            min_degree = W.sum(axis=0).min().item()
            nodes_with_min_degree = np.where(torch.eq(W.sum(axis=0), min_degree))[0]
            node_selected = np.random.choice(nodes_with_min_degree)
            indices = torch.nonzero(torch.logical_not(W[node_selected]))
            min_degree = W.sum(axis=0)[indices].min().item()
            cross_node = indices[np.random.choice(np.where(torch.eq(W.sum(axis=0)[indices], min_degree))[0])].item()
            W[node_selected][cross_node] = 1
            W[cross_node][node_selected] = 1
            nedges_rem -= 1

        if (W.sum(axis=0).min().item() < 2):
            raise ValueError("Some nodes remained isolated, enter a larger value of nedges to promote collabortion among nodes")
            
        if (agr == 'mean'):
            for i in range(nrows): W[i] = W[i]/W[i].sum()
        elif (agr == 'random_static'):
            for i in range(nrows):
                for j in range(nrows):
                    if (W[i][j]): W[i][j] = np.random.rand()
                W[i] = W[i]/W[i].sum()
        elif (agr == 'adjacency_static'): return W
        return W

    # If nodes can have different roles of hubs and spokes.
    else:
        W_hs = torch.zeros((nrows, ncols))
        W_sh = torch.zeros((ncols, nrows))
        num_idx = int(nedges) 
        if (num_idx < 2*ncols):
            print("Cannot ensure at least two hub connection per spoke with such low k values, please exit and restart")
            pdb.set_trace() ##Restart
            return W_hs, W_sh
        for i in range(ncols):
            chosen_rows = np.random.choice(nrows, 2, replace=False)
            W_hs[chosen_rows[0]][i] = 1
            W_sh[i][chosen_rows[0]] = 1
            W_hs[chosen_rows[1]][i] = 1
            W_sh[i][chosen_rows[1]] = 1
        ## Every spoke connected to at least 2 hubs for a feasible value of k
        ## Now if there are more connections to be made for this value of k, we do that again with proper random sampling below

        edges_rem = int(nedges - 2*ncols)
        idx_rem = torch.nonzero(torch.logical_not(W_hs))
        idx_chosen = np.random.choice(len(idx_rem), edges_rem, replace=False)
        for idx in idx_rem[idx_chosen]:
            W_hs[idx[0]][idx[1]] = 1
            W_sh[idx[1]][idx[0]] = 1

        if (agr == 'mean'):
            for i in range(nrows): W_hs[i] = W_hs[i]/W_hs[i].sum()
            for i in range(ncols): W_sh[i] = W_sh[i]/W_sh[i].sum()
        elif (agr == 'random_static'):
            for i in range(nrows):
                for j in range(ncols):
                    if (W_hs[i][j]): W_hs[i][j] = np.random.rand()
                    if (W_sh[j][i]): W_sh[j][i] = np.random.rand()
                W_hs[i] = W_hs[i]/W_hs[i].sum()
            for j in range(ncols): W_sh[j] = W_sh[j]/W_sh[j].sum()
        elif (agr == 'adjacency_static'): return W_hs, W_sh 
        return W_hs, W_sh

def cluster_graph(label_distr, k, prob): #probability of inter-cluster edges

    idx = np.argmax(label_distr, axis=1)
    nnodes = len(idx)
    W = torch.zeros((nnodes, nnodes))
    for i in range(nnodes):
        #W[i][i] = 1
        for j in range(i):
            if idx[i] == idx[j]:
                W[i][j] = 1
                W[j][i] = 1
            else:
                rd = np.random.rand()
                if (rd > 1-prob):
                    W[i][j] = 1
                    W[j][i] = 1
    #k_vals = W.sum(axis=0)
    
    for i in range((nnodes)):
        nz_idx = torch.nonzero(W[i])
        random.shuffle(nz_idx)
        sel_idx = nz_idx[:min(len(nz_idx), k-1)]
        W[i].fill_(0)
        W[i][i] = 1
        for idx in sel_idx: W[i][idx] = 1

        W[i] = W[i]/W[i].sum()
    return W

def local_W(label_distr, device):
    norm_distr = np.zeros(label_distr.shape)
    n = len(label_distr)
    for i in range(n): norm_distr[i] = label_distr[i]/np.sum(label_distr[i])
    W = torch.zeros((n,n)).to(device)
    majority = np.argmax(label_distr, axis=1)
    for i in range(n):
        for j in range(n):
            cos = np.sum((norm_distr[i]*norm_distr[j]))/(np.linalg.norm(norm_distr[i])*np.linalg.norm(norm_distr[j]))
            if (cos > 1): cos = 1
            rd = np.random.rand()
            if (rd < cos):
                W[i][j] = 1 #cos
        W[i] = W[i]/torch.sum(W[i])
    pdb.set_trace()
    return W

def self_wt(n, k, w, device):
    W = torch.zeros((n,n)).to(device)
    if (k%2 == 1): k_size = [int((k-1)/2), int((k-1)/2)]
    else: k_size = [int(k/2), int(k/2)-1]
    for i in range(n):
        for j in range (-k_size[0], k_size[1]+1):
            idx = i+j
            if (idx<0): idx += n
            if (idx>=n): idx -= n
            if (idx == i): W[i][idx] = w
            else: W[i][idx] = (1-w)/(k-1)
    return W

def k_regular_W(n, k, device):
    W = torch.zeros((n,n)).to(device)
    if (k%2 == 1): k_size = [int((k-1)/2), int((k-1)/2)]
    else: k_size = [int(k/2), int(k/2)-1]
    for i in range(n):
        for j in range (-k_size[0], k_size[1]+1):
            idx = i+j
            if (idx<0): idx += n
            if (idx>=n): idx -= n
            W[i][idx] = 1.0/k
    return W

def random_connection(n1, n2, k1, k2, device):
    
    W = torch.zeros((n1, n2)).to(device)
    for i in range(len(W)):
        num_conn = random.randint(k1, k2)
        idx = random.sample(range(0, n2), num_conn)
        W[i][idx] = 1.0/num_conn
    return W

def simulate_hsl_W(nhubs, nspokes, device):

    W_hs = random_connection(nhubs, nspokes, int(nspokes/nhubs), 2*int(nspokes/nhubs), device)
    W_h = k_regular_W(nhubs, int(nhubs/2)+1, device)
    W_sh = random_connection(nspokes, nhubs, 2, nhubs, device)
    return W_hs, W_h, W_sh

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

def char_tensor(string_):
    all_characters = string.printable
    tensor = torch.zeros(len(string_)).long()
    for c in range(len(string_)):
        try:
            tensor[c] = all_characters.index(string_[c])
        except:
            continue
    return tensor

def compute_cdist(client_wts):

    avg_wts = torch.mean(client_wts, dim=0)
    return torch.mean(torch.norm(client_wts-avg_wts, dim=1)).item()

def distribute_malicious(n_clients, n_mals, distr_type='first_few', n_groups=0, group_ids=0):

    is_mal = np.zeros(n_clients)
    #n_mals = math.floor(fbyz*n_clients)   
    if (distr_type == 'random'):
        mal_idx = np.random.choice(n_clients, n_mals, replace=False)
        is_mal[mal_idx] = 1
    elif (distr_type == 'first_few'): is_mal[:n_mals] = 1
    elif (distr_type == 'group'): is_mal[np.where(group_ids < math.floor(fbyz*n_groups))[0]] = 1
    return is_mal

def model_to_vec(net):

    return (torch.cat([x.detach().reshape((-1)) for x in net.parameters() if x.requires_grad != 'null'], dim=0)).squeeze(0)

def vec_to_model(vec, net_name, num_inp, num_out, device):

    net = load_net(net_name, num_inp, num_out, device)
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data = vec[idx:(idx+param[1].nelement())].reshape(param[1].shape).detach() ##assigned not updated
                idx += param[1].nelement()
    return net

def create_model(net, net_name, num_inp, num_outp, seed=None):

    created_net = load_net(net_name, num_inp, num_outp, torch.device('cuda'), seed)
    with torch.no_grad():
        idx = 0
        for param1, param2 in zip(created_net.parameters(), net.parameters()):
            if param1[1].requires_grad:
                param1[1].data = param2[1].data.clone().detach()

    return created_net

def update_model(message, net, aggregated_grads, test_data, device):

    if (message.find('weights') != -1):
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data = aggregated_grads[idx:(idx+param[1].nelement())].reshape(param[1].shape).detach() ##assigned not updated
                    idx += param[1].nelement()
    
    if (message.find('gradients') != -1):
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += aggregated_grads[idx:(idx+param[1].nelement())].reshape(param[1].shape).detach()
                    idx += param[1].nelement()

    if (message == 'shakespeare'):
        criterion = nn.CrossEntropyLoss()
        net.eval()
        ln = int(int(len(test_data[0])/200)*200)
        data = test_data[0][:ln].reshape((-1, 200))
        labels = test_data[1][:ln].reshape((-1, 200))
        with torch.no_grad():
            hdn = net.init_hidden(data.shape[0])
            hidden = (hdn[0].to(device), hdn[1].to(device))
            test_loss = 0
            for c in range(200):
                output, hidden = net(data[:, c], hidden)
                test_loss += criterion(output.view(data.shape[0], -1), labels[:,c])
            test_loss /= 200
        return test_loss

    test_acc = -1
    if (message.find('evaluate') != -1):
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_acc = correct/total
            print(correct, total, test_acc)
    return test_acc

def create_batch(data_size, batch_size, rr_idx, cl, sample_type='round_robin'):
   
    if (sample_type == 'random'): return np.random.choice(data_size, batch_size, replace=False)
    elif (sample_type == 'same'): return np.arange(batch_size)
    elif (sample_type == 'round_robin'): 
        if (rr_idx[cl] + batch_size < data_size): 
            batch_idx = np.asarray(list(range(int(rr_idx[cl]), int(rr_idx[cl])+batch_size)))
            rr_idx[cl] = rr_idx[cl] + batch_size
        else: 
            batch_idx = np.asarray(list(range(int(rr_idx[cl]), data_size)) + list(range(0, batch_size - (data_size-int(rr_idx[cl])))))
            rr_idx[cl] = batch_size - (data_size-int(rr_idx[cl]))
        return batch_idx

def cluster_clients(n_clients, n_groups, cluster_type="uniform"):

    if (cluster_type == "uniform"): return np.random.randint(0, n_groups, n_clients)

def num_params(net):

    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()
    return P

def load_attack(attack_name):

    if attack_name == 'benign':  byz = attack.benign
    elif attack_name == 'shejwalkar':  byz = attack.shejwalkar
    #elif attack_name == 'shej_attack':  byz = attack.shej_attack
    #elif attack_name == 'shej_agnostic':  byz = attack.shej_agnostic

    return byz

def load_net(net_name, num_inputs, num_outputs, device, seed=None):

    if (seed): torch.manual_seed(seed)
    if (net_name == 'resnet18'):
        #net = nets.ResNet18()
        net = resnet_cifar.ResNet18()
    elif (net_name == 'resnet-tiny'):
        net = resnet_tinyim_net.ResNet18()
    elif(net_name == 'dnn'):
        net = nets.DNN()
    elif(net_name == 'lstm'):
        n_characters = len(string.printable)
        net = nets.CharRNN(n_characters, 128, n_characters, 'lstm', 2)
    net.to(device) 
    if seed == 0:
        net.load_state_dict(torch.load('outputs/_model.pth'))
    return net

def load_byz(byz_name):

    if byz_name == 'benign':
        byz = attack.benign
    elif byz_name == 'full_trim':
        byz = attack.full_trim
    elif byz_name == 'full_krum':
        byz = attack.full_krum
    elif byz_name == 'adaptive_trim':
        byz = attack.adaptive_trim
    elif byz_name == 'adaptive_krum':
        byz = attack.adaptive_krum
    elif byz_name == 'shejwalkar':
        byz = attack.shejwalkar
    elif byz_name == 'shej_agnostic':
        byz = attack.shej_agnostic

    return byz

def create_test_img_folder_for_imagenet():

    path = 'data/tiny-imagenet-200/val/'
    img_dir = os.path.join(path, 'images')
    fp = open(os.path.join(path, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    img_dict = {}
    for line in data:
        words = line.split('\t')
        img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def load_data(dataset_name):

    if (dataset_name == 'mnist'):
        print("Loading MNIST")
        num_inputs, num_outputs= 28 * 28, 10
        net_name = 'dnn'
        batch_size, lr = 32, 0.01

        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]) 
        trainset = torchvision.datasets.MNIST(
                       root='./data',
                       train=True,
                       download='True', 
                       transform=transform
                    )
        train_data = torch.utils.data.DataLoader(
                         trainset,
                         batch_size=batch_size,
                         shuffle=False
                     )
        testset = torchvision.datasets.MNIST(
                      root='./data',
                      train=False,
                      download='True',
                      transform=transform
                  )
        test_data = torch.utils.data.DataLoader(
                        testset,
                        batch_size=batch_size,
                        shuffle=False
                    )

        del trainset, testset        
         
    elif (dataset_name == 'fmnist'):
        print("Loading FMNIST")
        num_inputs = 28 * 28
        num_outputs = 10
        net_name = 'dnn'
        batch_size, lr = 32, 0.01

        transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ]) 
        trainset = torchvision.datasets.FashionMNIST(
                       root='./data',
                       train=True,
                       download='True',
                       transform=transform
                   )
        train_data = torch.utils.data.DataLoader(
                        trainset,
                        batch_size=batch_size,
                        shuffle=False
                     )
        testset = torchvision.datasets.FashionMNIST(
                      root='./data',
                      train=False,
                      download='True',
                      transform=transform
                  )
        test_data = torch.utils.data.DataLoader(
                        testset,
                        batch_size=batch_size,
                        shuffle=False
                    )

        del trainset, testset        

    elif dataset_name == 'cifar10':
        print("Loading CIFAR-10")
        num_inputs = 32*32*3
        num_outputs = 10
        net_name = 'resnet18'
        batch_size, lr = 128, 0.01

        transform_train = transforms.Compose([
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ])
        transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ])        
        trainset = torchvision.datasets.CIFAR10(
                       root='./data',
                       train=True,
                       download='True',
                       transform=transform_train
                   )
        train_data = torch.utils.data.DataLoader(
                         trainset,
                         batch_size=batch_size,
                         shuffle=False
                     )
        testset = torchvision.datasets.CIFAR10(
                      root='./data',
                      train=False,
                      download='True',
                      transform=transform_test
                  )
        test_data = torch.utils.data.DataLoader(
                        testset,
                        batch_size=batch_size,
                        shuffle=False
                    )

        del trainset, testset
    
    elif dataset_name == 'imagenet':

        #create_test_img_folder_for_imagenet()
        num_inputs, num_outputs = 64*64*3, 200
        net_name, batch_size, lr = 'resnet-tiny', 256, 0.001
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        trainset = torchvision.datasets.ImageFolder(
                root='./data/tiny-imagenet-200/train/',
                transform=transform_train
                )
        train_data = torch.utils.data.DataLoader(
                         trainset,
                         batch_size=batch_size,
                         shuffle=True
                     )
        testset = torchvision.datasets.ImageFolder(
                root='./data/tiny-imagenet-200/val/images/',
                transform=transform_test
                )
        test_data = torch.utils.data.DataLoader(
                         testset,
                         batch_size=batch_size,
                         shuffle=True
                     )
        del trainset, testset

    elif dataset_name == 'shakespeare':
        print("Loading Shakespeare dataset")
        file, file_len = read_file('shakespeare.txt')
        return file, None, None, None, None

    else:
        sys.exit('Not Implemented Dataset!')

    trainObject = TrainObject(dataset_name, net_name, train_data, test_data, num_inputs, num_outputs, lr, batch_size)

    return trainObject



def distribute_data_by_label(device,
                             batch_size,
                             bias_weight,
                             train_data,
                             num_workers,
                             num_inputs,
                             num_outputs,
                             net_name
                             ):

    #if bias_weight == 1:

    if net_name == 'lstm':
        distributed_data = []
        distributed_label = []
        wts = torch.ones(num_workers).to(device) / num_workers
        for i in range(num_workers+1):
            if (i<num_workers):
                chunk = train_data[i*200*32: (i+1)*200*32+1]
            else: 
                chunk = train_data[i*200*32+1:]
            distributed_data.append(char_tensor(chunk[:-1]).to(device))
            distributed_label.append(char_tensor(chunk[1:]).to(device))
        return distributed_data, distributed_label, None

    other_group_size = (1-bias_weight) / (num_outputs-1)
    worker_per_group = num_workers / (num_outputs)
    distributed_data = [[] for _ in range(num_workers)]
    distributed_label = [[] for _ in range(num_workers)] 
    batch_ctr = 0
    for _, (data, label) in enumerate(train_data):
        sample_ctr = 0
        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1-bias_weight) / (num_outputs-1) + bias_weight
            lower_bound = (y.item()) * (1-bias_weight) / (num_outputs-1)
            np.random.seed(batch_size*batch_ctr + sample_ctr)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()
            
            # assign a data point to a worker
            sample_ctr += 1
            np.random.seed(batch_size*batch_ctr + sample_ctr)
            rd = np.random.random_sample()
            selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
            if (bias_weight == 0): selected_worker = np.random.randint(num_workers)
            distributed_data[selected_worker].append(x.to(device))
            distributed_label[selected_worker].append(y.to(device))
        batch_ctr += 1
    # concatenate the data for each worker
    distributed_data = [(torch.stack(data, dim=0)).squeeze(0) for data in distributed_data] 
    distributed_label = [(torch.stack(labels, dim=0)).squeeze(0) for labels in distributed_label]
    
    # random shuffle the workers
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    distributed_data = [distributed_data[i] for i in random_order]
    distributed_label = [distributed_label[i] for i in random_order]
    wts = torch.zeros(len(distributed_data)).to(device)
    for i in range(len(distributed_data)):
        wts[i] = len(distributed_data[i])
    wts = wts/torch.sum(wts)
    distributedData = DistributedData(distributed_data, distributed_label, wts)
    return distributedData
