import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import models
import torch.nn as nn
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
import pdb

class TrainObject:
    def __init__(self, dataset_name, net_name, train_data, train_labels,
                 test_data, num_inputs, num_outputs, lr, batch_size):
        self.dataset_name = dataset_name
        self.net_name = net_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
        self.batch_size = batch_size

class DistributedData:
    def __init__(self, distributed_input, distributed_output, wts, label_distribution):
        self.distributed_input = distributed_input
        self.distributed_output = distributed_output
        self.wts = wts
        self.label_distribution = label_distribution

def load_data(dataset_name, batch_size, lr, fraction=1.0):
    if dataset_name=='mnist':
        num_inputs, num_outputs = 28*28, 10
        net_name = 'lenet'
        if batch_size is None: batch_size=32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
        if fraction < 1.0:
            size = int(len(full_train)*fraction)
            full_train, _ = random_split(full_train, [size, len(full_train)-size])
        loader = torch.utils.data.DataLoader(full_train,batch_size=len(full_train),shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)
        test_data = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    elif dataset_name=='cifar10':
        num_inputs, num_outputs = 32*32*3,10
        net_name='cifar_cnn'
        if batch_size is None: batch_size=128
        transform_train=transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        full_train=torchvision.datasets.CIFAR10('./data',train=True,
                                                transform=transform_train,download=True)
        if fraction<1.0:
            size=int(len(full_train)*fraction)
            full_train,_=random_split(full_train,[size,len(full_train)-size])
        loader=torch.utils.data.DataLoader(full_train,batch_size=len(full_train),shuffle=False)
        data,labels=next(iter(loader))
        testset=torchvision.datasets.CIFAR10('./data',train=False,
                                             transform=transform_test,download=True)
        test_data=torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False)

    elif dataset_name=='agnews':
        """
        We will:
        1) Download AG_NEWS dataset
        2) Tokenize and build vocab
        3) Convert each example into a padded tensor of token IDs
        4) Return TrainObject with data/labels in-memory, ready for distribution
        """
        if batch_size is None:
            batch_size = 32  # A reasonable default for text

        # 1) Load raw AG_NEWS via TorchText
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))

        # 2) Tokenize and build vocabulary
        tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        raw_train = []
        for (lbl, txt) in train_iter:
            # lbl in {1,2,3,4}, txt is raw string
            raw_train.append((lbl, txt))
            counter.update(tokenizer(txt))
        vocab = list(counter.keys())
        # Keep tokens that appear at least once (already in Counter)
        # Or you could do min_freq=2, etc. to filter more aggressively

        stoi = {word: i+2 for i, word in enumerate(vocab)}  # +2 to leave room for PAD=0, UNK=1
        stoi["<PAD>"] = 0
        stoi["<UNK>"] = 1

        def text_pipeline(txt):
            return [stoi.get(token, 1) for token in tokenizer(txt)]

        # 3) Convert each example => (tensor_of_token_ids, label)
        #    We'll also track the maximum sequence length to pad
        max_len = 0
        train_data_list, train_label_list = [], []
        for (lbl, txt) in raw_train:
            token_ids = text_pipeline(txt)
            max_len = max(max_len, len(token_ids))
            train_data_list.append(token_ids)
            # AG_NEWS labels are 1..4 => use them as is (code does int(lbl) - 1 later if needed)
            train_label_list.append(lbl)

        # convert to padded Tensors
        def pad_sequence(seq, max_len):
            # pad or truncate to max_len
            if len(seq) >= max_len:
                return seq[:max_len]
            else:
                return seq + [0]*(max_len - len(seq))  # 0 => <PAD>

        padded_train_data = []
        for token_ids in train_data_list:
            padded_train_data.append(torch.tensor(pad_sequence(token_ids, max_len), dtype=torch.long))

        data = torch.stack(padded_train_data)  # shape [N, max_len]
        labels = torch.tensor(train_label_list, dtype=torch.long)  # shape [N]
        labels = labels - 1 # Now labels are 0..3
        # 4) For the test set, we'll store it as a DataLoader similar to MNIST/CIFAR
        #    so we can evaluate with your existing evaluate_global_metrics
        #    That function expects (images, labels) from each batch, but we'll pass (text, label).
        #    We'll do something minimal here:
        raw_test = list(test_iter)
        # find max_len in test set if we want separate
        max_len_test = 0
        test_data_list, test_label_list = [], []
        for (lbl, txt) in raw_test:
            token_ids = text_pipeline(txt)
            max_len_test = max(max_len_test, len(token_ids))
            test_data_list.append(token_ids)
            test_label_list.append(lbl)

        # We'll match training max_len for simplicity or choose the largest of the two
        final_max_len = max_len  # or max(max_len, max_len_test)
        padded_test_data = []
        for token_ids in test_data_list:
            padded_test_data.append(torch.tensor(pad_sequence(token_ids, final_max_len), dtype=torch.long))

        test_data_tensors = torch.stack(padded_test_data)  # shape [N_test, final_max_len]
        test_label_tensors = torch.tensor(test_label_list, dtype=torch.long)
        test_label_tensors = test_label_tensors - 1
        # We'll pass a custom dataset => DataLoader
        test_ds = torch.utils.data.TensorDataset(test_data_tensors, test_label_tensors)
        test_data = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        num_inputs = final_max_len       # "input dimension" = sequence length
        num_outputs = 4                  # AG_NEWS has 4 classes
        net_name = 'agnews_net'          # We'll define this in models.py  

        

    
    return TrainObject(dataset_name, net_name, data, labels, test_data,
                       num_inputs, num_outputs, lr, batch_size)

def distribute_data_noniid(data_tuple, num_nodes, non_iidness, inp_dim, out_dim,
                           net_name, device, min_samples=0):
    """
    Distribute data among 'num_nodes' using a Dirichlet approach.
    If min_samples>0, ensure each node gets at least 'min_samples'.
    """
    data, labels = data_tuple
    class_indices = [[] for _ in range(out_dim)]
    for i,lbl in enumerate(labels):
        class_indices[lbl.item()].append(i)

    # set alpha based on non_iidness
    if non_iidness == 0:
        alpha = 100.0
    else:
        alpha = 1.0/(non_iidness+1e-6)

    rng = np.random.default_rng(42)  # fixed seed for data distribution
    portions = [rng.dirichlet([alpha]*num_nodes) for _ in range(out_dim)]

    distributed_input = [[] for _ in range(num_nodes)]
    distributed_output= [[] for _ in range(num_nodes)]

    # 1) assign samples of each class via Dirichlet fractions
    for cls, p in enumerate(portions):
        cls_idx = class_indices[cls]
        rng.shuffle(cls_idx)
        cumsum=(np.cumsum(p)*len(cls_idx)).astype(int)
        prev=0
        for node_id,boundary in enumerate(cumsum):
            node_indices = cls_idx[prev:boundary]
            prev=boundary
            for idxd in node_indices:
                distributed_input[node_id].append(data[idxd])
                distributed_output[node_id].append(labels[idxd])

    # 2) Now we have a certain distribution. Next, enforce at least min_samples.
    if min_samples > 0:
        # find nodes with less than min_samples and reassign from big nodes
        # This is a simple approach: for each "small" node, we transfer from the largest node
        done = False
        while True:
            sizes = [len(distributed_input[i]) for i in range(num_nodes)]
            small_nodes = [i for i,s in enumerate(sizes) if s<min_samples]
            if not small_nodes:
                break
            # pick the largest node
            largest_node = max(range(num_nodes), key=lambda j: sizes[j])
            if sizes[largest_node] <= min_samples:
                # can't fix it further, break
                break
            for snode in small_nodes:
                needed = min_samples - sizes[snode]
                if needed<=0:
                    continue

                # transfer 'needed' samples from largest_node if possible
                # for simplicity, pop from the *end*
                if len(distributed_input[largest_node])>= needed:
                    to_move_x = distributed_input[largest_node][-needed:]
                    to_move_y = distributed_output[largest_node][-needed:]
                    distributed_input[largest_node] = distributed_input[largest_node][:-needed]
                    distributed_output[largest_node]= distributed_output[largest_node][:-needed]

                    distributed_input[snode].extend(to_move_x)
                    distributed_output[snode].extend(to_move_y)

                # recalc
                sizes[snode] += needed
                sizes[largest_node] -= needed
            # repeat until no small_nodes remain or we can't fix it further

    # 3) Convert to Tensors on 'device' (none will be empty if min_samples>0)
    for i in range(num_nodes):
        # stack them
        distributed_input[i] = torch.stack(distributed_input[i]).to(device)
        distributed_output[i] = torch.stack(distributed_output[i]).to(device)

    # 4) compute weighting for FedSGD
    wts = torch.tensor([len(distributed_input[i]) for i in range(num_nodes)],
                       dtype=torch.float32)
    if wts.sum() > 0:
        wts = wts / wts.sum()
    else:
        wts = torch.ones(num_nodes, dtype=torch.float32)/num_nodes

    # label distribution stats
    label_distribution = np.zeros((num_nodes, out_dim))
    for i in range(num_nodes):
        for y in distributed_output[i]:
            label_distribution[i][y.item()] += 1

    return DistributedData(distributed_input, distributed_output, wts, label_distribution)

def model_to_vec(net):
    return torch.cat([p.data.view(-1) for p in net.parameters()])

def vec_to_model(vec, net_name, num_inp, num_out, device):
    net = models.load_net(net_name, num_inp, num_out, device)
    idx=0
    with torch.no_grad():
        for p in net.parameters():
            size=p.numel()
            p.data=vec[idx:idx+size].view(p.shape).to(device)
            idx+=size
    return net

def evaluate_global_metrics(net, test_data, device):
    criterion=nn.CrossEntropyLoss()
    net.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images,lab in test_data:
            images, lab = images.to(device), lab.to(device)
            out=net(images)
            loss=criterion(out,lab)
            total_loss+=loss.item()
            _,pred=torch.max(out,1)
            correct+=(pred==lab).sum().item()
            total+=lab.size(0)
    return total_loss,(correct/total)*100

def get_input_shape(dataset_name):
    if dataset_name=='mnist':
        return [1,28,28]
    elif dataset_name=='cifar10':
        return [3,32,32]
    elif dataset_name=='agnews':
        return [207]

def create_k_random_regular_graph(n, k, device):
    W = torch.zeros((n,n),device=device)
    for i in range(n):
        W[i,i]=1
        half=k//2
        for j in range(1,half+1):
            W[i,(i+j)%n]=1
            W[i,(i-j)%n]=1
        if k%2==1:
            W[i,(i+half+1)%n]=1
    W = W / (W.sum(dim=1,keepdim=True))
    perm = torch.randperm(n, device=device)
    W = W[perm][:,perm]
    return W
