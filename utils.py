import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import models
import torch.nn as nn

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

def load_data(dataset_name, batch_size, fraction=1.0):
    """
    Loads MNIST or CIFAR10, returns entire dataset in a single batch (data, labels).
    If fraction < 1.0, we only keep that fraction of the training set.
    """
    if dataset_name=='mnist':
        num_inputs, num_outputs = 28*28, 10
        net_name = 'lenet'
        lr = 0.01
        if batch_size is None: batch_size=32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = torchvision.datasets.MNIST('./data', train=True, 
                                                transform=transform, download=True)
        if fraction < 1.0:
            size = int(len(full_train)*fraction)
            full_train, _ = random_split(full_train, [size, len(full_train)-size])
        loader = torch.utils.data.DataLoader(full_train,batch_size=len(full_train),shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.MNIST('./data', train=False, 
                                             transform=transform, download=True)
        test_data = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    elif dataset_name=='cifar10':
        num_inputs, num_outputs = 32*32*3, 10
        net_name = 'cifar_cnn'
        lr = 0.01
        if batch_size is None: batch_size=128
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        full_train = torchvision.datasets.CIFAR10('./data', train=True, 
                                                  transform=transform_train, download=True)
        if fraction < 1.0:
            size=int(len(full_train)*fraction)
            full_train, _=random_split(full_train,[size,len(full_train)-size])
        loader=torch.utils.data.DataLoader(full_train,batch_size=len(full_train),shuffle=False)
        data, labels=next(iter(loader))
        testset=torchvision.datasets.CIFAR10('./data', train=False, 
                                             transform=transform_test, download=True)
        test_data=torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False)

    return TrainObject(dataset_name, net_name, data, labels, 
                       test_data, num_inputs, num_outputs, lr, batch_size)

def distribute_data_noniid(data_tuple, num_nodes, non_iidness, 
                           inp_dim, out_dim, net_name, device):
    """
    Dirichlet-based non-iid data partitioning among num_nodes.
    non_iidness=0 => alpha=100 => fairly uniform
    otherwise alpha=1/(non_iidness+1e-6)
    """
    data, labels = data_tuple
    class_indices = [[] for _ in range(out_dim)]
    for i,l in enumerate(labels):
        class_indices[l.item()].append(i)
    if non_iidness==0:
        alpha=100.0
    else:
        alpha=1.0/(non_iidness+1e-6)
    rng=np.random.default_rng(42)
    portions=[rng.dirichlet([alpha]*num_nodes) for _ in range(out_dim)]

    distributed_input=[[] for _ in range(num_nodes)]
    distributed_output=[[] for _ in range(num_nodes)]

    for cls,p in enumerate(portions):
        cls_idx=class_indices[cls]
        rng.shuffle(cls_idx)
        cumsum=(np.cumsum(p)*len(cls_idx)).astype(int)
        prev=0
        for node_id,boundary in enumerate(cumsum):
            node_indices=cls_idx[prev:boundary]
            prev=boundary
            for idxd in node_indices:
                distributed_input[node_id].append(data[idxd])
                distributed_output[node_id].append(labels[idxd])

    for i in range(num_nodes):
        if len(distributed_input[i])>0:
            distributed_input[i]=torch.stack(distributed_input[i]).to(device)
            distributed_output[i]=torch.stack(distributed_output[i]).to(device)
        else:
            distributed_input[i]=torch.empty(0,*data.shape[1:]).to(device)
            distributed_output[i]=torch.empty(0,dtype=torch.long).to(device)

    wts=torch.tensor([len(distributed_input[i]) for i in range(num_nodes)],dtype=torch.float32)
    wts=wts/wts.sum()

    label_distribution=np.zeros((num_nodes,out_dim))
    for i in range(num_nodes):
        for y in distributed_output[i]:
            label_distribution[i][y.item()]+=1

    return DistributedData(distributed_input, distributed_output, wts, label_distribution)

def model_to_vec(net):
    """Flatten model parameters into a single 1D vector."""
    return torch.cat([p.data.view(-1) for p in net.parameters()])

def vec_to_model(vec, net_name, num_inp, num_out, device):
    """Load a 1D parameter vector back into a fresh net."""
    net = models.load_net(net_name, num_inp, num_out, device)
    idx=0
    with torch.no_grad():
        for p in net.parameters():
            size=p.numel()
            p.data=vec[idx:idx+size].view(p.shape).to(device)
            idx+=size
    return net

def evaluate_global_metrics(net, test_data, device):
    """
    Evaluate model on test set. 
    Returns (total_loss, accuracy).
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images,lab in test_data:
            images, lab = images.to(device), lab.to(device)
            out = net(images)
            loss = criterion(out, lab)
            total_loss += loss.item()
            _, pred = torch.max(out,1)
            correct += (pred==lab).sum().item()
            total += lab.size(0)
    return total_loss, (correct/total)*100

def get_input_shape(dataset_name):
    """Returns [channels, height, width] for trace shape."""
    if dataset_name=='mnist':
        return [1,28,28]
    elif dataset_name=='cifar10':
        return [3,32,32]

def create_k_random_regular_graph(n, k, device):
    """
    Create a random k-regular adjacency for n nodes. 
    We then row-normalize for aggregator usage.
    """
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
    # random permutation for "random" regular graph
    perm = torch.randperm(n, device=device)
    W = W[perm][:,perm]
    return W
