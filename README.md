# Hubs and Spokes Learning
The repository contains the files required to run a simulation of HSL and compare it with relevant baselines as described in the work [here](https://ieeexplore.ieee.org/abstract/document/10206535).

Default run: python main.py

Args:

--aggregation: ['p2p', 'hsl', 'fedsgd'].

--num_spokes: Number of spokes in HSL, clients in FL, or peers in P2PL.

--num_hubs: Number of hubs if aggregation used is HSL.

--num_rounds: Number of communication rounds, 500 by default.

--num_local_iters: Number of local iterations of training at the spokes in every round.

--g: Number of gossip steps in every round of communication. Use when aggregation type is P2P or HSL.



--dataset: ['mnist', 'cifar10'].

--bias: Non-IID bias for data distribution among spokes, 0.5 by default. Use 0 for IID distribution.



--W_type: ['franke_wolfe', 'random_graph', 'erdos_renyi']. Use when aggregation type is P2P.

--num_edges: Number of undirected edges among the spokes. Use when W_type is 'random_graph' or 'erdos_renyi'.

--max_degree: Maximum degree a node can have. Use when W_type is 'franke_wolfe'.

--spoke_budget: Number of hubs a spoke can connect to. Use when aggregation type used is HSL.

--k: Number of hubs a hub connects to in a k-regular manner. Use when aggregation type is HSL.



--save_cdist: Set to 1 if consensus distance in every round is to be computed and saved. 

--eval_time: Number of rounds after which local test accuracy at the spokes will be evaluated periodically, 10 by default.

--save_time: Number of rounds after which metrics are saved periodically, 100 by default.

--exp: Experiment name for the current run after which the metric files will be saved.

--gpu: Use -1 to disable GPU, all other values expect an available GPU.



Output:
_test_acc.npy when aggregation used is FedSGD.

_local_test_acc.npy, _consensus_test_acc.npy when aggregation used is P2P or HSL containing the local test accuracy of every spoke and that of the average model, computed every eval_time rounds.

_predist.npy, _postdist.npy when save_cdist is set to 1 contains the consensus distance among the spokes before and after gossiping in every eval_time number of rounds.

_args.txt: A record of all arguments run in the current experiment. 


Example run: python main.py --aggregation p2p --W_type franke_wolfe --max_degree 2   --num_spokes 100 --dataset cifar10 --num_rounds 500 --num_local_iters 3 --bias 0.8 --exp example

This command will run P2PL optimized by Franke Wolfe algorithm with 100 nodes, each of which can have a maximum degree of 2. CIFAR-10 is distributed among the 100 nodes with a bias of 0.8, meaning that every node has 80 percent data samples of one kind of label, and the rest of the labels are distributed randomly among the rest 20 percent of the samples. Every node runs 3 local iterations of minibatch gradient descent before communicating with other nodes, repeating the process 500 times. The final results are saved with a prefix example.
