Below is an **updated README** that incorporates the **new P2P Local** (EL Local) aggregation and the **revised HSL** three-stage aggregation scheme, as well as example commands to run each mode. Comments are added throughout to clarify the roles of each hyperparameter and aggregator.


# README

## Overview

This code implements various distributed and federated learning paradigms:

1. **Federated Learning (FL / `fedsgd`)**  
   A server aggregates updates from multiple clients (spokes) via weighted averaging.

2. **Peer-to-Peer (P2P) Learning**  
   - **`p2p`** (EL Oracle): A *centrally coordinated* \(k\)-regular graph. Each node has a fixed indegree and outdegree = \(k\).  
   - **`p2p_local`** (EL Local): A purely *distributed* approach without a centrally maintained graph. Each node i **chooses** exactly \(k\) neighbors (outdegree = \(k\)), but the indegree is unconstrained (i.e., you can be chosen by any number of neighbors).  

3. **Hubs-and-Spokes Learning (HSL / `hsl`)**  
   Multiple hubs and spokes coordinate in **three** steps each round:
   1. Hubs gather from \(b_{hs}\) spokes (the hub’s *fixed indegree*).  
   2. Hubs run an “EL Local” mixing among themselves, each with an outdegree = \(b_{hh}\).  
   3. Spokes gather from \(b_{sh}\) hubs (the spoke’s *fixed indegree*).

This code supports non-IID data distributions, multiple datasets (MNIST and CIFAR-10), different sampling strategies (round-robin or random), and flexible configurations for the number of nodes, hubs, local training steps, and more.  

---

## Inputs and Arguments

A summary of the key command-line options:

- `--exp`: **Experiment name** (used to name the output JSON metrics file).
- `--dataset`: Which dataset to use. Currently supports:
  - `mnist`
  - `cifar10`
- `--fraction`: Fraction of the dataset to use (e.g. `1.0` uses the entire training set).
- `--bias`: Degree of non-IIDness:
  - `0.0` is effectively IID
  - Higher values → more skewed or “non-IID” distribution across nodes
- `--aggregation`: **Type of aggregation** among nodes:
  - `fedsgd` for Federated Learning
  - `p2p` for EL Oracle (centrally-coordinated P2P)
  - `p2p_local` for EL Local (fully distributed P2P)
  - `hsl` for Hubs-and-Spokes Learning (3-step aggregation)
- `--num_spokes`: Number of “spoke” (client) nodes.
- `--num_hubs`: Number of hubs (for HSL). Ignored if `aggregation` is not `hsl`.
- `--num_rounds`: Number of global training rounds.
- `--num_local_iters`: Number of local gradient steps (mini-batches) each node runs per round.
- `--batch_size`: Local batch size during training at each node.
- `--eval_time`: Evaluate and record metrics every `eval_time` rounds.
- `--gpu`: GPU index to use (e.g. `0` or `1`). Use `-1` to force CPU training.
- `--lr`: Learning rate for **local** training at each node.
- `--sample_type`: Method for selecting local mini-batches:
  - `round_robin`
  - `random`

### P2P-Specific Arguments

- `--k`:  
  - For **`p2p`** (EL Oracle): The outdegree (and indegree) is exactly \(k\), enforced by a centrally generated \(k\)-regular graph.  
  - For **`p2p_local`** (EL Local): The outdegree = \(k\), but the indegree is unconstrained.

### HSL-Specific Arguments

- `--b_hs`: **Step 1** (hub “indegree”) → Each hub randomly selects `b_hs` spokes and averages them.  
- `--b_hh`: **Step 2** (hub-to-hub EL Local) → Each hub picks `b_hh` neighbors among the hubs, plus itself, and averages all their models.  
- `--b_sh`: **Step 3** (spoke “indegree”) → Each spoke randomly selects `b_sh` hubs and averages their models.

> **Note**: The older HSL flags `--bidirectional` and `--self_weights` from the legacy code are retained for backward compatibility but are not used in the *new* 3-step HSL logic.

---

## HSL Logic (3 Steps)

1. **Hubs gather from `b_hs` spokes**:  
   - *Hub indegree* = `b_hs`; each hub picks `b_hs` spokes at random (or with replacement if `b_hs` > #spokes).  
   - The hub’s model becomes the average of these `b_hs` spoke models.

2. **Hubs mix among themselves with “EL Local”**:  
   - Each hub has outdegree = `b_hh`. The hubs pick `b_hh` other hubs at random, plus themselves, and average those models.  
   - Indegree is not fixed; some hub might be chosen by many peers, or none.

3. **Spokes gather from `b_sh` hubs**:  
   - *Spoke indegree* = `b_sh`; each spoke picks `b_sh` hubs at random, averages them, and that becomes the spoke’s updated model.

---

## Outputs

- A JSON file `<exp>_metrics.json` containing:
  - **Round indices**
  - **Accuracy & Loss** (depending on aggregator type)
    - FL (`fedsgd`): Global model metrics
    - P2P (`p2p` or `p2p_local`): Each node’s local metrics
    - HSL (`hsl`): Spoke metrics
- Console logs per evaluation round.

---

## Example Runs

### 1. **Federated Learning (FL) Example**

```bash
python main.py \
  --exp fed_mnist \
  --dataset mnist \
  --fraction 1.0 \
  --aggregation fedsgd \
  --num_spokes 10 \
  --num_rounds 100 \
  --num_local_iters 2 \
  --batch_size 32 \
  --eval_time 10 \
  --gpu 0 \
  --lr 0.01 \
  --bias 0.1
```

- Federated learning on MNIST with 10 spokes and nearly IID data (`bias=0.1`).
- 2 local gradient steps each round, batch size 32.
- Evaluates every 10 rounds; logs final metrics in `fed_mnist_metrics.json`.

---

### 2. **EL Oracle (P2P) Example**

```bash
python main.py \
  --exp p2p_cifar_oracle \
  --dataset cifar10 \
  --fraction 0.5 \
  --aggregation p2p \
  --num_spokes 20 \
  --num_rounds 200 \
  --num_local_iters 1 \
  --batch_size 64 \
  --eval_time 20 \
  --gpu 0 \
  --lr 0.005 \
  --k 2 \
  --sample_type random \
  --bias 0.5
```

- **`p2p`** uses a *centrally generated* \(k=2\) regular graph among the 20 nodes.
- Non-IIDness = 0.5, random sampling for local batches, CIFAR-10 but only 50% of the data.
- The system logs local accuracies for each node every 20 rounds.

---

### 3. **EL Local (P2P Local) Example**

```bash
python main.py \
  --exp p2p_local_mnist \
  --dataset mnist \
  --aggregation p2p_local \
  --num_spokes 10 \
  --num_rounds 50 \
  --num_local_iters 2 \
  --batch_size 32 \
  --eval_time 5 \
  --gpu -1 \
  --lr 0.01 \
  --k 3 \
  --sample_type round_robin \
  --bias 0.2
```

- **`p2p_local`** uses *outdegree = 3* for each node, with no centralized graph.
- 10 nodes, MNIST dataset, 50 rounds. Evaluates every 5 rounds.
- Runs on CPU (`--gpu -1`).

---

### 4. **HSL (Hubs-and-Spokes Learning) with New 3-Step Aggregation**

```bash
python main.py \
  --exp hsl_cifar_new \
  --dataset cifar10 \
  --aggregation hsl \
  --fraction 1.0 \
  --num_spokes 10 \
  --num_hubs 3 \
  --num_rounds 100 \
  --num_local_iters 3 \
  --batch_size 64 \
  --eval_time 10 \
  --gpu 0 \
  --bias 0.5 \
  --lr 0.01 \
  --b_hs 2 \
  --b_hh 1 \
  --b_sh 3
```

In this run:

1. **Step 1**: Each of the 3 hubs picks `b_hs=2` spokes, averages them.  
2. **Step 2**: Each hub picks `b_hh=1` other hub (plus itself) and averages → “EL Local” among the 3 hubs.  
3. **Step 3**: Each of the 10 spokes picks `b_sh=3` hubs and averages them to update.

- Biased (non-IID) CIFAR-10 with `bias=0.5`.
- 3 local gradient steps each round, batch size 64.
- Evaluates every 10 rounds.

---

## Tips

- Adjust `num_local_iters`, `batch_size`, `bias`, or `fraction` to tune training speed and difficulty.
- If training is slow or memory-limited, reduce `num_spokes`, `num_hubs`, or switch to CPU by using `--gpu -1`.
- For debugging or faster prototyping, reduce `eval_time` to evaluate more often (at the cost of runtime overhead).

---

**Enjoy experimenting with FL, P2P, and HSL methods!**