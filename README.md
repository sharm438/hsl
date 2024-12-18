Below is an updated `README.md` with the requested changes and examples.

---

# README

## Overview

This code implements various distributed and federated learning paradigms:

- **Federated Learning (FL)**: A server aggregates updates from multiple clients (spokes).
- **Peer-to-Peer (P2P) Learning**: Nodes form a k-regular graph and aggregate by averaging their models with neighbors.
- **Hubs-and-Spokes Learning (HSL)**: Multiple hubs coordinate with spokes. Hubs first aggregate from spokes and among themselves, and then spokes aggregate from the hubs. This configuration is more complex and allows various topologies and weighting schemes.

This code supports non-IID data distributions, multiple datasets (MNIST and CIFAR-10), different sampling strategies (round-robin or random), and flexible configurations for number of nodes, hubs, local training steps, and more.

## Inputs and Arguments

- `--exp`: Experiment name (for output files).
- `--dataset`: `mnist` or `cifar10`.
- `--fraction`: Fraction of the dataset to use for training.
- `--bias`: Degree of non-IIDness. 0.0 is nearly IID; higher values lead to more skewed data distribution.
- `--aggregation`: Type of aggregation:
  - `fedsgd` for Federated Learning.
  - `p2p` for Peer-to-Peer Learning.
  - `hsl` for Hubs-and-Spokes Learning.
- `--num_spokes`: Number of spokes (clients).
- `--num_hubs`: Number of hubs (for HSL).
- `--num_rounds`: Number of global training rounds.
- `--num_local_iters`: Number of local training steps (batches) each node runs per round.
- `--batch_size`: Batch size for local training.
- `--eval_time`: Evaluate and record metrics every `eval_time` rounds.
- `--gpu`: GPU index to use (`-1` for CPU).
- `--k`: Parameter for k-regular graph among hubs (HSL) or peers (P2P).
- `--spoke_budget`: Number of hubs each spoke connects to in HSL.
- `--lr`: Learning rate for local training.
- `--sample_type`: `round_robin` or `random` batch sampling at the nodes.
- `--bidirectional`: For HSL, if `1`, spokes choose the same hubs in the final step as in the initial step; otherwise they choose randomly each time.
- `--self_weights`: For HSL, if `1`, spokes include their own model in the final averaging step after receiving models from chosen hubs.

## HSL-Specific Flags and Logic

- **`--bidirectional`**:  
  In HSL, the aggregation occurs in multiple steps. If `bidirectional=1`, spokes connect to the same set of hubs in both the initial and final aggregation steps. If `0`, they randomly choose new hubs in the final step.
  
- **`--self_weights`**:  
  If `self_weights=1`, spokes incorporate their own model into the final aggregation step along with the hubs' models. If `0`, spokes only average the hubs' models.

## Outputs

- Metrics (accuracy, loss) are recorded every `eval_time` rounds and saved to a `metrics.json` file after training completes.
- Depending on the aggregation:
  - FL: Global accuracy and loss.
  - P2P: Local accuracies and losses of each node.
  - HSL: Accuracy at the spokes (and potentially other metrics).

## Example Runs

### HSL Example

This example runs HSL on CIFAR-10 with non-IID data, self weights enabled, multiple hubs, and a moderate number of spokes:

```bash
python main.py \
  --eval_time 1 \
  --num_spokes 10 \
  --num_hubs 4 \
  --num_local_iters 3 \
  --num_rounds 500 \
  --aggregation hsl \
  --spoke_budget 2 \
  --k 2 \
  --exp hsl_s10h4_b2k2 \
  --dataset cifar10 \
  --bias 0.5 \
  --self_weights 1
```

Explanation:
- Runs for 500 rounds.
- Evaluates after every round (`eval_time=1`).
- HSL with 10 spokes, 4 hubs, each spoke connects to 2 hubs.
- `k=2` for hub-to-hub connections.
- Self weights are enabled, so each spoke includes its own model in the final aggregation step.
- Non-IIDness `bias=0.5` ensures moderately skewed data distribution across spokes.
- Uses CIFAR-10, which is more complex and tests the robustness of this configuration.

### Federated Learning (FL) Example

```bash
python main.py \
  --exp fed_mnist \
  --dataset mnist \
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

Explanation:
- Federated learning on MNIST with 10 spokes and a low bias (nearly IID data).
- Each spoke runs 2 local iterations per round.
- Evaluates every 10 rounds.
- Uses GPU 0 if available.
- Low non-IIDness (`bias=0.1`).

### P2P Example

```bash
python main.py \
  --exp p2p_cifar \
  --dataset cifar10 \
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

Explanation:
- P2P scenario with 20 peers on CIFAR-10.
- k=2 regular graph among peers.
- Slightly non-IID data (`bias=0.5`).
- Evaluates every 20 rounds.
- Random sampling strategy for local batches.

## Tips

- If training is too fast or doesn't progress, adjust `num_local_iters`, `batch_size`, `bias`, or `fraction`.
- If CUDA memory errors occur, consider running local training on CPU (`--gpu -1`) or reducing the number of nodes.
- For subtle experiments, `eval_time` can be increased or decreased to manage overhead.

---

This `README.md` now includes details on the newly introduced `bidirectional` and `self_weights` flags for HSL and provides example runs for all three aggregation methods.