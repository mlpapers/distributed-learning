# Distributed Learning and Parallel Training

## Parallelism and synchronization

- **Data-parallel SGD** - Replicates the model across workers, computes gradients on different data and synchronizes updates. PyTorch DDP averages gradient contributions; effective batch size, loss reduction and synchronization determine the update. [docs](<https://docs.pytorch.org/docs/2.14/notes/ddp.html>).
- **HOGWILD!** - Runs lock-free SGD updates in shared memory. Recht et al. analyze sparse update conflicts; its assumptions do not imply that arbitrary dense, asynchronous distributed training behaves like serial SGD. [Niu et al. 2011](<https://arxiv.org/abs/1106.5730> "HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent; NeurIPS").
- **DistBelief / Downpour SGD** - Dean et al. combine model partitioning with asynchronous training replicas. Downpour lets workers update shared parameters without lockstep iterations, trading synchronization for delayed parameter information. [Dean et al. 2012](<https://research.google/pubs/large-scale-distributed-deep-networks/> "Large Scale Distributed Deep Networks; NeurIPS").
- **Parameter servers** - Li et al. separate workers that compute updates from servers holding shared parameters. Flexible consistency models support asynchronous execution; consistency and fault tolerance are explicit system choices. [Li et al. 2014](<https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu> "Scaling Distributed Machine Learning with the Parameter Server; OSDI").
- **Decentralized SGD** - Workers mix parameters through a communication graph rather than a central server. Lian et al. identify bandwidth/latency regimes favoring this approach; graph connectivity and mixing affect convergence. [Lian et al. 2017](<https://arxiv.org/abs/1705.09056> "Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent; NeurIPS").
- **Local SGD** - Workers take several local steps before averaging parameters. Stich derives communication reductions for analyzed convex problems; less frequent synchronization changes both communication cost and optimization behavior. [Stich 2019](<https://arxiv.org/abs/1805.09767> "Local SGD Converges Fast and Communicates Little; ICLR").
- **Backup-worker SGD** - Waits for the first m workers and discards stale gradients. With fixed heterogeneous worker speeds and identical gradient distributions, an appropriately chosen m achieves optimal time complexity within a logarithmic factor. Rapid speed changes can defeat it. [Begunov and Tyurin 2026](<https://arxiv.org/abs/2602.03802v2> "Do We Need Asynchronous SGD? On the Near-Optimality of Synchronous Methods; NeurIPS").

## Communication compression

- **QSGD** - Quantizes and encodes gradient updates, trading communication bits against added estimation variance. Alistarh et al. provide convergence results under assumptions; compression ratio alone does not determine time to accuracy. [Alistarh et al. 2017](<https://arxiv.org/abs/1610.02132> "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding; NeurIPS").
- **Error feedback** - Accumulates compression residuals and adds them to later updates. Karimireddy et al. give counterexamples for plain signSGD and show how error feedback repairs analyzed failures; compressor conditions still matter. [Karimireddy et al. 2019](<https://proceedings.mlr.press/v97/karimireddy19a.html> "Error Feedback Fixes SignSGD and other Gradient Compression Schemes; ICML").
- **PowerSGD** - Approximates matrix gradients by low-rank factors compatible with all-reduce. Vogels et al. report wall-clock gains against an optimized SGD backend on studied CNN/LSTM workloads, accounting for compression overhead. [Vogels et al. 2019](<https://arxiv.org/abs/1905.13727> "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization; NeurIPS").

## Partitioning models and training state

- **PipeDream** - Overlaps forward and backward work across stages and versions weights for consistent passes. Its asynchronous pipeline schedule differs from GPipe's minibatch-level synchronization. [Harlap et al. 2018](<https://arxiv.org/abs/1806.03377> "PipeDream: Fast and Efficient Pipeline Parallel DNN Training; arXiv").
- **Tensor parallelism / Megatron-LM** - Splits transformer layer computations across devices. Shoeybi et al.'s intra-layer parallelism complements pipeline partitioning; communication occurs within layer execution rather than only between complete replicas. [Shoeybi et al. 2019](<https://arxiv.org/abs/1909.08053> "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism; arXiv").
- **GPipe** - Partitions a sequence of layers into stages and divides minibatches into microbatches. Pipelined execution increases model capacity across devices; stage imbalance and pipeline bubbles affect utilization. [Huang et al. 2019](<https://arxiv.org/abs/1811.06965> "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism; NeurIPS").
- **ZeRO** - Partitions optimizer states, then gradients and parameters across data-parallel workers in successive stages. This reduces replicated training state; activations and temporary buffers remain separate memory costs. [Rajbhandari et al. 2020](<https://arxiv.org/abs/1910.02054> "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models; SC").
    - **ZeRO-Infinity** - Extends sharded training with GPU, CPU and NVMe memory plus data movement. Offloading expands capacity, but useful throughput depends on bandwidth and overlap; fitting a model does not establish fast training. [Rajbhandari et al. 2021](<https://arxiv.org/abs/2104.07857> "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning; arXiv").
- **Zero Bubble pipelines** - Separate backward input-gradient and weight-gradient work to fill idle slots while retaining synchronous updates. The paper reports up to 23% throughput gain over 1F1B at similar memory limits; schedules depend on stage timings and memory. [Qi et al. 2023](<https://arxiv.org/abs/2401.10241> "Zero Bubble Pipeline Parallelism; arXiv").

## Activation memory, attention and experts

- **Sequence parallelism** - Shards operations such as layer normalization and dropout alongside tensor parallelism. Korthikanti et al. combine this with selective recomputation to reduce replicated activations; optimizer-state sharding addresses a different cost. [Korthikanti et al. 2023](<https://arxiv.org/abs/2205.05198> "Reducing Activation Recomputation in Large Transformer Models; MLSys").
- **Ring Attention** - Distributes sequence blocks and circulates keys and values for exact blockwise attention. Overlapping transfers with computation extends context capacity; hiding communication depends on block size, device throughput and interconnect bandwidth. [Liu et al. 2023](<https://arxiv.org/abs/2310.01889> "Ring Attention with Blockwise Transformers for Near-Infinite Context; arXiv").
- **MegaBlocks** - Implements mixture-of-experts layers with block-sparse kernels so uneven token routing needs neither token dropping nor padded dense batches. Variable workloads use sparse matrix multiplication instead of equal-sized batches for every expert. [Gale et al. 2023](<https://arxiv.org/abs/2211.15841> "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts; MLSys").

## Automatic parallelization

- **GSPMD** - Infers distributed tensor partitioning from a small set of sharding annotations and transforms the computation graph. Supports mixed parallelism while preserving a largely single-device programming interface. [Xu et al. 2021](<https://arxiv.org/abs/2105.04663> "GSPMD: General and Scalable Parallelization for ML Computation Graphs; arXiv").
- **Alpa** - Searches hierarchical inter- and intra-operator parallelization plans, combining data, operator and pipeline choices. [Zheng et al. 2022](<https://arxiv.org/abs/2201.12023> "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning; arXiv").

## Scheduling and recovery

- **Gradient coding** - Replicates data blocks and encodes gradients so the server can recover a full gradient despite a bounded number of stragglers. Trades redundant worker computation for synchronous progress; recovery depends on the code's tolerance. [Tandon et al. 2017](<https://proceedings.mlr.press/v70/tandon17a.html> "Gradient Coding: Avoiding Stragglers in Distributed Learning; ICML").
- **Pollux** - Jointly adapts GPU allocation and training configuration using goodput: throughput adjusted for statistical efficiency. Its cluster experiments reduce average completion time, linking resource scheduling to useful training progress rather than steps alone. [Qiao et al. 2020](<https://arxiv.org/abs/2008.12260> "Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning; arXiv").
- **CheckFreq** - Adapts checkpoint frequency from online profiling and overlaps persistence with training. Saving data-loader state preserves per-epoch sample coverage after recovery; checkpoint cost and lost work must both enter training-time comparisons. [Mohan et al. 2021](<https://www.usenix.org/conference/fast21/presentation/mohan> "{CheckFreq}: Frequent, {Fine-Grained} {DNN} Checkpointing; FAST").

## Scaling evidence and measurement

- **Large-batch ImageNet training** - Goyal et al. match small-batch ResNet-50 accuracy at batch size 8,192 using learning-rate scaling and warmup. This is a tested training recipe, not a universal rule for scaling batch size. [Goyal et al. 2017](<https://arxiv.org/abs/1706.02677> "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour; arXiv").
- **MLPerf Training** - Compares time to reach specified quality targets under controlled workload and timing rules. Repeated runs address stochastic training time; throughput gains that delay convergence need not improve this metric. [Mattson et al. 2020](<https://arxiv.org/abs/1910.01500> "MLPerf Training Benchmark; MLSys").

## Software

- **PyTorch DDP** - Synchronizes gradients across replicated models and can overlap communication with backpropagation. It does not shard the model or automatically divide input data; sampling remains part of the training setup. [docs](<https://docs.pytorch.org/docs/2.14/notes/ddp.html>).
- **PyTorch FSDP** - Shards parameters, gradients and optimizer state across workers, gathering parameters when needed. Sharding, resharding and CPU-offload choices trade memory against communication and transfer costs. [docs](<https://docs.pytorch.org/docs/2.14/fsdp.html>).
- **Horovod** - Uses collective communication, including ring reduction, to support distributed training with limited changes to training code. Sergeev and Del Balso's original paper describes TensorFlow integration. [Sergeev and Del Balso 2018](<https://arxiv.org/abs/1802.05799> "Horovod: fast and easy distributed deep learning in TensorFlow; arXiv").

## Related topics

- [Federated learning](https://mlpapers.org/federated-learning/)
- [Optimization](https://mlpapers.org/optimization/)
- [Neural networks](https://mlpapers.org/neural-nets/)
- [Generalization](https://mlpapers.org/generalization/)
- [Evaluation](https://mlpapers.org/evaluation/)
