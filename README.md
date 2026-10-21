# Distributed Learning and Parallel Training

## Parallelism and synchronization

- **[Data-parallel SGD](https://docs.pytorch.org/docs/2.14/notes/ddp.html)**. Replicates the model across workers, computes gradients on different data and synchronizes updates. PyTorch DDP averages gradient contributions; effective batch size, loss reduction and synchronization determine the update.
- **[DistBelief / Downpour SGD](https://research.google/pubs/large-scale-distributed-deep-networks/)** — 2012. Dean et al. combine model partitioning with asynchronous training replicas. Downpour lets workers update shared parameters without lockstep iterations, trading synchronization for delayed parameter information.
- **[Parameter servers](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu)** — 2014. Li et al. separate workers that compute updates from servers holding shared parameters. Flexible consistency models support asynchronous execution; consistency and fault tolerance are explicit system choices.
- **[HOGWILD!](https://arxiv.org/abs/1106.5730)** — 2011. Runs lock-free SGD updates in shared memory. Recht et al. analyze sparse update conflicts; its assumptions do not imply that arbitrary dense, asynchronous distributed training behaves like serial SGD.
- **[Local SGD](https://arxiv.org/abs/1805.09767)** — 2018. Workers take several local steps before averaging parameters. Stich derives communication reductions for analyzed convex problems; less frequent synchronization changes both communication cost and optimization behavior.
- **[Decentralized SGD](https://arxiv.org/abs/1705.09056)** — 2017. Workers mix parameters through a communication graph rather than a central server. Lian et al. identify bandwidth/latency regimes favoring this approach; graph connectivity and mixing affect convergence.

## Communication compression

- **[QSGD](https://arxiv.org/abs/1610.02132)** — 2016. Quantizes and encodes gradient updates, trading communication bits against added estimation variance. Alistarh et al. provide convergence results under assumptions; compression ratio alone does not determine time to accuracy.
- **[Error feedback](https://proceedings.mlr.press/v97/karimireddy19a.html)** — 2019. Accumulates compression residuals and adds them to later updates. Karimireddy et al. give counterexamples for plain signSGD and show how error feedback repairs analyzed failures; compressor conditions still matter.
- **[PowerSGD](https://arxiv.org/abs/1905.13727)** — 2019. Approximates matrix gradients by low-rank factors compatible with all-reduce. Vogels et al. report wall-clock gains against an optimized SGD backend on studied CNN/LSTM workloads, accounting for compression overhead.

## Partitioning models and training state

- **[Tensor parallelism / Megatron-LM](https://arxiv.org/abs/1909.08053)** — 2019. Splits transformer layer computations across devices. Shoeybi et al.’s intra-layer parallelism complements pipeline partitioning; communication occurs within layer execution rather than only between complete replicas.
- **[GPipe](https://arxiv.org/abs/1811.06965)** — 2018. Partitions a sequence of layers into stages and divides minibatches into microbatches. Pipelined execution increases model capacity across devices; stage imbalance and pipeline bubbles affect utilization.
- **[PipeDream](https://arxiv.org/abs/1806.03377)** — 2018. Overlaps forward and backward work across stages and versions weights for consistent passes. Its asynchronous pipeline schedule differs from GPipe’s minibatch-level synchronization.
- **[Zero Bubble pipelines](https://arxiv.org/abs/2401.10241)** — 2023/2024. Separate backward input-gradient and weight-gradient work to fill idle slots while retaining synchronous updates. The paper reports up to 23% throughput gain over 1F1B at similar memory limits; schedules depend on stage timings and memory.
- **[ZeRO](https://arxiv.org/abs/1910.02054)** — 2019. Partitions optimizer states, then gradients and parameters across data-parallel workers in successive stages. This reduces replicated training state; activations and temporary buffers remain separate memory costs.
- **[ZeRO-Infinity](https://arxiv.org/abs/2104.07857)** — 2021. Extends sharded training with GPU, CPU and NVMe memory plus data movement. Offloading expands capacity, but useful throughput depends on bandwidth and overlap; fitting a model does not establish fast training.

## Activation memory, attention and experts

- **[Sequence parallelism](https://arxiv.org/abs/2205.05198)** — 2022. Shards operations such as layer normalization and dropout alongside tensor parallelism. Korthikanti et al. combine this with selective recomputation to reduce replicated activations; optimizer-state sharding addresses a different cost.
- **[Ring Attention](https://arxiv.org/abs/2310.01889)** — 2023. Distributes sequence blocks and circulates keys and values for exact blockwise attention. Overlapping transfers with computation extends context capacity; hiding communication depends on block size, device throughput and interconnect bandwidth.
- **[MegaBlocks](https://arxiv.org/abs/2211.15841)** — 2022. Implements mixture-of-experts layers with block-sparse kernels so uneven token routing needs neither token dropping nor padded dense batches. Variable workloads use sparse matrix multiplication instead of equal-sized batches for every expert.

## Automatic parallelization

- **[GSPMD](https://arxiv.org/abs/2105.04663)** — 2021. Infers distributed tensor partitioning from a small set of sharding annotations and transforms the computation graph. Supports mixed parallelism while preserving a largely single-device programming interface.
- **[Alpa](https://arxiv.org/abs/2201.12023)** — 2022. Searches hierarchical inter- and intra-operator parallelization plans, combining data, operator and pipeline choices. Its comparisons concern the evaluated models and clusters; a plan is tied to the execution environment.

## Scheduling and recovery

- **[Pollux](https://arxiv.org/abs/2008.12260)** — 2020. Jointly adapts GPU allocation and training configuration using goodput: throughput adjusted for statistical efficiency. Its cluster experiments reduce average completion time, linking resource scheduling to useful training progress rather than steps alone.
- **[CheckFreq](https://www.usenix.org/conference/fast21/presentation/mohan)** — 2021. Adapts checkpoint frequency from online profiling and overlaps persistence with training. Saving data-loader state preserves per-epoch sample coverage after recovery; checkpoint cost and lost work must both enter training-time comparisons.
- **[Gradient coding](https://proceedings.mlr.press/v70/tandon17a.html)** — 2017. Replicates data blocks and encodes gradients so the server can recover a full gradient despite a bounded number of stragglers. Trades redundant worker computation for synchronous progress; recovery depends on the code’s tolerance.

## Scaling evidence and measurement

- **[Large-batch ImageNet training](https://arxiv.org/abs/1706.02677)** — 2017. Goyal et al. match small-batch ResNet-50 accuracy at batch size 8,192 using learning-rate scaling and warmup. This is a tested training recipe, not a universal rule for scaling batch size.
- **[MLPerf Training](https://arxiv.org/abs/1910.01500)** — 2019. Compares time to reach specified quality targets under controlled workload and timing rules. Repeated runs address stochastic training time; throughput gains that delay convergence need not improve this metric.
- **[Time to target quality](https://arxiv.org/abs/1905.13727)**. Report wall-clock time at matched quality alongside throughput, memory and communication. Include initialization, compression and synchronization costs; faster steps can require more steps to reach the target.
- **[Scaling protocol](https://arxiv.org/abs/1706.02677)**. Distinguish fixed-total-work speedup from growth in batch or model size. Record workers, topology, precision, accumulation and failures; a larger parallel workload is not the same experiment as a faster fixed workload.

## Software

- **[PyTorch DDP](https://docs.pytorch.org/docs/2.14/notes/ddp.html)**. Synchronizes gradients across replicated models and can overlap communication with backpropagation. It does not shard the model or automatically divide input data; sampling remains part of the training setup.
- **[PyTorch FSDP](https://docs.pytorch.org/docs/2.14/fsdp.html)**. Shards parameters, gradients and optimizer state across workers, gathering parameters when needed. Sharding, resharding and CPU-offload choices trade memory against communication and transfer costs.
- **[Horovod](https://arxiv.org/abs/1802.05799)** — 2018. Uses collective communication, including ring reduction, to support distributed training with limited changes to training code. Sergeev and Del Balso’s original paper describes TensorFlow integration.

## Related topics

[Federated learning](https://mlpapers.org/federated-learning/) · [Optimization](https://mlpapers.org/optimization/) · [Neural networks](https://mlpapers.org/neural-nets/) · [Generalization](https://mlpapers.org/generalization/) · [Evaluation](https://mlpapers.org/evaluation/).
