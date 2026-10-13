# Trained Neural Cellular Automata — Rust CPU Inference Benchmark 02

## Results (single run)

| Metric | Sequential Rust | Rayon |
|---|---:|---:|
| Inference latency, 30 steps | 380.474776 ms | 57.886926 ms |
| Cell inferences | 7,864,320 | 7,864,320 |
| Derived throughput | 20.67 million/s | 135.86 million/s |

**Measured speedup: 6.573×.** Grid: **512 × 512**; steps: **30**; exact output equality: **true**. These are one-shot release-build timings, not medians.

## Model and training

- Rust (edition 2024), 9-input → 16-ReLU-hidden → 1-sigmoid-output MLP.
- Supervised training on all 512 possible 3×3 binary neighborhoods of Conway's Game of Life; manual backpropagation and full-batch gradient descent.
- 4,000 epochs, learning rate 0.5, deterministic initialization.
- Training wall time: **161.197944 ms** (one run).
- Truth-table training accuracy: **100.00%**; this is **not independent generalization accuracy**.
- Weights saved as `models/nca_weights.txt` and reloaded before inference.
- Each inference step evaluates the trained model for every neighborhood with zero padding at boundaries.
- Double buffers avoid read-write interference; Rayon uses `par_iter_mut`.
- Both paths warm up once before 30 timed steps. Sequential always runs first.
- CPU: Intel Core i7-11700F (8 cores / 16 logical CPUs). **No GPU/CUDA execution**.

## Recorded output

```text
trained network: 9 input -> 16 ReLU -> 1 sigmoid
training task: supervised nonlinear Game of Life update; not image-growth training
truth-table accuracy: 100.00%
grid 512x512 steps 30 cell inferences 7864320
sequential 380.474776 ms
rayon 57.886926 ms
speedup 6.573x
identical outputs true
```

## Reproduction

```bash
cargo test --bin neural_ca
cargo run --release --bin neural_ca -- train
cargo run --release --bin neural_ca -- bench
```

## Limitations

This benchmark measures genuine **trained neural-rule inference**, but **not target-image-trained growing neural cellular automata**. It does not yet feature continuous multichannel state, residual updates, stochastic cell masks, differentiable unrolled growth, regeneration objectives, or CUDA execution. The accuracy dataset is also the training dataset. Only one release timing per variant was recorded; repeated samples and controlled thread counts are required for robust conclusions.
