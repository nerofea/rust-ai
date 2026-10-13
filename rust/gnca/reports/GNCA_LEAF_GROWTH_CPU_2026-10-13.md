# Lemon Leaf GNCA — CPU Benchmark 03

## Performance results

| Execution mode | Threads | Median of 7 (ms) | Minimum (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|
| Sequential | 1 | 1.2927 | 1.0061 | 2.4698 |
| Rayon | 1 | 1.0103 | 0.7546 | 1.1789 |
| Rayon | 2 | 0.4935 | 0.4311 | 1.0969 |
| Rayon | 4 | 0.4570 | 0.3668 | 1.0413 |
| Rayon | 8 | 3.0537 | 0.3866 | 7.0580 |
| Rayon | 16 | 1.8894 | 0.5090 | 14.2574 |

**Best measured median speedup: 2.83× (sequential / Rayon 4 threads).** The thread-count scaling is nonmonotonic. Seven runs is too small for a generalizable hardware performance claim.

## Workload

- Simplified procedural lemon-leaf silhouette and central-vein objective; not real photographic training data.
- Rust-only CPU computation, 24×24 cells, 12 simulation steps, stochastic per-cell updates with a deterministic 75% update mask.
- Four state values per cell: growth, vein, x coordinate, y coordinate (two state channels and two fixed spatial features).
- Residual update with tanh activation, read-only old state, next-state buffers.
- 2 trainable 9-feature nonlinear update rules (18 weights).
- Loss after six and 12 steps, finite-difference training across unrolled simulation (48 iterations, deterministic masks). No analytic/reverse-mode autodifferentiation.
- Training time: 926.262774 ms (one recorded run).
- Initial reconstruction loss: 0.281250; final: 0.275190.
- Damage before regrowth loss: 0.278137; after regrowth: 0.284374 (**failed to regenerate**).
- Sequential/Rayon final states matched exactly in unit testing.
- Seven inference run samples per variant; median, minimum, and maximum recorded. Sequential run measured before Rayon variants.
- No GPU or CUDA execution.

## Interpretation

This is a **CPU neural-cellular leaf-growth prototype**, not a trained 16-channel growing neural cellular automaton. Its weights are trained end-to-end with finite differences across an unrolled simulation but there is no reverse-mode differentiation, no demonstrated leaf reconstruction quality, and no successful regeneration objective. Loss is evaluated on the same procedural target used for training, not an independent evaluation set.

## Next implementation requirements

1. Upgrade state from two dynamic channels to 16 continuous dynamic channels.
2. Train a true neural perception/update network through unrolled differentiable steps with reverse-mode gradients.
3. Add damage-and-recovery training samples to the objective and verify recovery on held-out masks.
4. Train with an actual standardized lemon-leaf reference and separate silhouette/venation/pigmentation labels.
5. Add warm-up iterations, alternate variant ordering, record raw timing samples, and evaluate larger 128²–512² grids.

## Reproduce

```bash
cargo test --bin leaf_growth
cargo run --release --bin leaf_growth
```