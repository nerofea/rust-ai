# GNCA Rust CPU Parallelism — Baseline Benchmark Report

**Benchmark ID:** GNCA-CPU-2026-10-13-001  
**Date:** 2026-10-13  
**Project:** Growing Neural Cellular Automata (GNCA)  
**Source directory on Anna:** `/home/nerofea/nerofea/rust/gnca`  
**Report type:** Single-run, optimized CPU baseline (not a statistical benchmark)

## Executive results

| Measurement | Sequential Rust | Rayon parallel Rust |
|---|---:|---:|
| Wall-clock time for 30 steps | 96.821905 ms | 29.058906 ms |
| Grid size | 512 × 512 | 512 × 512 |
| Cells per step | 262,144 | 262,144 |
| Total cell updates | 7,864,320 | 7,864,320 |
| Average time per step | 3.227397 ms | 0.968630 ms |
| Approximate cell updates per second | 81.22 million | 270.63 million |

**Observed speedup:** 3.33× (96.821905 / 29.058906).  
**Observed time reduction:** approximately 69.99%.  
**Correctness check:** identical cell states after all 30 steps = `true`.  
**Unit test result:** 10 passed, 0 failed.  
**Release process exit code:** 0.

## Machine and software context

- Host: `DESKTOP-GVIP9TT` ("Anna"), running the program under WSL/Linux.
- CPU: Intel Core i7-11700F @ 2.50 GHz, 8 physical cores / 16 logical CPUs reported by `lscpu`.
- GPU: NVIDIA GeForce RTX 3070 was detected, **not used in this test**.
- Rust toolchain observed during setup: `rustc 1.95.0`, `cargo 1.95.0`.
- Rust crate: `gnca` version `0.1.0`, edition 2024.
- Parallel crate: `rayon 1.12.0`.
- Build/run command: `cargo run --release`.
- Build mode: Rust `release` profile (optimized); no custom RUSTFLAGS documented.
- Rayon worker thread count: default configuration, not explicitly pinned or measured.

## Algorithm and experimental procedure

The program initializes two equivalent 512 × 512 `Grid` instances, each containing `Cell { energy: f32, alive: bool, age: u32 }`, initially with energy 0.3, except the center cell, which starts at 1.0.

Both implementations execute 30 synchronous steps with rate `0.25`. Each cell evaluates its available 3 × 3 neighborhood excluding itself (up to eight neighbors, fewer on edges). It computes the mean neighbor energy and applies:

```text
input       = rate * (mean_neighbor_energy - current_energy)
next_energy = clamp(current_energy + input, 0.0, 1.0)
next_alive  = next_energy > 0.5
next_age    = current_age + 1 if next_alive, otherwise current_age
```

The model uses **two preallocated vectors** (current/next). Each step reads exclusively from the current vector, writes exclusively to the next vector, then swaps buffers. This avoids read/write races and makes each generation synchronous.

- **Sequential:** nested row/column loops update `next` in row-major order.
- **Parallel:** Rayon `par_iter_mut().enumerate()` divides the `next` buffer among worker threads; immutable reads use the previous `cells` buffer.
- The sequential run was timed **first**, followed by Rayon, using `std::time::Instant`.
- Both timed sections included all 30 simulation steps and buffer swaps, but excluded grid allocation and initialization, release compilation, printing, and the final equality comparison.
- The program checked `sequential.cells == parallel.cells`, comparing `energy`, `alive`, and `age` for every cell.

## Captured console output

```text
GNCA baseline: 512x512, 30 steps
Sequential: 96.821905ms
Rayon:      29.058906ms
Speedup:    3.33x
Identical cell states: true
```

The tests reported:

```text
running 10 tests
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Interpretation and limitations

The observed 3.33× advantage applies **only to this particular single optimized run on Anna**. It is not evidence of a universal Rayon speedup or a stable estimate of mean throughput.

Limitations:
1. One measured execution per implementation: no warm-ups, repetitions, distribution, variance, confidence interval, or CPU-affinity control.
2. Sequential always ran before parallel, so order, thermal state, power management, OS scheduler and cache effects could bias results.
3. The number of Rayon worker threads was not captured; reported logical CPU count does not prove how many workers actually ran.
4. This is a **deterministic diffusion-style cellular automaton**, not a trained neural cellular automata inference workload.
5. There was **no CUDA execution, GPU kernel, host/device transfer, GPU profiling, or energy measurement**.
6. The update counts are derived from grid dimensions and step count; updates/second are calculated estimates, not independent hardware counters.
7. No Git commit hash for the original WSL project was recorded; the project files were initially untracked at the time of inspection.

## Reproduction

Inside Anna's WSL:

```bash
cd ~/nerofea/rust/gnca
cargo test
cargo run --release
```

The implementation prints both timings, calculated speedup, and whether the resulting cell states match. Results will vary by run.

## Follow-up benchmarks

1. Repeat each variant 20+ times after warm-up; report median, p5/p95, and variance.
2. Alternate execution order and control threads using `RAYON_NUM_THREADS` (e.g. 1, 2, 4, 8, 16).
3. Sweep grid sizes (128², 256², 512², 1024², 2048²) and fixed iteration counts.
4. Store machine metadata, complete raw samples, and source Git revision alongside each result.
5. Profile memory bandwidth, work scheduling, and CPU utilization before comparing to a CUDA kernel.

**Evidence source:** stdout from the GNCA release run and 10-test suite on `DESKTOP-GVIP9TT`, observed through Remote Desktop Commander on 2026-10-13.
