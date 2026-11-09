# Rust AI — GPU technology foundation

Production-oriented Rust GPU technology, separate from the deferred-count benchmark in portable-compute-observability.

## Current runnable scope

CPU reference tensor shape checks and vector addition. GPU execution is **not yet implemented**; the CUDA/wgpu directories are design targets, not operational backends.

```bash
cd gpu
cargo test --workspace
cargo run -p buffer-demo
```

## Roadmap

1. Backend-independent buffers, shape/stride validation, dtype contracts.
2. wgpu allocation, transfer, compute dispatch and synchronization.
3. CUDA backend with explicit streams and memory lifetime.
4. Reduction, scan, matmul and sparse primitives.
5. Attention, inference and computer-vision pipelines.
6. Export repeatable workloads to external observability tooling without importing its benchmark implementation.
