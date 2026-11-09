# Architecture

- `gpu-core`: typed descriptors and validation.
- `gpu-runtime`: execution contracts, currently CPU reference only.
- `examples/buffer-demo`: runnable reference example.

Planned: `engines/wgpu`, `engines/cuda`, `primitives`, `ai`, `vision`. Do not claim GPU acceleration until backends are implemented and verified on hardware.
