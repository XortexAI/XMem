# XMem Benchmarks

This directory contains benchmark harnesses for XMem.

- `longmemeval/`: Python-only LongMemEval benchmark runner targeting the XMem HTTP API.
- `locomo/`: Python-only LoCoMo benchmark runner targeting the XMem HTTP API.
- `beam/`: Python-only BEAM runner, defaulting to the Hugging Face BEAM 1M split.

Benchmark runs can create large dataset and result artifacts. Keep those files under
each benchmark's `data`, `results`, or `outputs` directory; those paths are
intentionally ignored by git.
