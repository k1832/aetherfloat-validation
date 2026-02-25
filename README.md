# AetherFloat — Validation Source Code

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
<!-- TODO: Add screenshot(s) of key result(s) -->
<!-- ![AetherFloat Key Results](path_to_your_screenshot.png) -->

## ⚠️ License & IP Notice
This repository is released under a custom **Academic Evaluation License** to facilitate peer review and reproducibility. Commercial deployment, hardware synthesis integration, or utilization in proprietary architectures requires a separate Commercial IP license. See the LICENSE file for details. Algorithms and architectures described herein are patent pending (U.S. App. No. 63/987,398 and supplemental filings).

## Overview
Companion code for the paper:

> **The AetherFloat Family: Block-Scale-Free Quad-Radix Floating-Point Architectures for AI Accelerators**

This repository contains all scripts needed to reproduce the figures, tables, and validations reported in the paper.

## Quick Start

### Python Environment (uv)

```bash
uv sync
```

### Docker (hardware synthesis only)

```bash
docker build -t aetherfloat-synth -f Dockerfile .
```

### C++ (lexicographic sort validation)

```bash
g++ -O2 -o aether_core src/aether_core.cpp
```

## Reproducing Paper Results

### Figure 1 — SQNR Wobble Plot

Compares quantization noise (SQNR) between bfloat16 and AetherFloat-16.

```bash
uv run src/wobble_plot.py
# → wobble_plot.pdf
```

### Figure 2 — Stochastic Rounding Ablation (multi-GPU)

QAT ablation on Qwen2.5-7B comparing stochastic rounding chunk sizes (1, 16, 256) against a bfloat16 baseline (300 steps).

```bash
uv run src/train_ablation.py
# → sr_ablation_qwen_real_7b.pdf
```

### Figure 3 — QAT Convergence (multi-GPU)

Quantization-aware training comparing 8-bit AF8 (scale-free) vs FP8 E4M3 vs bfloat16 baseline on Qwen2.5-7B (200 steps).

```bash
uv run src/train_qat_af8.py
# → qat_8bit_convergence_ste_7b.pdf
```

### Table II — PTQ Evaluation (Qwen2.5-7B, multi-GPU)

Post-training quantization evaluation on WikiText-2, PIQA, and HellaSwag.

```bash
uv run src/eval_ptq_7b.py --fmt all
```

Run a single format with `--fmt bf16`, `--fmt fp8`, `--fmt af8`, or `--fmt af16`.

### Table III — Hardware Synthesis (Docker)

Synthesizes FP8 Base-2 and AF8 Base-4 MAC datapaths and compares area, delay, and power using Yosys and OpenSTA.

> **Note:** This script requires Yosys and OpenSTA, which are EDA tools with complex build dependencies (Tcl, Boost, CUDD, etc.). The Dockerfile packages the entire toolchain so you don't need to install them on your host.

```bash
docker build -t aetherfloat-synth -f Dockerfile .
docker run --rm -v "$(pwd):/workspace" -w /workspace \
  aetherfloat-synth python3 src/synth_mac_datapath.py
```

### Lexicographic Sort Validation (C++)

Validates AetherFloat-16 encoding/decoding with 1 million random floats and verifies the O(1) lexicographic sorting property through monotonicity checks.

```bash
g++ -O2 -o aether_core src/aether_core.cpp
./aether_core
```

## File Overview

| File | Paper Reference | Description |
| --- | --- | --- |
| `src/aether_sim.py` | — | Core quantization library (AF8/AF16, FP8 baseline, PTQ/QAT patching) |
| `src/aether_core.cpp` | Section IV-A | Lexicographic sort validation (1M random floats) |
| `src/wobble_plot.py` | Figure 1 | SQNR wobble comparison: bfloat16 vs AetherFloat-16 |
| `src/train_ablation.py` | Figure 2 | Stochastic rounding ablation study on Qwen2.5-7B |
| `src/train_qat_af8.py` | Figure 3 | QAT convergence: AF8 vs FP8 vs bfloat16 |
| `src/eval_ptq_7b.py` | Table II | PTQ benchmarks (WikiText-2, PIQA, HellaSwag) |
| `src/synth_mac_datapath.py` | Table III | MAC datapath synthesis (area / delay / power) |
| `Dockerfile` | Table III | Build environment for Yosys + OpenSTA |

## Requirements

* **Python** >= 3.11, [uv](https://docs.astral.sh/uv/)
* **Multi-GPU** setup for training/eval scripts (Qwen2.5-7B)
* **Docker** for hardware synthesis (Yosys + OpenSTA are built inside the container)
* **C++ compiler** (g++ or clang++) for `aether_core.cpp`

## Citation

If you find this code or our paper useful in your research, please consider citing:

<!-- TODO: Update after the arXiv paper is published -->
```bibtex
@misc{morisaki2026aetherfloat,
  title={The AetherFloat Family: Block-Scale-Free Quad-Radix Floating-Point Architectures for AI Accelerators},
  author={Keita Morisaki},
  year={2026},
  eprint={...},
  archivePrefix={arXiv},
  primaryClass={cs.AR}
}
```
