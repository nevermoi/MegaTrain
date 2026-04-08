<div align="center">

# MegaTrain

### Full Precision Training of 100B+ Parameter LLMs on a Single GPU

[![Paper](https://img.shields.io/badge/Paper-arXiv%202602.04816-red)](https://arxiv.org/abs/2602.04816)
[![Paper](https://img.shields.io/badge/Paper-arXiv%202604.05091-red)](https://arxiv.org/abs/2604.05091)
[![GitHub Stars](https://img.shields.io/github/stars/DLYuanGod/MegaTrain?style=social)](https://github.com/DLYuanGod/MegaTrain/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**A RAM-centric architecture that stores parameters in host memory and treats GPUs as transient compute engines, enabling full-precision training of 100B+ models on a single GPU.**

[Quick Start](#quick-start) | [Supported Models](#supported-models) | [Data Preparation](#data-preparation) | [Performance](#performance) | [Citation](#citation)

</div>

---

## Features

- **Single GPU, Massive Models** -- Train 120B+ models on one GPU by leveraging CPU RAM for parameter storage
- **Universal Model Support** -- Any HuggingFace decoder-only model works out of the box via `AutoModelForCausalLM`
- **Hybrid Architecture** -- Automatic handling of mixed attention (linear + full) and MoE layers
- **LlamaFactory-style Data** -- Flexible `dataset_info.json` registry with alpaca/sharegpt format support
- **1.84x Faster** -- Outperforms DeepSpeed ZeRO-3 on 14B models through pipelined double-buffered execution
- **YAML Configuration** -- Easy model/dataset/hyperparameter setup with 25+ ready-made configs

## Quick Start

```bash
# Install
git clone https://github.com/DLYuanGod/MegaTrain.git
cd MegaTrain
pip install -e .

# Train with built-in demo data
python examples/train.py --config examples/configs/llama3_8b.yaml

# Train any supported model
python examples/train.py --config examples/configs/qwen3_5_27b.yaml
python examples/train.py --config examples/configs/deepseek_v3.yaml
```

## Supported Models

| Model Family | Model Sizes | Architecture |
|:-------------|:------------|:-------------|
| [Qwen2/Qwen2.5](https://huggingface.co/Qwen) | 0.5B/1.5B/3B/7B/14B/32B/72B | Dense |
| [Qwen3](https://huggingface.co/Qwen) | 0.6B/1.7B/4B/8B/14B/32B | Dense |
| [Qwen3.5](https://huggingface.co/Qwen) | 0.8B/2B/4B/9B/27B | Hybrid (linear+full attn) |
| [Qwen3.5 MoE](https://huggingface.co/Qwen) | 35B-A3B/122B-A10B/397B-A17B | Hybrid + MoE |
| [Qwen3-Next](https://huggingface.co/Qwen) | 80B-A3B | Hybrid + MoE |
| [Llama 2](https://huggingface.co/meta-llama) | 7B/13B/70B | Dense |
| [Llama 3/3.1/3.2/3.3](https://huggingface.co/meta-llama) | 1B/3B/8B/70B | Dense |
| [Llama 4](https://huggingface.co/meta-llama) | Scout-17B-16E/Maverick | MoE |
| [Mistral](https://huggingface.co/mistralai) | 7B | Dense |
| [Mixtral](https://huggingface.co/mistralai) | 8x7B/8x22B | MoE |
| [DeepSeek (LLM/Code/R1)](https://huggingface.co/deepseek-ai) | 7B/16B/67B | Dense |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai) | 671B | MoE (256 experts) |
| [Phi-3/Phi-4](https://huggingface.co/microsoft) | 3.8B/14B | Dense |
| [Gemma 2/3](https://huggingface.co/google) | 2B/7B/9B/27B | Dense |
| [GLM-4/GLM-4.5](https://huggingface.co/THUDM) | 9B/32B | Dense |
| [InternLM 2/2.5](https://huggingface.co/internlm) | 7B/20B | Dense |
| [Yi 1.5](https://huggingface.co/01-ai) | 6B/9B/34B | Dense |
| [Baichuan 2](https://huggingface.co/baichuan-inc) | 7B/13B | Dense |
| [GPT-OSS](https://huggingface.co/openai) | 20B/120B | Dense |
| Any HF decoder-only model | Any size | Auto-detected |

> MegaTrain uses HuggingFace's `AutoModelForCausalLM` with automatic model structure discovery. Any decoder-only transformer model is supported without code changes.

## Data Preparation

MegaTrain supports a **LlamaFactory-compatible data system** with flexible format support.

### Option 1: Dataset Registry (Recommended)

Register datasets in [`data/dataset_info.json`](data/dataset_info.json) and reference by name:

```yaml
dataset:
  name: "alpaca_en_demo"    # name from dataset_info.json
  dataset_dir: "data"
  max_seq_len: 1024
```

Supports **alpaca format**, **sharegpt format**, local JSON/JSONL files, and HuggingFace Hub datasets. See [`data/README.md`](data/README.md) for details.

### Option 2: Direct Path (Legacy)

```yaml
dataset:
  path: "/path/to/arrow/dataset"
  query_field: "query"
  response_field: "response"
```

### Provided Datasets

| Dataset | Source | Format |
|:--------|:-------|:-------|
| [alpaca_en_demo](data/alpaca_en_demo.json) | Built-in | Alpaca |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | HuggingFace Hub | Alpaca |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | HuggingFace Hub | Alpaca |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | HuggingFace Hub | Alpaca |
| [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | HuggingFace Hub | Alpaca |
| [ShareGPT4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) | HuggingFace Hub | ShareGPT |
| [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | HuggingFace Hub | ShareGPT |
| [OpenThoughts-114k](https://huggingface.co/datasets/llamafactory/OpenThoughts-114k) | HuggingFace Hub | ShareGPT |
| [OpenR1-Math-94k](https://huggingface.co/datasets/llamafactory/OpenR1-Math-94k) | HuggingFace Hub | ShareGPT |

## Configuration

```yaml
model:
  name: "Qwen/Qwen3.5-27B"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

dataset:
  name: "metamath"
  max_seq_len: 1024

training:
  batch_size: 64
  num_steps: 500
  learning_rate: 1.0e-5

optimizer:
  type: "deepspeed_adam"
```

See [`examples/configs/`](examples/configs/) for 25+ ready-made configurations.

<details><summary><b>All Available Configs</b></summary>

| Config | Model | Architecture | Batch Size |
|:-------|:------|:-------------|:-----------|
| `qwen_7b.yaml` | Qwen 2.5 7B | Dense | 128 |
| `qwen_32b.yaml` | Qwen 2.5 32B | Dense | 96 |
| `qwen3_8b.yaml` | Qwen 3 8B | Dense | 148 |
| `qwen3_5_7b.yaml` | Qwen 3.5 7B | Hybrid | 148 |
| `qwen3_5_9b.yaml` | Qwen 3.5 9B | Hybrid | 128 |
| `qwen3_5_27b.yaml` | Qwen 3.5 27B | Hybrid | 64 |
| `qwen3_5_moe_35b.yaml` | Qwen 3.5 35B-A3B | Hybrid + MoE | 64 |
| `qwen3_next_80b.yaml` | Qwen3-Next 80B-A3B | Hybrid + MoE | 16 |
| `llama2_7b.yaml` | Llama 2 7B | Dense | 148 |
| `llama3_8b.yaml` | Llama 3.1 8B | Dense | 148 |
| `llama3_70b.yaml` | Llama 3.1 70B | Dense | 32 |
| `llama4_scout.yaml` | Llama 4 Scout | MoE | 16 |
| `mistral_7b.yaml` | Mistral 7B | Dense | 148 |
| `mixtral_8x7b.yaml` | Mixtral 8x7B | MoE | 32 |
| `deepseek_r1_8b.yaml` | DeepSeek-R1 8B | Dense | 148 |
| `deepseek_v3.yaml` | DeepSeek-V3 671B | MoE | 4 |
| `phi4_14b.yaml` | Phi-4 14B | Dense | 96 |
| `gemma2_9b.yaml` | Gemma 2 9B | Dense | 128 |
| `gemma3_27b.yaml` | Gemma 3 27B | Dense | 64 |
| `glm4_9b.yaml` | GLM-4 9B | Dense | 128 |
| `glm4_32b.yaml` | GLM-4 32B | Dense | 96 |
| `internlm2_7b.yaml` | InternLM 2.5 7B | Dense | 148 |
| `yi_34b.yaml` | Yi 1.5 34B | Dense | 64 |

</details>

## Performance

| Model | GPU | TFLOPS | CPU RAM | GPU VRAM |
|:------|:----|:-------|:--------|:---------|
| Qwen 2.5 32B | 1x H100 | ~259 | ~327 GB | ~40 GB |
| Qwen 3.5 27B | 1x H100 | ~157 | ~275 GB | ~38 GB |

### Key Techniques

- **Double buffering** for overlapped weight transfer between CPU and GPU
- **Per-layer structure grouping** for hybrid/MoE architectures
- **Gradient checkpointing** every K layers to reduce GPU memory
- **Async gradient collection** with slab pool
- **Manual gradient computation** (no autograd overhead)
- **HuggingFace native Flash Attention** integration
- **DeepSpeed CPUAdam** for 5-7x faster optimizer steps

## Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| GPU | 40GB+ VRAM (A100) | 80GB (H100/H200) |
| CPU RAM | 128GB (7B models) | 256GB+ (32B+ models) |
| CUDA | 11.8 | 12.0+ |
| PyTorch | 2.0 | 2.4+ |
| Python | 3.9 | 3.10+ |

### Installation

```bash
git clone https://github.com/DLYuanGod/MegaTrain.git
cd MegaTrain
pip install -e .

# Optional: faster attention & optimizer
pip install flash-attn
pip install flash-linear-attention causal-conv1d  # for Qwen3.5 linear attention
pip install deepspeed                              # for CPUAdam optimizer
```

## Troubleshooting

<details><summary><b>Out of Memory?</b></summary>

- Reduce `batch_size` in config
- Increase `checkpoint_interval`
- Reduce `max_seq_len`

</details>

<details><summary><b>Slow Training?</b></summary>

- Use `deepspeed_adam` optimizer (5-7x faster than PyTorch AdamW)
- Install Flash Attention
- Install `flash-linear-attention` + `causal-conv1d` for Qwen3.5 models
- Increase `num_workers` for data loading

</details>

<details><summary><b>New Model Not Working?</b></summary>

- Ensure it's a decoder-only model (not encoder-decoder like T5)
- Check `trust_remote_code: true` in config if the model requires it
- Try `attn_implementation: "sdpa"` or `"eager"` if flash attention fails

</details>

## Citation

If you use MegaTrain in your research, please cite:

```bibtex
@misc{yuan2026megatrainprecisiontraining100b,
      title={MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU}, 
      author={Zhengqing Yuan and Hanchi Sun and Lichao Sun and Yanfang Ye},
      year={2026},
      eprint={2604.05091},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05091}, 
}
```

## Acknowledgement

This project benefits from the following open-source works:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) -- Our data loading system (`dataset_info.json` registry, alpaca/sharegpt format support) is inspired by LlamaFactory's elegant dataset management design. Thanks to [@hiyouga](https://github.com/hiyouga) and all contributors.
- [HuggingFace Transformers](https://github.com/huggingface/transformers) -- Universal model loading and native Flash Attention integration.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) -- SIMD-accelerated CPUAdam optimizer.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) -- Memory-efficient attention and cross-entropy loss.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) -- Efficient linear attention kernels for hybrid models like Qwen3.5.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).
