# MegaTrain: A RAM-Centric Architecture for LLM Training
</a> <a href='https://arxiv.org/pdf/2602.04816'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## Quick Start

```bash
# Install
git clone https://github.com/DLYuanGod/MegaTrain.git
cd MegaTrain
pip install -e .

# Train (any supported model)
python examples/train.py --config examples/configs/qwen_7b.yaml
python examples/train.py --config examples/configs/llama3_8b.yaml
python examples/train.py --config examples/configs/mistral_7b.yaml
```

## Features

- **Single GPU Training**: Train 120B+ models on one GPU
- **CPU-Backed Parameters**: FP32 master weights on CPU, BF16 working copy on GPU
- **Universal Model Support**: Any HuggingFace decoder-only model works out of the box
- **Hybrid Architecture Support**: Handles mixed attention types (linear + full) and MoE
- **YAML Configuration**: Easy model/dataset/hyperparameter configuration
- **HF Native Flash Attention**: Uses HuggingFace's built-in `attn_implementation`
- **Auto CUDA Extension**: Automatically builds optimized CUDA kernels

## Supported Models

| Model Family | Model Sizes | Status |
|--------------|-------------|--------|
| **Qwen2/Qwen2.5** | 0.5B/1.5B/3B/7B/14B/32B/72B | Fully supported |
| **Qwen3** | 0.6B/1.7B/4B/8B/14B/32B | Fully supported |
| **Qwen3.5** (hybrid linear+full attn) | 0.8B/2B/4B/9B/27B | Fully supported |
| **Qwen3.5 MoE** (hybrid attn + MoE) | 35B-A3B/122B-A10B/397B-A17B | Fully supported |
| **Qwen3-Next** (hybrid attn + MoE) | 80B-A3B | Fully supported |
| **Llama 2** | 7B/13B/70B | Fully supported |
| **Llama 3/3.1/3.2/3.3** | 1B/3B/8B/70B | Fully supported |
| **Llama 4** (MoE) | Scout-17B-16E/Maverick | Fully supported |
| **Mistral** | 7B | Fully supported |
| **Mixtral** (MoE) | 8x7B/8x22B | Fully supported |
| **DeepSeek** (LLM/Code/R1) | 7B/16B/67B | Fully supported |
| **DeepSeek-V3** (MoE) | 671B | Fully supported |
| **Phi-3/Phi-4** | 3.8B/14B | Fully supported |
| **Gemma 2/3** | 2B/7B/9B/27B | Fully supported |
| **GLM-4/GLM-4.5** | 9B/32B | Fully supported |
| **InternLM 2/2.5** | 7B/20B | Fully supported |
| **Yi 1.5** | 6B/9B/34B | Fully supported |
| **Baichuan 2** | 7B/13B | Fully supported |
| **GPT-OSS** | 20B/120B | Fully supported |
| Any HF decoder-only model | Any size | Auto-detected |

**How it works**: MegaTrain uses HuggingFace's `AutoModelForCausalLM` with automatic model structure discovery. Any decoder-only transformer model is supported without code changes — just provide the model name in your config.

### Hybrid & MoE Architecture Support

MegaTrain automatically handles:
- **Hybrid attention** (e.g., Qwen3.5): Layers with different attention types (linear vs full) are grouped by structure, with separate GPU templates per group
- **MoE layers** (e.g., Mixtral, DeepSeek-MoE, Qwen3-Next): Per-layer buffer sizing accommodates varying parameter counts across dense and MoE layers

## Usage

### 1. Configure Training

Edit or create a YAML config file:

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"  # or "sdpa" or "eager"

dataset:
  path: "/path/to/your/dataset"
  max_seq_len: 1024
  query_field: "query"        # field name in your dataset
  response_field: "response"  # field name in your dataset

training:
  batch_size: 148
  num_steps: 1000
  learning_rate: 2.0e-5

optimizer:
  type: "deepspeed_adam"  # or "adamw"
```

### 2. Train

```bash
# Use config file
python examples/train.py --config examples/configs/llama3_8b.yaml

# Override parameters
python examples/train.py \
    --config examples/configs/qwen_7b.yaml \
    --batch-size 64 \
    --num-steps 500
```

### 3. Use in Python

```python
from infinity import CPUMasterModel, CPUMasterConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load any model
config = CPUMasterConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="/path/to/dataset",
    batch_size=148,
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation=config.attn_implementation,
)

model = CPUMasterModel(hf_model, config)

# Train...
loss, tokens, timing = model.forward_and_backward(
    input_ids, attention_mask, labels
)
```

## Available Configs

| Config | Model | Architecture | Batch Size |
|--------|-------|-------------|------------|
| `qwen_7b.yaml` | Qwen 2.5 7B | Dense | 148 |
| `qwen_32b.yaml` | Qwen 2.5 32B | Dense | 96 |
| `qwen3_8b.yaml` | Qwen 3 8B | Dense | 148 |
| `qwen3_5_7b.yaml` | Qwen 3.5 7B | Hybrid (linear+full) | 148 |
| `qwen3_5_9b.yaml` | Qwen 3.5 9B | Hybrid (linear+full) | 128 |
| `qwen3_5_27b.yaml` | Qwen 3.5 27B | Hybrid (linear+full) | 64 |
| `qwen3_5_moe_35b.yaml` | Qwen 3.5 35B-A3B | Hybrid + MoE | 64 |
| `qwen3_next_80b.yaml` | Qwen3-Next 80B-A3B | Hybrid + MoE | 16 |
| `llama2_7b.yaml` | Llama 2 7B | Dense | 148 |
| `llama3_8b.yaml` | Llama 3.1 8B | Dense | 148 |
| `llama3_70b.yaml` | Llama 3.1 70B | Dense | 32 |
| `llama4_scout.yaml` | Llama 4 Scout | MoE (16 experts) | 16 |
| `mistral_7b.yaml` | Mistral 7B | Dense | 148 |
| `mixtral_8x7b.yaml` | Mixtral 8x7B | MoE (8 experts) | 32 |
| `deepseek_r1_8b.yaml` | DeepSeek-R1 8B | Dense | 148 |
| `deepseek_v3.yaml` | DeepSeek-V3 671B | MoE (256 experts) | 4 |
| `phi4_14b.yaml` | Phi-4 14B | Dense | 96 |
| `gemma2_9b.yaml` | Gemma 2 9B | Dense | 128 |
| `gemma3_27b.yaml` | Gemma 3 27B | Dense | 64 |
| `glm4_9b.yaml` | GLM-4 9B | Dense | 128 |
| `glm4_32b.yaml` | GLM-4 32B | Dense | 96 |
| `internlm2_7b.yaml` | InternLM 2.5 7B | Dense | 148 |
| `yi_34b.yaml` | Yi 1.5 34B | Dense | 64 |

See `examples/configs/README.md` for detailed configuration guide.

## Requirements

- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100, H100, GH200)
- **CPU RAM**: 256GB+ for 32B models
- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **Python**: 3.9+


**Key Techniques:**
- Double buffering for overlapped weight transfer
- Per-layer structure grouping (handles hybrid/MoE architectures)
- Gradient checkpointing every K layers
- Async gradient collection with slab pool
- Manual gradient computation (no autograd overhead)
- HuggingFace native Flash Attention integration

## Performance

Training Qwen 2.5 32B on single H100:
- **Memory**:  ~327GB CPU
- **Throughput**: ~259 TFLOPS
- **Batch Size**: 96 (seq_len=1024)

## Installation Details

When you run `pip install -e .`:
1. Installs Python dependencies (PyTorch, Transformers, etc.)
2. Builds CUDA extension automatically (if CUDA available)
3. Sets up `infinity` library for import

Optional dependencies:
```bash
# Flash Attention (recommended)
pip install flash-attn

# DeepSpeed CPUAdam (5-7x faster optimizer)
pip install deepspeed
```

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in config
- Increase `checkpoint_interval`
- Reduce `max_seq_len`

**Slow Training?**
- Use `deepspeed_adam` optimizer
- Install Flash Attention
- Increase `num_workers` for data loading

**CUDA Extension Failed?**
- Training still works without it (slightly slower)
- Check CUDA version: `nvcc --version`
- Manually build: `cd infinity/cuda_pipeline && python setup.py install`

**New Model Not Working?**
- Ensure it's a decoder-only model (not encoder-decoder like T5)
- Check `trust_remote_code: true` in config if the model requires it
- Try `attn_implementation: "sdpa"` or `"eager"` if flash attention fails

## Citation

If you use MegaTrain in your research, please cite:

```bibtex
@misc{yuan2026horizonlmramcentricarchitecturellm,
      title={MegaTrain: A RAM-Centric Architecture for LLM Training}, 
      author={Zhengqing Yuan and Lichao Sun and Yanfang Ye},
      year={2026},
      eprint={2602.04816},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2602.04816}, 
}
```
