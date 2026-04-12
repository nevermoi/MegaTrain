"""
MegaTrain Engine for VERL — CPU-memory-centric single-GPU training backend.

MegaTrain stores model parameters in CPU RAM and uses the GPU as a transient
compute engine with double-buffered weight transfer. This makes it possible
to train 100B+ parameter models on a single GPU for post-training workloads
(SFT, RLHF, DPO, GRPO) where memory is the bottleneck.

This engine wraps MegaTrain's CPUMasterModel to implement VERL's BaseEngine
interface, enabling it as a training backend for VERL's RL pipelines.
"""

import gc
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseEngine, EngineRegistry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegaTrainEngine(BaseEngine):
    """
    Engine implementation using MegaTrain's CPU-memory-centric architecture.

    MegaTrain stores all parameters in CPU RAM and streams them to GPU
    layer-by-layer with double-buffered execution. This enables training
    models far larger than GPU memory (100B+ on a single GPU).

    Key differences from FSDP/Megatron backends:
    - Single-GPU, single-process (no torch.distributed required)
    - Parameters always live on CPU; GPU is transient compute
    - Forward and backward are fused (layer-by-layer with recompute)
    - No separate model.train()/eval() modes needed
    """

    def __init__(
        self,
        model_config,
        engine_config,
        optimizer_config,
        checkpoint_config,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.rank = 0  # Single-GPU, always rank 0

        self._is_offload_param = True  # MegaTrain always offloads params to CPU
        self._is_offload_optimizer = True  # Optimizer always on CPU

        self.cpu_master = None
        self.optimizer = None
        self.lr_scheduler = None
        self.hf_model = None

    @property
    def is_param_offload_enabled(self) -> bool:
        return True  # MegaTrain always keeps params on CPU

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return True  # Optimizer always on CPU

    def initialize(self):
        """Build the MegaTrain CPUMasterModel, optimizer, and LR scheduler."""
        from infinity.model.cpu_master import CPUMasterModel
        from infinity.config.training import CPUMasterConfig

        # Build MegaTrain config from VERL configs
        megatrain_config = self._build_megatrain_config()

        # Load HF model
        model_path = self.model_config.path if hasattr(self.model_config, 'path') else self.model_config.model_path
        trust_remote_code = getattr(self.model_config, 'trust_remote_code', True)
        dtype_str = getattr(self.engine_config, 'dtype', 'bfloat16')
        dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        is_vlm = hasattr(hf_config, 'vision_config')

        if is_vlm:
            try:
                from transformers import AutoModelForImageTextToText
                load_class = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForCausalLM
                load_class = AutoModelForCausalLM
        else:
            from transformers import AutoModelForCausalLM
            load_class = AutoModelForCausalLM

        attn_impl = getattr(self.engine_config, 'attn_implementation', 'flash_attention_2')

        logger.info(f"Loading model from {model_path} with attn={attn_impl}")
        hf_model = load_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )

        # Create CPUMasterModel
        self.cpu_master = CPUMasterModel(hf_model, megatrain_config)
        del hf_model

        # Create optimizer
        if not getattr(self.engine_config, 'forward_only', False):
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None

        logger.info(f"MegaTrain engine initialized. "
                     f"Model params on CPU, GPU device={megatrain_config.device}")

    def _build_megatrain_config(self):
        """Convert VERL configs to MegaTrain CPUMasterConfig."""
        from infinity.config.training import CPUMasterConfig

        model_path = self.model_config.path if hasattr(self.model_config, 'path') else self.model_config.model_path
        dtype_str = getattr(self.engine_config, 'dtype', 'bfloat16')
        dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16

        device = getattr(self.engine_config, 'device', 0)
        attn_impl = getattr(self.engine_config, 'attn_implementation', 'flash_attention_2')
        trust_remote_code = getattr(self.model_config, 'trust_remote_code', True)

        # Extract training hyperparams from optimizer config
        lr = getattr(self.optimizer_config, 'lr', 1e-5)
        weight_decay = getattr(self.optimizer_config, 'weight_decay', 0.01)
        max_grad_norm = getattr(self.optimizer_config, 'clip_grad', 1.0)
        max_seq_len = getattr(self.engine_config, 'max_seq_len', 2048)
        checkpoint_interval = getattr(self.engine_config, 'checkpoint_interval', 4)
        num_grad_slabs = getattr(self.engine_config, 'num_grad_slabs', 12)

        config = CPUMasterConfig(
            model_name=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
            dataset_path="__verl__",  # Placeholder; data comes from VERL
            dataset_name="",
            max_seq_len=max_seq_len,
            batch_size=1,  # VERL controls batching
            gradient_accumulation_steps=1,
            num_steps=1,
            learning_rate=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            checkpoint_interval=checkpoint_interval,
            num_grad_slabs=num_grad_slabs,
        )
        return config

    def _build_optimizer(self):
        """Build optimizer for CPU master parameters."""
        lr = getattr(self.optimizer_config, 'lr', 1e-5)
        weight_decay = getattr(self.optimizer_config, 'weight_decay', 0.01)
        clip_grad = getattr(self.optimizer_config, 'clip_grad', 1.0)
        betas = getattr(self.optimizer_config, 'betas', (0.9, 0.999))
        eps = getattr(self.optimizer_config, 'eps', 1e-8)

        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                self.cpu_master.get_parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adamw_mode=True,
            )
            logger.info("Using DeepSpeed CPUAdam optimizer")
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.cpu_master.get_parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            logger.info("Using PyTorch AdamW optimizer (DeepSpeed CPUAdam not available)")

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        """Build LR scheduler."""
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        total_steps = getattr(self.optimizer_config, 'total_training_steps', 1000)
        num_warmup_steps = getattr(self.optimizer_config, 'lr_warmup_steps', 0)
        lr_scheduler_type = getattr(self.optimizer_config, 'lr_scheduler_type', 'cosine')

        if num_warmup_steps <= 0:
            ratio = getattr(self.optimizer_config, 'lr_warmup_steps_ratio', 0.0)
            num_warmup_steps = int(ratio * total_steps)

        if lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = getattr(self.optimizer_config, 'min_lr_ratio', 0.01)
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)

    def train_mode(self, **kwargs):
        """Context manager for training mode. MegaTrain doesn't need mode switching."""
        @contextmanager
        def _ctx():
            prev_mode = self.mode
            self.mode = "train"
            try:
                yield
            finally:
                self.mode = prev_mode
        return _ctx()

    def eval_mode(self, **kwargs):
        """Context manager for eval mode. MegaTrain doesn't need mode switching."""
        @contextmanager
        def _ctx():
            prev_mode = self.mode
            self.mode = "eval"
            try:
                yield
            finally:
                self.mode = prev_mode
        return _ctx()

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """
        Run forward (and optionally backward) on a batch of data.

        For MegaTrain, forward and backward are fused: the CPU-offloaded
        layer-by-layer execution means we can't separate them.

        Args:
            data: TensorDict with input_ids, attention_mask, etc.
            loss_function: VERL loss function (PPO, DPO, etc.)
            forward_only: If True, only compute forward (inference mode).
        """
        # Extract tensors from VERL's TensorDict
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        if forward_only:
            return self._forward_only(data, input_ids, attention_mask, loss_function)
        else:
            return self._forward_backward(data, input_ids, attention_mask, loss_function)

    def _forward_only(self, data, input_ids, attention_mask, loss_function):
        """Inference-only forward pass returning log_probs."""
        with torch.no_grad():
            logits = self.cpu_master.forward_logits(input_ids, attention_mask)

            # Compute log_probs from logits (both must be on GPU)
            input_ids_gpu = input_ids.to(logits.device)
            log_probs = logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])

            model_output = {"log_probs": log_probs}

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=data, dp_group=None
                )
            else:
                loss = torch.tensor(0.0)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item() if isinstance(loss, torch.Tensor) else loss,
                "metrics": metrics,
            }

            return output

    def _forward_backward(self, data, input_ids, attention_mask, loss_function):
        """Training forward+backward with VERL loss function."""
        def _loss_fn_adapter(logits, input_ids_gpu):
            """Adapt VERL's loss function to MegaTrain's interface."""
            # Compute log_probs from logits
            log_probs = logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])
            model_output = {"log_probs": log_probs}
            loss, metrics = loss_function(
                model_output=model_output, data=data, dp_group=None
            )
            return loss, metrics

        loss_val, num_tokens, timing, meta = self.cpu_master.forward_and_backward_custom_loss(
            input_ids, attention_mask, _loss_fn_adapter
        )

        output = {
            "model_output": {"log_probs": None},  # Already consumed by loss
            "loss": loss_val,
            "metrics": meta if meta else {},
        }

        return output

    def train_batch(self, data: TensorDict, loss_function: Callable) -> dict:
        """Full training step: zero_grad, forward+backward, optimizer step."""
        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if self.is_mp_src_rank_with_outputs():
            if "metrics" not in outputs:
                outputs["metrics"] = {}
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def optimizer_zero_grad(self):
        """Zero gradients on CPU master parameters."""
        self.cpu_master.zero_grad()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        """Clip gradients, step optimizer, sync params to GPU."""
        clip_grad = getattr(self.optimizer_config, 'clip_grad', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.cpu_master.get_parameters(), clip_grad
        )
        if self.optimizer is not None:
            self.optimizer.step()
        self.cpu_master._sync_params_to_gpu()
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def lr_scheduler_step(self):
        """Step the LR scheduler."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return 0.0

    def get_per_tensor_param(self):
        """Yield (name, tensor) pairs for all model parameters."""
        params = {}
        for name, p in zip(
            self._get_param_names(),
            self.cpu_master.get_parameters()
        ):
            params[name] = p.data
        return params.items(), None

    def _get_param_names(self):
        """Generate parameter names for checkpoint compatibility."""
        names = []
        for i, p in enumerate(self.cpu_master.get_parameters()):
            names.append(f"param_{i}")
        return names

    def get_data_parallel_size(self):
        return 1  # Single GPU

    def get_data_parallel_rank(self):
        return 0  # Single GPU

    def get_data_parallel_group(self):
        return None  # No distributed group

    def is_mp_src_rank_with_outputs(self):
        return True  # Single GPU, always has outputs

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """No-op for MegaTrain — params always on CPU, compute always on GPU."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ):
        """Save model checkpoint (CPU master parameters)."""
        os.makedirs(local_path, exist_ok=True)
        state = {
            'model_state': {
                name: p.data.clone()
                for name, p in zip(self._get_param_names(), self.cpu_master.get_parameters())
            },
            'global_step': global_step,
        }
        if self.optimizer is not None:
            state['optimizer_state'] = self.optimizer.state_dict()
        torch.save(state, os.path.join(local_path, f"checkpoint_{global_step}.pt"))
        logger.info(f"Saved checkpoint at step {global_step} to {local_path}")

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: bool = True,
        **kwargs,
    ):
        """Load model checkpoint."""
        ckpt_file = os.path.join(local_path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            # Try finding latest
            import glob
            ckpts = sorted(glob.glob(os.path.join(local_path, "checkpoint_*.pt")))
            if ckpts:
                ckpt_file = ckpts[-1]
            else:
                logger.warning(f"No checkpoint found in {local_path}")
                return

        state = torch.load(ckpt_file, map_location="cpu")

        if 'model_state' in state:
            for name, p in zip(self._get_param_names(), self.cpu_master.get_parameters()):
                if name in state['model_state']:
                    p.data.copy_(state['model_state'][name])

        if self.optimizer is not None and 'optimizer_state' in state:
            self.optimizer.load_state_dict(state['optimizer_state'])

        self.cpu_master._sync_params_to_gpu()
        logger.info(f"Loaded checkpoint from {ckpt_file}")

    def cleanup(self):
        """Clean up MegaTrain resources."""
        if self.cpu_master is not None:
            self.cpu_master.cleanup()


@EngineRegistry.register(
    model_type="language_model",
    backend="megatrain",
    device="cuda",
)
class MegaTrainEngineWithLMHead(MegaTrainEngine):
    """MegaTrain engine registered for language model training with VERL."""

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        """Process a single micro-batch through MegaTrain."""
        return self.forward_backward_batch(micro_batch, loss_function, forward_only)
