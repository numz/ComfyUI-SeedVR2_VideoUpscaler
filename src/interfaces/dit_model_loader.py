"""
SeedVR2 DiT Model Loader Node
Configure DiT (Diffusion Transformer) model with memory optimization
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_dit_models, DEFAULT_DIT
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadDiTModel(io.ComfyNode):
    """
    Configure DiT (Diffusion Transformer) model loader with memory optimization
    
    Provides configuration for:
    - Model selection and device placement
    - BlockSwap memory optimization for limited VRAM
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_DIT configuration dictionary for main upscaler node
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:        
        devices = get_device_list()
        dit_models = get_available_dit_models()
        
        return io.Schema(
            node_id="SeedVR2LoadDiTModel",
            display_name="SeedVR2 (Down)Load DiT Model",
            category="SEEDVR2",
            description=(
                "Load and configure SeedVR2 DiT (Diffusion Transformer) model for video upscaling. "
                "Supports BlockSwap memory optimization for low VRAM systems, model caching for batch processing, "
                "multi-GPU offloading, and torch.compile acceleration. \n\n"
                "Connect to Video Upscaler node."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=dit_models,
                    default=DEFAULT_DIT,
                    tooltip=(
                        "DiT (Diffusion Transformer) model for video upscaling.\n"
                        "Models automatically download on first use.\n"
                        "Additional models can be added to the ComfyUI models folder."
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for DiT model inference (upscaling phase)"
                ),
                io.Int.Input("blocks_to_swap",
                    default=0,
                    min=0,
                    max=36,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Number of transformer blocks to swap between devices for VRAM optimization.\n"
                        "• 0: Disabled (default)\n"
                        "• 3B model: 0-32 blocks\n"
                        "• 7B model: 0-36 blocks\n"
                        "\n"
                        "Requires offload_device to be set and different from device.\n"
                        "Not available on macOS (unified memory architecture)."
                    )
                ),
                io.Boolean.Input("swap_io_components",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Offload input/output embeddings and normalization layers to reduce VRAM.\n"
                        "Requires offload_device to be set and different from device.\n"
                        "Not available on macOS (unified memory architecture)."
                    )
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="none",
                    optional=True,
                    tooltip=(
                        "Device to offload DiT model when not actively processing.\n"
                        "• 'none': Keep model on inference device (default, fastest)\n"
                        "• 'cpu': Offload to system RAM (reduces VRAM usage)\n"
                        "• 'cuda:X': Offload to another GPU (good balance if available)\n"
                        "\n"
                        "Required for BlockSwap (blocks_to_swap or swap_io_components)."
                    )
                ),
                io.Boolean.Input("cache_model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Keep DiT model loaded on offload_device between workflow runs.\n"
                        "Useful for batch processing to avoid repeated loading.\n"
                        "Requires offload_device to be set."
                    )
                ),
                io.Combo.Input("attention_mode",
                    options=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                    default="sdpa",
                    optional=True,
                    tooltip=(
                        "Attention computation backend:\n"
                        "• sdpa: PyTorch scaled_dot_product_attention (default, stable, always available)\n"
                        "• flash_attn_2: Flash Attention 2 (Ampere+, requires flash-attn package)\n"
                        "• flash_attn_3: Flash Attention 3 (Hopper+, requires flash-attn with FA3 support)\n"
                        "• sageattn_2: SageAttention 2 (requires sageattention package)\n"
                        "• sageattn_3: SageAttention 3 (Blackwell/RTX 50xx only, requires sageattn3 package)\n"
                        "\n"
                        "SDPA is recommended - stable and works everywhere.\n"
                        "Flash Attention and SageAttention provide speedup through optimized CUDA kernels on compatible GPUs."
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node.\n"
                        "Provides 20-40% speedup with compatible PyTorch 2.0+ and Triton installation."
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_DIT").Output(
                    tooltip="DiT model configuration containing model path, device settings, BlockSwap parameters, and compilation options. Connect to Video Upscaler node."
                )
            ]
        )
    
    @classmethod
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, blocks_to_swap: int = 0, 
                     swap_io_components: bool = False, attention_mode: str = "sdpa",
                     torch_compile_args: Dict[str, Any] = None) -> io.NodeOutput:
        """
        Create DiT model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            blocks_to_swap: Number of transformer blocks to swap (requires offload_device != device)
            swap_io_components: Whether to offload I/O components (requires offload_device != device)
            attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is not set
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached DiT model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "attention_mode": attention_mode,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
        }
        
        return io.NodeOutput(config)