"""
SeedVR2 ComfyUI Nodes
Central registry for all SeedVR2 nodes
"""

from comfy_api.latest import ComfyExtension, io

from .video_upscaler import SeedVR2VideoUpscaler
from .dit_model_loader import SeedVR2LoadDiTModel
from .vae_model_loader import SeedVR2LoadVAEModel
from .torch_compile_settings import SeedVR2TorchCompileSettings


class SeedVR2Extension(ComfyExtension):
    """SeedVR2 ComfyUI Extension"""
    
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of all SeedVR2 nodes"""
        return [
            SeedVR2VideoUpscaler,
            SeedVR2LoadDiTModel,
            SeedVR2LoadVAEModel,
            SeedVR2TorchCompileSettings,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """ComfyUI V3 entry point"""
    return SeedVR2Extension()


__all__ = [
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
    'SeedVR2LoadVAEModel',
    'SeedVR2TorchCompileSettings',
    'SeedVR2Extension',
    'comfy_entrypoint',
]