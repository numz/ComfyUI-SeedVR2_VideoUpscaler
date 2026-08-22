"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension

try:
	from aiohttp import web
	from server import PromptServer
	from .src.utils.model_registry import get_local_dit_models, get_local_vae_models

	@PromptServer.instance.routes.get("/seedvr2/model-list")
	async def seedvr2_model_list(request):
		model_type = request.query.get("type", "")
		if model_type == "dit":
			models = get_local_dit_models()
		elif model_type == "vae":
			models = get_local_vae_models()
		else:
			return web.json_response({"error": "type must be dit or vae"}, status=400)
		return web.json_response({"models": models})
except Exception:
	pass

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]

WEB_DIRECTORY = "web"