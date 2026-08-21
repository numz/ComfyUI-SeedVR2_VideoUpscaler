# SeedVR2 Comfy Desktop Shared Model Directory Adaptation

## Pull Request Summary

This branch adapts SeedVR2 model management to the standard ComfyUI/Comfy Desktop model registration mechanism. DiT and VAE models no longer share the plugin-specific `models/SEEDVR2` directorydirectory (while maintaining compatibility). They are resolved independently through the `diffusion_models` and `vae` categories exposed by `folder_paths`.

The integration can use Comfy Desktop shared model directories, instance model directories, and paths registered through `extra_model_paths.yaml`, while retaining a fallback for standalone CLI usage.

## Problem and Goals

The previous implementation treated DiT and VAE as one SeedVR2 model type and constructed paths inside a fixed directory. This prevented reliable use of shared model configuration, allowed cross-category lookup, and could cause existing local models to trigger registry downloads or hash-based replacement. Nodes also had no way to show only models that actually exist locally.

This change resolves paths through official ComfyUI categories, keeps workflow values as filenames or category-relative paths, prefers registered local models, adds a local-only option with server-side validation, and preserves the standalone CLI `--model_dir` override and fallback behavior.

## Final Model Path Strategy

| Model | ComfyUI category | Lookup API |
| --- | --- | --- |
| DiT / diffusion model | `diffusion_models` | `folder_paths.get_full_path("diffusion_models", name)` |
| VAE | `vae` | `folder_paths.get_full_path("vae", name)` |

The plugin does not read Comfy Desktop configuration files or guess installation directories, AppData paths, or drive letters. Desktop/ComfyUI registers directories; the plugin consumes the unified `folder_paths` API.

```text
<shared models root>/diffusion_models/seedvr2_ema_3b_fp16.safetensors
<shared models root>/vae/ema_vae_fp16.safetensors
```

Node values remain stable filenames such as `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors`. Category-relative subdirectories returned by ComfyUI are preserved.

## Changes

### Category-aware path utilities

`src/utils/constants.py` adds `DIT_MODEL_TYPE` and `VAE_MODEL_TYPE` and makes path utilities category-aware. `get_model_paths`, `get_model_dir`, `get_all_model_paths`, and `get_all_model_files` operate on a requested category. `find_model_file` tries `folder_paths.get_full_path()` first, then registered directories, then the fallback. `models/SEEDVR2` remains available as a read fallback for legacy standalone and ComfyUI installations, but is not a new download target in a ComfyUI environment. To handle older ComfyUI/Desktop category mappings, registered `unet` paths remain readable as the official legacy alias, while new DiT downloads prefer `diffusion_models` and fall back to the current instance's `models/diffusion_models`; they are never newly written to `models/unet`. Compatibility discovery also reads paths registered under `seedvr2` or `SEEDVR2` in `folder_names_and_paths`, then registers those paths in both `diffusion_models` and `vae` before scanning.

### DiT and VAE loading

Both initial loading and cached-runner restoration in `src/core/model_configuration.py` now pass the explicit category: DiT uses `diffusion_models` and VAE uses `vae`. The two model loader interfaces and `src/interfaces/video_upscaler.py` follow the same categorized path flow, and the video node no longer passes the old shared model directory.

### Model lists and same-name priority

`src/utils/model_registry.py` separates preset model lists from local model lists. `get_available_dit_models` and `get_available_vae_models` return only registry presets by default, while `get_local_dit_models` and `get_local_vae_models` return the complete local list, including compatible legacy `SEEDVR2` files. Same-name local files keep the original workflow name and are preferred at runtime.

The local lists also merge files from the legacy `models/SEEDVR2` directory. Existing DiT and VAE files placed there by older versions therefore remain selectable and are resolved through the corresponding model category.

### Download behavior and local model protection

`src/utils/downloads.py` attaches a category to every task: DiT maps to `diffusion_models` and VAE maps to `vae`. Without `model_dir`, each file is downloaded to its category's primary directory; an explicit CLI override remains supported.

Before downloading, the code calls `folder_paths.get_full_path(category, filename)`. If the registered path exists, it is used immediately and registry hash validation and downloading are skipped, even when the local revision has a different hash. Fallback lookup, cache validation, and downloading occur only when no registered file exists. Category names are kept separate from human-readable log labels to prevent lookup failures.

### Local-only model filtering

The DiT and VAE loader nodes expose an optional `local_models_only` Boolean input. Disabled shows only registry preset models; a same-named local file is still preferred at runtime. Enabled shows the complete local list from ComfyUI registered paths and the compatible legacy `SEEDVR2` directories. The execute methods enforce the local-only rule server-side.

Both loader nodes also implement the ComfyUI V3 `validate_inputs()` hook. If a custom model saved in a workflow has been deleted or cannot be resolved, the hook returns a clear missing-model message and ComfyUI's standard validation flow marks the node as invalid. Registry presets remain valid before their first download so the existing automatic-download behavior is preserved.

The new `web/seedvr2_model_sources.js` extension requests `/seedvr2/model-list?type=dit|vae` after node creation and updates Combo options when the toggle changes. It preserves the workflow's model value in both modes; missing values are never replaced and remain available for ComfyUI's standard invalid-state display. `__init__.py` registers the route and sets `WEB_DIRECTORY = "web"`.

The extension reapplies the toggle after node creation, workflow configuration restoration, and asynchronous model-list loading, so initialization follows the workflow's saved `local_models_only` value instead of always retaining the schema default.

Whether local mode is enabled or disabled, the current workflow value is never replaced, allowing missing models to remain invalid and show ComfyUI's warning state.

### Standalone CLI compatibility

`inference_cli.py` no longer injects the old unified directory when no argument is supplied. Without `--model_dir`, it uses ComfyUI categories or falls back to `./models/SEEDVR2` when `folder_paths` is unavailable. With `--model_dir`, the explicit CLI override is preserved.

## File Change List

### Modified

- `README.md`: document standard categories and shared-directory usage.
- `__init__.py`: register the local model-list API and enable the web directory.
- `inference_cli.py`: remove forced injection of the old unified directory.
- `src/core/model_configuration.py`: distinguish categories during initial loading and cache restoration.
- `src/interfaces/dit_model_loader.py`: use categorized model paths.
- `src/interfaces/vae_model_loader.py`: use categorized model paths.
- `src/interfaces/video_upscaler.py`: stop passing the old unified model directory.
- `src/utils/constants.py`: provide category-aware path and lookup utilities.
- `src/utils/downloads.py`: download by category and protect registered local models.
- `src/utils/model_registry.py`: separate preset and local lists, preserve same-name local replacement, and expose local-only lists.

### Added

- `web/seedvr2_model_sources.js`: frontend extension for local model filtering.

## Compatibility and Upgrade Notes

- Existing workflow model names are unchanged; no `[Local]` or `[Preset]` prefixes are introduced.
- Workflows without `local_models_only` default to disabled.
- Comfy Desktop/ComfyUI must be restarted to register the route and `WEB_DIRECTORY`; a hard browser refresh is recommended.
- Without shared configuration, the plugin uses the instance's standard `diffusion_models` and `vae` directories.
- The old `models/SEEDVR2` directory remains a read fallback for DiT and VAE, but is not used as a new download target in ComfyUI.
- Legacy `unet` category mappings remain readable but are not selected as new DiT download destinations.
- Node initialization follows the workflow's saved `local_models_only` value: disabled uses presets, enabled uses the complete local list.

## Validation Checklist

Run this in the Python environment used by Comfy Desktop:

```powershell
python -m py_compile `
  "inference_cli.py" `
  "src\utils\constants.py" `
  "src\utils\downloads.py" `
  "src\utils\model_registry.py" `
  "src\core\model_configuration.py" `
  "src\interfaces\dit_model_loader.py" `
  "src\interfaces\vae_model_loader.py" `
  "src\interfaces\video_upscaler.py" `
  "__init__.py"
```

Also verify that DiT and VAE resolve only from their own categories; same-name local models appear once and take priority; legacy `models/SEEDVR2` files appear and load; registered local files do not trigger downloads or hash replacement; `local_models_only` filters and restores options; missing custom files trigger the standard node validation error; invalid local-only selections are rejected by the backend; and CLI behavior is correct with and without `--model_dir`.

## Design Conclusion

The change separates model name, model category, and physical path. ComfyUI owns directory registration, nodes store stable filenames, and loading and downloading resolve paths by category. Shared directories, instance directories, and the standalone CLI fallback can therefore coexist without DiT/VAE cross-matching or overwriting an existing local model with a preset download.
