# SeedVR2 Comfy Desktop 共享模型目录适配

## PR 摘要

本分支将 SeedVR2 的模型管理方式适配为 ComfyUI/Comfy Desktop 的标准模型注册机制。DiT 和 VAE 不再共用插件私有的 `models/SEEDVR2` 目录（保留兼容性），而是分别通过 `folder_paths` 查询 `diffusion_models` 和 `vae` 类别。

这使插件可以直接使用 Comfy Desktop 配置的共享模型目录、当前实例目录和 `extra_model_paths.yaml` 注册目录，同时保留独立 CLI 的兼容 fallback。

## 问题与目标

原实现把 DiT 和 VAE 作为同一种 SeedVR2 模型处理，并将路径拼接到一个固定目录。该方式存在以下问题：

- 无法正确使用 Comfy Desktop 的共享模型配置。
- DiT 与 VAE 可能在错误的目录中互相查找。
- 工作流和节点逻辑依赖物理路径，而不是 ComfyUI 管理的模型类别。
- 已经存在的本地模型可能仍被 registry 下载或哈希校验逻辑覆盖。
- 节点无法只显示当前已经存在的本地模型。

本次改动的目标是：

1. 使用 ComfyUI 官方模型类别解析模型路径。
2. 保持节点工作流中的模型值为文件名或类别相对路径。
3. 本地已注册模型优先，只有找不到时才执行下载。
4. 为节点提供“仅显示本地模型”选项，并在后端进行同等校验。
5. 保持独立 CLI 的 `--model_dir` 和非 ComfyUI fallback 行为。

## 最终模型路径策略

| 模型 | ComfyUI 类别 | 查询 API |
| --- | --- | --- |
| DiT / diffusion model | `diffusion_models` | `folder_paths.get_full_path("diffusion_models", name)` |
| VAE | `vae` | `folder_paths.get_full_path("vae", name)` |

插件不读取或解析 Comfy Desktop 的配置文件，也不猜测安装目录、AppData 或盘符。Desktop/ComfyUI 负责注册目录，插件只使用 `folder_paths` 提供的统一接口。

典型共享目录结构如下：

```text
<shared models root>/diffusion_models/seedvr2_ema_3b_fp16.safetensors
<shared models root>/vae/ema_vae_fp16.safetensors
```

节点保存的值仍然是：

```text
seedvr2_ema_3b_fp16.safetensors
ema_vae_fp16.safetensors
```

如果 ComfyUI 返回子目录相对路径，插件会完整保留该相对路径，不使用 `basename` 丢失目录信息。

## 变更内容

### 1. 类别化路径工具

`src/utils/constants.py` 新增 `DIT_MODEL_TYPE` 和 `VAE_MODEL_TYPE`，并将路径工具改为按类别工作：

- `get_model_paths(type)` 获取该类别全部注册目录。
- `get_model_dir(type)` 获取首选目录，用于下载目标。
- `get_all_model_paths(type)` 去除大小写和路径格式造成的重复项。
- `get_all_model_files(type)` 扫描指定类别的模型文件。
- `find_model_file(name, fallback_dir, model_type)` 先调用 `folder_paths.get_full_path()`，再扫描注册目录，最后使用 fallback。

`models/SEEDVR2` 作为独立 CLI 和旧 ComfyUI 安装的读取 fallback 保留，不再作为 Comfy Desktop 节点的新下载目录。

为兼容不同 ComfyUI/Desktop 版本的旧类别映射，DiT 仍可从官方注册的 `unet` 别名读取，但新的下载目标会优先使用 `diffusion_models`；如果注册列表首项是 `unet`，则改用当前实例的 `models/diffusion_models`，不会把新模型写入 `models/unet`。读取模型时仍会扫描旧的 `models/SEEDVR2` 目录，因此原目录中的模型可以继续使用。

兼容路径发现还会读取 ComfyUI 的 `folder_names_and_paths` 中以 `seedvr2` 或 `SEEDVR2` 注册的路径，并在首次扫描前将这些旧路径注册到 `diffusion_models` 和 `vae` 两个类别。这样即使共享目录不是当前实例的 `models_dir`，旧目录中的 DiT/VAE 文件也能被两个模型节点发现。

### 2. DiT/VAE 加载链路

`src/core/model_configuration.py` 的首次加载和缓存恢复分支都显式传入对应类别：

- DiT 的 checkpoint 始终使用 `diffusion_models`。
- VAE 的 checkpoint 始终使用 `vae`。

`src/interfaces/dit_model_loader.py`、`src/interfaces/vae_model_loader.py` 和 `src/interfaces/video_upscaler.py` 与新的分类路径链路保持一致。视频放大节点不再传入统一的旧缓存目录，而是让下载器和配置器自行按类别解析。

### 3. 模型列表与同名优先级

`src/utils/model_registry.py` 现在区分预设模型列表和本地模型列表：

- `get_available_dit_models()` 使用 `diffusion_models`。
- `get_available_vae_models()` 使用 `vae`。
- `get_local_dit_models()` 和 `get_local_vae_models()` 只返回实际注册的本地文件。

本地列表同时合并旧的 `models/SEEDVR2` 目录，因此旧版本已经放置的 DiT/VAE 模型会在开启本地模式后出现在下拉框中，并通过对应类别的查找逻辑被加载使用。

默认可用列表只返回 registry 中的预设配置模型。共享目录或旧 `SEEDVR2` 目录中存在与预设同名文件时，名称保持不变，运行时仍由本地路径优先使用该文件。开启 `local_models_only` 后才显示完整本地模型列表。

### 4. 下载行为与本地模型保护

`src/utils/downloads.py` 为每个下载任务携带独立的模型类别：

```text
DiT -> diffusion_models
VAE -> vae
```

当 `model_dir` 未提供时，分别下载到两个类别的首选目录；CLI 显式提供 `--model_dir` 时仍使用该覆盖目录。

下载前先调用 `folder_paths.get_full_path(category, filename)`。如果该路径存在，则直接使用并跳过 registry 哈希校验和下载，即使本地文件与预设哈希不同，也不会被删除或覆盖。只有未找到已注册文件时，才会继续检查 fallback 文件、验证缓存或下载缺失模型。

实现中将 ComfyUI 类别名与日志显示名分开，避免 `diffusion_models`/`vae` 被 `DiT`/`VAE` 日志文本覆盖后导致查找失败。

### 5. “仅显示本地模型”

DiT 和 VAE 加载节点新增可选布尔输入 `local_models_only`：

- 关闭时：只显示插件 registry 中的预设配置模型；如果本地存在同名文件，执行时优先使用本地文件。
- 开启时：只显示当前已注册或位于兼容 `SEEDVR2` 目录中的本地模型。

节点执行函数也会校验该选项。通过旧工作流、API 或手工请求传入不存在的模型时，后端会拒绝执行，而不仅依赖前端过滤。

两个模型加载节点还实现了 ComfyUI V3 的 `validate_inputs()`。当工作流中保存的自定义模型文件已经被删除或无法解析时，节点会返回明确的缺失模型信息，由 ComfyUI 官方校验链路显示红色失效状态和警告。registry 预设模型仍允许进入首次下载流程，因此“尚未下载的预设”不会被误判为自定义失效文件。

新增 `web/seedvr2_model_sources.js`：节点创建后请求 `/seedvr2/model-list?type=dit|vae`，并根据开关动态更新 Combo 选项。无论开关状态如何，扩展都保留工作流中的模型值；缺失值不会被替换，交由 ComfyUI 显示标准失效状态。

`__init__.py` 注册该 HTTP 路由，并通过 `WEB_DIRECTORY = "web"` 让 ComfyUI 自动加载前端扩展。路由只调用模型注册表，不读取固定路径。

前端扩展会在节点创建、工作流配置恢复以及本地模型列表异步返回后重新应用 `local_models_only` 状态，避免初始化时始终使用 schema 的默认列表而忽略工作流中保存的开关值。

本地模式开启或关闭时都不会自动替换当前工作流值，以便缺失模型始终保留为失效输入并显示 ComfyUI 警告。

### 6. 独立 CLI 兼容性

`inference_cli.py` 不再在未指定参数时强制注入旧的 `models/SEEDVR2` 路径：

- 不指定 `--model_dir`：按 ComfyUI 类别解析；无 `folder_paths` 时回退到 `./models/SEEDVR2`。
- 指定 `--model_dir`：保留 CLI 手动覆盖行为。

因此节点使用共享模型机制，独立 CLI 仍可使用原有目录结构。

## 文件变更清单

### 已修改

- `README.md`：补充标准模型类别和共享目录使用说明。
- `__init__.py`：注册本地模型列表 API，并启用 `web` 目录。
- `inference_cli.py`：移除 CLI 对旧统一目录的强制注入。
- `src/core/model_configuration.py`：在首次加载和缓存恢复时区分 DiT/VAE 类别。
- `src/interfaces/dit_model_loader.py`：接入分类模型路径。
- `src/interfaces/vae_model_loader.py`：接入分类模型路径。
- `src/interfaces/video_upscaler.py`：不再传入统一旧模型目录。
- `src/utils/constants.py`：提供类别化目录、扫描和查找工具。
- `src/utils/downloads.py`：按类别下载，并保护已注册本地模型。
- `src/utils/model_registry.py`：读取本地列表、保留预设 fallback，并支持本地列表查询。

### 新增

- `web/seedvr2_model_sources.js`：前端本地模型过滤扩展。

## 兼容性与升级说明

- 现有工作流中的模型名称不变，不添加 `[Local]` 或 `[Preset]` 前缀。
- 旧工作流没有 `local_models_only` 时默认关闭，不改变原有可选模型范围。
- Comfy Desktop/ComfyUI 需要重启以注册新路由和 `WEB_DIRECTORY`；浏览器端建议强制刷新以加载新 JavaScript。
- 没有共享模型配置时，插件会使用当前实例的标准 `diffusion_models` 和 `vae` 目录。
- 旧的 `models/SEEDVR2` 目录会作为 DiT/VAE 的读取 fallback 保留；在 ComfyUI 环境中不会作为新的下载目标。
- 旧的 `unet` 类别映射可以继续读取，但不会作为新的 DiT 下载目录，避免模型被写入错误位置。
- 节点初始化会遵循工作流保存的 `local_models_only` 值：关闭使用预设列表，开启使用完整本地列表。

## 验证清单

建议在 Comfy Desktop 使用的 Python 环境中执行：

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

在 ComfyUI 环境中还应确认：

1. `diffusion_models` 和 `vae` 返回各自注册目录。
2. DiT 不会从 `vae` 返回，VAE 不会从 `diffusion_models` 返回。
3. 本地同名模型只出现一次，并优先于预设模型。
4. 已注册本地模型不会触发下载或哈希覆盖。
5. `local_models_only` 开关能正确过滤并恢复模型列表。
6. 后端拒绝不存在的本地模型选择。
7. 未提供 `--model_dir` 的 CLI 和显式覆盖路径的 CLI 均能保持预期行为。
8. 删除已选择的自定义模型后，节点校验返回缺失模型错误并显示失效状态。

## 设计结论

该改动将“模型名称”“模型类别”和“物理路径”彻底分离：ComfyUI 负责目录注册，节点保存稳定的文件名，加载和下载阶段按类别解析完整路径。这样共享目录、实例目录和独立 CLI fallback 可以共存，同时避免 DiT/VAE 误匹配及本地模型被预设下载覆盖。
