# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Configuration utility functions
"""

import importlib
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
from ..utils.model_registry import MODEL_CLASSES

#imports for Dynamic VRAM patching
import comfy
import torch
import torch.nn as nn
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception as e:
    if "already registered" not in str(e):
        raise

def swap_layers_recursively(model, target_ops):
    """
    Recursively replaces standard torch.nn modules with custom ComfyUI operations 
    from the provided target_ops class (e.g., manual_cast, fp8_ops).
    """
    for name, child in model.named_children():
        new_layer = None
        
        # 1. Handle Linear Layers
        if isinstance(child, nn.Linear) and hasattr(target_ops, "Linear"):
            new_layer = target_ops.Linear(
                child.in_features, 
                child.out_features, 
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            
        # 2. Handle Convolutional Layers (1D, 2D)
        elif isinstance(child, (nn.Conv1d, nn.Conv2d)):
            dim = 2 if isinstance(child, nn.Conv2d) else 1
            op_name = f"Conv{dim}d"
            
            if hasattr(target_ops, op_name):
                target_cls = getattr(target_ops, op_name)
                new_layer = target_cls(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )

        # 2b. Handle InflatedCausalConv3d
        elif isinstance(child, InflatedCausalConv3d) and hasattr(target_ops, "Conv3d"):

            class PatchedCausalConv3d(InflatedCausalConv3d, target_ops.Conv3d):
                def __init__(self, layer):
                    # 1. Initialize target_ops.Conv3d using the exact parameters from the old layer
                    super().__init__(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=(layer.temporal_padding, *layer.padding[1:]), # Restore true padding for init
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=(layer.bias is not None),
                        device=layer.weight.device,
                        dtype=layer.weight.dtype,
                        inflation_mode = layer.inflation_mode,
                        memory_device = layer.memory_device
                    )

            # Instantiate, share weight/bias references, and replace
            new_layer = PatchedCausalConv3d(child)
            new_layer.weight = child.weight
            if child.bias is not None:
                new_layer.bias = child.bias

            setattr(model, name, new_layer)

        # 3. Handle Normalization Layers
        elif isinstance(child, nn.LayerNorm) and hasattr(target_ops, "LayerNorm"):
            new_layer = target_ops.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                device=child.weight.device if child.elementwise_affine else None,
                dtype=child.weight.dtype if child.elementwise_affine else None
            )
            
        elif isinstance(child, nn.GroupNorm) and hasattr(target_ops, "GroupNorm"):
            new_layer = target_ops.GroupNorm(
                child.num_groups,
                child.num_channels,
                eps=child.eps,
                affine=child.affine,
                device=child.weight.device if child.affine else None,
                dtype=child.weight.dtype if child.affine else None
            )

        # 4. Handle Embeddings
        elif isinstance(child, nn.Embedding) and hasattr(target_ops, "Embedding"):
            new_layer = target_ops.Embedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=child.weight.device,
                dtype=child.weight.dtype
            )

        # 5. Apply replacement or recurse
        if new_layer is not None:
            setattr(model, name, new_layer)
        else:
            # If no replacement happened, recurse into children
            swap_layers_recursively(child, target_ops)

    return model

def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration. Will resolve inheritance.
    """
    
    #print(path)
    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Union[DictConfig, ListConfig]], Union[DictConfig, ListConfig]],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:
    """
    Import a python item, checking model registry first.
    
    Args:
        path: Module path
        name: Class/function name to import
        
    Returns:
        Imported object
    """
    # Simple lookup with path as key
    if path in MODEL_CLASSES:
        return MODEL_CLASSES[path]
    
    # Fallback to dynamic import for everything else
    try:
        return getattr(importlib.import_module(path), name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{name}' from '{path}': {e}")


def create_object(config: DictConfig) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)
    """
    
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    default_ops = comfy.ops.manual_cast
    if args == "as_config":
        return swap_layers_recursively(item(config), default_ops)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return swap_layers_recursively(item(**config), default_ops)
    raise NotImplementedError(f"Unknown args type: {args}")
