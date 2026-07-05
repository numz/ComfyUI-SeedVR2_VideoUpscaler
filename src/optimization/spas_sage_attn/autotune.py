"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note: This module provides advanced autotuning functionality.
Some features require the full SpargeAttn package to be installed globally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
from einops import rearrange

# Try to import optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Simple fallback

try:
    import numpy as np
except ImportError:
    np = None

# Use local imports instead of global package
try:
    from .utils import precision_metric
    from .core import spas_sage_attn_meansim_topk_cuda as spas_sage_attn_meansim_cuda
    from .core import spas_sage2_attn_meansim_topk_cuda as spas_sage2_attn_meansim_cuda
    AUTOTUNE_AVAILABLE = True
except ImportError:
    AUTOTUNE_AVAILABLE = False
    precision_metric = None
    spas_sage_attn_meansim_cuda = None
    spas_sage2_attn_meansim_cuda = None

def extract_sparse_attention_state_dict(model, verbose=False):
    saved_state_dict = {}
    for k, v in model.named_modules(): # enumerate all nn.Module instance in the model
        if isinstance(v, SparseAttentionMeansim):
            if verbose: print(k, 'is an instance of SparseAttentionMeansim')
            for model_key, model_param in model.state_dict().items(): # find the corresponding state_dict item
                if k in model_key:
                    if verbose: print(f'{model_key} is a substate_dict of {k}, we will save it.')
                    saved_state_dict[model_key] = model_param
    return saved_state_dict


def load_sparse_attention_state_dict(model, saved_state_dict, multigpu=False, verbose=False):
    if not multigpu:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    for k, v in model.named_modules():
        if isinstance(v, SparseAttentionMeansim): # find each SparseAttentionMeansim instance
            if verbose: print(k, 'is an instance of SparseAttentionMeansim, but it is empty now.')
            for sk, sv in saved_state_dict.items():
                if k in sk:
                    if verbose: print(f'{sk} is a substate_dict of {k}, we will load it.')
                    sub_name = sk.split(k)[1][1:]
                    if multigpu:
                        sv= sv.to(device=v.device)
                    else:
                        sv = sv.to(device=device, dtype=dtype)
                    setattr(v, sub_name, nn.Parameter(sv, requires_grad=False))
    if not multigpu:
        model = model.to(device)
    return model


def partition_points_into_line(points, block_size, min_dim1=-1, max_dim1=1):
    blocks = {}
    for point in points:
        dim1 = point['simthreshd1']
        # Calculate block indices for dim1 and dim2
        block_index_dim1 = int((dim1 - min_dim1) // block_size)
        key = (block_index_dim1,)
        # Initialize the block if it doesn't exist
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(point)
    return blocks

# GPUProcessPoolExecutor is an optional dependency for multi-GPU tuning
# It's not required for basic SpargeAttn functionality
try:
    from tools.gpu_process import GPUProcessPoolExecutor
    executor = GPUProcessPoolExecutor()
except (ImportError, ModuleNotFoundError):
    GPUProcessPoolExecutor = None
    executor = None

class SparseAttentionMeansim(nn.Module):
    def __init__(self, sim_rule="l1", l1=0.07, pv_l1=0.08, cos_sim=0.98, rmse=0.07, rearrange_kwargs={}, tune_pv=True):
        super(SparseAttentionMeansim, self).__init__()
        self.head_num = None
        assert l1 >= 0 and cos_sim <= 1 and rmse >= 0, "l1, cos_sim, rmse should be legal"
        assert pv_l1 > l1, 'pv l1 must greater than l1'
        self.l1 = l1
        self.pv_l1 = pv_l1
        self.cos_sim = cos_sim
        self.rmse = rmse
        self.is_sparse = None  # bool, shape of head number, decide whether to use sparse attention for each head
        self.cdfthreshd = None  # float, shape of head number, decide the threshold of cdf for each head
        self.simthreshd1 = None
        self.simthreshd2 = None
        self.pvthreshd = None
        self.tuning_sparsity = None
        self.num_data_passed = 0
        self.hyperparams_cache = {}
        self.sim_rule = sim_rule
        self.rearrange_kwargs = rearrange_kwargs
        self.tune_pv = tune_pv
    
    def is_sim(self, o_gt, o_sparse):
        if self.sim_rule == "cosine":
            return precision_metric(o_sparse, o_gt, verbose=False)["Cossim"] > self.cos_sim
        elif self.sim_rule == "rmse":
            return precision_metric(o_sparse, o_gt, verbose=False)["RMSE"] < self.rmse
        elif self.sim_rule == "l1":
            return precision_metric(o_sparse, o_gt, verbose=False)["L1"] < self.l1
        else:
            raise ValueError("sim_rule should be one of ['cosine', 'rmse', 'l1']")
    
    def init_hyperparams(self, head_num, device):
        self.head_num = head_num
        self.is_sparse = nn.Parameter(
            torch.ones(self.head_num, dtype=torch.bool, device=device),
            requires_grad=False,
        )
        self.cdfthreshd = nn.Parameter(
            torch.ones(self.head_num, device=device) * 0.1,
            requires_grad=False,
        )
        self.simthreshd1 = nn.Parameter(
            torch.ones(self.head_num, device=device) * -1,
            requires_grad=False,
        )
        self.simthreshd2 = nn.Parameter(
            torch.zeros(self.head_num, device=device),
            requires_grad=False,
        )
        self.pvthreshd = nn.Parameter(
            torch.ones(self.head_num, device=device) * 20,
            requires_grad=False,
        )
        self.tuning_sparsity = torch.zeros(self.head_num, device=device)
        self.num_data_passed = 0
        self.hyperparams_cache = {}

    def kernel_selection(self):
        sm = torch.cuda.get_device_capability()
        sm = 10*sm[0] + sm[1]
        if sm >= 89:
            return spas_sage2_attn_meansim_cuda
        else:
            warnings.warn(f'{sm=}, do not support sageattn2, using sageattn1 kernel')
            return spas_sage_attn_meansim_cuda

    @torch.no_grad()
    def tune_pvthreshd(self, qi, ki, vi, mask=None, is_causal=False, smooth_k=True, simthreshd1=None, cdfthreshd=None):
        gt_i = F.scaled_dot_product_attention(qi, ki, vi, mask, is_causal=is_causal)
        cur_min_pvthreshd = 0.0
        cur_max_pvthreshd = 50.0
        cur_pvthreshd = 20.0
        delta = 0.1
        while cur_max_pvthreshd - cur_min_pvthreshd > delta:
            kernel = self.kernel_selection()
            sparse_i, sparsity = kernel(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                cdfthreshd=cdfthreshd,
                smooth_k=smooth_k,
                return_sparsity=True,
                simthreshd1=simthreshd1 if simthreshd1 is not None else self.simthreshd1,
                pvthreshd=cur_pvthreshd,
            )
            if precision_metric(sparse_i, gt_i, verbose=False)["L1"] < self.pv_l1:
                cur_max_pvthreshd = cur_pvthreshd
                cur_pvthreshd = (cur_pvthreshd + cur_min_pvthreshd) / 2
            else:
                cur_min_pvthreshd = cur_pvthreshd
                cur_pvthreshd = (cur_pvthreshd + cur_max_pvthreshd) / 2
        return cur_pvthreshd, sparsity

    @torch.no_grad()
    def tune_cdfthreshd(
        self, qi, ki, vi, mask=None, is_causal=False, smooth_k=True, simthreshd1=None
    ):
        gt_i = F.scaled_dot_product_attention(qi, ki, vi, mask, is_causal=is_causal)
        cur_min_cdfthreshd = 0.0
        cur_max_cdfthreshd = 1.0
        cur_cdfthreshd = 0.1
        delta = 0.001
        while cur_max_cdfthreshd - cur_min_cdfthreshd > delta:
            kernel = self.kernel_selection()
            sparse_i, sparsity = kernel(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                cdfthreshd=cur_cdfthreshd,
                smooth_k=smooth_k,
                return_sparsity=True,
                simthreshd1=simthreshd1 if simthreshd1 is not None else self.simthreshd1,
            )
            if self.is_sim(gt_i, sparse_i):
                cur_max_cdfthreshd = cur_cdfthreshd
                cur_cdfthreshd = (cur_cdfthreshd + cur_min_cdfthreshd) / 2
            else:
                cur_min_cdfthreshd = cur_cdfthreshd
                cur_cdfthreshd = (cur_cdfthreshd + cur_max_cdfthreshd) / 2

        if cur_cdfthreshd > 1 - delta:  # could not reach precision, using full attention
            cur_cdfthreshd = 1
        elif cur_cdfthreshd < delta:
            # no sim block is already enough for precision, mostly not sparse, using full attention
            # suggest to use more data to tune or use full attention
            pass
        return cur_cdfthreshd, sparsity

    @torch.no_grad()
    def autotune(self, qi, ki, vi, head_idx, mask=None, is_causal=False, smooth_k=True):
        qi = qi.to(torch.cuda.current_device())
        ki = ki.to(torch.cuda.current_device())
        vi = vi.to(torch.cuda.current_device())
        all_hyperparams = []
        granularity = 16
        for simthreshd1 in range(int(-1 * granularity), int(1 * granularity)):
            simthreshd1 = simthreshd1 / granularity
            cur_cdfthreshd, sparsity = self.tune_cdfthreshd(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                smooth_k=smooth_k,
                simthreshd1=simthreshd1,
            )
            if self.tune_pv:
                pvthreshd, _ = self.tune_pvthreshd(
                    qi,
                    ki,
                    vi,
                    mask,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    simthreshd1=simthreshd1,
                    cdfthreshd=cur_cdfthreshd,
                )
            else:
                pvthreshd = 20
            all_hyperparams.append({
                "simthreshd1": simthreshd1,
                "cdfthreshd": cur_cdfthreshd,
                'pvthreshd': pvthreshd,
                "sparsity": sparsity,
                'data_idx': self.num_data_passed
            })
            if sparsity < 0.1:
                break  # no need to continue to raise threshold bound
        if self.hyperparams_cache.get(head_idx) is None:
            self.hyperparams_cache[head_idx] = []
        cache_hyper = self.hyperparams_cache[head_idx]
        all_hyperparams = all_hyperparams + cache_hyper
        self.hyperparams_cache[head_idx] = all_hyperparams
        
        grid = partition_points_into_line(all_hyperparams, 2/granularity)
        groups = list(grid.values())
        # sort by sum of sparsity, local smoothing
        groups = sorted(groups, key=lambda x: sum([y['sparsity'] for y in x]), reverse=True)
        final_group = groups[0]
        final_simthreshd1 = np.max([x['simthreshd1'] for x in final_group]).item()
        final_cdfthreshd = np.max([x['cdfthreshd'] for x in final_group]).item()
        final_pvthreshd = np.max([x['pvthreshd'] for x in final_group]).item()
        mean_sparsity = np.mean([x['sparsity'] for x in final_group]).item()
        return {
            'final_simthreshd1': final_simthreshd1,
            'final_cdfthreshd': final_cdfthreshd,
            'final_pvthreshd': final_pvthreshd,
            'mean_sparsity': mean_sparsity,
            'head_idx': head_idx
        }
        
    def fill_results(self, rtdict):
        head_idx = rtdict['head_idx']
        self.simthreshd1[head_idx] = rtdict['final_simthreshd1']
        self.cdfthreshd[head_idx] = rtdict['final_cdfthreshd']
        self.pvthreshd[head_idx] = rtdict['final_pvthreshd']
        self.is_sparse[head_idx] = rtdict['mean_sparsity'] > 0.1 and self.is_sparse[head_idx]
        self.tuning_sparsity[head_idx] = rtdict['mean_sparsity']
        if not self.is_sparse[head_idx]:
            self.cdfthreshd[head_idx] = 1
            self.simthreshd1[head_idx] = 1
        
    @torch.no_grad()
    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        is_causal=False,
        scale=None,
        tensor_layout="HND",
        tune_mode=False,
        smooth_k=True,
        return_sparsity=False,
    ):
        assert len(q.shape) == 4, "q should be 4-d tensor with B, H, L, D"
            
        if os.environ.get("TUNE_MODE", "") != "" or tune_mode:
            if tensor_layout == 'NHD':
                q = rearrange(q, '... L H D -> ... H L D')
                k = rearrange(k, '... L H D -> ... H L D')
                v = rearrange(v, '... L H D -> ... H L D')
            if self.is_sparse is None:  # init per head hyper parameters
                self.init_hyperparams(q.shape[1], q.device)
            if os.environ.get('PARALLEL_TUNE', '') == '':
                for i in tqdm(range(self.head_num)):
                    if not self.is_sparse[i].item():
                        continue
                    qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
                    rtdict = self.autotune(qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
                    self.fill_results(rtdict)
            else:
                futures = []
                for i in range(self.head_num):
                    if not self.is_sparse[i].item():
                        continue
                    qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
                    future = executor.submit(self.autotune, qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
                    futures.append(future)
                for future in tqdm(futures):
                    rtdict = future.result()
                    self.fill_results(rtdict)
                    
                
            self.num_data_passed += 1
            print(f'{self.cdfthreshd=}')
            print(f'{self.simthreshd1=}')
            print(f'{self.is_sparse=}')
            print(f'{self.pvthreshd=}')
            print(f'{self.tuning_sparsity=}')
            print(f'mean sparsity:{self.tuning_sparsity.mean().item()}')
            o = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
            if tensor_layout == 'NHD':
                o = rearrange(o, '... H L D -> ... L H D')
            torch.cuda.empty_cache()
        else:
            assert self.cdfthreshd is not None, "attention hyperparameters should be tuned first"
            kernel = self.kernel_selection()
            o = kernel(
                q,
                k,
                v,
                mask,
                is_causal=is_causal,
                smooth_k=smooth_k,
                scale=scale,
                tensor_layout=tensor_layout,
                cdfthreshd=self.cdfthreshd,
                simthreshd1=self.simthreshd1,
                pvthreshd=self.pvthreshd.float(),
                return_sparsity=return_sparsity,
                attention_sink= True,  # Only keep True when inference !!!!
            )
        
        if return_sparsity:
            o, total_sparsity = o
            return o, total_sparsity
        else:
            return o
