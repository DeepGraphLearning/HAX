# Adapted from flash-linear-attention v0.2.0 https://github.com/fla-org/flash-linear-attention/blob/v0.2.0/fla/models/mamba/modeling_mamba.py

# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBA model."""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm

from src.configuration_enhanced_mamba import EnhancedMambaConfig

logger = logging.get_logger(__name__)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    except ImportError:
        selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_update, causal_conv1d_fn = None, None
    is_fast_path_available = all((
        selective_state_update,
        selective_scan_fn,
        causal_conv1d_fn,
        causal_conv1d_update,
        mamba_inner_fn
    ))


def cumtopk(x, k, dim=-1):
    """
    Computes the cumulative top-k elements along a specified dimension,
    returning both the values and their indices.

    Parameters:
    - x (torch.Tensor): The input tensor.
    - k (int): The number of top elements to keep.
    - dim (int): The dimension along which to compute the cumulative top-k.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
        - values: The cumulative top-k values.
        - indices: The indices of the cumulative top-k values.
    """
    # Move the specified dimension to the last dimension
    x = x.transpose(dim, -1)
    *batch_shape, size = x.shape
    x_flat = x.reshape(-1, size)  # Flatten batch dimensions

    # Prepare output tensors
    out_shape = batch_shape + [size, k]
    values_out = x.new_empty(out_shape)
    indices_out = x.new_empty(out_shape, dtype=torch.long)

    # Create a mask to zero out elements beyond the current position
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=x.device))
    x_cumulative = x_flat.unsqueeze(1).expand(-1, size, -1)
    x_cumulative = x_cumulative.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Compute the top-k elements and their indices at each position
    topk_values, topk_indices = x_cumulative.topk(k, dim=2)
    values_out = topk_values.reshape(*out_shape)
    indices_out = topk_indices.reshape(*out_shape)

    # Adjust indices to match the original tensor's indexing
    # No adjustment needed since indices correspond to the last dimension positions

    # Move the last dimension back to its original position
    if dim != -1 and dim != x.dim() - 1:
        dims = list(range(values_out.dim()))
        dims[-2], dims[dim] = dims[dim], dims[-2]
        values_out = values_out.permute(*dims)
        indices_out = indices_out.permute(*dims)

    return values_out, indices_out


class EnhancedMambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        max_batch_size: Optional[int] = None,
    ):
        if max_batch_size is not None:
            logger.warning_once(
                f"The 'max_batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.46. Use the more precisely named 'batch_size' argument instead."
            )
        self.dtype = dtype
        self.batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: torch.Tensor = torch.zeros(
            config.num_hidden_layers,
            self.batch_size,
            self.intermediate_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states: torch.Tensor = torch.zeros(
            config.num_hidden_layers,
            self.batch_size,
            self.intermediate_size,
            self.ssm_state_size,
            device=device,
            dtype=dtype,
        )

        torch._dynamo.mark_static_address(self.conv_states)
        torch._dynamo.mark_static_address(self.ssm_states)

        self.hash_proj_2 = {
            i: None
            for i in range(config.num_hidden_layers)
        }
        self.query_states_2 = {
            i: None
            for i in range(config.num_hidden_layers)
        }
        self.key_states_2 = {
            i: None
            for i in range(config.num_hidden_layers)
        }
        self.val_states = {
            i: None
            for i in range(config.num_hidden_layers)
        }

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class EnhancedMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: EnhancedMambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias

        self.sparse_arch = config.sparse_arch
        if self.sparse_arch:
            self.total_sparse_keys = config.sparse_keys
            key_source = len(config.sparse_arch.split("+"))
            self.sparse_keys = config.sparse_keys // key_source
            if "dilated" in config.sparse_arch:
                self.dilation = config.data_max_length // self.sparse_keys
            if "lsh" in config.sparse_arch:
                self.num_hash = config.num_hash

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        if self.sparse_arch:
            self.xx_proj = nn.Linear(self.intermediate_size, self.ssm_state_size * 2, bias=False)
            self.E = nn.Parameter(torch.ones(self.intermediate_size))

        if "key_selection" in self.sparse_arch:
            self.dx_proj_1 = nn.Linear(self.ssm_state_size * 2, self.ssm_state_size * 2)
            self.dx_proj_2 = nn.Linear(self.ssm_state_size * 2, self.ssm_state_size * 2)
            self.dx_proj_3 = nn.Linear(self.ssm_state_size * 2, 1)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of "
                "`(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def ranking_loss(self, pred, target):
        target = ((target.unsqueeze(-1) > target.unsqueeze(-2)).float() + (target.unsqueeze(-1) >= target.unsqueeze(-2)).float()) / 2
        pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
        return loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[EnhancedMambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if cache_params is not None and cache_position[0] > 0:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

        A = -torch.exp(self.A_log.float())

        K2Q2 = self.xx_proj(hidden_states.transpose(1, 2))
        K2, Q2 = torch.split(K2Q2, [self.ssm_state_size, self.ssm_state_size], dim=-1)

        # 3.b. Discretize K and Q and forward with attention
        V = hidden_states
        sqrt_2 = math.sqrt(self.ssm_state_size)
        K2 = K2 / sqrt_2

        if self.sparse_arch:
            if self.sparse_arch == "full": # full attention

                if cache_params is not None:
                    if cache_params.seqlen_offset > 0:
                        K2 = torch.concat((cache_params.key_states_2[self.layer_idx], K2), dim=1)
                        V = torch.concat((cache_params.val_states[self.layer_idx], V), dim=2)

                    cache_params.key_states_2[self.layer_idx] = K2.clone().detach()
                    cache_params.val_states[self.layer_idx] = V.clone().detach()
                
                input_shape = (V.shape[0], V.shape[2])
                causal_seq_ids = torch.arange(input_shape[1], device=V.device)
                causal_mask = causal_seq_ids[None, None, :].repeat(input_shape[0], input_shape[1], 1) <= causal_seq_ids[None, :, None]
                causal_mask = torch.cat((torch.ones(input_shape[0], input_shape[1], K2.shape[1] - input_shape[1], device=V.device, dtype=V.dtype), causal_mask.to(V.dtype)), dim=-1)
                attn_mask = (1.0 - causal_mask) * torch.finfo(V.dtype).min

                attn_score = torch.matmul(Q2, K2.transpose(-1, -2))
                attn_score = attn_score + attn_mask
                attn_score = nn.functional.softmax(attn_score, dim=-1)
                attn_states = torch.matmul(V, attn_score.transpose(-1, -2))

                gate_loss = 0

            else: # sparse attention

                if cache_params is not None:
                    if cache_params.seqlen_offset > 0:
                        Q2 = torch.concat((cache_params.query_states_2[self.layer_idx], Q2), dim=1)
                        K2 = torch.concat((cache_params.key_states_2[self.layer_idx], K2), dim=1)
                        V = torch.concat((cache_params.val_states[self.layer_idx], V), dim=2)

                    cache_params.query_states_2[self.layer_idx] = Q2.clone().detach()
                    cache_params.key_states_2[self.layer_idx] = K2.clone().detach()
                    cache_params.val_states[self.layer_idx] = V.clone().detach()
                
                input_shape = (V.shape[0], V.shape[2])
                causal_seq_ids = torch.arange(input_shape[1], device=V.device)
                causal_mask = causal_seq_ids[None, None, :].repeat(input_shape[0], input_shape[1], 1) <= causal_seq_ids[None, :, None]

                if "sink" in self.sparse_arch:
                    idx = torch.arange(K2.shape[1], device=K2.device).unsqueeze(0).expand(Q2.shape[1], K2.shape[1])
                    sink_mask = (idx < self.sparse_keys).unsqueeze(0).repeat(input_shape[0], 1, 1) & causal_mask
                else:
                    sink_mask = torch.zeros_like(causal_mask)

                if "sliding_window" in self.sparse_arch:
                    idx_Q2 = torch.arange(Q2.shape[1], device=Q2.device).unsqueeze(1)
                    idx_K2 = torch.arange(K2.shape[1], device=K2.device).unsqueeze(0)
                    idx_d2 = idx_Q2 - idx_K2
                    sliding_window_mask = ((idx_d2 >= 0) & (idx_d2 < self.sparse_keys)).unsqueeze(0).repeat(input_shape[0], 1, 1)
                else:
                    sliding_window_mask = torch.zeros_like(causal_mask)
                
                if "dilated" in self.sparse_arch:
                    idx = torch.arange(K2.shape[1], device=K2.device).unsqueeze(0).expand(Q2.shape[1], K2.shape[1])
                    dilated_mask = (idx % self.dilation == self.dilation - 1).unsqueeze(0).repeat(input_shape[0], 1, 1) & causal_mask
                else:
                    dilated_mask = torch.zeros_like(causal_mask)

                if "argmax_lsh" in self.sparse_arch:
                    def lsh_sliding_window(tensor, B):
                        assert tensor.dtype == torch.bool, "Input tensor must be of boolean type."
                        reversed_tensor = tensor.flip(dims=[-1])
                        cumsum = torch.cumsum(reversed_tensor.int(), dim=-1)
                        mask = cumsum <= B
                        result = (reversed_tensor & mask).flip(dims=[-1])
                        return result

                    NQ2, NK2 = Q2 - Q2.mean(1, keepdim=True), K2 - K2.mean(1, keepdim=True)
                    NQ2, NK2 = nn.functional.normalize(NQ2), nn.functional.normalize(NK2)
                    H2 = torch.randn((self.ssm_state_size, self.num_hash), device=K2.device, dtype=K2.dtype)
                    HQ2, HK2 = torch.matmul(NQ2, H2), torch.matmul(NK2, H2)
                    HQ_idx, HK_idx = torch.argmax(HQ2, dim=-1), torch.argmax(HK2, dim=-1)
                    argmax_lsh_mask = HQ_idx.unsqueeze(-1) == HK_idx.unsqueeze(-2)
                    argmax_lsh_mask = lsh_sliding_window(causal_mask & argmax_lsh_mask, B=self.sparse_keys)
                else:
                    argmax_lsh_mask = torch.zeros_like(causal_mask)

                if "sign_bin_lsh" in self.sparse_arch:
                    assert self.num_hash < 64
                    def lsh_sliding_window(tensor, B):
                        assert tensor.dtype == torch.bool, "Input tensor must be of boolean type."
                        reversed_tensor = tensor.flip(dims=[-1])
                        cumsum = torch.cumsum(reversed_tensor.int(), dim=-1)
                        mask = cumsum <= B
                        result = (reversed_tensor & mask).flip(dims=[-1])
                        return result

                    NQ2, NK2 = Q2 - Q2.mean(1, keepdim=True), K2 - K2.mean(1, keepdim=True)
                    NQ2, NK2 = nn.functional.normalize(NQ2), nn.functional.normalize(NK2)
                    H2 = torch.randn((self.ssm_state_size, self.num_hash), device=K2.device, dtype=K2.dtype)
                    HQ2, HK2 = torch.matmul(NQ2, H2), torch.matmul(NK2, H2)
                    HQ_bits, HK_bits = (HQ2 > 0).to(torch.long), (HK2 > 0).to(torch.long)
                    weights = (1 << torch.arange(self.num_hash, device=HQ_bits.device, dtype=torch.long)).view(1, 1, self.num_hash)
                    HQ_idx, HK_idx = (HQ_bits * weights).sum(-1), (HK_bits * weights).sum(-1)
                    sign_bin_lsh_mask = HQ_idx.unsqueeze(-1) == HK_idx.unsqueeze(-2)
                    sign_bin_lsh_mask = lsh_sliding_window(causal_mask & sign_bin_lsh_mask, B=self.sparse_keys)
                else:
                    sign_bin_lsh_mask = torch.zeros_like(causal_mask)

                if "key_selection" in self.sparse_arch:
                    TK2, TQ2 = K2.detach(), Q2.detach().cumsum(dim=1)
                    TQ2 = nn.functional.normalize(TQ2)
                    T2 = torch.cat((TK2, TQ2), dim=-1)
                    T2 = nn.functional.relu(self.dx_proj_1(T2))
                    T2 = nn.functional.relu(self.dx_proj_2(T2))
                    key_score = self.dx_proj_3(T2).squeeze(-1)
                
                    key_selection_mask = torch.zeros_like(causal_mask).bool()
                    key_val, key_idx = cumtopk(key_score, k=self.sparse_keys)
                    key_bid = torch.arange(key_idx.shape[0], device=key_idx.device).view(-1, 1, 1).expand(key_idx.shape)
                    key_sid = torch.arange(key_idx.shape[1], device=key_idx.device).view(1, -1, 1).expand(key_idx.shape)
                    key_selection_mask[key_bid, key_sid, key_idx] = True
                    key_selection_mask = key_selection_mask & causal_mask

                    if self.training:
                        noise = torch.rand_like(key_score)
                        if attention_mask is not None:
                            noise = noise.masked_fill(attention_mask == 0, -1.0)
                        selected_idx = noise.topk(self.sparse_keys, dim=1).indices
                        selected_bid = torch.arange(selected_idx.shape[0], device=selected_idx.device).view(-1, 1).expand(selected_idx.shape)
                        selected_keys = key_score[selected_bid, selected_idx]
                        linear_attn_score = torch.matmul(Q2, K2[selected_bid, selected_idx, :].transpose(-1, -2))
                        linear_attn_sid = torch.arange(linear_attn_score.shape[1], device=linear_attn_score.device).view(1, -1, 1).expand(linear_attn_score.shape)
                        linear_attn_score = torch.where(linear_attn_sid >= selected_idx.unsqueeze(1), linear_attn_score.sigmoid(), 0)
                        selected_attn = linear_attn_score.detach().sum(dim=1)
                        gate_loss = self.ranking_loss(selected_keys, selected_attn)
                    else:
                        gate_loss = 0

                else:
                    key_selection_mask = torch.zeros_like(causal_mask)
                    gate_loss = 0

                sparse_attn_mask = sliding_window_mask | sink_mask | dilated_mask | key_selection_mask | argmax_lsh_mask | sign_bin_lsh_mask
                assert ((~causal_mask) & sparse_attn_mask).sum() == 0
                assert (sparse_attn_mask.sum(dim=2) > self.total_sparse_keys).sum() == 0
                sparse_attn_mask = (1.0 - sparse_attn_mask.to(V.dtype)) * torch.finfo(V.dtype).min

                sparse_attn_score = torch.matmul(Q2, K2.transpose(-1, -2))
                sparse_attn_score = sparse_attn_score + sparse_attn_mask
                sparse_attn_score = nn.functional.softmax(sparse_attn_score, dim=-1)

                attn_states = torch.matmul(V, sparse_attn_score.transpose(-1, -2))

        else:
            gate_loss = 0

        if cache_params is not None:
            attn_states = attn_states[:, :, cache_params.seqlen_offset:]

        attn_states = attn_states.to(hidden_states.dtype)

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        if cache_params is not None and cache_position[0] > 0:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(self.layer_idx, ssm_state)

        # 4. Final linear projection
        if self.sparse_arch:
            scan_outputs = scan_outputs + attn_states * self.E[None, :, None] * self.act(gate)
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states, gate_loss


class EnhancedMambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = EnhancedMambaMixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[EnhancedMambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states, gate_loss = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        return hidden_states, gate_loss


@dataclass
class EnhancedMambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[EnhancedMambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: Optional[torch.FloatTensor] = None


@dataclass
class EnhancedMambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[EnhancedMambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: Optional[torch.FloatTensor] = None


class EnhancedMambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EnhancedMambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["EnhancedMambaBlock", "EnhancedMambaMixer"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, EnhancedMambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.data = nn.Parameter(inv_dt.to(module.dt_proj.bias.device))
            module.dt_proj.bias._no_reinit = True
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class EnhancedMambaModel(EnhancedMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([EnhancedMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[EnhancedMambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, EnhancedMambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = EnhancedMambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_gate_loss = 0
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, gate_loss = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states, gate_loss = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            all_gate_loss = all_gate_loss + gate_loss

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states, all_gate_loss] if v is not None)

        return EnhancedMambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            gate_loss=all_gate_loss
        )


class EnhancedMambaForCausalLM(EnhancedMambaPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = EnhancedMambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[EnhancedMambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                if attention_mask is not None:
                    attention_mask = None

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'cache_params': cache_params,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'attention_mask': attention_mask,
            'logits_to_keep': logits_to_keep,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[EnhancedMambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, EnhancedMambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba_outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return (loss,) + output if loss is not None else output

        return EnhancedMambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
            gate_loss=mamba_outputs.gate_loss,
        )