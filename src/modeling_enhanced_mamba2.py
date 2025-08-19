# Adapted from flash-linear-attention v0.2.0 https://github.com/fla-org/flash-linear-attention/blob/v0.2.0/fla/models/mamba2/modeling_mamba2.py

# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
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
"""PyTorch MAMBA2 model."""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.layernorm_gated import RMSNormGated
from fla.models.mamba2.modeling_mamba2 import pad_tensor_by_size, reshape_into_chunks, segment_sum, apply_mask_to_padding_states

from src.configuration_enhanced_mamba2 import EnhancedMamba2Config

logger = logging.get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    except ImportError:
        (
            selective_state_update,
            mamba_chunk_scan_combined,
            mamba_split_conv1d_scan_combined,
        ) = (None, None, None)
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_update, causal_conv1d_fn = None, None
    is_fast_path_available = all((
        selective_state_update,
        causal_conv1d_fn,
        causal_conv1d_update
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


class EnhancedMamba2Cache:
    """
    Arguments:
        config: EnhancedMamba2Config
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config.
        n_groups: (`int`):
            Model's number of groups taken from the config - similar to tensor parallel in Transformer.
        state_size: (`int`):
            Model's SSM state size taken from config.
        num_heads: (`int`):
            The number of heads used in the linear attention / SSM.
        head_dim: (`int`):
            The respective dimension of the heads used in the linear attention / SSM.
        intermediate_size: (`int`):
            Model's intermediate_size based on (expand * hidden_dim) from config.
        conv_states: (`torch.Tensor`):
            A tensor of shape `[num_layers, batch_size, conv_kernel_size, intermediate_size + 2 * n_groups * state_size]`
            that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[num_layers, batch_size, num_heads, head_dim, state_size]` that holds ssm states.
    """

    def __init__(
        self,
        config: EnhancedMamba2Config,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )

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
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class EnhancedMamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: EnhancedMamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.sparse_arch = config.sparse_arch
        self.total_sparse_keys = config.sparse_keys
        key_source = len(config.sparse_arch.split("+"))
        self.sparse_keys = config.sparse_keys // key_source
        if "dilated" in config.sparse_arch:
            self.dilation = config.data_max_length // self.sparse_keys
        if "lsh" in config.sparse_arch:
            self.num_hash = config.num_hash

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        if self.sparse_arch:
            self.xx_proj = nn.Linear(self.intermediate_size, self.ssm_state_size * 2, bias=False)
            self.E = nn.Parameter(torch.ones(self.intermediate_size))

        if "key_selection" in self.sparse_arch:
            self.dx_proj_1 = nn.Linear(self.ssm_state_size * 2, self.ssm_state_size * 2)
            self.dx_proj_2 = nn.Linear(self.ssm_state_size * 2, self.ssm_state_size * 2)
            self.dx_proj_3 = nn.Linear(self.ssm_state_size * 2, 1)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = RMSNormGated(
            self.intermediate_size, eps=self.layer_norm_epsilon, norm_before_gate=False
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of "
                "`(selective_state_update, causal_conv1d_fn, causal_conv1d_update)` is None. "
                "Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation and"
                "https://github.com/Dao-AILab/causal-conv1d"
            )

    def ranking_loss(self, pred, target):
        target = ((target.unsqueeze(-1) > target.unsqueeze(-2)).float() + (target.unsqueeze(-1) >= target.unsqueeze(-2)).float()) / 2
        pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
        return loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[EnhancedMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        # Single step calculations via cache
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            raise NotImplementedError

        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            _, _, gate, hidden_states_B_C, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed,
                    (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                )
                cache_params.update_conv_state(
                    layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                )

            if self.activation not in ["silu", "swish"]:
                hidden_states_B_C = self.act(
                    self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                )
            else:
                hidden_states_B_C = causal_conv1d_fn(
                    x=hidden_states_B_C.transpose(1, 2),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            
            hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            K2Q2 = self.xx_proj(hidden_states)
            K2, Q2 = torch.split(K2Q2, [self.ssm_state_size, self.ssm_state_size], dim=-1)

            # 3.b. Discretize K and Q and forward with attention
            V = hidden_states.transpose(-1, -2)
            sqrt_2 = math.sqrt(self.ssm_state_size)
            K2 = K2 / sqrt_2

            if self.sparse_arch:
                if self.sparse_arch == "full": # full attention

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

            # 3. SSM transformation
            scan_output, ssm_state = mamba_chunk_scan_combined(
                hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                dt,
                A,
                B.view(batch_size, seq_len, self.n_groups, -1),
                C.view(batch_size, seq_len, self.n_groups, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=None,
                return_final_states=True,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                **dt_limit_kwargs,
            )

            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

            scan_output = scan_output.view(batch_size, seq_len, -1)
            if self.sparse_arch:
                scan_output = scan_output + attn_states.transpose(-1, -2) * self.E[None, None, :]
            # Multiply "gate" branch and apply extra normalization layer
            scan_output = self.norm(scan_output, gate)

            # 4. Final linear projection
            out = self.out_proj(scan_output)
        return out, gate_loss


class EnhancedMamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = EnhancedMamba2Mixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[EnhancedMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states, gate_loss = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        return hidden_states, gate_loss


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->Mamba2
class EnhancedMamba2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
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
    cache_params: Optional[EnhancedMamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaCausalLMOutput with Mamba->Mamba2
class EnhancedMamba2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
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
    cache_params: Optional[EnhancedMamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: Optional[torch.FloatTensor] = None


class EnhancedMamba2PreTrainedModel(PreTrainedModel, GenerationMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EnhancedMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["EnhancedMamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(
        self,
        module: nn.Module,
        num_residuals_per_layer: int = 1,
    ):
        """Initialize the weights."""
        if isinstance(module, EnhancedMamba2Mixer):

            # --- A_log ---
            A = torch.arange(1, module.num_heads + 1)
            with torch.no_grad():
                if not isinstance(module.A_log, torch.distributed.tensor.DTensor):
                    module.A_log.copy_(torch.log(A))
                else:
                    logger.warning_once("`A_log` is a DTensor, skipping initialization")
            module.A_log._no_weight_decay = True

            # --- D ---
            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            # --- dt_bias ---
            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if not isinstance(module.dt_bias, torch.distributed.tensor.DTensor):
                    module.dt_bias.copy_(inv_dt)
                else:
                    logger.warning_once("`dt_bias` is a DTensor, skipping initialization")
            module.dt_bias._no_reinit = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                # guard against deprecated behavior
                if hasattr(module.bias, "_no_reinit"):
                    raise ValueError("This is not supposed to happen")
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                # p = module.o_proj.weight
                # guard against deprecated behavior
                raise ValueError("This is not supposed to happen")
            elif hasattr(module, 'out_proj'):
                p = module.out_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class EnhancedMamba2Model(EnhancedMamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([EnhancedMamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

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
        cache_params: Optional[EnhancedMamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, EnhancedMamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = EnhancedMamba2Cache(
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
                    mixer_block.__call__,
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
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

        return EnhancedMamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            gate_loss=all_gate_loss,
        )


class EnhancedMamba2ForCausalLM(EnhancedMamba2PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = EnhancedMamba2Model(config)
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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[EnhancedMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
                input_ids = input_ids[:, -1][..., None]

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
            model_inputs = {"input_ids": input_ids}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'attention_mask': attention_mask,
            'cache_params': cache_params,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'logits_to_keep': logits_to_keep
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[EnhancedMamba2Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, EnhancedMamba2CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
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
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return EnhancedMamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
            gate_loss=mamba_outputs.gate_loss,
        )