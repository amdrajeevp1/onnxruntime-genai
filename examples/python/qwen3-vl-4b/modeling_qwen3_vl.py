# Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
# 
# This is the reference implementation from HuggingFace Transformers
# Used for understanding the Qwen3-VL architecture
#
# Key sections for vision model:
# - Lines 54-72: Qwen3VLVisionMLP
# - Lines 75-92: Qwen3VLVisionPatchEmbed  ← Conv3D patch embedding
# - Lines 95-109: Qwen3VLVisionRotaryEmbedding
# - Lines 112-127: Qwen3VLVisionPatchMerger  ← Spatial merge + project to text dim
# - Lines 198-245: Qwen3VLVisionAttention  ← Multi-head attention with rotary
# - Lines 248-266: Qwen3VLVisionBlock  ← Transformer block (Attn + MLP)
# - Lines 748-924: Qwen3VLVisionModel  ← Main vision model (forward pass)
#
# Vision model forward pass summary (lines 838-924):
# 1. Patch embedding (Conv3D)
# 2. Position embeddings (bilinear interpolation)
# 3. Rotary position embeddings
# 4. 24 transformer blocks
# 5. Patch merger (2x2 spatial merge + project to 2560)
# 6. Return pooler_output for text injection

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput


@dataclass
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    """Output with deepstack features from vision encoder layers 5, 11, 17"""
    deepstack_features: list[torch.FloatTensor] | None = None


# ============================================================================
# VISION MODEL COMPONENTS
# ============================================================================

class Qwen3VLVisionMLP(nn.Module):
    """
    MLP for vision transformer blocks
    
    Architecture:
        hidden_size (1024) → intermediate_size (4096) → hidden_size (1024)
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    """
    Patch embedding using 3D convolution
    
    Input: [432, 1536] (flattened patches)
    Process: Reshape → Conv3D → Flatten
    Output: [432, 1024] (patch embeddings)
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size  # 16
        self.temporal_patch_size = config.temporal_patch_size  # 2
        self.in_channels = config.in_channels  # 3 (RGB)
        self.embed_dim = config.hidden_size  # 1024

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, 
            self.embed_dim, 
            kernel_size=kernel_size, 
            stride=kernel_size,  # Non-overlapping
            bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # Reshape: [432, 1536] → [-1, 3, 2, 16, 16]
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # Conv3D: [-1, 3, 2, 16, 16] → [-1, 1024, 1, 1, 1]
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        # Flatten: → [432, 1024]
        return hidden_states.view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """
    Rotary position embeddings for vision attention - ONNX Compatible
    
    Pre-computes frequency table for all possible spatial positions.
    For images up to 1536px: max_positions = 96 (1536/16 patches)
    Memory overhead: ~75 KB
    """
    freq_table: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0, max_positions: int = 96) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_positions = max_positions
        
        # Pre-compute frequency table for all positions up to max_positions
        # This eliminates the need for dynamic torch.arange() during forward pass
        seq = torch.arange(max_positions, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(seq, inv_freq)  # [max_positions, dim/2]
        
        # Register as buffer so it's included in state_dict and moved to correct device
        self.register_buffer("freq_table", freqs, persistent=True)

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Get rotary embeddings for given sequence length
        
        Args:
            seqlen: Maximum spatial dimension (height or width in patches)
                    At export time, this is a concrete Python int from .item()
        
        Returns:
            freqs: [seqlen, dim/2] frequency table
        """
        # Simple indexing - ONNX compatible when seqlen is concrete
        # The caller (rot_pos_emb) extracts seqlen as int via .item()
        return self.freq_table[:seqlen]


class Qwen3VLVisionPatchMerger(nn.Module):
    """
    Spatial merge + projection to text dimension
    
    Input: [432, 1024] (18×24 patches)
    Process: 
        1. Reshape to merge 2×2 spatial neighbors: [108, 4096]
        2. LayerNorm
        3. MLP: 4096 → 4096 → 2560 (text dimension!)
    Output: [108, 2560] ← Ready for text injection!
    """
    def __init__(self, config, use_postshuffle_norm=False) -> None:
        super().__init__()
        # After spatial merge: 2×2 patches = 4 * hidden_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)  # 1024 * 4 = 4096
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, 
            eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        # Project to text dimension!
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)  # 4096 → 2560

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [432, 1024] → view as [108, 4096] (due to position embedding permutation)
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        # MLP projection
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x  # [108, 2560]


def rotate_half(x):
    """Rotate half the hidden dims for rotary embeddings"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key
    
    q, k: [seq_len, num_heads, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionAttention(nn.Module):
    """
    Multi-head attention for vision transformer
    
    Input: [432, 1024]
    Process:
        1. QKV projection: [432, 1024] → [432, 3072]
        2. Split Q, K, V: each [432, 16, 64]
        3. Apply rotary embeddings
        4. Attention: [1, 16, 432, 432]
        5. Output projection: [432, 1024]
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size  # 1024
        self.num_heads = config.num_heads  # 16
        self.head_dim = self.dim // self.num_heads  # 64
        self.num_key_value_groups = 1
        
        # QKV projection in one linear layer
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)  # 1024 → 3072
        self.proj = nn.Linear(self.dim, self.dim)  # Output projection
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # [432, 1024]
        cu_seqlens: torch.Tensor,  # Cumulative sequence lengths
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]  # 432
        
        # QKV projection and split
        # [432, 1024] → [432, 3072] → [432, 3, 16, 64] → 3 × [432, 16, 64]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        
        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Reshape for attention: [seq, heads, dim] → [batch, heads, seq, dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1, 16, 432, 64]
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Attention computation (implementation varies: eager, sdpa, flash)
        # attn_weights: [1, 16, 432, 432]
        # attn_output: [1, 16, 432, 64]
        # ... (attention implementation details omitted for brevity)
        
        # Reshape and project output
        attn_output = attn_output.reshape(seq_length, -1).contiguous()  # [432, 1024]
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """
    Transformer block for vision model
    
    Architecture:
        Input [432, 1024]
        → LayerNorm → Attention → Residual
        → LayerNorm → MLP → Residual
        Output [432, 1024]
    """
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [432, 1024]
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ============================================================================
# MAIN VISION MODEL
# ============================================================================

class Qwen3VLVisionModel(nn.Module):
    """
    Complete Qwen3-VL Vision Model
    
    Forward pass:
        Input: pixel_values [432, 1536], grid_thw [1, 18, 24]
        
        1. Patch Embedding (Conv3D): [432, 1536] → [432, 1024]
        2. Position Embeddings (Interpolate + Add): [432, 1024]
        3. Rotary Embeddings: cos/sin [432, 128]
        4. 24 Transformer Blocks: [432, 1024] → [432, 1024]
           - At layers 5, 11, 17: Extract deepstack features
        5. Patch Merger: [432, 1024] → [108, 2560]
        
        Output:
            last_hidden_state: [432, 1024]
            pooler_output: [108, 2560] ← Used for text injection!
            deepstack_features: [[108, 2560], [108, 2560], [108, 2560]]
    """
    
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size  # 2
        self.patch_size = config.patch_size  # 16
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size  # 4

        # Patch embedding layer
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)

        # Learned position embeddings (48×48 = 2304)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)  # 48

        # Rotary position embeddings
        head_dim = config.hidden_size // config.num_heads  # 1024 / 16 = 64
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)  # 32

        # 24 transformer blocks
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        
        # Main patch merger (output)
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        # DeepStack: Separate mergers for layers 5, 11, 17
        self.deepstack_visual_indexes = config.deepstack_visual_indexes  # [5, 11, 17]
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings based on spatial positions
        
        For each patch, compute (row, col) coordinates considering
        the spatial merge size and grid dimensions.
        
        Output: [432, 128] (rotary embeddings for each patch)
        """
        merge_size = self.spatial_merge_size  # 2

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())  # 432
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:  # [1, 18, 24]
            merged_h, merged_w = height // merge_size, width // merge_size  # 9, 12

            # Compute full-resolution (row, col) positions for each patch
            # considering the spatial merge arrangement
            # ... (position computation details)

            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # Lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings  # [432, 128]

    def fast_pos_embed_interpolate(self, grid_thw):
        """
        Interpolate position embeddings from 48×48 grid to actual image size
        
        Uses bilinear interpolation:
            - Compute fractional indices for each patch
            - Get 4 neighboring positions (top-left, top-right, bottom-left, bottom-right)
            - Weighted average based on distance
        
        Also permutes patches to prepare for spatial merging:
            [T, H, W, hidden] → [T, H/2, 2, W/2, 2, hidden] → flatten
        
        Output: [432, 1024] (position embeddings for each patch)
        """
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        # ... (bilinear interpolation implementation)
        
        # CRITICAL: Permutation for spatial merging
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)  # Rearrange for 2×2 merge
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        
        return torch.cat(patch_pos_embeds_permute)  # [432, 1024]

    def forward(
        self, 
        hidden_states: torch.Tensor,  # [432, 1536]
        grid_thw: torch.Tensor,  # [1, 18, 24]
        **kwargs
    ) -> BaseModelOutputWithDeepstackFeatures:
        """
        Complete vision model forward pass
        """
        # 1. Patch embedding: [432, 1536] → [432, 1024]
        hidden_states = self.patch_embed(hidden_states)

        # 2. Add position embeddings (with spatial merge permutation)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)  # [432, 1024]
        hidden_states = hidden_states + pos_embeds

        # 3. Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [432, 128]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [432, 256]
        position_embeddings = (emb.cos(), emb.sin())

        # 4. Compute cu_seqlens (cumulative sequence lengths)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],  # height * width
            grid_thw[:, 0]  # temporal
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)  # [0, 432]

        # 5. Pass through 24 transformer blocks
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,  # [432, 1024]
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
            # Extract deepstack features at layers 5, 11, 17
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        # 6. Final patch merger: [432, 1024] → [108, 2560]
        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,  # [432, 1024]
            pooler_output=merged_hidden_states,  # [108, 2560] ← For text injection!
            deepstack_features=deepstack_feature_lists,  # [[108, 2560], ...]
        )


# ============================================================================
# NOTES FOR DATAFLOW DIAGRAM
# ============================================================================
"""
Complete dataflow for 400×300 image:

Input:
    pixel_values: [432, 1536]
    grid_thw: [1, 18, 24]

Step 1: Patch Embedding (Conv3D)
    [432, 1536] → reshape → [-1, 3, 2, 16, 16]
                → Conv3D  → [-1, 1024, 1, 1, 1]
                → flatten → [432, 1024]

Step 2: Position Embeddings (Bilinear Interpolation + Permutation)
    Interpolate from 48×48 grid → [432, 1024]
    Permute for spatial merge: [T, H/2, 2, W/2, 2, hidden] → flatten
    Add to hidden_states: [432, 1024]

Step 3: Rotary Embeddings
    Compute (row, col) positions → [432, 2]
    Lookup frequency table → [432, 128]
    Duplicate and get cos/sin → cos: [432, 128], sin: [432, 128]

Step 4: 24 Transformer Blocks
    Each block: [432, 1024] → LayerNorm → Attention → Residual
                              → LayerNorm → MLP → Residual
                              → [432, 1024]
    
    At layers 5, 11, 17:
        Apply deepstack merger → [108, 2560]

Step 5: Patch Merger (Spatial Merge + Project)
    [432, 1024] → view([108, 4096])  # Merge 2×2 spatial neighbors
                → LayerNorm
                → Linear(4096 → 4096) + GELU
                → Linear(4096 → 2560)  # Project to text dimension!
                → [108, 2560]

Output:
    last_hidden_state: [432, 1024]   # Raw patch features
    pooler_output: [108, 2560]       # MERGED, ready for text injection!
    deepstack_features: [[108, 2560], [108, 2560], [108, 2560]]
"""
