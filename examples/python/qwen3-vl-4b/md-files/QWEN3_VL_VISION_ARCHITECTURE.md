# Qwen3-VL Vision Model Architecture - Detailed Dataflow

## 📐 **Complete Dataflow Diagram**

```
================================================================================
                    QWEN3-VL VISION MODEL FORWARD PASS
                Example: 400×300 RGB Image → Vision Features
================================================================================

Configuration:
  - hidden_size: 1024
  - num_heads: 16
  - depth: 24 (transformer blocks)
  - patch_size: 16
  - temporal_patch_size: 2
  - spatial_merge_size: 2
  - in_channels: 3
  - intermediate_size: 4096
  - out_hidden_size: 2560 (matches text decoder)

================================================================================

INPUT
─────────────────────────────────────────────────────────────────────────
pixel_values:         [432, 1536]        (flattened patches)
image_grid_thw:       [1, 18, 24]        (temporal, height_patches, width_patches)
─────────────────────────────────────────────────────────────────────────

Explanation of input shapes:
  - Image: 400×300 → Resized to 288×384 (patch-aligned)
  - T (temporal): 1 frame
  - H (height patches): 288 ÷ 16 = 18 patches
  - W (width patches): 384 ÷ 16 = 24 patches
  - Total patches: 1 × 18 × 24 = 432
  - Channels: 3 (RGB) × 512 embedding = 1536


┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Qwen3VLVisionPatchEmbed (nn.Module)                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: nn.Conv3d(3, 1024, kernel_size=[2, 16, 16], stride=[2,16,16])│
│                                                                           │
│  Input:   pixel_values          [432, 1536]                             │
│           Reshaped to           [-1, 3, 2, 16, 16]                      │
│                                 (batch_patches, RGB, temp, h, w)         │
│                                                                           │
│  Process:                                                                 │
│    1. Reshape pixel values into 3D patches:                              │
│       [-1, 3, temporal_patch_size=2, patch_size=16, patch_size=16]      │
│    2. Apply 3D convolution to extract patch embeddings                  │
│    3. Flatten spatial dimensions                                         │
│                                                                           │
│  Output:  hidden_states         [432, 1024]                             │
│           (num_patches, hidden_size)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 2: Position Embeddings (nn.Embedding + Interpolation)           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: pos_embed = nn.Embedding(2304, 1024)                        │
│             Bilinear interpolation from grid                             │
│                                                                           │
│  Input:   hidden_states         [432, 1024]                             │
│           image_grid_thw        [1, 18, 24]                             │
│                                                                           │
│  Process:                                                                 │
│    1. fast_pos_embed_interpolate():                                      │
│       - Create spatial grid: 48×48 learned positions                    │
│       - Interpolate to match image patches (18×24)                      │
│       - Bilinear weights: (1-dh)(1-dw), (1-dh)dw, dh(1-dw), dh*dw      │
│    2. Permute for spatial merging later:                                │
│       - Reorder: [T, H/2, 2, W/2, 2, hidden] → [T*H*W, hidden]         │
│                                                                           │
│  Output:  patch_pos_embeds      [432, 1024]                             │
│           Added to hidden_states                                         │
│                                                                           │
│  Result:  hidden_states         [432, 1024]  (with position info)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: Rotary Position Embeddings                                   │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: Qwen3VLVisionRotaryEmbedding                                │
│             Creates cos/sin embeddings for attention                     │
│                                                                           │
│  Input:   grid_thw              [1, 18, 24]                             │
│                                                                           │
│  Process:                                                                 │
│    1. rot_pos_emb():                                                     │
│       - For each patch, compute (row, col) coordinates                  │
│       - Apply rotary encoding: freq_table[pos_ids]                      │
│       - Concatenate and duplicate: cat([rotary, rotary], dim=-1)        │
│    2. Extract cos/sin:                                                   │
│       - position_embeddings = (cos, sin)                                │
│                                                                           │
│  Output:  rotary_pos_emb        [432, 128]  (head_dim × 2)             │
│           position_embeddings    (cos: [432, 128], sin: [432, 128])     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Prepare cu_seqlens (Cumulative Sequence Lengths)                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: Create sequence boundary markers for attention              │
│                                                                           │
│  Process:                                                                 │
│    cu_seqlens = [0, 432]  (for single image)                           │
│    Used by attention to handle variable-length sequences                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
         ╔══════════════════════════════════════════════════╗
         ║         24 TRANSFORMER BLOCKS (depth=24)         ║
         ║   Each block: Qwen3VLVisionBlock (nn.Module)    ║
         ╚══════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┴───────────────┐
                    │   BLOCK i (i = 0 to 23)       │
                    └───────────────┬───────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4.i.1: Pre-Attention LayerNorm                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: nn.LayerNorm(1024, eps=1e-6)                                │
│                                                                           │
│  Input:   hidden_states         [432, 1024]                             │
│  Output:  normed_hidden         [432, 1024]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4.i.2: Qwen3VLVisionAttention (nn.Module)                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Sub-modules:                                                             │
│    - self.qkv = nn.Linear(1024, 3072, bias=True)  # 1024 * 3           │
│    - self.proj = nn.Linear(1024, 1024)                                  │
│                                                                           │
│  Input:   normed_hidden         [432, 1024]                             │
│           cu_seqlens            [0, 432]                                 │
│           position_embeddings   (cos: [432, 128], sin: [432, 128])      │
│                                                                           │
│  Process:                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 1: QKV Projection                                       │       │
│  │   qkv = self.qkv(normed_hidden)  [432, 3072]               │       │
│  │   Reshape: [432, 3, 16, 64]                                 │       │
│  │            (seq, qkv, num_heads, head_dim)                  │       │
│  │   Split into Q, K, V: each [432, 16, 64]                    │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 2: Apply Rotary Position Embeddings                    │       │
│  │   apply_rotary_pos_emb_vision(Q, K, cos, sin)              │       │
│  │   Q_rot, K_rot: [432, 16, 64]                               │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 3: Reshape for Attention                                │       │
│  │   Q: [1, 16, 432, 64]  (batch, heads, seq, head_dim)       │       │
│  │   K: [1, 16, 432, 64]                                        │       │
│  │   V: [1, 16, 432, 64]                                        │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 4: Attention Computation (eager or flash)              │       │
│  │   attn_weights = (Q @ K.T) * scale                          │       │
│  │                = [1, 16, 432, 432]                           │       │
│  │   attn_weights = softmax(attn_weights)                      │       │
│  │   attn_output = attn_weights @ V                            │       │
│  │               = [1, 16, 432, 64]                             │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 5: Output Projection                                    │       │
│  │   attn_output = attn_output.reshape([432, 1024])           │       │
│  │   attn_output = self.proj(attn_output)  [432, 1024]        │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Output:  attn_output           [432, 1024]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Residual Connection 1                                                   │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: hidden_states = hidden_states + attn_output                 │
│                                                                           │
│  Input:   hidden_states (before attn)  [432, 1024]                      │
│           attn_output                   [432, 1024]                      │
│  Output:  hidden_states                 [432, 1024]                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4.i.3: Pre-MLP LayerNorm                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: nn.LayerNorm(1024, eps=1e-6)                                │
│                                                                           │
│  Input:   hidden_states         [432, 1024]                             │
│  Output:  normed_hidden         [432, 1024]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4.i.4: Qwen3VLVisionMLP (nn.Module)                             │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Sub-modules:                                                             │
│    - self.linear_fc1 = nn.Linear(1024, 4096, bias=True)                │
│    - self.act_fn = GELU                                                  │
│    - self.linear_fc2 = nn.Linear(4096, 1024, bias=True)                │
│                                                                           │
│  Input:   normed_hidden         [432, 1024]                             │
│                                                                           │
│  Process:                                                                 │
│    x = self.linear_fc1(normed_hidden)    [432, 4096]                    │
│    x = self.act_fn(x)                    [432, 4096]                    │
│    x = self.linear_fc2(x)                [432, 1024]                    │
│                                                                           │
│  Output:  mlp_output            [432, 1024]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Residual Connection 2                                                   │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Operation: hidden_states = hidden_states + mlp_output                  │
│                                                                           │
│  Input:   hidden_states (after attn)   [432, 1024]                      │
│           mlp_output                    [432, 1024]                      │
│  Output:  hidden_states                 [432, 1024]                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                       │
         │ *** DEEPSTACK FEATURE EXTRACTION ***                 │
         │ At layers 5, 11, 17: Extract and save features      │
         │                                                       │
         │ If layer_num in [5, 11, 17]:                        │
         │   Apply Qwen3VLVisionPatchMerger                    │
         │   (separate merger for each deepstack layer)        │
         │                                                       │
         └───────────────────────────┬──────────────────────────┘
                                    │
                                    ↓
                    [Repeat for all 24 blocks]
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  After 24 Blocks: hidden_states                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Shape: [432, 1024]                                                      │
│  (All 432 patches, each with 1024-dim features)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 5: Qwen3VLVisionPatchMerger (nn.Module)                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Purpose: Spatial merge (2×2 patches → 1 patch) to reduce sequence len  │
│                                                                           │
│  Sub-modules:                                                             │
│    - self.norm = nn.LayerNorm(4096, eps=1e-6)                           │
│    - self.linear_fc1 = nn.Linear(4096, 4096)                            │
│    - self.act_fn = GELU                                                  │
│    - self.linear_fc2 = nn.Linear(4096, 2560)  ← Output matches text!    │
│                                                                           │
│  Input:   hidden_states         [432, 1024]                             │
│                                                                           │
│  Process:                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 1: Reshape for Spatial Merge                           │       │
│  │   Implicit reshape based on position embedding permutation  │       │
│  │   [432, 1024] → view as [T×H/2×W/2, 2×2, 1024]             │       │
│  │                         [108, 4, 1024]                       │       │
│  │   (108 merged patches, each contains 2×2=4 original patches)│       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 2: Flatten Spatial Dimension                            │       │
│  │   x = x.view(-1, 4096)     [108, 4096]                      │       │
│  │   (Each merged patch: 4 patches × 1024 = 4096 dims)         │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 3: Normalize                                            │       │
│  │   x = self.norm(x)         [108, 4096]                       │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Step 4: MLP Projection                                       │       │
│  │   x = self.linear_fc1(x)   [108, 4096]                      │       │
│  │   x = self.act_fn(x)       [108, 4096]                      │       │
│  │   x = self.linear_fc2(x)   [108, 2560]  ← Match text dim!  │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Output:  merged_hidden_states  [108, 2560]                             │
│           (pooler_output)                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT (BaseModelOutputWithDeepstackFeatures)                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                           │
│  Returns:                                                                 │
│    last_hidden_state:     [432, 1024]   (raw patch features)            │
│    pooler_output:         [108, 2560]   (merged, ready for injection!)  │
│    deepstack_features:    List[[108, 2560], [108, 2560], [108, 2560]]  │
│                           (from layers 5, 11, 17)                        │
│                                                                           │
│  *** pooler_output is what gets injected into text embeddings! ***      │
└─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           KEY OBSERVATIONS
================================================================================

1. **Spatial Merging Strategy**:
   - Position embeddings are permuted BEFORE transformer blocks
   - Permutation: [T, H/2, 2, W/2, 2, hidden] → [T*H*W, hidden]
   - This pre-arranges patches so 2×2 spatial neighbors are sequential
   - PatchMerger can then simply reshape and merge

2. **Patch Count Reduction**:
   - Input patches: 18 × 24 = 432
   - After spatial merge: 9 × 12 = 108 (4× reduction)
   - This matches the number of <|image_pad|> tokens in text sequence!

3. **Dimension Matching**:
   - Vision hidden_size: 1024
   - Text hidden_size: 2560
   - PatchMerger outputs: 2560 (matches text!)
   - This allows direct injection into text embeddings

4. **Rotary Position Embeddings**:
   - Applied per-patch based on (row, col) coordinates
   - Provides spatial awareness during attention
   - head_dim = 64, so rotary uses 32 dims (half)

5. **DeepStack Features**:
   - Extracted at layers 5, 11, 17 (early, middle, late)
   - Each processed through a separate PatchMerger
   - Used to inject visual context into early text decoder layers
   - Enables better vision-text integration

6. **Attention Pattern**:
   - Full self-attention across all 432 patches
   - cu_seqlens allows variable-length sequences (multiple images)
   - Complexity: O(432²) = ~186K attention pairs per head

7. **Memory Requirements**:
   - Hidden states: 432 × 1024 × 4 bytes = 1.77 MB
   - Attention weights: 16 heads × 432² × 4 bytes = 12 MB
   - Activations: intermediate_size creates 432 × 4096 = 7.08 MB

================================================================================
```

---

## 🔍 **Detailed Module Breakdown**

### Module 1: Patch Embedding (`Qwen3VLVisionPatchEmbed`)

**Code Reference**: Lines 130-145 in modeling_qwen3_vl.py

```python
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.in_channels = 3
        self.embed_dim = 1024
        
        kernel_size = [2, 16, 16]  # [temporal, height, width]
        self.proj = nn.Conv3d(
            3, 1024,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True
        )
    
    def forward(self, hidden_states):
        # hidden_states: [432, 1536]
        # Reshape to: [-1, 3, 2, 16, 16]
        hidden_states = hidden_states.view(
            -1, 3, 2, 16, 16
        )
        # Conv3d produces: [-1, 1024, 1, 1, 1]
        hidden_states = self.proj(hidden_states)
        # Flatten: [432, 1024]
        return hidden_states.view(-1, 1024)
```

**Purpose**: Convert RGB patches into learned embeddings

**Key Points**:
- Uses 3D convolution (not 2D) to handle temporal dimension
- Stride equals kernel size → non-overlapping patches
- Each 16×16×3 patch → 1024-dim embedding

---

### Module 2: Position Embeddings

**Code Reference**: Lines 838-856 in modeling_qwen3_vl.py

```python
# Learned position embeddings
self.pos_embed = nn.Embedding(2304, 1024)
self.num_grid_per_side = 48  # sqrt(2304)

def fast_pos_embed_interpolate(self, grid_thw):
    # For each image, interpolate position embeddings
    # from 48×48 grid to actual patch grid (e.g., 18×24)
    
    # Bilinear interpolation
    h_idxs = torch.linspace(0, 47, height_patches)
    w_idxs = torch.linspace(0, 47, width_patches)
    
    # Get integer indices and fractional parts
    h_floor, w_floor = h_idxs.int(), w_idxs.int()
    h_ceil, w_ceil = h_floor + 1, w_floor + 1
    dh, dw = h_idxs - h_floor, w_idxs - w_floor
    
    # Bilinear weights
    weights = [
        (1-dh) * (1-dw),  # top-left
        (1-dh) * dw,       # top-right
        dh * (1-dw),       # bottom-left
        dh * dw            # bottom-right
    ]
    
    # Weighted sum of 4 neighbors
    pos_embeds = sum(weights[i] * pos_embed[indices[i]] for i in range(4))
    
    # Permute for spatial merging
    pos_embeds = permute_for_merge(pos_embeds)
    
    return pos_embeds  # [432, 1024]
```

**Purpose**: Add spatial position information

**Key Points**:
- Learned embeddings for 48×48 grid (2304 positions)
- Bilinear interpolation for arbitrary image sizes
- Permutation prepares for 2×2 spatial merge

---

### Module 3: Rotary Position Embeddings

**Code Reference**: Lines 148-163 in modeling_qwen3_vl.py

```python
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim=32, theta=10000.0):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, seqlen):
        # seqlen: maximum position
        seq = torch.arange(seqlen)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs  # [seqlen, dim/2]

def rot_pos_emb(self, grid_thw):
    # For each patch, compute (row, col) position
    # Based on spatial merge size and grid dimensions
    
    for t, h, w in grid_thw:
        # Compute full-resolution positions
        row_idx = block_rows * merge_size + intra_row
        col_idx = block_cols * merge_size + intra_col
        
        # Get rotary embeddings
        embeddings = freq_table[row_idx, col_idx]
    
    # Concatenate and duplicate
    emb = torch.cat([embeddings, embeddings], dim=-1)
    return emb.cos(), emb.sin()  # [432, 128] each
```

**Purpose**: Encode spatial relationships for attention

**Key Points**:
- Creates (row, col) position IDs for each patch
- Rotary encoding: better relative position encoding
- Applied during attention Q/K computation

---

### Module 4: Transformer Block (`Qwen3VLVisionBlock`)

**Code Reference**: Lines 321-343 in modeling_qwen3_vl.py

**Structure**:
```
Input [432, 1024]
  │
  ├─ LayerNorm [432, 1024]
  │    │
  │    └─ Attention [432, 1024]
  │         - QKV projection: [432, 1024] → [432, 3072]
  │         - Split Q, K, V: each [432, 16, 64]
  │         - Apply rotary: Q_rot, K_rot [432, 16, 64]
  │         - Attention: [1, 16, 432, 432]
  │         - Output proj: [432, 1024]
  │
  ├─ Residual: hidden + attn_out
  │
  ├─ LayerNorm [432, 1024]
  │    │
  │    └─ MLP [432, 1024]
  │         - FC1: [432, 1024] → [432, 4096]
  │         - GELU
  │         - FC2: [432, 4096] → [432, 1024]
  │
  └─ Residual: hidden + mlp_out
  │
Output [432, 1024]
```

---

### Module 5: Patch Merger (`Qwen3VLVisionPatchMerger`)

**Code Reference**: Lines 203-219 in modeling_qwen3_vl.py

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config):
        # After merge: 2x2 patches = 4 * 1024 = 4096
        self.hidden_size = 1024 * 4  # 4096
        self.norm = nn.LayerNorm(4096, eps=1e-6)
        self.linear_fc1 = nn.Linear(4096, 4096)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(4096, 2560)  # Match text!
    
    def forward(self, x):
        # x: [432, 1024]
        # Due to position embedding permutation,
        # patches are arranged: [108 groups of 4, 1024]
        
        # Reshape: [108, 4*1024] = [108, 4096]
        x = x.view(-1, 4096)
        
        # Normalize
        x = self.norm(x)
        
        # MLP: 4096 → 4096 → 2560
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        
        return x  # [108, 2560]
```

**Purpose**: Reduce spatial resolution and match text dimension

**Key Points**:
- Merges 2×2 spatial neighbors into single token
- 432 patches → 108 merged patches (4× reduction)
- Projects to 2560 dims (text decoder dimension)
- Output directly replaces `<|image_pad|>` tokens!

---

## 📏 **Shape Transformations Summary**

| Stage | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| **Input** | Raw pixels | [432, 1536] | - |
| **Patch Embed** | Conv3D | [432, 1536] | [432, 1024] |
| **Pos Embed** | Interpolate + Add | [432, 1024] | [432, 1024] |
| **Rotary** | Compute cos/sin | grid_thw | cos/sin: [432, 128] |
| **24 Blocks** | Attention + MLP | [432, 1024] | [432, 1024] |
| **Patch Merge** | Spatial reduce + project | [432, 1024] | [108, 2560] |
| **Output** | Return dict | - | pooler: [108, 2560] |

---

## 🔗 **Vision → Text Connection**

```
Vision Output:        [108, 2560]  ← pooler_output
                          ↓
Text Token Sequence:  [1, 2, 3, <img>, <img>, ..., <img>, 4, 5]
                                  └──── 108 tokens ─────┘
                          ↓
Text Embeddings:      [1, seq_len, 2560]
                          ↓
Vision Injection:     Replace <img> embeddings with vision features
                          ↓
Merged Embeddings:    [1, seq_len, 2560]  ← Has vision!
```

---

## 📚 **References**

- **Source Code**: [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Vision Model**: Lines 748-924 (Qwen3VLVisionModel)
- **Patch Embed**: Lines 130-145
- **Attention**: Lines 254-319
- **Patch Merger**: Lines 203-219

---

## 💡 **Key Takeaways**

1. **Two-stage spatial reduction**:
   - Position embedding permutation (preparation)
   - Patch merger (actual merge: 2×2 → 1)

2. **Dimension matching is critical**:
   - Vision: 1024 → Merge: 4096 → Project: 2560
   - Text expects 2560-dim embeddings

3. **DeepStack features**:
   - Multi-scale visual features from layers 5, 11, 17
   - Injected into early text decoder layers
   - Enhances vision-text integration

4. **Efficient attention**:
   - Full self-attention across all patches
   - Rotary embeddings for spatial awareness
   - cu_seqlens for variable-length sequences

5. **Output flexibility**:
   - `last_hidden_state`: Raw patch features [432, 1024]
   - `pooler_output`: Merged features [108, 2560] ← **Used for injection!**
   - `deepstack_features`: Multi-scale features for text decoder
