# Qwen3-VL Vision Model - Quick Reference

## 📊 **Simplified Dataflow (Example: 400×300 Image)**

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                         │
│  pixel_values: [432, 1536]  (flattened 18×24 patches)          │
│  grid_thw:     [1, 18, 24]  (1 frame, 18×24 patches)           │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 1: PatchEmbed (Conv3D)                                  │
│  ───────────────────────────────────────────────────────────── │
│  nn.Conv3d(3 → 1024, kernel=[2,16,16], stride=[2,16,16])      │
│                                                                  │
│  [432, 1536] → reshape → [-1, 3, 2, 16, 16]                    │
│              → conv3d  → [-1, 1024, 1, 1, 1]                    │
│              → flatten → [432, 1024]                            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 2: Position Embeddings                                  │
│  ───────────────────────────────────────────────────────────── │
│  Bilinear interpolation from 48×48 learned grid                │
│  Permute for spatial merging                                    │
│                                                                  │
│  [432, 1024] + pos_embeds → [432, 1024]                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 3: Rotary Position Embeddings                           │
│  ───────────────────────────────────────────────────────────── │
│  Compute (cos, sin) for each patch position                    │
│                                                                  │
│  grid_thw → compute (row,col) → [432, 128] cos, sin            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
         ╔═════════════════════════════════╗
         ║   24 Transformer Blocks         ║
         ║   Each block:                   ║
         ║   ┌──────────────────────────┐  ║
         ║   │ LayerNorm                │  ║
         ║   │ Attention (with rotary)  │  ║
         ║   │ Residual                 │  ║
         ║   │ LayerNorm                │  ║
         ║   │ MLP (1024→4096→1024)     │  ║
         ║   │ Residual                 │  ║
         ║   └──────────────────────────┘  ║
         ║                                  ║
         ║   [432, 1024] → [432, 1024]     ║
         ║                                  ║
         ║   DeepStack: Save features at   ║
         ║   layers 5, 11, 17              ║
         ╚═════════════════════════════════╝
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 5: PatchMerger                                          │
│  ───────────────────────────────────────────────────────────── │
│  Spatial merge (2×2 → 1) + Project to text dimension           │
│                                                                  │
│  [432, 1024] → view([108, 4096])  (merge 2×2 patches)          │
│              → LayerNorm                                        │
│              → Linear(4096 → 4096) + GELU                       │
│              → Linear(4096 → 2560)                              │
│              → [108, 2560]                                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                        │
│  last_hidden_state:  [432, 1024]   (raw patches)               │
│  pooler_output:      [108, 2560]   (MERGED, ready for text!)   │
│  deepstack_features: [[108,2560], [108,2560], [108,2560]]      │
│                       (from layers 5, 11, 17)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔢 **Shape Transformations at a Glance**

| Module | Input Shape | Output Shape | Key Operation |
|--------|-------------|--------------|---------------|
| PatchEmbed | [432, 1536] | [432, 1024] | Conv3D projection |
| Pos Embed | [432, 1024] | [432, 1024] | Add interpolated positions |
| Rotary | grid_thw | cos/sin [432, 128] | Compute rotation matrices |
| 24 Blocks | [432, 1024] | [432, 1024] | Attention + MLP |
| PatchMerger | [432, 1024] | [108, 2560] | **2×2 merge + project** |

---

## 🎯 **Critical Points**

### 1. **Patch Count Math**
```
Image: 400×300 → Resized to 288×384
Patches: 288÷16 × 384÷16 = 18×24 = 432 patches
After merge: 9×12 = 108 patches (4× reduction)
```

### 2. **Dimension Matching**
```
Vision hidden:  1024
Text hidden:    2560
Merger output:  2560  ← Matches text!
```

### 3. **Attention Details**
```
QKV Linear:  [432, 1024] → [432, 3072]
Split Q/K/V: [432, 3072] → 3 × [432, 16, 64]
Attn output: [1, 16, 432, 64] → [432, 1024]
```

### 4. **Spatial Merging Strategy**
```
Position embedding permutation PREPARES for merge:
  [T, H, W, hidden] → [T, H/2, 2, W/2, 2, hidden]
  
PatchMerger EXECUTES the merge:
  [432, 1024] → view([108, 4*1024]) → [108, 4096]
              → project → [108, 2560]
```

---

## 🔗 **Connection to Text Model**

```
Vision Model Output:
  pooler_output: [108, 2560]
         ↓
Text Tokenizer creates:
  [1, 2, 3, <img>, <img>, ..., <img>, 4, 5, 6]
            └──── 108 tokens ─────┘
         ↓
Embedding Layer:
  [batch, seq_len, 2560]
  All tokens get embeddings, including <img> tokens
         ↓
Vision Injection (masked_scatter):
  Replace <img> token embeddings with vision features
  inputs_embeds[mask] = pooler_output  # [108, 2560]
         ↓
Merged Embeddings:
  [batch, seq_len, 2560]
  Now has REAL vision features instead of generic <img> embeddings!
         ↓
Text Decoder:
  Processes merged embeddings with full vision context
```

---

## 📐 **Memory and Computation**

### Per-Image Memory (400×300)
- **Hidden states**: 432 × 1024 × 4 bytes = **1.77 MB**
- **Attention weights**: 16 heads × 432² × 4 bytes = **12 MB**
- **Intermediate activations**: 432 × 4096 × 4 bytes = **7.08 MB**
- **Total per layer**: ~20 MB
- **Total for 24 layers**: ~480 MB

### Computation Complexity
- **Attention**: O(432²) = 186,624 operations per head
- **16 heads**: ~3M attention operations per layer
- **24 layers**: ~72M attention operations total
- **MLP**: 432 × (1024 × 4096 + 4096 × 1024) = ~7B FLOPs per layer

---

## 🎨 **Visual Representation**

```
Image (400×300)
    │
    ├─ Resize to patch-aligned (288×384)
    │
    ├─ Split into patches (18×24 = 432)
    │
    ├─ Conv3D → Embeddings [432, 1024]
    │
    ├─ + Position Embeddings
    │
    ├─ Transformer × 24 layers
    │   │
    │   ├─ Self-Attention (with rotary)
    │   ├─ MLP (1024→4096→1024)
    │   │
    │   └─ @ layers 5,11,17: Save DeepStack features
    │
    ├─ Spatial Merge (2×2 → 1)
    │   │
    │   ├─ 432 patches → 108 merged patches
    │   └─ 4×1024 → 2560 (project to text dim)
    │
    └─ Output [108, 2560] → Ready for text injection!
```

---

## 🔍 **Module-by-Module Details**

### Module 1: PatchEmbed
- **Type**: `nn.Conv3d`
- **Params**: kernel=[2,16,16], stride=[2,16,16], bias=True
- **Purpose**: Extract patch features
- **Input**: Flattened patches [432, 1536]
- **Output**: Embeddings [432, 1024]

### Module 2: Position Embeddings
- **Type**: `nn.Embedding` + interpolation
- **Params**: 2304 learned positions (48×48 grid)
- **Purpose**: Add spatial location info
- **Method**: Bilinear interpolation to image size

### Module 3: Rotary Embeddings
- **Type**: Custom computation
- **Params**: freq_table based on (row, col)
- **Purpose**: Relative position encoding for attention
- **Output**: cos/sin [432, 128] each

### Module 4: Transformer Block (×24)
- **Components**:
  - LayerNorm [1024]
  - Attention [1024, num_heads=16]
  - LayerNorm [1024]
  - MLP [1024 → 4096 → 1024]

### Module 5: PatchMerger
- **Type**: LayerNorm + 2× Linear
- **Params**: 
  - norm [4096]
  - fc1 [4096 → 4096]
  - fc2 [4096 → 2560]
- **Purpose**: Reduce patches & match text dim
- **Input**: [432, 1024]
- **Output**: [108, 2560]

---

## 📚 **Quick Reference Card**

```
┌────────────────────────────────────────────────────────┐
│ QWEN3-VL VISION MODEL CHEAT SHEET                     │
├────────────────────────────────────────────────────────┤
│ Config:                                                │
│   hidden_size: 1024                                    │
│   num_heads: 16                                        │
│   depth: 24                                            │
│   patch_size: 16                                       │
│   spatial_merge: 2                                     │
│   out_hidden: 2560                                     │
│                                                         │
│ Shapes (400×300 image):                                │
│   Input:        [432, 1536]                            │
│   Patches:      [432, 1024]                            │
│   After blocks: [432, 1024]                            │
│   After merge:  [108, 2560] ← Ready for text!         │
│                                                         │
│ Key Operations:                                        │
│   1. Conv3D patch embedding                            │
│   2. Position interpolation                            │
│   3. 24× Transformer (Attn + MLP)                      │
│   4. 2×2 spatial merge                                 │
│   5. Project to text dimension (2560)                  │
│                                                         │
│ Output:                                                │
│   pooler_output: [108, 2560]                           │
│   → Replaces 108 <|image_pad|> tokens in text!        │
└────────────────────────────────────────────────────────┘
```
