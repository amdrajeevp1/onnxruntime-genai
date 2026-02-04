# Image Preprocessing Pipeline: 400×300 → [432, 1536]

## 🎯 **Question: How is a 400×300 image converted to [432, 1536]?**

---

## 📊 **Complete Preprocessing Pipeline**

```
Original Image (400×300 RGB)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Resize to Patch-Aligned Dimensions                      │
│ ───────────────────────────────────────────────────────────────│
│                                                                  │
│ Goal: Make dimensions divisible by patch_size (16)              │
│                                                                  │
│ Calculation:                                                     │
│   - Target grid: 18×24 patches                                  │
│   - Height: 18 patches × 16 pixels/patch = 288 pixels          │
│   - Width:  24 patches × 16 pixels/patch = 384 pixels          │
│                                                                  │
│ Input:  400×300×3                                               │
│ Output: 288×384×3                                               │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Add Temporal Dimension                                  │
│ ───────────────────────────────────────────────────────────────│
│                                                                  │
│ Qwen3-VL uses temporal_patch_size=2 for video support          │
│ For static images, duplicate the frame                          │
│                                                                  │
│ Input:  [1, 3, 288, 384]    (T, C, H, W)                       │
│ Output: [2, 3, 288, 384]    (duplicate frame for temporal=2)   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Create 3D Patches                                       │
│ ───────────────────────────────────────────────────────────────│
│                                                                  │
│ Split into patches of size:                                     │
│   temporal_patch_size × patch_size × patch_size                │
│   = 2 × 16 × 16                                                 │
│                                                                  │
│ Number of patches:                                              │
│   T: 2 ÷ 2 = 1 temporal patch                                  │
│   H: 288 ÷ 16 = 18 height patches                              │
│   W: 384 ÷ 16 = 24 width patches                               │
│   Total: 1 × 18 × 24 = 432 patches                             │
│                                                                  │
│ Each patch contains:                                            │
│   2 (temporal) × 16 (height) × 16 (width) × 3 (RGB)            │
│   = 1536 values                                                 │
│                                                                  │
│ Input:  [2, 3, 288, 384]                                        │
│ Output: [432, 2, 16, 16, 3]  (conceptual)                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Flatten Each Patch                                      │
│ ───────────────────────────────────────────────────────────────│
│                                                                  │
│ Flatten the last 4 dimensions: [2, 16, 16, 3] → [1536]         │
│                                                                  │
│ Input:  [432, 2, 16, 16, 3]                                     │
│ Output: [432, 1536]  ← This is the input to PatchEmbed!        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔢 **Detailed Calculation**

### Original Image
```
Shape: 400 × 300 × 3 (RGB)
Pixels: 400 × 300 = 120,000 pixels
Total values: 120,000 × 3 = 360,000 values
```

### After Resize (Step 1)
```
Shape: 288 × 384 × 3
Pixels: 288 × 384 = 110,592 pixels
Total values: 110,592 × 3 = 331,776 values

Why these dimensions?
  - Must be divisible by patch_size (16)
  - Aspect ratio preserved approximately: 384/288 = 1.33, 400/300 = 1.33 ✓
```

### After Temporal Duplication (Step 2)
```
Shape: 2 × 3 × 288 × 384
Total values: 2 × 331,776 = 663,552 values
```

### After Patching (Step 3)
```
Number of patches:
  Temporal: 2 ÷ temporal_patch_size(2) = 1
  Height:   288 ÷ patch_size(16) = 18
  Width:    384 ÷ patch_size(16) = 24
  Total:    1 × 18 × 24 = 432 patches

Each patch size:
  2 × 16 × 16 × 3 = 1536 values

Shape: [432, 2, 16, 16, 3]
```

### After Flattening (Step 4)
```
Shape: [432, 1536]
Total values: 432 × 1536 = 663,552 ✓ (matches step 2!)

This is the input to the vision model!
```

---

## 🎨 **Visual Representation**

### Image to Grid
```
Original Image (400×300)
┌────────────────────────────────┐
│                                │
│         400 pixels             │
│                                │
│  300 pixels                    │
│                                │
└────────────────────────────────┘

           ↓ Resize

Patch-Aligned Image (288×384)
┌────────────────────────────────────────┐
│  [16] [16] [16] ... [16]  (24 patches) │
│  [16] [16] [16] ... [16]               │
│   ...                                   │  18 patches
│  [16] [16] [16] ... [16]               │
└────────────────────────────────────────┘

           ↓ Add temporal + Patch

432 Patches (18×24×1)
Each patch: [2, 16, 16, 3] = 1536 values
```

### Patch Structure
```
One 3D Patch:
┌─────────────────┐
│  Frame 1        │  16×16×3 = 768 values
│  ┌───────────┐  │
│  │ 16×16 RGB │  │
│  └───────────┘  │
├─────────────────┤
│  Frame 2        │  16×16×3 = 768 values
│  ┌───────────┐  │
│  │ 16×16 RGB │  │
│  └───────────┘  │
└─────────────────┘
Total: 2×768 = 1536 values per patch
```

---

## 💻 **Code Implementation**

### Processor Code (Conceptual)
```python
from transformers import AutoProcessor
from PIL import Image

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# Load image
image = Image.open("test_image.jpg")  # 400×300

# Process image
inputs = processor(
    images=[image],
    return_tensors="pt"
)

# Check shapes
print(f"pixel_values: {inputs['pixel_values'].shape}")
# Output: torch.Size([432, 1536])

print(f"image_grid_thw: {inputs['image_grid_thw']}")
# Output: tensor([[1, 18, 24]])
```

### What the Processor Does
```python
def process_image(image):
    # Step 1: Resize to patch-aligned dimensions
    # Target: make H and W divisible by 16
    image = smart_resize(image)  # 400×300 → 288×384
    
    # Step 2: Convert to tensor and add temporal dimension
    # [C, H, W] → [T, C, H, W]
    image = to_tensor(image)  # [3, 288, 384]
    image = image.unsqueeze(0)  # [1, 3, 288, 384]
    image = image.repeat(2, 1, 1, 1)  # [2, 3, 288, 384] (duplicate for temporal=2)
    
    # Step 3: Create 3D patches
    # [T, C, H, W] → [num_patches, T_patch, H_patch, W_patch, C]
    patches = create_3d_patches(
        image,
        temporal_patch_size=2,  # 2 frames → 1 temporal patch
        patch_size=16           # 16×16 spatial patches
    )  # [432, 2, 16, 16, 3]
    
    # Step 4: Flatten each patch
    # [432, 2, 16, 16, 3] → [432, 1536]
    flattened = patches.reshape(432, -1)
    
    return flattened, grid_thw=[1, 18, 24]
```

---

## 🔍 **Why These Numbers?**

### Why 432 patches?
```
Grid: 18 × 24 = 432 spatial locations
Temporal: 1 (after grouping 2 frames into 1 temporal patch)
Total: 432 patches
```

### Why 1536 dimensions per patch?
```
temporal_patch_size × patch_size × patch_size × channels
= 2 × 16 × 16 × 3
= 1536
```

### Why temporal_patch_size = 2?
```
Qwen3-VL is designed for both images AND videos
- For videos: groups 2 consecutive frames
- For images: duplicates the single frame to create 2 frames
- Provides temporal consistency in the architecture
```

### Why grid_thw = [1, 18, 24]?
```
T (temporal): 1 temporal patch (after grouping 2 frames)
H (height):   18 spatial patches (288 ÷ 16)
W (width):    24 spatial patches (384 ÷ 16)
```

---

## 🔗 **Connection to Vision Model**

After preprocessing, the vision model receives:

```
Input to Qwen3VLVisionModel.forward():
  pixel_values:  [432, 1536]
  grid_thw:      [[1, 18, 24]]

Step in Vision Model:
  1. PatchEmbed (Conv3D):
     Input:  [432, 1536]
     Reshape to: [-1, 3, 2, 16, 16]  (restore 3D patch structure)
     Conv3D: kernel=[2, 16, 16], stride=[2, 16, 16]
     Output: [432, 1024]  (embedded patches)
```

---

## 📊 **Shape Summary Table**

| Stage | Shape | Description |
|-------|-------|-------------|
| **Original** | `[400, 300, 3]` | Input image (H×W×C) |
| **Resize** | `[288, 384, 3]` | Patch-aligned (divisible by 16) |
| **Add Temporal** | `[2, 3, 288, 384]` | Duplicate frame (T×C×H×W) |
| **Patch** | `[432, 2, 16, 16, 3]` | 3D patches |
| **Flatten** | `[432, 1536]` | **Input to vision model** |
| **PatchEmbed** | `[432, 1024]` | Embedded patches |
| **After 24 Blocks** | `[432, 1024]` | Transformer output |
| **PatchMerger** | `[108, 2560]` | Spatial merge + project to text dim |

---

## 🎯 **Key Takeaways**

1. **400×300 is resized to 288×384** to be divisible by patch_size (16)
2. **Temporal dimension added**: Single frame duplicated → 2 frames
3. **432 patches created**: 18×24 spatial grid, 1 temporal group
4. **1536 values per patch**: 2 (temporal) × 16 × 16 (spatial) × 3 (RGB)
5. **Final input shape**: [432, 1536] ready for vision model

The key insight is that Qwen3-VL treats **even static images as 2-frame sequences** to maintain consistency with its video processing architecture!
