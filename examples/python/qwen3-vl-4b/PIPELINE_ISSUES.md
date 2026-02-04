# Qwen3-VL ONNX Pipeline - Issues and Status

## Current Status

### ✅ WORKING
1. **Text-only inference** - PASSED
   - Embeddings ONNX model works correctly
   - Text decoder ONNX model works correctly  
   - Int64 dtype fix applied
   - Generated output (single token greedy decoding)

### ❌ IN PROGRESS
2. **Image + text inference** - FAILED
   - Image preprocessing works (using HuggingFace processor)
   - Vision encoder ONNX model fails during inference

---

## Issues Found and Fixed

### Issue 1: Data Type Mismatch ✅ FIXED
**Error:** `Unexpected input data type. Actual: (tensor(int32)) , expected: (tensor(int64))`

**Fix Applied:**
```python
# Ensure int64 for ONNX compatibility
input_ids = inputs["input_ids"][0].astype(np.int64)
```

**Status:** ✅ RESOLVED

---

### Issue 2: Image Preprocessing ✅ FIXED (Partially)
**Error:** `Only returning PyTorch tensors is currently supported`

**Fix Applied:**
- Use HuggingFace's `Qwen3VLImageProcessor` for preprocessing
- Convert PyTorch tensors to NumPy for ONNX inference

```python
# Get PyTorch tensors from HF processor
inputs = self.image_processor(images=images, return_tensors="pt")

# Convert to NumPy for ONNX
pixel_values = inputs["pixel_values"].cpu().numpy()
grid_thw = inputs["image_grid_thw"].cpu().numpy()
```

**Status:** ✅ RESOLVED

---

### Issue 3: Vision Encoder Broadcasting Error ❌ ACTIVE
**Error:**
```
[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while 
running Mul node. Name:'/vision_model/blocks.0/attn/Mul_3' 
Status Message: Attempting to broadcast an axis by a dimension other than 1. 64 by 144
```

**Details:**
- Image preprocessing produces: `grid_thw: [[1, 16, 16]]` (256 patches)
- Vision encoder was exported with: `grid_thw: [[1, 24, 24]]` (576 patches)
- The vision encoder has hardcoded shape expectations from export time

**Root Cause:**
The vision encoder ONNX model was exported with specific dimensions during tracing:
- Export used: 24×24 patches (384×384 pixel image)
- Runtime used: 16×16 patches (224×224 pixel image)

**Problem:**
ONNX traced models can have shape constraints that don't generalize to different input sizes. The vision encoder's attention mechanism has hardcoded assumptions about patch dimensions.

---

## Possible Solutions for Issue 3

### Option A: Re-export Vision Encoder with Dynamic Shapes ⭐ RECOMMENDED
Re-export the vision encoder with proper dynamic axis specifications:
```python
dynamic_axes={
    "pixel_values": {0: "num_patches"},  # Allow variable number of patches
    "image_grid_thw": {0: "num_images", 1: "thw_dim"},
    "pooled_embeds": {0: "num_merged_patches"}
}
```

**Pros:**
- Supports any image size
- Proper solution
- Once fixed, works forever

**Cons:**
- Requires re-export
- May have additional tracing issues

---

### Option B: Use Fixed Input Size
Force all images to resize to the export dimensions (384×384 → 24×24 patches):
```python
# In image processor config
image_processor = processor.image_processor
# Override min_pixels/max_pixels to force 384×384
```

**Pros:**
- Works immediately with current ONNX model
- Simple fix

**Cons:**
- Less flexible
- All images must be same size
- May affect accuracy for different aspect ratios

---

### Option C: Export Multiple Vision Encoders
Export separate vision encoders for common sizes:
- vision_encoder_16x16.onnx (256 patches)
- vision_encoder_24x24.onnx (576 patches)
- vision_encoder_32x32.onnx (1024 patches)

Select at runtime based on image size.

**Pros:**
- Flexible for different sizes
- Guaranteed to work

**Cons:**
- Multiple model files
- More complex logic
- Storage overhead

---

## Detailed Error Analysis

### Vision Encoder Export Specifications

**What we used during export:**
```python
# From builder_qwen3vl.py
grid_t, grid_h, grid_w = 1, 24, 24  # 384x384 image → 24x24 patches
num_patches = grid_t * grid_h * grid_w  # Must match!
patch_features = 3 * 2 * 16 * 16  # RGB * temporal_patch * spatial_patch^2
```

**What runtime is providing:**
```
pixel_values: (256, 1536)  # 16×16 = 256 patches
grid_thw: [[1, 16, 16]]     # 224×224 pixel image
```

**The mismatch:**
- Export: 576 patches (24×24)
- Runtime: 256 patches (16×16)

This causes internal broadcasting errors in the vision encoder's attention layers because the positional embeddings and attention masks were traced with 24×24 expectations.

---

## Recommended Action Plan

### Phase 1: Quick Fix (Option B)
1. Force image resize to 384×384 in the processor
2. Test end-to-end pipeline
3. Verify generation works

### Phase 2: Proper Fix (Option A)
1. Modify `builder_qwen3vl.py` to properly handle dynamic shapes
2. Re-export vision encoder with dynamic axes
3. Test with various image sizes
4. Document supported size ranges

### Phase 3: Optimization
1. Export vision encoder for multiple common sizes
2. Implement smart model selection
3. Profile performance across sizes

---

## Current Pipeline Architecture

```
┌─────────────────────────────────────────────────┐
│  HuggingFace Preprocessing (PyTorch)            │
│  ├── Image Processor (smart_resize, patches)   │
│  └── Tokenizer (text → input_ids)              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  ONNX Inference (onnxruntime)                   │
│  ├── vision_encoder.onnx  ❌ (shape mismatch)   │
│  ├── embeddings.onnx      ✅ (working)          │
│  └── model.onnx           ✅ (working)          │
└─────────────────────────────────────────────────┘
```

---

## Next Steps

**Immediate:**
1. Decide on solution approach (A, B, or C)
2. Implement chosen fix
3. Test with real images

**Short-term:**
1. Add autoregressive generation loop
2. Implement beam search / sampling
3. Add streaming output

**Long-term:**
1. Quantize models (INT4, FP16)
2. Optimize for CUDA/DirectML
3. Create demo app

---

## Test Results Summary

| Test | Status | Details |
|------|--------|---------|
| Text-only inference | ✅ PASS | Generated single token |
| Image preprocessing | ✅ PASS | HF processor works |
| Vision encoder | ❌ FAIL | Shape/broadcast mismatch |
| Full multimodal | ❌ BLOCKED | Waiting for vision fix |

---

**Status:** Waiting for decision on Issue 3 solution approach.
