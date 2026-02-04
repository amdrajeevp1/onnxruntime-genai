# Qwen3-VL ONNX Export Setup - COMPLETE ✓

## What Has Been Created

I've set up a complete pipeline for exporting Qwen3-VL-4B to ONNX format, following the Phi4-MM reference implementation.

### Files Created

#### 1. Documentation
- **README.md** - Quick start guide
- **SETUP_GUIDE.md** - Detailed step-by-step instructions
- **IMPLEMENTATION_REFERENCE.md** - Technical reference comparing with Phi4-MM
- **SETUP_COMPLETE.md** - This file

#### 2. Setup Scripts
- **copy_hf_files.py** - Downloads HuggingFace modeling files locally
- **modify_rotary_embedding.py** - Modifies rotary embedding for ONNX export
- **setup_qwen3vl.py** - Master script that runs everything

#### 3. Export Scripts
- **builder_qwen3vl.py** - Exports vision, embeddings, and text models to ONNX

#### 4. Inference Scripts
- **test_qwen3vl_inference.py** - Complete inference pipeline for testing

## Quick Start (3 Steps)

### Step 1: Run Master Setup Script

```powershell
cd C:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b
python setup_qwen3vl.py
```

This will:
1. Download HuggingFace model files to `pytorch_modified/`
2. Modify rotary embedding for ONNX compatibility
3. Copy files to `pytorch/` directory
4. Export ONNX models (vision + embeddings + text)

### Step 2: Choose Export Target

When prompted, select:
- **Option 1:** FP32 for CPU (testing)
- **Option 2:** FP16 for CUDA (GPU inference)
- **Option 3:** FP16 for DirectML (Windows GPU)

### Step 3: Test Inference

```powershell
# Test with an image
python test_qwen3vl_inference.py ^
  --model_path ./cpu ^
  --image_path ./test_images/sample.jpg ^
  --prompt "Describe this image"
```

## Manual Setup (If Automated Fails)

### Option A: Step-by-Step

```powershell
# Step 1: Download HF files
python copy_hf_files.py

# Step 2: Modify rotary embedding
python modify_rotary_embedding.py

# Step 3: Copy modified files
cp pytorch_modified\*.py pytorch\

# Step 4: Export ONNX
python builder_qwen3vl.py ^
  --input ./pytorch ^
  --output ./cpu ^
  --precision fp32 ^
  --execution_provider cpu
```

### Option B: Use Existing Builder (Text Only)

```powershell
cd C:\Users\rajeevp\Documents\onnxruntime-genai-1

python -m src.python.py.models.builder ^
  --input examples\python\qwen3-vl-4b\pytorch ^
  --output examples\python\qwen3-vl-4b\cpu-text ^
  --precision fp32 ^
  --execution_provider cpu ^
  --extra_options exclude_embeds=true
```

This exports **only** the text decoder (no vision or embeddings).

## Key Modifications Explained

### 1. Rotary Embedding Fix

**Problem:** HuggingFace implementation uses dynamic decisions that prevent ONNX export.

**Solution:** Modified `Qwen3VLTextRotaryEmbedding.forward()` to:
- Remove `@dynamic_rope_update` decorator
- Assume `position_ids` is always 3D: `[3, batch_size, seq_len]`
- Remove conditional expansion logic

**File:** `pytorch_modified/modular_qwen3_vl.py` (lines 269-286)

### 2. Component-Based Export

Following Phi4-MM approach, we export 3 separate ONNX models:

1. **vision_encoder.onnx**
   - Input: `pixel_values` [num_patches, features]
   - Input: `image_grid_thw` [num_images, 3]
   - Output: `image_embeds` [num_patches, 2560]

2. **embeddings.onnx**
   - Input: `input_ids` [batch, seq_len]
   - Output: `inputs_embeds` [batch, seq_len, 2560]

3. **model.onnx** (text decoder)
   - Input: `inputs_embeds` [batch, seq_len, 2560]
   - Input: `position_ids` [3, batch, seq_len]
   - Output: `logits` [batch, seq_len, vocab_size]

### 3. Inference Pipeline

The inference pipeline:
1. Preprocesses image with `Qwen3VLImageProcessor`
2. Runs vision encoder to get `image_embeds`
3. Tokenizes text prompt
4. Runs embeddings to get `text_embeds`
5. Merges `image_embeds` and `text_embeds`
6. Generates text with decoder

## Architecture Summary

### Qwen3-VL Components

```
┌─────────────────┐
│  Image (RGB)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Encoder  │  ← 24 layers, 1024 hidden
│ - 3D Conv Patch │  ← Conv3D(3→1024, kernel=2×16×16)
│ - 2D RoPE       │  ← 2D rotary position embeddings
│ - Spatial Merge │  ← 2×2→1 merge + project to 2560
└────────┬────────┘
         │
         ▼
    image_embeds [N, 2560]
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  Text Tokens    │   │  Merge Logic    │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ▼                     │
┌─────────────────┐            │
│  Embeddings     │            │
│  [B, S, 2560]   │            │
└────────┬────────┘            │
         │                     │
         └─────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Merged Embeds  │
         │  [B, S', 2560]  │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Text Decoder   │  ← 28 layers, 2560 hidden
         │  - 3D MRoPE     │  ← Multi-axis RoPE
         │  - GQA          │  ← Grouped Query Attention
         └────────┬────────┘
                  │
                  ▼
              logits [B, S', vocab]
```

### Key Differences from Phi4-MM

| Feature | Phi4-MM | Qwen3-VL |
|---------|---------|----------|
| Modalities | Text + Image + Audio | Text + Image |
| Text Layers | 32 | 28 |
| Hidden Size | 3072 | 2560 |
| Vision Layers | 27 | 24 |
| Vision Hidden | 1152 | 1024 |
| RoPE Type | 1D | **3D MRoPE** |
| Position IDs | [B, S] | **[3, B, S]** |
| Patch Embed | 2D Conv | **3D Conv** |

## Directory Structure After Setup

```
qwen3-vl-4b/
├── README.md                      ← Start here
├── SETUP_GUIDE.md                 ← Detailed guide
├── SETUP_COMPLETE.md              ← This file
├── IMPLEMENTATION_REFERENCE.md    ← Technical details
│
├── setup_qwen3vl.py              ← Run this for automated setup
├── copy_hf_files.py              ← Downloads HF files
├── modify_rotary_embedding.py    ← Modifies for ONNX
├── builder_qwen3vl.py            ← ONNX export
├── test_qwen3vl_inference.py     ← Inference test
│
├── pytorch/                       ← Original model (already exists)
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── tokenizer files
│
├── pytorch_modified/              ← Modified files (created by setup)
│   ├── modeling_qwen3_vl.py      ← Modified
│   ├── modular_qwen3_vl.py       ← Modified (key file!)
│   ├── processing_qwen3_vl.py    ← Copied
│   ├── configuration_qwen3_vl.py ← Copied
│   └── video_processing_qwen3_vl.py ← Copied
│
├── cpu/                           ← ONNX models (created by export)
│   ├── vision_encoder.onnx       ← Vision model
│   ├── embeddings.onnx           ← Embedding layer
│   ├── model.onnx                ← Text decoder
│   ├── vision_processor.json     ← Image preprocessing
│   ├── genai_config.json         ← ONNX Runtime config
│   └── tokenizer files           ← Copied from pytorch/
│
└── md-files/                      ← Your experiment docs (keep)
```

## Troubleshooting

### Issue 1: "transformers module not found"

```powershell
pip install --upgrade transformers
# or
pip install git+https://github.com/huggingface/transformers.git
```

### Issue 2: "dynamic_rope_update not defined"

This means the modification didn't work. Check:
1. `pytorch_modified/modular_qwen3_vl.py` was modified
2. Files were copied to `pytorch/` directory
3. The decorator was removed from `Qwen3VLTextRotaryEmbedding.forward()`

### Issue 3: "ONNX export fails"

Common causes:
- Dynamic shapes in vision encoder → Use representative input sizes
- Position IDs not 3D → Ensure shape is [3, batch, seq_len]
- Missing modified files → Re-run `modify_rotary_embedding.py`

### Issue 4: "Model not found"

Make sure you're in the correct directory:
```powershell
cd C:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b
```

## Next Steps

### 1. Test Text-Only (Quick Test)

```powershell
# Export text decoder only (uses existing builder)
cd C:\Users\rajeevp\Documents\onnxruntime-genai-1

python -m src.python.py.models.builder ^
  --input examples\python\qwen3-vl-4b\pytorch ^
  --output examples\python\qwen3-vl-4b\cpu-text ^
  --precision fp32 ^
  --execution_provider cpu ^
  --extra_options exclude_embeds=true

# Test with GenAI
python examples\python\model-generate.py ^
  -m examples\python\qwen3-vl-4b\cpu-text
```

### 2. Full Multimodal Export

```powershell
cd examples\python\qwen3-vl-4b
python setup_qwen3vl.py
```

### 3. Optimize Models

```powershell
# Quantize text decoder to INT4
python -m onnxruntime.quantization.quantize ^
  --model cpu\model.onnx ^
  --output cpu\model_int4.onnx ^
  --quantization_mode int4
```

### 4. Deploy with ONNX Runtime GenAI

Use the exported models in your application:
```python
import onnxruntime_genai as og

# Load model
model = og.Model("./cpu")

# Create generator
generator = og.Generator(model)

# Generate text
# (Note: Need to implement image injection)
```

## Resources

### Documentation
- [README.md](./README.md) - Quick start
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - Detailed setup
- [IMPLEMENTATION_REFERENCE.md](./IMPLEMENTATION_REFERENCE.md) - Technical details

### Reference Implementations
- [Phi4-MM Guide](../phi-4-multi-modal.md) - Our baseline
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B) - Original model

### Your Experiments
- [md-files/](./md-files/) - All your documented experiments

## Summary

✅ **Setup Complete!**

You now have:
1. ✅ Scripts to download and modify HuggingFace files
2. ✅ Builder to export vision + embeddings + text to ONNX
3. ✅ Inference pipeline for testing
4. ✅ Complete documentation

**To run:**
```powershell
python setup_qwen3vl.py
```

**To test:**
```powershell
python test_qwen3vl_inference.py --model_path ./cpu --image_path <image.jpg>
```

Good luck with your ONNX export! 🚀
