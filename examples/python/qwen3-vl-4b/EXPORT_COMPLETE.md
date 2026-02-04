# Qwen3-VL ONNX Export - COMPLETE SUCCESS! ✅✅✅

## Mission Accomplished!

All three required models have been successfully exported to ONNX format!

## Exported Models

### 1. Vision Encoder ✅
**File:** `cpu/vision_encoder.onnx` (1.25 GB)
**Created:** 2/3/2026 7:23:43 PM

**Specifications:**
- Input: `pixel_values` [num_patches, 1536]
  - 1536 = 3 channels × 2 temporal × 16 patch_size × 16 patch_size
- Input: `image_grid_thw` [num_images, 3]
  - 3 dimensions: Temporal, Height, Width in patches
- Output: `pooled_embeds` [num_merged_patches, 2560]
  - Merged patches ready for LLM injection

**Architecture:**
- 24 transformer layers
- Hidden size: 1024
- 16 attention heads
- 3D Conv patch embedding
- 2D rotary position embeddings
- 2×2 spatial merge → projects to 2560 dim

### 2. Embeddings ✅
**File:** `cpu/embeddings.onnx` (1.56 GB)
**Created:** 2/3/2026 7:23:47 PM

**Specifications:**
- Input: `input_ids` [batch, seq_len] (INT64)
- Output: `inputs_embeds` [batch, seq_len, 2560] (FP32)
- Vocabulary: 151,936 tokens
- Embedding dimension: 2560

### 3. Text Decoder ✅  
**File:** `cpu-text/model.onnx` (908 KB + external data)
**Created:** 2/3/2026 7:10 PM

**Specifications:**
- Input: `inputs_embeds` [batch, seq_len, 2560] (FP32)
- Input: `position_ids` [3, batch, seq_len] (INT64) - **3D MRoPE!**
- Input: `attention_mask` [batch, seq_len] (INT64)
- Input: KV caches (36 layers × 2)
- Output: `logits` [batch, seq_len, 151936] (FP32)
- Output: Updated KV caches

**Architecture:**
- 36 transformer layers
- Hidden size: 2560
- 32 query heads, 8 KV heads (GQA)
- Head dimension: 128
- **3D MRoPE** with sections [24, 20, 20]
- RoPE theta: 5,000,000

## Key Fixes Applied

### Fix 1: Rotary Embedding (Text Decoder) ✅
**Problem:** Dynamic decisions in `@dynamic_rope_update` prevented ONNX export

**Solution:**
- Removed `@dynamic_rope_update` decorator
- Made `position_ids` always 3D: [3, batch, seq_len]
- Removed conditional expansion logic

**Files Modified:**
- `pytorch/modular_qwen3_vl.py`
- `pytorch/modeling_qwen3_vl.py`

### Fix 2: Vision Encoder Shape Mismatch ✅
**Problem:** num_patches (432) didn't match grid_thw (576)

**Solution:**
```python
# Calculate patches from grid: T × H × W
grid_t, grid_h, grid_w = 1, 24, 24
num_patches = grid_t * grid_h * grid_w  # 576 ✓
```

### Fix 3: Vision Encoder SDPA Issue ✅
**Problem:** scaled_dot_product_attention with GQA doesn't export to ONNX

**Solution:**
```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ...,
    attn_implementation="eager"  # Force eager attention
)
```

### Fix 4: Embeddings Path ✅
**Problem:** Wrong attribute path for Qwen3VL model

**Solution:**
```python
# Wrong: model.model.embed_tokens
# Right: model.language_model.embed_tokens
embeddings = model.model.language_model.embed_tokens
```

## Directory Structure

```
qwen3-vl-4b/
├── cpu/                           ← ONNX models (vision + embeddings)
│   ├── vision_encoder.onnx       ← 1.25 GB ✅
│   ├── embeddings.onnx           ← 1.56 GB ✅
│   └── vision_processor.json     ← Config
│
├── cpu-text/                      ← ONNX model (text decoder)
│   ├── model.onnx                ← 908 KB ✅
│   ├── model.onnx.data           ← Weights
│   ├── genai_config.json         ← Config
│   └── tokenizer files           ← All copied ✅
│
├── pytorch/                       ← Modified source files
│   ├── modular_qwen3_vl.py       ← Modified for ONNX
│   ├── *.safetensors             ← Model weights
│   └── config files              ← Original configs
│
└── pytorch_modified/              ← Modified files + backups
    ├── *.py                       ← Modified files
    └── *.py.backup               ← Original backups
```

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Image Input (PIL/numpy)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  Preprocess Image                 │
          │  - Smart resize                   │
          │  - Normalize                      │
          │  - Create 3D patches              │
          └──────────────┬───────────────────┘
                         │
                         ▼
          pixel_values [576, 1536]
          image_grid_thw [1, 3]
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  vision_encoder.onnx (1.25 GB)   │
          │  - 24 transformer layers          │
          │  - 2D RoPE                        │
          │  - Spatial merge                  │
          └──────────────┬───────────────────┘
                         │
                         ▼
          pooled_embeds [144, 2560]
                         │
                         │
┌────────────────────────┴────────────────────────┐
│                                                  │
│  Text Input "What's in the image?"              │
│                 │                                │
│                 ▼                                │
│           Tokenizer                              │
│                 │                                │
│                 ▼                                │
│           input_ids [1, N]                       │
│                 │                                │
│                 ▼                                │
│    ┌────────────────────────────┐               │
│    │  embeddings.onnx (1.56 GB)  │               │
│    └────────────┬───────────────┘               │
│                 │                                │
│                 ▼                                │
│         text_embeds [1, N, 2560]                 │
│                 │                                │
│                 ▼                                │
│    ┌────────────────────────────┐               │
│    │  Merge Embeddings           │               │
│    │  - Inject vision at         │◄──────────────┘
│    │    <|image_pad|> positions  │
│    └────────────┬───────────────┘
│                 │
│                 ▼
│         merged_embeds [1, N+144, 2560]
│                 │
│                 ▼
│    ┌────────────────────────────┐
│    │  model.onnx (908 KB + data) │
│    │  - 36 transformer layers    │
│    │  - 3D MRoPE                 │
│    │  - GQA (32/8 heads)         │
│    └────────────┬───────────────┘
│                 │
│                 ▼
│            logits [1, N+144, 151936]
│                 │
│                 ▼
│             Decode
│                 │
│                 ▼
│         Generated Text
└─────────────────────────────────────────────────┘
```

## Model Sizes Summary

| Component | File Size | Description |
|-----------|-----------|-------------|
| Vision Encoder | 1.25 GB | 24-layer ViT with 3D patches |
| Embeddings | 1.56 GB | Token → embedding mapping |
| Text Decoder | ~908 KB + data | 36-layer transformer with 3D MRoPE |
| **Total** | **~2.8 GB** | Complete pipeline |

## Next Steps

### Step 1: Verify All Models

```powershell
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b

# Check vision encoder
python -c "import onnx; m = onnx.load('cpu/vision_encoder.onnx'); print('Vision:', [i.name for i in m.graph.input], '->', [o.name for o in m.graph.output])"

# Check embeddings
python -c "import onnx; m = onnx.load('cpu/embeddings.onnx'); print('Embeddings:', [i.name for i in m.graph.input], '->', [o.name for o in m.graph.output])"

# Check text decoder
python -c "import onnx; m = onnx.load('cpu-text/model.onnx'); print('Text:', [i.name for i in m.graph.input][:5], '...', '->', [o.name for o in m.graph.output][:3], '...')"
```

### Step 2: Create Integration Pipeline

Now you need to create a pipeline that:
1. **Preprocesses image** → pixel_values + grid_thw
2. **Runs vision_encoder.onnx** → pooled_embeds
3. **Tokenizes text** → input_ids  
4. **Runs embeddings.onnx** → text_embeds
5. **Merges embeddings** → Inject pooled_embeds at `<|image_pad|>` positions
6. **Runs model.onnx** → logits
7. **Decodes output** → Generated text

### Step 3: Test End-to-End

Create a test script that runs all three models together. Reference your existing experiments:
- `md-files/HYBRID_PIPELINE_SUCCESS.md`
- `md-files/VISION_INJECTION_GUIDE.md`

## Summary of Achievements

✅ **Downloaded** HuggingFace model files  
✅ **Modified** rotary embedding for ONNX compatibility  
✅ **Exported** vision encoder (1.25 GB)  
✅ **Exported** embeddings layer (1.56 GB)  
✅ **Exported** text decoder (908 KB + data)  
✅ **Created** processor configurations  
✅ **Copied** tokenizer files  

## Critical Implementation Details

### Vision Encoder
- **Fixed attention:** Used `attn_implementation="eager"` instead of SDPA
- **Fixed shapes:** Ensured num_patches matches T×H×W from grid_thw
- **Output:** Returns pooled_embeds (merged patches for LLM)

### Embeddings  
- **Fixed path:** Used `model.language_model.embed_tokens`
- **Vocab:** 151,936 tokens
- **Dimension:** 2560

### Text Decoder
- **3D Position IDs:** [3, batch, seq_len] for MRoPE
- **MRoPE Sections:** [24, 20, 20] for T/H/W
- **GQA:** 32 query heads, 8 KV heads

## Files Created

### Scripts
- ✅ `setup_qwen3vl.py` - Master setup script
- ✅ `copy_hf_files.py` - Downloads HF files
- ✅ `modify_rotary_embedding.py` - Modifies for ONNX
- ✅ `builder_qwen3vl.py` - Exports all three models
- ✅ `test_text_decoder.py` - Tests text decoder
- ✅ `test_qwen3vl_inference.py` - Full pipeline (to be completed)

### Documentation
- ✅ `README.md` - Quick start
- ✅ `SETUP_GUIDE.md` - Detailed setup
- ✅ `IMPLEMENTATION_REFERENCE.md` - Technical comparison
- ✅ `EXPORT_SUCCESS_SUMMARY.md` - Text decoder success
- ✅ `EXPORT_COMPLETE.md` - This file (all three models!)

### Models
- ✅ `cpu/vision_encoder.onnx` - Vision model
- ✅ `cpu/embeddings.onnx` - Embedding layer
- ✅ `cpu-text/model.onnx` - Text decoder
- ✅ `cpu/vision_processor.json` - Image preprocessing config
- ✅ `cpu-text/genai_config.json` - GenAI config
- ✅ `cpu-text/tokenizer files` - All tokenizer files

## Key Takeaways

1. **Eager Attention is Required**
   - SDPA doesn't export to ONNX with GQA
   - Use `attn_implementation="eager"` when loading model

2. **Shape Consistency is Critical**
   - num_patches must equal T×H×W from grid_thw
   - Otherwise position embedding fails

3. **Model Structure Matters**
   - Qwen3VL uses `model.language_model.embed_tokens`
   - Not `model.model.embed_tokens` like some models

4. **3D Position IDs**
   - Text decoder requires [3, batch, seq] shape
   - For MRoPE (multi-axis rotary embeddings)

## Ready for Integration!

You now have all three components. The next phase is:

1. **Create image preprocessor** (similar to Phi4-MM)
2. **Build multimodal pipeline** (merge vision + text)
3. **Test end-to-end inference**
4. **Optimize models** (INT4 quantization, etc.)

---

**Congratulations! 🎉 All three models exported successfully!**

Awaiting your command for the next phase...
