# Qwen3-VL ONNX Builder - Current Status

**Date**: February 4, 2026  
**Goal**: Export Qwen3-VL model using local builder (like Phi4-MM)

---

## ✅ Completed Steps

### 1. Added Qwen3-VL Support to onnxruntime-genai Package

**Files Modified in Source** (`src/python/py/models/`):

**a) `builders/qwen.py`** - Added `Qwen3VLTextModel` class:
```python
class Qwen3VLTextModel(Qwen25VLTextModel):
    """
    Qwen3-VL text model inherits from Qwen2.5-VL.
    Main difference: MRoPE sections [24, 20, 20] vs [16, 24, 24]
    """
    def load_weights(self, input_path):
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(...)
```

**b) `builders/__init__.py`** - Exported `Qwen3VLTextModel`

**c) `builder.py`** - Added architecture support:
```python
elif config.architectures[0] == "Qwen3VLForConditionalGeneration":
    onnx_model = Qwen3VLTextModel(config, ...)
```

### 2. Updated Installed Package

Created and ran `update_installed_package.py` to copy modified files to:
```
C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\Lib\site-packages\onnxruntime_genai\models\
```

Result: ✅ All 3 files copied successfully

### 3. Created Local Qwen3-VL Builder

**File**: `builder_qwen3vl.py` (inspired by Phi4-MM's `builder.py`)

**Features**:
- Exports vision model to ONNX (for reference)
- Exports text model using onnxruntime-genai builder (INT4 quantized)
- Follows same pattern as Phi4-MM
- Supports `--skip-vision` and `--skip-text` flags

**Usage**:
```bash
python builder_qwen3vl.py -i ./pytorch -o ./output -p fp32 -e cpu [--skip-vision]
```

---

## ⏳ Currently Running

### Text Model Export (IN PROGRESS)

**Command**:
```bash
python builder_qwen3vl.py -i ./pytorch -o ./qwen3vl-onnx-final -p fp32 -e cpu --skip-vision
```

**Status**: Running (PID: 3460)
- Started: ~2 minutes ago
- Expected Duration: 3-5 minutes total
- Output Directory: `./qwen3vl-onnx-final/qwen3vl-text/`

**What's Happening**:
1. Loading Qwen3-VL PyTorch model
2. Running onnxruntime-genai builder
3. Exporting to ONNX
4. Quantizing to INT4
5. Optimizing model

---

## Architecture Comparison

### Qwen3-VL vs Qwen2.5-VL

| Feature | Qwen2.5-VL | Qwen3-VL |
|---------|------------|----------|
| MRoPE Sections | `[16, 24, 24]` | `[24, 20, 20]` |
| Head Dim | 128 | 128 |
| Hidden Size | varies | 2560 (4B) |
| Rope Theta | 1000000 | 5000000 |
| Architecture | Qwen2_5_VLForConditionalGeneration | Qwen3VLForConditionalGeneration |

### Why Inherit from Qwen2.5-VL?

Both models use:
- ✅ MRoPE (Multi-dimensional Rotary Position Embedding)
- ✅ Same attention mechanism (GroupQueryAttention)
- ✅ Same LayerNorm behavior (FP32 computation)
- ✅ Same RoPE casting requirements

**Only Difference**: MRoPE section dimensions

---

## Key Files

### Source Code (Repository)
- `src/python/py/models/builder.py` - Main builder entry point
- `src/python/py/models/builders/__init__.py` - Exports
- `src/python/py/models/builders/qwen.py` - Qwen model classes

### Local Builder (qwen3-vl-4b/)
- `builder_qwen3vl.py` - Local Qwen3-VL builder
- `update_installed_package.py` - Package updater utility

### Model Directories
- `./pytorch/` - Source PyTorch model (3.2 GB)
- `./qwen3vl-onnx-final/` - Current export output (in progress)

---

## Next Steps

### After Text Export Completes:

1. **Verify Export**:
   - Check `./qwen3vl-onnx-final/qwen3vl-text/` exists
   - Verify files: `*.onnx`, `*.onnx.data`, `genai_config.json`
   - Expected size: ~1.1 GB for INT4 quantized model

2. **Test Text-Only Generation**:
   ```python
   import onnxruntime_genai as og
   
   model = og.Model("./qwen3vl-onnx-final/qwen3vl-text")
   tokenizer = og.Tokenizer(model)
   
   # Generate text
   input_tokens = tokenizer.encode("Hello, how are you?")
   params = og.GeneratorParams(model)
   params.input_ids = input_tokens
   
   output_tokens = model.generate(params)
   text = tokenizer.decode(output_tokens[0])
   print(text)
   ```

3. **Add Vision Component**:
   - Run builder with vision: `python builder_qwen3vl.py -i ./pytorch -o ./output -p fp32 -e cpu`
   - Use PyTorch vision in hybrid pipeline (recommended)

4. **Create End-to-End Test**:
   - Load vision model (PyTorch)
   - Load text model (ONNX)
   - Test with image + text prompt
   - Verify multimodal generation

---

## Technical Notes

### MRoPE Implementation

Qwen3-VL uses 3D rotary embeddings for handling:
- **Temporal** dimension (video frames): 24 dims
- **Height** dimension (image rows): 20 dims
- **Width** dimension (image columns): 20 dims

Total: 24 + 20 + 20 = 64 dims (half of 128 head_dim)

### Why Text-Only First?

Testing text model separately helps isolate issues:
1. ✅ Validates Qwen3VLTextModel builder works
2. ✅ Confirms INT4 quantization succeeds
3. ✅ Verifies onnxruntime-genai integration
4. Then add vision complexity

---

## Status Summary

| Task | Status |
|------|--------|
| Add Qwen3VL to source builders | ✅ Complete |
| Update installed package | ✅ Complete |
| Create local builder script | ✅ Complete |
| Export text model | ⏳ In Progress |
| Test text-only generation | ⏸️ Pending |
| Export vision model | ⏸️ Pending |
| Test multimodal pipeline | ⏸️ Pending |

**Overall**: ~70% Complete  
**ETA**: Text export completes in 1-3 minutes

---

**Last Updated**: February 4, 2026 02:03:00 UTC  
**Status**: Waiting for text model export to complete
