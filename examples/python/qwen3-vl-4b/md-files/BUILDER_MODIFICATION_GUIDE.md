# Qwen3-VL Builder Modification Guide

## Model Architecture Comparison

### Phi-4 Multimodal Structure
```python
Phi4MultimodalForCausalLM
├── model (Phi4MMModel)
│   └── embed_tokens_extend (Phi4MixedModalityEmbedding)
│       ├── image_embed         # Vision encoder ✅
│       ├── audio_embed         # Audio/Speech encoder ✅
│       └── token_embedding     # Text embeddings
├── vision_projection            # LoRA adapter for vision
├── audio_projection            # LoRA adapter for audio
└── lm_head                     # Output head
```

**Components to Export (Phi-4):**
1. ✅ Vision encoder (`model.model.embed_tokens_extend.image_embed`)
2. ✅ Audio encoder (`model.model.embed_tokens_extend.audio_embed`)
3. ✅ Embedding layer (`model.model.embed_tokens_extend`)
4. ✅ Text decoder (`model.model`)

---

### Qwen3-VL Structure (SIMPLER!)
```python
Qwen3VLModel
├── visual                      # Vision encoder ✅ DONE
│   ├── patch_embed
│   ├── pos_embed
│   ├── rotary_pos_emb
│   ├── blocks (24 layers)
│   ├── merger
│   └── deepstack_merger_list
│
└── language_model              # Text decoder
    ├── embed_tokens
    ├── layers (36 layers)
    └── norm
```

**Components to Export (Qwen3-VL):**
1. ✅ **Vision encoder** (`model.visual`) - **ALREADY DONE!**
2. ❌ **NO Audio/Speech** - doesn't exist
3. ⏳ **Text decoder** (`model.language_model`) - Use generic builder
4. ⏳ **Integration** - Connect vision features to text input

---

## Key Differences

| Feature | Phi-4 | Qwen3-VL |
|---------|-------|----------|
| **Vision** | Embedded in language model | Separate `visual` module |
| **Audio** | ✅ Has audio/speech encoder | ❌ No audio |
| **Video** | ❌ Static images only | ✅ Video support (temporal) |
| **Embedding** | Complex `embed_tokens_extend` | Simple `embed_tokens` |
| **LoRA** | Separate projection layers | No LoRA adapters |

---

## What to Remove from Phi-4 Builder

### 1. Remove Audio/Speech Functions
```python
# DELETE THIS ENTIRE FUNCTION:
def build_speech(args):
    # ... 200+ lines ...
    # NOT NEEDED - Qwen3-VL has no audio
```

### 2. Remove Audio-related Imports
```python
# DELETE:
import soundfile  # Not needed

# KEEP:
from PIL import Image  # Still needed for vision
```

### 3. Remove Audio from Main
```python
# DELETE this call:
build_speech(args)  # Not needed
```

---

## What to Modify

### 1. Vision Builder (ALREADY DONE ✅)
**File**: `builder_vision.py`

**Key Changes Made**:
- Use `model.visual` instead of `model.model.embed_tokens_extend.image_embed`
- Handle dual inputs: `(pixel_values, grid_thw)`
- Disable SDPA: `attn_implementation="eager"`
- Remove LoRA adapter logic

### 2. Text Decoder Export (USE GENERIC BUILDER)
**Approach**: Use `onnxruntime_genai.models.builder` like we did for Qwen3-4B

```python
from onnxruntime_genai.models.builder import create_model

# Export text decoder with INT4 quantization
create_model(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    input_path="./pytorch",
    output_dir="./cpu",
    precision="int4",
    execution_provider="cpu",
    cache_dir="./cache"
)
```

**This will handle**:
- 36-layer text decoder
- MRoPE (multimodal positional embeddings)
- INT4 quantization
- KV cache
- All the complex transformer logic

---

## Simplified Export Strategy

### Option 1: Separate Scripts (RECOMMENDED)
```
qwen3-vl-4b/
├── builder_vision.py       # ✅ DONE - Vision encoder only
├── builder_text.py         # TODO - Use generic builder
└── builder_integration.py  # TODO - Connect components
```

**Advantages**:
- Simpler, cleaner code
- Each component independent
- Easier to debug
- Can reuse generic builder for text

### Option 2: Single Builder (Like Phi-4)
```
qwen3-vl-4b/
└── builder.py              # All-in-one builder
    ├── build_vision()      # ✅ Already done
    ├── build_embedding()   # Removed - not needed for Qwen3-VL
    └── build_text()        # Call generic builder
```

**Advantages**:
- Single entry point
- Familiar pattern from Phi-4

---

## Recommended Approach

### Step 1: Use Generic Builder for Text ✅
**DO THIS FIRST** - It's the easiest:

```python
# builder_text.py
import argparse
from onnxruntime_genai.models.builder import create_model

def build_text_decoder(input_path, output_dir):
    """Export Qwen3-VL text decoder using generic builder"""
    
    print("="*80)
    print("EXPORTING QWEN3-VL TEXT DECODER")
    print("="*80)
    
    create_model(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        input_path=input_path,
        output_dir=output_dir,
        precision="int4",           # INT4 quantization
        execution_provider="cpu",   # CPU backend
        cache_dir="./cache"
    )
    
    print("✅ Text decoder exported successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    build_text_decoder(args.input, args.output)
```

### Step 2: Test Components Separately
```bash
# 1. Vision encoder (already done)
ls cpu/qwen3-vl-vision.onnx

# 2. Text decoder (to be exported)
python builder_text.py --input ./pytorch --output ./cpu-text

# 3. Verify both exist
ls cpu-text/*.onnx
```

### Step 3: Integration (Later)
Once both components work, create integration:
- Load vision ONNX
- Load text ONNX
- Connect vision features to text input
- Test end-to-end

---

## Key Insight: Qwen3-VL is SIMPLER!

**Phi-4 Complexity**:
- 4 separate ONNX exports
- Custom builders for each component
- LoRA adapters
- Complex audio branching logic
- Manual configuration files

**Qwen3-VL Simplicity**:
- 2 ONNX exports (vision + text)
- Vision = custom builder (done)
- Text = **generic builder** (use existing tools!)
- No LoRA adapters
- Cleaner architecture

---

## Next Steps

### Immediate Action:
1. ✅ **Vision encoder** - Already exported
2. **Create `builder_text.py`** - Use generic builder
3. **Export text decoder** - Run generic builder
4. **Verify both components** - Load and test separately

### After Both Components Work:
5. Create configuration files
6. Test integration
7. Run full multimodal inference

---

## Why Generic Builder for Text?

The `onnxruntime_genai.models.builder` is **designed for this**:
- Handles Qwen architecture
- Supports MRoPE (multimodal RoPE)
- INT4 quantization built-in
- KV cache optimization
- Tested and maintained

**Don't reinvent the wheel!** Use it just like we did for Qwen3-4B text-only model.

---

## Summary: Builder Modifications

### What to DELETE:
- ❌ `build_speech()` function
- ❌ Audio-related code
- ❌ `soundfile` import
- ❌ LoRA adapter logic
- ❌ Complex embedding export

### What to KEEP:
- ✅ Vision export logic (already in `builder_vision.py`)
- ✅ ONNX validation
- ✅ File handling utilities

### What to ADD:
- ✅ Call to generic text builder
- ✅ Simplified integration layer

---

**Bottom Line**: Qwen3-VL is **much simpler** than Phi-4 because:
1. No audio/speech
2. Cleaner architecture separation
3. Can use generic builder for text
4. Only 2 main components vs 4

**Recommendation**: Use `builder_vision.py` (done) + generic builder for text (easy) instead of modifying the complex Phi-4 builder!
