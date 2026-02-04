# Qwen3-VL vs Phi-4: Module Comparison

## TL;DR Answer to Your Question

**"What are different modules other than the text model?"**

For **Qwen3-VL-4B**, there are only **2 main modules**:
1. ✅ **`visual`** - Vision encoder (images + video) - **ALREADY EXPORTED**
2. ✅ **`language_model`** - Text decoder (36 layers)

**NO AUDIO/SPEECH MODULE** - You're absolutely right!

---

## Detailed Comparison

### Phi-4 Multimodal Modules

```
Phi4MultimodalForCausalLM
│
├── [1] Vision Encoder
│   Location: model.model.embed_tokens_extend.image_embed
│   Function: Process images
│   Size: ~614 MB
│   Export: Custom builder required
│
├── [2] Audio/Speech Encoder ⚠️
│   Location: model.model.embed_tokens_extend.audio_embed
│   Function: Process audio/speech
│   Size: ~896 MB
│   Export: Custom builder required
│
├── [3] Embedding Layer
│   Location: model.model.embed_tokens_extend
│   Function: Merge vision/audio/text tokens
│   Size: ~2.4 GB
│   Export: Custom builder required
│
└── [4] Text Decoder
    Location: model.model
    Function: Generate text from multimodal tokens
    Size: ~3.1 GB (INT4)
    Export: Custom builder required
```

**Total Modules: 4**
**Total Complexity: Very High**

---

### Qwen3-VL-4B Modules

```
Qwen3VLModel
│
├── [1] Vision Encoder ✅ DONE
│   Location: model.visual
│   Function: Process images AND video
│   Components:
│     ├── patch_embed (convert image to patches)
│     ├── pos_embed (positional embeddings)
│     ├── rotary_pos_emb (RoPE for vision)
│     ├── blocks (24 transformer layers)
│     ├── merger (combine patches)
│     └── deepstack_merger_list (multi-level features at layers 5,11,17)
│   Size: ~1.66 GB
│   Export: ✅ DONE with builder_vision.py
│   Handles: Static images + video sequences
│
└── [2] Text Decoder
    Location: model.language_model
    Function: Generate text from vision+text tokens
    Components:
      ├── embed_tokens (text + vision token embeddings)
      ├── layers (36 transformer layers with MRoPE)
      └── norm (output normalization)
    Size: ~2-3 GB (INT4 estimated)
    Export: Use generic builder (simple!)
    Handles: Text generation with multimodal understanding
```

**Total Modules: 2**
**Total Complexity: Low**
**No Audio/Speech: ❌ Doesn't exist!**

---

## Key Differences Table

| Aspect | Phi-4 MM | Qwen3-VL |
|--------|----------|----------|
| **Vision** | ✅ Images only | ✅ Images + Video |
| **Audio** | ✅ Audio + Speech | ❌ **NONE** |
| **Video** | ❌ Not supported | ✅ Temporal processing |
| **Modules** | 4 separate | 2 separate |
| **Architecture** | Embedded in language model | Clean separation |
| **Export Complexity** | High (all custom) | Low (reuse generic) |
| **LoRA Adapters** | Yes (vision + audio) | No |
| **Builder Type** | All custom | Vision=custom, Text=generic |

---

## What This Means for Builder.py

### Original Phi-4 builder.py has:

```python
def build_vision(args):
    # Export vision encoder
    # ~150 lines of code
    pass

def build_speech(args):      # ❌ DELETE THIS
    # Export audio/speech encoder
    # ~200 lines of code
    # NOT NEEDED FOR QWEN3-VL!
    pass

def build_embedding(args):   # ❌ DELETE THIS
    # Export embedding layer
    # ~150 lines of code
    # NOT NEEDED FOR QWEN3-VL!
    pass

def build_text(args):        # ✅ SIMPLIFY THIS
    # Export text decoder
    # ~200 lines of code
    # REPLACE WITH GENERIC BUILDER CALL!
    pass
```

### For Qwen3-VL, we need:

```python
# File 1: builder_vision.py ✅ DONE
def build_vision(args):
    # Export visual encoder
    # Handles images + video
    # ~200 lines of code
    # ✅ ALREADY WORKING!
    pass

# File 2: builder_text.py ✅ READY TO USE
def build_text_decoder(args):
    # Just call generic builder!
    # ~20 lines of code
    create_model(...)  # That's it!
    pass

# NO SPEECH BUILDER NEEDED!
# NO EMBEDDING BUILDER NEEDED!
```

---

## Why Qwen3-VL is Simpler

### 1. **Cleaner Architecture**
- Vision and text are **separate top-level modules**
- No complex embedding layer that merges 3 modalities
- No LoRA adapters to handle

### 2. **No Audio/Speech**
- Don't need speech encoder export
- Don't need audio projection layers
- Don't need complex audio branching logic

### 3. **Can Reuse Generic Builder**
- Text decoder is standard Qwen3 architecture
- Generic builder already handles:
  - MRoPE (multimodal RoPE)
  - Vision token embeddings
  - KV cache
  - INT4 quantization

### 4. **Video Support Built-In**
- Vision encoder handles both images and video
- Temporal processing with `temporal_patch_size: 2`
- Grid dimensions in `grid_thw` parameter
- No separate video encoder needed!

---

## Module Interaction Flow

### Phi-4 Flow (Complex)
```
Input → Modality Detection
        ├─ Image → Vision Encoder → Vision LoRA → \
        ├─ Audio → Speech Encoder → Audio LoRA → → Embedding Layer → Text Decoder → Output
        └─ Text  → Token Embedding ────────────→ /
```

### Qwen3-VL Flow (Simple)
```
Input → Modality Detection
        ├─ Image/Video → Vision Encoder → \
        └─ Text ──────────────────────────→ → Text Decoder → Output
                                              (embed_tokens handles merging)
```

---

## Export Strategy

### ❌ DON'T: Modify Phi-4 builder.py
**Why not?**
- 650 lines of code
- 70% is audio/speech logic (not needed)
- Complex embedding logic (not needed)
- LoRA adapter handling (not needed)

### ✅ DO: Use simple scripts
**Why?**
- Vision: `builder_vision.py` ✅ Already done (258 lines, working)
- Text: `builder_text.py` ✅ Ready to use (120 lines, calls generic builder)
- Total: ~380 lines vs 650 lines
- Cleaner, simpler, maintainable

---

## Summary: Answer to Your Questions

### Q: "What are different modules other than the text model?"
**A:** Only **1 module**: The `visual` encoder (already exported!)

### Q: "How should we modify the builder.py?"
**A:** **Don't modify it!** Instead:
- ✅ Use `builder_vision.py` (done)
- ✅ Create `builder_text.py` (ready)
- ✅ Both simpler than modifying 650-line Phi-4 builder

### Q: "Looks like speech doesn't exist"
**A:** **Correct!** Qwen3-VL has:
- ✅ Vision (images + video)
- ✅ Text
- ❌ NO Audio/Speech (that's Phi-4 only)

---

## What to Do Next

1. **Export text decoder** - Run `builder_text.py`
   ```bash
   python builder_text.py \
     --input ./pytorch \
     --output ./cpu-text \
     --precision int4
   ```

2. **Verify both components**
   - Vision: `cpu/qwen3-vl-vision.onnx` ✅
   - Text: `cpu-text/*.onnx` (after export)

3. **Test separately** before integration
   - Vision: Load and process test image
   - Text: Load and generate text

4. **Integrate** once both work

---

**Bottom Line**: 
- Qwen3-VL has **2 modules** (vision + text)
- **No audio/speech** module
- **Don't modify** the complex Phi-4 builder
- **Use simple scripts** instead - much easier!
