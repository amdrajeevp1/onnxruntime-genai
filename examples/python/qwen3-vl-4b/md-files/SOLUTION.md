# Qwen3-VL Export Problem & Solution

## The Problem

**Cannot reuse Qwen2.5-VL builder** because architectures are different:

| Component | Qwen2.5-VL | Qwen3-VL |
|-----------|------------|----------|
| Vision hidden size | 640 | **1024** |
| Vision layers | ? | 24 |
| MRoPE sections | [16, 24, 24] | **[24, 20, 20]** |
| Text layers | 28 | 36 |

**Error**: Weight shape mismatch (640 vs 1024)

---

## Three Options Forward

### Option 1: Skip Generic Builder - Export Manually ⚡ FASTEST
**Status**: Can start immediately  
**Time**: 2-3 hours  
**Approach**: Similar to how we exported Phi-4 components

Since we already have the vision encoder exported, just export the text decoder using PyTorch directly:

1. Load `model.language_model` from Qwen3-VL
2. Export to ONNX using `torch.onnx.export` (like we did for vision)
3. Apply INT4 quantization separately
4. Create configuration files manually

**Pros**: 
- Works around the generic builder limitations
- Full control over the export
- We already did this successfully for Phi-4

**Cons**:
- More manual work
- Need to handle quantization ourselves

### Option 2: Implement Qwen3VLTextModel Builder 🔧 PROPER
**Status**: Needs implementation  
**Time**: 1-2 hours  
**Approach**: Add proper support to the codebase

This is the "right" way but requires modifying the builder codebase.

### Option 3: Use Qwen3-4B Builder Instead 💡 HYBRID
**Status**: Can try now  
**Time**: 15 minutes  
**Approach**: Export text-only using Qwen3 builder (not Qwen3-VL)

The text decoder in Qwen3-VL is very similar to standalone Qwen3. We could:
1. Export just the `language_model` using Qwen3ForCausalLM builder
2. Manually extract and save just the language_model weights
3. Use generic Qwen3 export

---

## RECOMMENDED: Option 3 (Hybrid Approach)

Let's try exporting the text component as a standalone Qwen3 model!

### Step 1: Extract Language Model Weights

```python
# extract_language_model.py
import torch
from transformers import AutoModel

# Load Qwen3-VL
model = AutoModel.from_pretrained(
    "./pytorch",
    trust_remote_code=True
)

# Extract just the language_model
language_model = model.language_model

# Save as standalone model
language_model.save_pretrained("./pytorch-text-only")
```

### Step 2: Create Text-Only Config

```python
# The language_model uses Qwen3 architecture
# We can export it as Qwen3ForCausalLM
```

### Step 3: Export with Generic Builder

```bash
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-4B \
  -i ./pytorch-text-only \
  -o ./cpu-text \
  -p int4 \
  -e cpu
```

This might work because the text decoder IS just a Qwen3 model!

---

## Alternative: Direct PyTorch Export (Most Reliable)

If Option 3 doesn't work, we do what worked for Phi-4:

1. ✅ Vision encoder - DONE (builder_vision.py)
2. Export text decoder with PyTorch directly
3. Manually create genai_config.json
4. Test integration

Would you like me to implement Option 3 (extract language_model and export as Qwen3)?
