# ONNX Runtime GenAI Builder - Qwen3-VL Support Needed

## Problem

The generic `onnxruntime_genai.models.builder` does **NOT** support Qwen3-VL yet!

**Error**: `NotImplementedError: The ./pytorch model is not currently supported.`

## Root Cause

### Current Support in `builder.py`
```python
# Line 304-305: Qwen3 text-only ✅
elif config.architectures[0] == "Qwen3ForCausalLM":
    onnx_model = Qwen3Model(...)

# Line 308-317: Qwen 2.5 VL ✅
elif config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
    onnx_model = Qwen25VLTextModel(...)
```

### What Qwen3-VL Has
```json
{
  "architectures": ["Qwen3VLForConditionalGeneration"],
  "model_type": "qwen3_vl"
}
```

**Missing**: `Qwen3VLForConditionalGeneration` mapping!

---

## Files That Need Modification

### 1. `src/python/py/models/builders/qwen.py`

**Add new class** (after line 23):

```python
class Qwen3VLTextModel(Model):
    """
    Qwen3-VL text decoder builder.
    
    Similar to Qwen2.5-VL but with differences:
    - MRoPE sections: [24, 20, 20] (vs Qwen2.5-VL's [16, 24, 24])
    - Head size: 128 (vs 128)
    - Layers: 36 (vs varies)
    """
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        
        # Same as Qwen2.5-VL: Force LayerNorm to float32
        print("Forcing LayerNorm computation to float32 for Qwen3-VL parity.")
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True
        
        # Same as Qwen2.5-VL: Force RoPE to float32
        print("Forcing RoPE computation to float32 for Qwen3-VL parity.")
        if "rope_cast" not in self.attention_attrs:
            self.attention_attrs["rope_cast"] = {}
        self.attention_attrs["rope_cast"]["use_fp32"] = True
        
        # Check rope type - Qwen3-VL uses mrope
        if config.rope_scaling and "type" in config.rope_scaling:
            assert config.rope_scaling["type"] in ["mrope", "default"]
        
        # Qwen3-VL applies RoPE manually before attention
        self.attention_attrs["use_rope_in_attn"] = False
        self.attention_attrs["use_packed_matmul"] = False
        
        # Add position_ids if needed
        if "position_ids" not in self.input_names:
            print("Re-adding 'position_ids' to self.input_names.")
            self.input_names.append("position_ids")
        
        # Get MRoPE sections from config
        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")
        
        # Qwen3-VL: [24, 20, 20] * 2 = [24, 20, 20, 24, 20, 20]
        self.mrope_splits = self.mrope_sections * 2
        
        if sum(self.mrope_splits) != self.head_size:
            raise ValueError(
                f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) "
                f"does not match head size ({self.head_size})"
            )
        
        # Force GroupQueryAttention
        self.attention_attrs["op_type"] = "GroupQueryAttention"
        
        if not self.is_gqa_supported():
            print(f"Warning: {self.ep} does not support GQA for {self.io_dtype}")
        
        # Create inv_freq tensor
        self.make_inv_freq_tensor()
        
    # Copy these methods from Qwen25VLTextModel:
    # - make_inv_freq_tensor()
    # - make_position_ids_and_mask()  
    # - make_3d_position_ids()
    # - make_attention()
    # - make_output()
    # - load_model_from_name()
    # (They should be identical or very similar)
```

### 2. `src/python/py/models/builder.py`

**Add mapping** (after line 317):

```python
    elif config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
        # ... existing Qwen2.5-VL code ...
        onnx_model = Qwen25VLTextModel(...)
        
    elif config.architectures[0] == "Qwen3VLForConditionalGeneration":  # NEW!
        text_config = config.text_config
        for key in text_config:
            if not hasattr(config, key):
                setattr(config, key, getattr(text_config, key))
        print(
            "WARNING: This is only generating the text component of the model. "
            "Setting `--extra_options exclude_embeds=true` by default."
        )
        extra_options["exclude_embeds"] = True
        onnx_model = Qwen3VLTextModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
        
    elif config_only:
        # ... existing code ...
```

**Import the new class** (at top of file):

```python
from .builders.qwen import QwenModel, Qwen3Model, Qwen25VLTextModel, Qwen3VLTextModel  # Add Qwen3VLTextModel
```

---

## Key Differences: Qwen2.5-VL vs Qwen3-VL

| Feature | Qwen2.5-VL | Qwen3-VL |
|---------|-----------|----------|
| **Architecture** | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| **MRoPE Sections** | `[16, 24, 24]` | `[24, 20, 20]` |
| **Text Layers** | 28 (varies by model size) | 36 |
| **Hidden Size** | 2048/2560 | 2560 |
| **Video Support** | Limited | Full (temporal_patch_size=2) |
| **Head Size** | 128 | 128 |

### MRoPE Splits Calculation

**Qwen2.5-VL**:
```python
mrope_sections = [16, 24, 24]
mrope_splits = [16, 24, 24, 16, 24, 24]  # sections * 2
sum = 128 (matches head_size)
```

**Qwen3-VL**:
```python
mrope_sections = [24, 20, 20]
mrope_splits = [24, 20, 20, 24, 20, 20]  # sections * 2
sum = 128 (matches head_size)
```

---

## Alternative: Reuse Qwen2.5-VL Implementation?

**Can we just add an alias?**

```python
# Quick fix option in builder.py:
elif config.architectures[0] == "Qwen3VLForConditionalGeneration":
    # Reuse Qwen2.5-VL implementation
    text_config = config.text_config
    for key in text_config:
        if not hasattr(config, key):
            setattr(config, key, getattr(text_config, key))
    print("Using Qwen2.5-VL builder for Qwen3-VL (experimental)")
    extra_options["exclude_embeds"] = True
    onnx_model = Qwen25VLTextModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
```

**Pros**:
- Quick fix (2 minutes)
- Minimal code changes
- Might just work™

**Cons**:
- Not officially supported
- Might have subtle bugs
- MRoPE sections are different

**Verdict**: **Worth trying first!** The implementations are very similar.

---

## Recommended Approach

### Option 1: Quick Alias (Try First) ⚡

1. Edit `builder.py` line 317
2. Add Qwen3-VL mapping that reuses Qwen2.5-VL class
3. Test if it works
4. **Time**: 5 minutes

### Option 2: Proper Implementation (If Option 1 Fails) 🔧

1. Create `Qwen3VLTextModel` class in `qwen.py`
2. Copy methods from `Qwen25VLTextModel`
3. Adjust for Qwen3-VL specifics
4. Add mapping in `builder.py`
5. Test thoroughly
6. **Time**: 30-60 minutes

---

## Testing After Fix

```bash
# Try text decoder export again
python builder_text.py \
  --input ./pytorch \
  --output ./cpu-text \
  --precision int4 \
  --execution_provider cpu
```

**Expected**:
- ✅ Model loads successfully
- ✅ Text decoder exports (7-10 minutes)
- ✅ INT4 quantization works
- ✅ ONNX files created

---

## Summary

**Question**: Do we need to change implementation?

**Answer**: **YES**, but we have 2 options:

1. **Quick fix**: Add 5-line alias to reuse Qwen2.5-VL builder
   - Fast to implement
   - Might just work
   - Worth trying first

2. **Proper fix**: Create dedicated Qwen3-VL builder class
   - More robust
   - Officially supported
   - Better long-term

**Recommendation**: Try Option 1 first (alias). If it works, great! If not, do Option 2 (proper implementation).

---

## Files to Modify

**Option 1 (Quick)**:
- `src/python/py/models/builder.py` (add 8 lines after line 317)

**Option 2 (Proper)**:
- `src/python/py/models/builders/qwen.py` (add ~50 lines + copy methods)
- `src/python/py/models/builder.py` (add 8 lines + import)

Would you like me to implement Option 1 (quick alias) first?
