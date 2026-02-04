# Qwen3-VL Vision Model - Module ONNX Export Analysis

## 🎯 **Executive Summary**

**Test Results**: Systematically tested ONNX export for all 267 modules in Qwen3-VL vision model

### **Key Findings**

| Export Method | Success Rate | Notes |
|---------------|--------------|-------|
| **torch.onnx.export** | **216/267 (81%)** | Individual components mostly work |
| **torch.export + dynamo** | **0/267 (0%)** | Script issue - all failed |

### **Surprising Discovery**

🎉 **The full VisionModel (Qwen3VLVisionModel) SUCCEEDED with torch.onnx.export!**

This contradicts our earlier attempts and suggests that:
1. Individual problematic modules can sometimes be bypassed when exporting the complete model
2. ONNX export may optimize away certain dynamic operations
3. Our previous failures might have been due to input/output configuration issues

---

## 📊 **Detailed Breakdown**

### **Modules That CANNOT Be Exported (torch.onnx.export)**

#### 1. **Qwen3VLVisionRotaryEmbedding** ❌
```
Module: rotary_pos_emb
Type: Qwen3VLVisionRotaryEmbedding
Status: FAILED

Error: arange() received an invalid combination of arguments
       got (Tensor, dtype=torch.dtype, device=torch.device)
```

**Root Cause**:
```python
def forward(self, seqlen: int) -> torch.Tensor:
    seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    #              ^^^^^^
    #              seqlen is a Python int during inference,
    #              but becomes a Tensor during ONNX tracing!
    freqs = torch.outer(seq, self.inv_freq)
    return freqs
```

**Why it fails**: `torch.arange()` expects a scalar integer, but during ONNX tracing, `seqlen` becomes a tensor. The ONNX tracer can't handle dynamic `arange` with tensor inputs.

**Location in code**: Lines 107-112 in `modeling_qwen3_vl.py`

---

#### 2. **Qwen3VLVisionBlock** (All 24 blocks) ❌
```
Module: blocks.0, blocks.1, ..., blocks.23
Type: Qwen3VLVisionBlock
Status: FAILED (all 24 blocks)

Error: cannot unpack non-iterable NoneType object
```

**Root Cause**: The forward method expects `position_embeddings` as a tuple, but during isolated testing, we passed `None` for some parameters, causing unpacking errors.

**Not a fundamental ONNX issue** - just a testing artifact. The blocks themselves contain exportable operations.

**Location in code**: Lines 290-313 in `modeling_qwen3_vl.py`

---

#### 3. **Qwen3VLVisionAttention** (All 24 attention modules) ❌
```
Module: blocks.0.attn, blocks.1.attn, ..., blocks.23.attn
Type: Qwen3VLVisionAttention
Status: FAILED (all 24 attention modules)

Error: cannot unpack non-iterable NoneType object
```

**Root Cause**: Same as VisionBlock - the attention module expects proper `position_embeddings` tuple, but test provided `None`.

**Not a fundamental ONNX issue** - these are standard attention operations that ONNX supports.

**Location in code**: Lines 198-287 in `modeling_qwen3_vl.py`

---

#### 4. **ModuleList** (2 instances) ❌
```
Module: blocks, deepstack_merger_list
Type: ModuleList
Status: FAILED

Error: Module [ModuleList] is missing the required "forward" function
```

**Root Cause**: `nn.ModuleList` is a container, not an actual layer with a forward pass. Cannot be exported standalone.

**Expected behavior** - containers don't have forward methods.

---

### **Modules That CAN Be Exported (torch.onnx.export)** ✅

All these exported successfully:

| Module Type | Count | Notes |
|-------------|-------|-------|
| **Conv3d** | 1 | Patch embedding |
| **Embedding** | 1 | Position embeddings (2304 positions) |
| **LayerNorm** | 48 | 2 per block × 24 blocks |
| **Linear** | 144 | QKV proj, output proj, MLP layers |
| **GELU / GELUTanh** | 28 | Activation functions |
| **Qwen3VLVisionMLP** | 24 | MLP modules |
| **Qwen3VLVisionPatchMerger** | 4 | Main merger + 3 deepstack mergers |
| **Qwen3VLVisionPatchEmbed** | 1 | Patch embedding wrapper |
| **Qwen3VLVisionModel** | 1 | **FULL MODEL!** 🎉 |

**Total**: 216/267 modules (81%) ✅

---

## 🔍 **Critical Analysis**

### **Why Did the Full VisionModel Succeed?**

The full `Qwen3VLVisionModel` exported successfully even though some individual components failed during isolated testing. This suggests:

1. **Context matters**: When exporting the complete model with proper inputs, ONNX can trace through operations that fail in isolation

2. **The problematic modules might not execute**: Some code paths may not be taken during the forward pass with our specific inputs

3. **ONNX optimizations**: The exporter might optimize away or constant-fold certain dynamic operations

### **What About Our Previous Failures?**

Our earlier attempts to export the vision model failed with errors like:

```
RuntimeError: Can't get valid min_val and max_val
```

This was during ACTUAL inference, not just export. The difference:
- **Module export test**: Tests if ONNX graph can be CREATED
- **Our previous tests**: Tested if ONNX graph can be EXECUTED with runtime inference

**Hypothesis**: The vision model CAN be exported to ONNX, but the resulting ONNX model might:
- Contain ops that don't run correctly
- Have hardcoded shapes that break during inference
- Contain unsupported operations that fail at runtime (not during export)

---

## 🧪 **The Real Test: Does the Exported ONNX Work?**

The export test shows we CAN create an ONNX graph. Now we need to test:

### Test 1: Does the ONNX file load?
```python
import onnxruntime as ort
session = ort.InferenceSession("VisionModel_torch_onnx.onnx")
```

### Test 2: Does it run inference?
```python
outputs = session.run(None, {
    "input_0": pixel_values_np,
    "input_1": grid_thw_np
})
```

### Test 3: Do outputs match PyTorch?
```python
pytorch_output = vision_model(pixel_values, grid_thw)
onnx_output = ort_session.run(...)
np.allclose(pytorch_output, onnx_output, rtol=1e-3)
```

---

## 📋 **Summary of Findings**

### **Modules That Cannot Export (Fundamental Issues)**

**1. Qwen3VLVisionRotaryEmbedding** ❌
- **Error**: `torch.arange()` with dynamic tensor input
- **Why**: ONNX can't handle data-dependent operations
- **Location**: Line 107-112 in modeling_qwen3_vl.py
- **Code**:
  ```python
  def forward(self, seqlen: int) -> torch.Tensor:
      seq = torch.arange(seqlen, device=..., dtype=...)
      #              ^^^^^^^ 
      #              Becomes a tensor during ONNX tracing!
  ```

### **Modules That Failed Due to Test Setup** ⚠️

**1. Qwen3VLVisionBlock** (×24) ⚠️
- **Error**: "cannot unpack non-iterable NoneType object"
- **Why**: Test passed `None` for required `position_embeddings` parameter
- **Not a fundamental ONNX issue**: These are standard transformer blocks

**2. Qwen3VLVisionAttention** (×24) ⚠️
- **Error**: Same as VisionBlock
- **Why**: Same as VisionBlock
- **Not a fundamental ONNX issue**: Standard multi-head attention

### **Modules That Exported Successfully** ✅

- **All primitive layers**: Linear, LayerNorm, Conv3d, Embedding, GELU
- **All MLP modules**: Qwen3VLVisionMLP (×24)
- **All Patch Mergers**: Main + 3 deepstack mergers
- **Patch Embedding**: Qwen3VLVisionPatchEmbed
- **🎉 THE FULL VISION MODEL**: Qwen3VLVisionModel

---

## 🎯 **Key Takeaways**

### 1. **Most Individual Components Export Successfully**
   - 216/267 modules (81%) can be exported
   - Only fundamental issue: `RotaryEmbedding` with dynamic `torch.arange`

### 2. **The Full VisionModel Exports!**
   - This is surprising given our previous failures
   - Suggests the issue is in **runtime execution**, not export

### 3. **Dynamo Export Has Script Issues**
   - All 267 modules failed with the same error
   - Likely a problem with our test script, not the modules
   - Error: "cannot access local variable 'torch'"

### 4. **Next Steps to Investigate**
   - ✅ Module export: DONE (this test)
   - ⏳ Runtime inference: Does the exported ONNX model actually run?
   - ⏳ Numerical accuracy: Do outputs match PyTorch?
   - ⏳ Dynamic shapes: Can it handle different image sizes?

---

## 📈 **Success Rate by Module Type**

| Module Type | Success Rate | Notes |
|-------------|--------------|-------|
| Linear | 144/144 (100%) | All exported ✅ |
| LayerNorm | 48/48 (100%) | All exported ✅ |
| GELU/GELUTanh | 28/28 (100%) | All exported ✅ |
| Conv3d | 1/1 (100%) | Exported ✅ |
| Embedding | 1/1 (100%) | Exported ✅ |
| MLP | 24/24 (100%) | All exported ✅ |
| PatchMerger | 4/4 (100%) | All exported ✅ |
| PatchEmbed | 1/1 (100%) | Exported ✅ |
| **RotaryEmbedding** | **0/1 (0%)** | **FAILED** ❌ |
| VisionBlock | 0/24 (0%) | Test setup issue ⚠️ |
| VisionAttention | 0/24 (0%) | Test setup issue ⚠️ |
| ModuleList | 0/2 (0%) | Expected (no forward) |
| **VisionModel (Full)** | **1/1 (100%)** | **EXPORTED!** ✅ |

---

## 🔬 **The Problematic Module: RotaryEmbedding**

### Code (Lines 95-109 in modeling_qwen3_vl.py)

```python
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def forward(self, seqlen: int) -> torch.Tensor:
        # This line fails during ONNX export!
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        #                  ^^^^^^^ seqlen becomes a Tensor during tracing
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

### Why It Fails

**During normal execution**:
```python
rotary_emb = RotaryEmbedding()
output = rotary_emb(100)  # seqlen=100 (Python int)
# Works fine!
```

**During ONNX tracing**:
```python
torch.onnx.export(rotary_emb, (100,), ...)
# ONNX tracer wraps 100 as a Tensor
# torch.arange(Tensor) is not supported!
```

### The Problem

`torch.arange()` requires a **scalar value at graph construction time**, but during ONNX export, the tracer wraps inputs as **symbolic tensors**, making the operation data-dependent and incompatible with ONNX.

---

## 🎉 **Good News: Full Model Exports!**

Despite the `RotaryEmbedding` failure in isolation, the **full VisionModel exported successfully**.

Possible reasons:
1. **Pre-computed rotary embeddings**: The model might compute rotary embeddings once with known dimensions and reuse them
2. **Constant folding**: ONNX exporter might evaluate `rot_pos_emb()` at export time with concrete inputs
3. **Different code path**: The full model might use a different execution path that avoids the problematic operation

### Verification Needed

To confirm the exported model works:
```bash
# Test runtime inference with the exported ONNX model
python test_vision_onnx_inference.py --onnx module_export_tests/VisionModel_torch_onnx.onnx
```

---

## 📊 **Comparison: Individual vs Full Model**

| Aspect | Individual Modules | Full VisionModel |
|--------|-------------------|------------------|
| **Export Success** | 216/267 (81%) | 1/1 (100%) ✅ |
| **Problematic Components** | RotaryEmbedding | ??? (Need to test inference) |
| **Blocks + Attention** | Failed (test issue) | Exported ✅ |
| **Reason for Failure** | Isolated testing | Holistic tracing |

---

## 🔍 **Detailed Module Categories**

### **Category 1: Fully Exportable** ✅

These exported successfully with torch.onnx.export:

**Atomic Operations**:
- `nn.Linear` (144 instances) - Matrix multiplication
- `nn.LayerNorm` (48 instances) - Normalization
- `nn.Conv3d` (1 instance) - 3D convolution for patches
- `nn.Embedding` (1 instance) - Position embeddings
- `nn.GELU / GELUTanh` (28 instances) - Activation functions

**Composite Modules**:
- `Qwen3VLVisionMLP` (24 instances) - MLP blocks
- `Qwen3VLVisionPatchMerger` (4 instances) - Spatial merge + projection
- `Qwen3VLVisionPatchEmbed` (1 instance) - Patch embedding

**The Complete Model**:
- `Qwen3VLVisionModel` (1 instance) - **FULL MODEL** 🎉

---

### **Category 2: Failed in Isolation (Test Issue)** ⚠️

These failed due to test setup, not fundamental ONNX limitations:

- `Qwen3VLVisionBlock` (24 instances)
- `Qwen3VLVisionAttention` (24 instances)

**Reason**: Test provided `None` for required parameters like `position_embeddings`, causing unpacking errors. Not a true ONNX incompatibility.

---

### **Category 3: Fundamentally Incompatible** ❌

Only ONE module has a fundamental ONNX incompatibility:

- `Qwen3VLVisionRotaryEmbedding` (1 instance)

**Issue**: Uses `torch.arange()` with dynamic tensor input, which ONNX doesn't support.

---

## 🧩 **The Mystery: How Does Full Model Succeed?**

Despite `RotaryEmbedding` failing in isolation, the full `VisionModel` exported. Let's investigate:

### **Hypothesis 1: Pre-computation**

In the full model forward pass (lines 838-924):
```python
def forward(self, hidden_states, grid_thw):
    # ...
    rotary_pos_emb = self.rot_pos_emb(grid_thw)  # Call rot_pos_emb
    # ...
```

The `rot_pos_emb()` method (lines 740-766) is more complex:
```python
def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    merge_size = self.spatial_merge_size
    max_hw = int(grid_thw[:, 1:].max().item())  # Convert to Python int!
    freq_table = self.rotary_pos_emb(max_hw)    # Call with Python int
    # ...
```

**Key**: `max_hw` is extracted as a **Python int** using `.item()`, so `rotary_pos_emb()` receives a scalar, not a tensor!

### **Hypothesis 2: Constant Folding**

ONNX exporter might:
1. See that `grid_thw` is provided as a concrete value during export
2. Evaluate `rot_pos_emb()` completely at export time
3. Embed the resulting frequency table directly in the ONNX graph

### **Hypothesis 3: Different Code Path**

The full model might use caching or pre-computed rotary embeddings that bypass the dynamic computation.

---

## 📝 **Recommendations**

### **For Vision Model ONNX Export**

Based on test results:

**✅ DO**: Export the full `Qwen3VLVisionModel`
- 81% of components work
- Full model exported successfully
- Worth testing runtime inference

**⚠️ VERIFY**: Test the exported ONNX model
- Check if it loads with ONNX Runtime
- Run inference with real inputs
- Compare outputs with PyTorch
- Test with different image sizes (dynamic shapes)

**❌ DON'T**: Try to fix individual modules in isolation
- Only 1 module (`RotaryEmbedding`) has a fundamental issue
- Full model export bypasses this somehow
- Focus on end-to-end validation instead

### **Next Steps**

1. **Test Runtime Inference** (Priority 1)
   ```bash
   python test_vision_onnx_runtime.py
   ```

2. **Numerical Validation** (Priority 2)
   - Compare PyTorch vs ONNX outputs
   - Check for accuracy differences

3. **Dynamic Shape Testing** (Priority 3)
   - Test with various image sizes
   - Verify shape handling

4. **Integrate with Text Decoder** (Priority 4)
   - If vision ONNX works, connect to ONNX text model
   - Complete hybrid pipeline with vision injection

---

## 🎯 **Conclusion**

### **Main Findings**

1. **81% of modules export successfully** with torch.onnx.export
2. **Only 1 module has fundamental ONNX incompatibility**: `RotaryEmbedding`
3. **The full VisionModel exported!** This is the biggest discovery
4. **Dynamo export needs script fixes** (all failed with same error)

### **Action Items**

**Immediate**:
- ✅ Module export test: COMPLETE
- ⏳ Test runtime inference of exported VisionModel ONNX
- ⏳ Validate numerical accuracy

**If Runtime Works**:
- Create production-ready vision encoder ONNX
- Integrate with ONNX text decoder
- Complete hybrid pipeline with vision features

**If Runtime Fails**:
- Analyze runtime errors
- Investigate shape mismatches
- Consider alternative approaches

---

## 📂 **Test Artifacts**

- **Test script**: `test_module_onnx_export.py`
- **Detailed report**: `module_export_tests/export_test_report.txt`
- **ONNX files**: `module_export_tests/*.onnx` (216 successfully exported modules)
- **Analysis**: This document

---

**Test Date**: January 30, 2026  
**Status**: Module export test COMPLETE ✅  
**Next**: Runtime inference validation
