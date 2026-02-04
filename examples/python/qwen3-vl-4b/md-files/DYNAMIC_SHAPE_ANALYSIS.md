# Dynamic Shape Strategies: Phi-4 vs Qwen3-VL

## Analysis: How Phi-4 MM Handled Dynamic Shapes

### Phi-4 Vision Encoder

```python
# Phi-4 approach (builder.py lines 61-66)
dynamic_axes = {
    "pixel_values": {0: "num_images", 1: "max_num_crops", 3: "height", 4: "width"},
    "image_attention_mask": {0: "num_images", 1: "max_num_crops"},
    "image_sizes": {0: "num_images"},
    "image_features": {0: "num_image_tokens"},
}

torch.onnx.export(
    model.model.embed_tokens_extend.image_embed,
    args=dummy_inputs,
    dynamic_axes=dynamic_axes,
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,  # TorchScript
)
```

**Key Points**:
1. ✅ Used TorchScript (`dynamo=False`) - same as us
2. ✅ Defined dynamic axes for ALL variable dimensions
3. ✅ Applied ORT transformer optimizer afterward
4. ✅ Then quantized to INT4

### Our Qwen3-VL Approach

```python
# Our approach (builder_vision.py lines 131-135)
dynamic_axes = {
    "pixel_values": {0: "num_patches"},
    "grid_thw": {0: "num_images"},
    "vision_features": {0: "num_patches"}
}
```

**The Problem**: We only marked the **batch/patch count** as dynamic, but **not the spatial dimensions** inside the patches!

---

## Root Cause: Hidden Spatial Operations

### Qwen3-VL Vision Architecture

```
Input: pixel_values [num_patches, 1536]
                ↓
        Reshape to 3D
      (uses grid_thw info)
                ↓
      [1, height, width, 1024]
                ↓
         DeepStack merger
    (operates on spatial dims)
                ↓
    Output: [num_patches, 2560]
```

The **Reshape** and **DeepStack** operations are **hardcoding spatial dimensions** during TorchScript tracing!

When we exported with an 11x15 grid (from test image), it captured:
```python
# Hardcoded in ONNX graph
reshape_shape = [1, 11, 2, 15, 2, -1]  # ← grid-specific!
```

This only works for that exact grid size.

---

## Options to Fix Qwen3-VL Dynamic Shapes

### 🔵 **Option 1: Pre-compute Spatial Dimensions (EASIEST)**

**Idea**: Pass spatial dimensions as explicit inputs instead of having them computed internally.

**Changes Needed**:
```python
# In builder_vision.py - modify export

# Current:
inputs = (pixel_values, grid_thw)

# New:
height_patches = grid_thw[0, 1]  # Extract height
width_patches = grid_thw[0, 2]   # Extract width
inputs = (pixel_values, height_patches, width_patches)

dynamic_axes = {
    "pixel_values": {0: "num_patches"},
    "height_patches": {},  # scalar
    "width_patches": {},   # scalar
    "vision_features": {0: "num_patches"}
}
```

**Pros**: ✅ Simple, no model code changes  
**Cons**: ❌ Still might have internal reshape issues

---

### 🟢 **Option 2: Post-Export ONNX Graph Surgery (RECOMMENDED)**

**Idea**: Modify the exported ONNX graph to make Reshape operations dynamic.

**Steps**:
1. Export model as we did
2. Load ONNX graph
3. Find problematic Reshape nodes
4. Replace hardcoded shape with computed shape from inputs
5. Save modified graph

**Implementation**:
```python
import onnx
from onnx import helper, numpy_helper
import onnx.shape_inference

def make_reshape_dynamic(onnx_model_path, output_path):
    """
    Make Reshape operations dynamic in Qwen3-VL vision encoder
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Find all Reshape nodes
    for node in graph.node:
        if node.op_type == "Reshape":
            # Check if it has a constant shape
            shape_input = node.input[1]
            
            # Find the initializer with this shape
            for init in graph.initializer:
                if init.name == shape_input:
                    # This is a constant shape - make it dynamic!
                    shape_values = numpy_helper.to_array(init)
                    
                    # If shape has spatial dimensions, replace with computation
                    if len(shape_values) == 6 and shape_values[1] > 1:
                        # This looks like: [1, H, 2, W, 2, -1]
                        # Need to compute H, W from grid_thw
                        
                        # Create shape computation subgraph
                        # ... (detailed implementation below)
    
    onnx.save(model, output_path)
    return model
```

**Pros**: ✅ Clean, proper solution  
**Cons**: ⚠️ Requires ONNX graph knowledge

---

### 🟡 **Option 3: Use Torch Export (Dynamo) Instead**

**Idea**: Use `torch.export` with Dynamo instead of TorchScript.

**Changes**:
```python
# In builder_vision.py

# Replace TorchScript export with Dynamo
torch._dynamo.config.capture_scalar_outputs = True

# Define dynamic shapes
from torch.export import Dim
ep = torch.export.export(
    vision_encoder,
    args=(dummy_pixel_values, dummy_grid_thw),
    strict=False,
    dynamic_shapes=[
        {0: Dim.AUTO},  # num_patches
        {0: Dim.AUTO},  # num_images
    ]
)

# Convert to ONNX
onnx_program = torch.onnx.export(
    ep, (),
    input_names=["pixel_values", "grid_thw"],
    output_names=["vision_features"]
)
onnx_program.save(output_path, external_data=True)
```

**Pros**: ✅ Better dynamic shape support  
**Cons**: ❌ May require newer PyTorch, more errors to fix

---

### 🔴 **Option 4: Fixed Input Size Only**

**Idea**: Accept only specific input dimensions, no dynamic shapes.

**Implementation**:
```python
# In test_qwen3vl.py - standardize all images

def preprocess_image(image_path):
    """Resize ALL images to fixed 336x336"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((336, 336), Image.BICUBIC)  # Fixed!
    # ... rest of preprocessing
```

**Pros**: ✅ Guaranteed to work  
**Cons**: ❌ Loses flexibility, must pad/crop all images

---

### 🟣 **Option 5: Apply ORT Transformer Optimizer (LIKE PHI-4)**

**Idea**: Use the same post-processing pipeline as Phi-4.

**Implementation**:
```python
# After initial export, apply ORT optimizer

import subprocess
import sys

subprocess.run([
    f"{sys.executable}", "-m", "onnxruntime.transformers.optimizer",
    "--input", "qwen3-vl-vision.onnx",
    "--output", "qwen3-vl-vision-opt.onnx",
    "--model_type", "clip",  # Similar to vision encoder
    "--num_heads", str(16),
    "--hidden_size", str(1024),
    "--use_external_data_format",
    "--opt_level", str(0),
    "--disable_shape_inference",
])
```

**Pros**: ✅ Industry-standard optimization  
**Cons**: ⚠️ May not fix the root reshape issue

---

## Comparison: Phi-4 vs Qwen3-VL

| Aspect | Phi-4 MM | Qwen3-VL | Impact |
|--------|----------|----------|---------|
| **Export Method** | TorchScript | TorchScript | ✅ Same |
| **Dynamic Axes** | 4D input | 2D input | ⚠️ Different |
| **Spatial Ops** | In flattened form | Grid-based reshape | ❌ Problem |
| **Post-processing** | ORT optimizer | None | ⚠️ Missing |
| **Quantization** | After optimizer | Direct | ⚠️ Different order |

**Key Difference**: Phi-4's vision encoder operates on already-spatialized inputs (`[batch, crops, C, H, W]`), while Qwen3-VL reconstructs spatial dimensions from flattened patches internally.

---

## 🎯 RECOMMENDED SOLUTION PATHS

### Path A: Quick Fix (1 hour)
1. ✅ **Option 4** - Fixed input size (336x336)
2. Test with standardized images
3. Document limitation

**Result**: Working pipeline, limited flexibility

### Path B: Proper Fix (2-3 hours)
1. 🟢 **Option 2** - ONNX graph surgery
2. Make Reshape nodes dynamic
3. Test with various sizes
4. 🟣 **Option 5** - Apply ORT optimizer

**Result**: Fully dynamic, production-ready

### Path C: Modern Approach (2-4 hours)
1. 🟡 **Option 3** - Switch to Torch Dynamo export
2. Debug any compatibility issues
3. Apply same post-processing as Phi-4

**Result**: Best dynamic shape support, future-proof

---

## Detailed Implementation: Option 2 (ONNX Graph Surgery)

```python
def fix_qwen3vl_vision_dynamic_shapes(input_model, output_model):
    """
    Fix Reshape operations in Qwen3-VL vision encoder to be dynamic
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    
    model = onnx.load(input_model, load_external_data=True)
    graph = model.graph
    
    print("Analyzing ONNX graph for Reshape nodes...")
    
    # Step 1: Find problematic Reshape operations
    reshape_nodes_to_fix = []
    for i, node in enumerate(graph.node):
        if node.op_type == "Reshape" and "Reshape" in node.name:
            print(f"Found Reshape node: {node.name}")
            
            # Check if shape input is a constant
            shape_input = node.input[1]
            for init in graph.initializer:
                if init.name == shape_input:
                    shape = numpy_helper.to_array(init)
                    print(f"  Current shape: {shape}")
                    
                    # Is this a spatial reshape? (6D with specific pattern)
                    if len(shape) == 6 and shape[1] > 1 and shape[3] > 1:
                        reshape_nodes_to_fix.append((node, init, shape))
    
    print(f"\nFound {len(reshape_nodes_to_fix)} Reshape nodes to fix\n")
    
    # Step 2: For each Reshape, create dynamic shape computation
    for node, init, original_shape in reshape_nodes_to_fix:
        print(f"Fixing {node.name}...")
        
        # Remove the constant initializer
        graph.initializer.remove(init)
        
        # Create a subgraph to compute the shape from grid_thw
        # Shape should be: [1, H, 2, W, 2, -1]
        # Where H = grid_thw[0,1] and W = grid_thw[0,2]
        
        # Add shape computation nodes
        # (This requires extracting H and W from grid_thw and building shape tensor)
        
        # For now, simpler approach: make the -1 dimension truly dynamic
        # by computing it from input shape
        
        # Create a Shape node to get input dimensions
        shape_node_name = f"{node.name}_compute_shape"
        shape_node = helper.make_node(
            "Shape",
            inputs=[node.input[0]],
            outputs=[f"{node.name}_input_shape"]
        )
        graph.node.insert(graph.node.index(node), shape_node)
        
        print(f"  ✓ Added dynamic shape computation\n")
    
    # Step 3: Update graph metadata
    print("Updating graph metadata...")
    model.graph.ClearField('value_info')  # Clear cached shape info
    
    # Step 4: Run shape inference
    print("Running shape inference...")
    model = onnx.shape_inference.infer_shapes(model)
    
    # Save
    print(f"Saving to {output_model}...")
    onnx.save(
        model,
        output_model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_model}.data",
        size_threshold=0
    )
    
    print("✓ Done!")
    return model
```

---

## Testing Strategy

After implementing any fix:

```python
# Test with different image sizes
test_sizes = [
    (224, 224),   # Small
    (336, 336),   # Medium (original export size)
    (448, 448),   # Large
    (512, 384),   # Non-square
]

for width, height in test_sizes:
    print(f"\nTesting {width}x{height}...")
    
    # Create test image
    image = Image.new('RGB', (width, height), color='white')
    image.save(f"test_{width}x{height}.jpg")
    
    # Run inference
    try:
        result = test_vision_encoder(f"test_{width}x{height}.jpg")
        print(f"✓ SUCCESS: {result.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
```

---

## Summary

**Why Phi-4 works and Qwen3-VL doesn't**:
- Phi-4: Spatial operations happen **before** ONNX export (in preprocessing)
- Qwen3-VL: Spatial operations happen **inside** the ONNX model (Reshape nodes)

**Best Options** (ranked):
1. 🥇 **Option 2** - ONNX graph surgery (proper fix)
2. 🥈 **Option 3** - Torch Dynamo export (modern approach)
3. 🥉 **Option 5** - ORT optimizer (may help, worth trying)
4. 🎖️ **Option 4** - Fixed size (quick workaround)

**Recommendation**: Start with **Option 2 (ONNX graph surgery)** - it's the cleanest solution that doesn't require re-export and gives us full control.

---

## Next Steps

1. Implement Option 2 (ONNX graph surgery script)
2. Test with various image sizes
3. If Option 2 is too complex, fall back to Option 4 (fixed size)
4. Apply Option 5 (ORT optimizer) regardless for performance

Would you like me to implement Option 2 or try Option 4 first for a quick win?
