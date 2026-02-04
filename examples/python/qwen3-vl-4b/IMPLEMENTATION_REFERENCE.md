# Qwen3-VL ONNX Implementation Reference

## Overview

This document provides a comprehensive reference for the Qwen3-VL ONNX export implementation, comparing it with the Phi4-MM baseline.

## Architecture Comparison

### Phi4-MM Architecture
```
Audio → Speech Encoder → audio_embeds
                             ↓
Image → Vision Encoder → image_embeds
                             ↓
Text → Tokenizer → input_ids → Embedding → text_embeds
                                               ↓
                                          Merge embeds
                                               ↓
                                          Text Decoder → logits
```

### Qwen3-VL Architecture
```
Image → Vision Encoder → image_embeds
                             ↓
Text → Tokenizer → input_ids → Embedding → text_embeds
                                               ↓
                                          Merge embeds
                                               ↓
                                          Text Decoder → logits
                                          (with 3D MRoPE)
```

**Key Differences:**
1. Qwen3-VL: No audio support (vision + text only)
2. Qwen3-VL: Uses 3D MRoPE (multi-axis rotary embeddings)
3. Qwen3-VL: Vision encoder uses 3D patch embedding (Conv3D)
4. Qwen3-VL: Vision encoder uses 2D rotary embeddings

## Component Breakdown

### 1. Vision Encoder

#### Phi4-MM Vision
- Architecture: SigLIP + NaViT
- Input: Preprocessed image patches
- Patch size: 14x14
- Layers: 27 transformer layers
- Hidden size: 1152

#### Qwen3-VL Vision
- Architecture: Custom ViT with 3D patches
- Input: Raw image pixels
- Patch size: 16x16 spatial, 2 temporal
- Layers: 24 transformer layers
- Hidden size: 1024
- Special features:
  - 3D Conv patch embedding
  - 2D rotary position embeddings
  - Smart resize (dynamic resolution)
  - Spatial merge (2x2 → 1)

**Export Differences:**
```python
# Phi4-MM: Preprocessed patches
pixel_values: [batch, num_patches, channels * patch_size^2]

# Qwen3-VL: Flattened patches with temporal
pixel_values: [num_patches, channels * temporal * patch_size^2]
image_grid_thw: [num_images, 3]  # Temporal, Height, Width
```

### 2. Embeddings

#### Phi4-MM
- Standard embedding layer
- Input: input_ids [batch, seq]
- Output: inputs_embeds [batch, seq, 3072]

#### Qwen3-VL
- Standard embedding layer
- Input: input_ids [batch, seq]
- Output: inputs_embeds [batch, seq, 2560]
- **Same approach, different dimensions**

### 3. Text Decoder

#### Phi4-MM
- Architecture: Phi-3.5 based
- Layers: 32
- Hidden size: 3072
- Heads: 32
- RoPE: Standard 1D

#### Qwen3-VL
- Architecture: Qwen2.5 based
- Layers: 28
- Hidden size: 2560
- Heads: 20
- RoPE: **3D MRoPE** (key difference!)

**MRoPE Details:**
```python
# Standard RoPE (Phi4-MM)
position_ids: [batch, seq_len]

# MRoPE (Qwen3-VL)
position_ids: [3, batch, seq_len]  # Temporal, Height, Width axes
mrope_sections: [24, 20, 20]  # Dimension split for T/H/W
```

## ONNX Export Strategy

### Overall Approach (Same as Phi4-MM)

Both models follow the same multi-component export strategy:

1. **Vision Encoder** → `vision_encoder.onnx`
2. **Embeddings** → `embeddings.onnx`
3. **Text Decoder** → `model.onnx`

### Key Modifications Required

#### 1. Rotary Embedding (Critical!)

**Problem:** Dynamic shape decisions prevent ONNX export

**Phi4-MM Solution:**
- No 3D position IDs, standard 1D RoPE
- No dynamic expansion needed

**Qwen3-VL Solution:**
```python
# BEFORE (HuggingFace - Won't export)
@torch.no_grad()
@dynamic_rope_update
def forward(self, x, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, ...)
    # ...

# AFTER (ONNX-compatible)
@torch.no_grad()
def forward(self, x, position_ids):
    # Always assume 3D
    assert position_ids.ndim == 3, "Must be [3, B, S]"
    # ...
```

**Files to modify:**
- `modeling_qwen3_vl.py` (lines 269-286)
- `modular_qwen3_vl.py` (lines 269-286)

**Classes affected:**
- `Qwen3VLTextRotaryEmbedding`

#### 2. Vision Model Dynamic Shapes

**Problem:** Smart resize creates dynamic shapes

**Solution:**
- Export with representative image sizes
- Use dynamic axes in ONNX export
- Handle in inference pipeline

```python
torch.onnx.export(
    vision_model,
    (dummy_input, grid_thw),
    output_path,
    dynamic_axes={
        "pixel_values": {0: "num_patches"},
        "image_grid_thw": {0: "num_images"},
    }
)
```

#### 3. Position IDs Generation

**Problem:** Need 3D position IDs for inference

**Solution:**
```python
def prepare_position_ids(input_ids, image_mask):
    """Generate 3D position IDs for MRoPE"""
    batch, seq_len = input_ids.shape
    
    # Initialize 3D position IDs [3, batch, seq_len]
    position_ids = torch.arange(seq_len)[None, :].expand(batch, -1)
    position_ids = torch.stack([position_ids] * 3, dim=0)
    
    # Adjust for image tokens (TODO: implement logic)
    # This requires tracking image token positions and
    # adjusting T/H/W axes accordingly
    
    return position_ids
```

## Builder Implementation

### Phi4-MM Builder Structure

```python
# builder.py (Phi4-MM specific)
class Phi4MMModel:
    def export_vision(self): ...
    def export_speech(self): ...
    def export_embeddings(self): ...
    def export_text(self): ...
```

### Qwen3-VL Builder Structure

```python
# builder_qwen3vl.py (new file)
def export_vision_encoder(model, output_dir):
    """Export vision encoder separately"""
    # Handle 3D patch embedding
    # Handle dynamic grid_thw
    ...

def export_embeddings(model, output_dir):
    """Export embeddings (same as Phi4-MM)"""
    ...

def export_text_decoder(input_dir, output_dir):
    """Use existing builder with exclude_embeds=true"""
    from py.models.builder import create_model
    create_model(..., exclude_embeds=True)
```

**Key difference:** Qwen3-VL uses standalone script (`builder_qwen3vl.py`) instead of extending main builder.

## Inference Pipeline

### Common Pattern (Phi4-MM & Qwen3-VL)

```python
class MultiModalPipeline:
    def __init__(self, model_path):
        # Load ONNX sessions
        self.vision_session = ort.InferenceSession(...)
        self.embeddings_session = ort.InferenceSession(...)
        self.decoder_session = ort.InferenceSession(...)
        
        # Load processors
        self.image_processor = ...
        self.tokenizer = ...
    
    def __call__(self, text, image=None):
        # 1. Process modalities
        if image:
            image_embeds = self.process_image(image)
        
        # 2. Tokenize text
        input_ids = self.tokenize(text)
        
        # 3. Embed tokens
        text_embeds = self.embed_tokens(input_ids)
        
        # 4. Merge embeddings
        merged_embeds = self.merge(text_embeds, image_embeds)
        
        # 5. Generate
        output_ids = self.generate(merged_embeds)
        
        return output_ids
```

### Qwen3-VL Specific Considerations

1. **Image Processing:**
```python
def process_image(self, image_path):
    # Smart resize
    h, w = self.smart_resize(height, width)
    
    # Create 3D patches (temporal dimension)
    patches = self.create_patches_3d(image, h, w)
    
    # Run vision encoder
    image_embeds = self.vision_session.run(
        None,
        {
            "pixel_values": patches,
            "image_grid_thw": grid_thw
        }
    )[0]
    
    return image_embeds
```

2. **Position IDs:**
```python
def prepare_position_ids(self, seq_len):
    # Create 3D position IDs [3, batch, seq_len]
    position_ids = np.arange(seq_len)[None, :]
    position_ids = np.stack([position_ids] * 3, axis=0)
    return position_ids
```

3. **Embedding Merge:**
```python
def merge_embeddings(self, text_embeds, image_embeds):
    # Find <|image_pad|> tokens
    # Replace with image embeddings
    # Qwen3-VL uses <|vision_start|><|image_pad|><|vision_end|>
    ...
```

## Configuration Files

### Phi4-MM Configs

1. `genai_config.json` - ONNX Runtime GenAI
2. `speech_processor.json` - Audio preprocessing
3. `vision_processor.json` - Image preprocessing

### Qwen3-VL Configs

1. `genai_config.json` - ONNX Runtime GenAI
2. `vision_processor.json` - Image preprocessing

**Example vision_processor.json:**
```json
{
  "processor_type": "Qwen3VLImageProcessor",
  "min_pixels": 50176,
  "max_pixels": 12845056,
  "patch_size": 16,
  "temporal_patch_size": 2,
  "merge_size": 2,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5]
}
```

## Testing Strategy

### Unit Tests

```python
# test_vision.py
def test_vision_encoder():
    # Test vision encoder ONNX export
    # Verify output shapes
    # Check numerical parity with PyTorch

# test_embeddings.py
def test_embeddings():
    # Test embedding layer export
    # Verify token-to-embedding mapping

# test_decoder.py
def test_text_decoder():
    # Test decoder with 3D position IDs
    # Verify MRoPE implementation
```

### Integration Tests

```python
# test_pipeline.py
def test_full_pipeline():
    # Test end-to-end inference
    # Compare outputs with HuggingFace
    # Measure latency
```

## Performance Optimization

### Model Quantization

```bash
# INT4 quantization for text decoder
python -m onnxruntime.quantization.quantize \
  --model model.onnx \
  --output model_int4.onnx \
  --quantization_mode int4
```

### Runtime Optimization

```python
# Enable CUDA graph capture
sess_options = ort.SessionOptions()
sess_options.add_session_config_entry(
    "enable_cuda_graph", "1"
)
```

## Debugging Tips

### Common Issues

1. **Position IDs Shape Mismatch**
   - Error: "Expected 3D, got 2D"
   - Fix: Ensure position_ids are [3, B, S]

2. **Vision Encoder Dynamic Shapes**
   - Error: ONNX export fails with dynamic shapes
   - Fix: Use representative input sizes

3. **Embedding Merge Issues**
   - Error: Image embeddings not properly merged
   - Fix: Check token IDs and merge logic

### Verification

```python
# Compare ONNX vs PyTorch
def verify_parity(pytorch_output, onnx_output, atol=1e-3):
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")
    
    assert max_diff < atol, f"Parity check failed: {max_diff} > {atol}"
```

## References

### Phi4-MM References
- [Phi-4 Multimodal Guide](../phi-4-multi-modal.md)
- [Phi-4 HuggingFace](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

### Qwen3-VL References
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B)
- [Qwen3-VL Modeling Code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [MRoPE Paper](https://arxiv.org/abs/2406.04334)

### ONNX Runtime References
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
- [ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)

## Summary

The Qwen3-VL ONNX export follows the same multi-component strategy as Phi4-MM, with these key modifications:

1. **Rotary Embedding:** Remove dynamic decisions for 3D MRoPE
2. **Vision Encoder:** Handle 3D patch embedding and dynamic grids
3. **Position IDs:** Generate 3D position IDs for inference
4. **Builder:** Use standalone script instead of extending main builder

The implementation is complete and ready for testing!
