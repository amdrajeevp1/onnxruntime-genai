# Qwen3-VL-4B ONNX Export Setup Guide

## Overview
This guide will help you export Qwen3-VL-4B to ONNX format for use with ONNX Runtime GenAI.

## Directory Structure
```
qwen3-vl-4b/
├── pytorch/                    # Original PyTorch model (already downloaded)
│   ├── config.json
│   ├── *.safetensors
│   └── tokenizer files
├── pytorch_modified/           # Modified PyTorch files for ONNX export
│   ├── modeling_qwen3_vl.py   # Modified modeling file
│   ├── processing_qwen3_vl.py # Copied from HF
│   └── configuration_qwen3_vl.py # Copied from HF
├── builder_qwen3vl.py         # Builder script for export
├── test_qwen3vl_inference.py  # Inference pipeline test
└── cpu/                        # Output ONNX models (after export)

## Step 1: Copy HuggingFace Model Files Locally

Run the following script to download the necessary modeling files from HuggingFace:

```bash
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b
python copy_hf_files.py
```

This will:
1. Download modeling_qwen3_vl.py
2. Download processing_qwen3_vl.py  
3. Download configuration_qwen3_vl.py
4. Download video_processing_qwen3_vl.py
5. Save them to pytorch_modified/ directory

## Step 2: Modify Rotary Embedding for ONNX Export

The key modification is in `Qwen3VLTextRotaryEmbedding.forward()` to remove dynamic decisions:

**Original HF Code (lines 269-286 in modular_qwen3_vl.py):**
```python
@torch.no_grad()
@dynamic_rope_update  # Dynamic behavior - not ONNX friendly!
def forward(self, x, position_ids):
    # Dynamic expansion based on position_ids dimensions
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    # ... rest of implementation
```

**Modified Code for ONNX:**
```python
@torch.no_grad()
def forward(self, x, position_ids):
    # Assume position_ids is always 3D [3, batch_size, seq_len]
    # Remove dynamic_rope_update decorator
    # Remove conditional expansion
    assert position_ids.ndim == 3, "position_ids must be 3D for ONNX export"
    # ... rest of implementation
```

The script `modify_rotary_embedding.py` will handle this automatically.

## Step 3: Export Models with Builder

### 3.1 Export Text Model Only (for testing)

```bash
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1

python -m src.python.py.models.builder ^
  --input examples\python\qwen3-vl-4b\pytorch ^
  --output examples\python\qwen3-vl-4b\cpu-text ^
  --precision fp32 ^
  --execution_provider cpu ^
  --extra_options exclude_embeds=true
```

This exports only the text/decoder component.

### 3.2 Export Vision + Embedding + Text (Full Pipeline)

Use the custom builder script:

```bash
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b

python builder_qwen3vl.py ^
  --input ./pytorch ^
  --output ./cpu ^
  --precision fp32 ^
  --execution_provider cpu
```

This will export:
1. `vision_encoder.onnx` - Vision model (patch embed → transformer → merger)
2. `embeddings.onnx` - Token embedding layer
3. `model.onnx` - Text decoder (with inputs_embeds instead of input_ids)

## Step 4: Create Processor Configuration

Create `vision_processor.json` in the output directory with image preprocessing parameters:

```json
{
  "processor_type": "Qwen3VLImageProcessor",
  "min_pixels": 50176,
  "max_pixels": 12845056,
  "patch_size": 16,
  "temporal_patch_size": 2,
  "merge_size": 2,
  "spatial_factor": 32,
  "temporal_factor": 2,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5]
}
```

## Step 5: Run Inference Pipeline

Test the exported models:

```bash
python test_qwen3vl_inference.py ^
  --model_path ./cpu ^
  --image_path ./test_images/sample.jpg ^
  --prompt "Describe this image"
```

## Key Modifications Summary

### 1. Rotary Embedding (modeling_qwen3_vl.py)

- **Class**: `Qwen3VLTextRotaryEmbedding`
- **Method**: `forward()`
- **Changes**:
  - Remove `@dynamic_rope_update` decorator
  - Remove dynamic `position_ids.ndim == 2` check
  - Assume position_ids is always 3D: `[3, batch_size, seq_len]`
  - Remove conditional expansion logic

### 2. Vision Model Export (builder_qwen3vl.py)

Export the vision model separately:
- Input: `pixel_values` [batch, channels, height, width]
- Output: `image_embeds` [batch*seq_len, hidden_dim]

### 3. Embedding Export (builder_qwen3vl.py)

Export the embedding layer:
- Input: `input_ids` [batch, seq_len]
- Output: `inputs_embeds` [batch, seq_len, hidden_dim]

### 4. Text Model Export (existing builder)

Already supported via `Qwen3VLTextModel` class:
- Input: `inputs_embeds` [batch, seq_len, hidden_dim]
- Output: `logits` [batch, seq_len, vocab_size]

## Architecture Overview

```
Image → Vision Encoder → image_embeds
                             ↓
Text → Tokenizer → input_ids → Embedding → text_embeds
                                               ↓
                                          Merge embeds
                                               ↓
                                          Text Decoder → logits
```

## Troubleshooting

### Issue 1: "dynamic_rope_update not found"
- **Cause**: Using wrong transformers version
- **Fix**: Install transformers from source or use modified files

### Issue 2: "position_ids must be 3D"
- **Cause**: Not preprocessing position_ids correctly
- **Fix**: Ensure position_ids has shape [3, B, S] before passing to model

### Issue 3: "Vision model export fails"
- **Cause**: Dynamic shapes in vision encoder
- **Fix**: Use fixed image sizes during export or add shape constraints

## References

1. [Phi-4 Multimodal Guide](../phi-4-multi-modal.md) - Reference implementation
2. [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B) - Original model
3. [ONNX Runtime GenAI Docs](https://github.com/microsoft/onnxruntime-genai)

## Next Steps

After successful export:
1. Optimize ONNX models with ONNX Runtime
2. Quantize to INT4 for better performance
3. Test on CUDA/DirectML if needed
4. Deploy with ONNX Runtime GenAI API
