# Qwen3-VL-4B ONNX Export

This directory contains scripts and documentation for exporting Qwen3-VL-4B to ONNX format for use with ONNX Runtime GenAI.

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the master setup script:

```bash
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b
python setup_qwen3vl.py
```

This will automatically:
1. Download HuggingFace model files
2. Modify them for ONNX export
3. Export vision encoder, embeddings, and text decoder
4. Create processor configurations

### Option 2: Manual Setup

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for detailed step-by-step instructions.

## Directory Structure

```
qwen3-vl-4b/
├── README.md                      # This file
├── SETUP_GUIDE.md                 # Detailed setup guide
├── setup_qwen3vl.py              # Master setup script
├── copy_hf_files.py              # Download HF files
├── modify_rotary_embedding.py    # Modify for ONNX export
├── builder_qwen3vl.py            # ONNX export builder
├── test_qwen3vl_inference.py     # Inference pipeline test
├── pytorch/                       # Original PyTorch model
│   ├── config.json
│   ├── *.safetensors
│   └── tokenizer files
├── pytorch_modified/              # Modified files for ONNX
├── cpu/                           # Exported ONNX models (FP32)
├── cuda/                          # Exported ONNX models (FP16 CUDA)
└── md-files/                      # Documentation from experiments
```

## Exported Models

After running the export, you will have:

1. **vision_encoder.onnx** - Vision model
   - Input: `pixel_values` [num_patches, features]
   - Input: `image_grid_thw` [num_images, 3]
   - Output: `image_embeds` [num_patches, hidden_dim]

2. **embeddings.onnx** - Token embedding layer
   - Input: `input_ids` [batch, seq_len]
   - Output: `inputs_embeds` [batch, seq_len, hidden_dim]

3. **model.onnx** - Text decoder
   - Input: `inputs_embeds` [batch, seq_len, hidden_dim]
   - Input: `position_ids` [3, batch, seq_len]
   - Output: `logits` [batch, seq_len, vocab_size]

4. **vision_processor.json** - Image preprocessing config
5. **genai_config.json** - ONNX Runtime GenAI config
6. **tokenizer files** - Copied from original model

## Usage

### Testing Inference

```bash
python test_qwen3vl_inference.py \
  --model_path ./cpu \
  --image_path ./test_images/sample.jpg \
  --prompt "Describe this image"
```

### Using in Your Application

```python
from test_qwen3vl_inference import Qwen3VLPipeline

# Create pipeline
pipeline = Qwen3VLPipeline("./cpu")

# Run inference
output_ids = pipeline(
    prompt="What's in this image?",
    image_path="image.jpg"
)
```

## Key Modifications

The main modification for ONNX export is in the rotary embedding:

**Original (HuggingFace):**
```python
@torch.no_grad()
@dynamic_rope_update  # Dynamic behavior
def forward(self, x, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, ...)
    # ...
```

**Modified (ONNX):**
```python
@torch.no_grad()
def forward(self, x, position_ids):
    # Assume position_ids is always 3D
    assert position_ids.ndim == 3
    # ...
```

This removes dynamic decisions that prevent ONNX export.

## Architecture

Qwen3-VL is a multimodal model with:

- **Vision Encoder**: 24-layer transformer (1024 hidden, 16 heads)
  - Processes images via 3D patch embedding
  - Uses 2D rotary embeddings
  - Outputs merged patches (2x2 spatial merge)

- **Text Decoder**: 28-layer transformer (2560 hidden, 20 heads)
  - Uses 3D rotary embeddings (MRoPE)
  - MRoPE sections: [24, 20, 20] for temporal/height/width
  - Accepts merged text + vision embeddings

## Troubleshooting

### Issue: Import Error for transformers

Make sure you have the latest transformers:
```bash
pip install --upgrade transformers
```

Or install from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

### Issue: ONNX Export Fails

Check that:
1. Rotary embedding was modified correctly
2. position_ids has 3D shape [3, batch, seq_len]
3. PyTorch version is compatible (2.0+)

### Issue: Vision Encoder Export Fails

The vision encoder uses dynamic shapes. You may need to:
1. Use fixed image sizes during export
2. Add shape constraints
3. Export with example inputs

## Performance

Expected model sizes:
- Vision Encoder: ~400MB (FP32)
- Embeddings: ~150MB (FP32)
- Text Decoder: ~8GB (FP32)

For better performance:
- Use FP16 precision on CUDA/DirectML
- Apply INT4 quantization to text decoder
- Enable CUDA/WebGPU graph capture

## References

- [Qwen3-VL Paper](https://arxiv.org/abs/2501.XXXXX)
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
- [Phi4-MM Reference](../phi-4-multi-modal.md)

## License

Same as the original Qwen3-VL model and ONNX Runtime GenAI.

## Contributing

If you find issues or improvements:
1. Document them in md-files/
2. Update this README
3. Share with the team
