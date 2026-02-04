# Qwen3-VL ONNX Model Builder Guide

This guide walks you through exporting the Qwen3-VL-4B-Instruct model to ONNX format for use with ONNX Runtime.

## Overview

The builder exports three separate ONNX models:
1. **Vision Encoder** (FP32) - Processes images into vision embeddings
2. **Embedding Layer** (FP32) - Converts token IDs to text embeddings
3. **Text Decoder** (FP32 or INT4) - Autoregressive language model with KV cache

## Prerequisites

### Environment Setup

1. **Conda Environment**:
```bash
conda create -n onnxruntime-genai python=3.10
conda activate onnxruntime-genai
```

2. **Install Dependencies**:
```bash
pip install onnxruntime-genai
pip install transformers torch torchvision pillow numpy
```

3. **Verify Installation**:
```bash
python -c "import onnxruntime_genai; print('ONNX Runtime GenAI:', onnxruntime_genai.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

## Step 1: Download PyTorch Model

Download the Qwen3-VL-4B-Instruct model from Hugging Face:

```bash
cd examples/python/qwen3-vl-4b

# Option A: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir pytorch

# Option B: Using Python
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', trust_remote_code=True)
model.save_pretrained('pytorch')
processor.save_pretrained('pytorch')
"
```

**Expected structure**:
```
pytorch/
├── config.json
├── modeling_qwen3_vl.py
├── modular_qwen3_vl.py
├── processing_qwen3_vl.py
├── model-*.safetensors
└── ... (other files)
```

## Step 2: Apply ONNX-Compatible Modifications

Copy the modified `modeling_qwen3_vl.py` from `pytorch_modified/` to your `pytorch/` directory:

```bash
# Backup original file (optional but recommended)
cp pytorch/modeling_qwen3_vl.py pytorch/modeling_qwen3_vl.py.original

# Copy modified version
cp pytorch_modified/modeling_qwen3_vl.py pytorch/modeling_qwen3_vl.py
```

### What Was Modified?

The key modification is in `Qwen3VLVisionRotaryEmbedding` class:

**Original** (dynamic shapes):
```python
def forward(self, seqlen: int) -> torch.Tensor:
    seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    freqs = torch.outer(seq, self.inv_freq)  # Dynamic operation
    return freqs
```

**Modified** (static shapes):
```python
def __init__(self, dim: int, theta: float = 10000.0, max_seqlen: int = 1024):
    super().__init__()
    # Precompute frequency table for ONNX export
    seq = torch.arange(max_seqlen, dtype=torch.float)
    freqs = torch.outer(seq, inv_freq)
    self.register_buffer("freq_table", freqs, persistent=False)

def forward(self, seqlen: int) -> torch.Tensor:
    # Use precomputed table (static shape)
    return self.freq_table[:seqlen]
```

**Why?** ONNX export fails with dynamic tensor operations that depend on runtime values. Precomputing the frequency table avoids `GuardOnDataDependentSymNode` errors.

## Step 3: Test PyTorch Pipeline (Recommended)

Before exporting to ONNX, verify the modified model works correctly:

```bash
python test_pytorch_pipeline.py
```

**Expected output**:
- Model loads successfully
- Test cases run without errors
- Generated text is coherent
- Output saved to `pytorch_test_outputs.txt`

**Example test cases**:
```
Test 1: Simple greeting
Test 2: Question answering
Test 3: Short story generation
Test 4: Code generation
```

If any test fails, do not proceed with ONNX export. Check that:
- Modified file was copied correctly
- PyTorch model loads with `trust_remote_code=True`
- All dependencies are installed

## Step 4: Run ONNX Export

### Option A: FP32 Text Model (Better Quality)

```bash
python builder_simple.py -i pytorch --text_precision fp32
```

**Output structure**:
```
cpu-fp32/
├── qwen3vl-vision.onnx         (~1.9 GB)
├── qwen3vl-embedding.onnx      (~600 MB)
├── model.onnx                  (~9.5 GB)
├── genai_config.json
├── tokenizer files...
```

### Option B: INT4 Text Model (Smaller, Faster)

```bash
python builder_simple.py -i pytorch --text_precision int4
```

**Output structure**:
```
cpu-fp32/
├── qwen3vl-vision.onnx         (~1.9 GB)
├── qwen3vl-embedding.onnx      (~600 MB)

cpu-int4/
├── model.onnx                  (~2.5 GB)
├── model.onnx.data            (~2.5 GB)
├── genai_config.json
├── tokenizer files...
```

**Export time**: 5-15 minutes depending on hardware

### Builder Arguments

```bash
python builder_simple.py -h

Arguments:
  -i, --input PATH          Path to PyTorch model directory (required)
  -p, --precision {fp16,fp32}  Precision for vision/embedding (default: fp32)
  -t, --text_precision {fp32,int4}  Precision for text model (default: int4)
  -e, --execution_provider {cpu,cuda,dml}  Target device (default: cpu)
  -c, --cache_dir PATH      Cache directory (default: ./cache_dir)
```

## Step 5: Verify Exported Models

Check that all required files exist:

```bash
# For FP32 text model
ls -lh cpu-fp32/*.onnx
ls cpu-fp32/genai_config.json

# For INT4 text model
ls -lh cpu-int4/model.onnx*
ls cpu-int4/genai_config.json
```

**File sizes** (approximate):
- `qwen3vl-vision.onnx`: 1.9 GB
- `qwen3vl-embedding.onnx`: 600 MB
- `model.onnx` (FP32): 9.5 GB
- `model.onnx` (INT4): 2.5 GB (+ 2.5 GB .data file)

## Troubleshooting

### Error: "GuardOnDataDependentSymNode"
- **Cause**: Using original `modeling_qwen3_vl.py` instead of modified version
- **Fix**: Ensure modified file is copied to `pytorch/` directory (Step 2)

### Error: "AttributeError: 'Qwen3VLConfig' object has no attribute 'rope_theta'"
- **Cause**: Missing `rope_theta` in config.json
- **Fix**: Add `"rope_theta": 5000000` to `pytorch/config.json` at root level

### Error: "conversion of scaled_dot_product_attention not implemented"
- **Cause**: SDPA attention not supported for ONNX export
- **Fix**: Builder automatically sets `_attn_implementation = 'eager'` - this is normal

### Vision export warnings about "Iterating over a tensor"
- **Cause**: TracerWarnings from dynamic operations in vision model
- **Impact**: Non-critical warnings, export still succeeds
- **Note**: These are expected and do not affect functionality

### Out of Memory during export
- **Fix**: Close other applications, or use smaller batch sizes
- **Alternative**: Export text model with INT4 quantization instead of FP32

## Next Steps

After successful export:
1. Test text-only inference: See `QWEN3VL_PIPELINE.md`
2. Test with your own prompts
3. Experiment with generation parameters (temperature, top_k, top_p)

## Technical Details

### Export Strategy

**Vision Model**:
- Fixed input size: 384×384 pixels (576 patches)
- Patch size: 16×16, temporal patch: 2
- Direct `torch.onnx.export` with eager attention
- Output: Pooled embeddings ready for LLM

**Embedding Model**:
- Simple text embedding layer export
- Input: Token IDs [batch, seq_len]
- Output: Embeddings [batch, seq_len, hidden_size]

**Text Model**:
- Uses `onnxruntime_genai.models.builder`
- Excludes embeddings (exported separately)
- Supports FP32 or INT4-RTN quantization
- Includes KV cache for efficient generation
- 3D position IDs for MRoPE (Multimodal Rotary Position Embedding)

### Model Architecture

```
Qwen3-VL-4B-Instruct:
- Vision: ViT-style encoder with 3D rotary embeddings
- Text: 36-layer transformer decoder
- Vocabulary: 151,936 tokens
- Hidden size: 2,560
- Attention heads: 20 (query), 8 (key/value) - GQA
- Context length: 32,768 tokens
```

## Reference Files

- `builder_simple.py` - Main export script (385 lines)
- `pytorch_modified/modeling_qwen3_vl.py` - Modified model with ONNX fixes
- `test_pytorch_pipeline.py` - PyTorch validation script
- `QWEN3VL_PIPELINE.md` - Inference usage guide
