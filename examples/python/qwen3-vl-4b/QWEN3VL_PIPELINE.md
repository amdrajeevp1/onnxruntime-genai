# Qwen3-VL ONNX Inference Pipeline Guide

This guide explains how to use the exported Qwen3-VL ONNX models for inference.

## Overview

The Qwen3-VL ONNX pipeline provides two inference scripts:

1. **`qwen3-vl-text.py`** - Text-only inference (simpler, faster for pure text tasks)
2. **`qwen3-vl.py`** - Full multimodal pipeline (text-only AND vision+text)

Both scripts support:
- ✅ FP32 and INT4 text models
- ✅ Streaming token generation
- ✅ Customizable sampling parameters
- ✅ Qwen3 architecture with standard RoPE

## Prerequisites

1. **Exported ONNX models** - Follow `MODEL_BUILDER.md` to export models
2. **Required packages**:
```bash
pip install onnxruntime transformers numpy
```

3. **Directory structure**:
```
qwen3-vl-4b/
├── cpu-fp32/                    # Vision, embedding, and optionally text models
│   ├── qwen3vl-vision.onnx     # Vision encoder (1.9 GB)
│   ├── qwen3vl-embedding.onnx  # Token embeddings (600 MB)
│   ├── model.onnx              # FP32 text decoder (9.5 GB)
│   └── genai_config.json
├── cpu-int4/                    # Optional: INT4 text model
│   ├── model.onnx              # INT4 text decoder (2.5 GB)
│   ├── model.onnx.data         # INT4 weights
│   └── genai_config.json
├── pytorch/                     # Tokenizer and processor files
│   ├── tokenizer.json
│   ├── preprocessor_config.json
│   └── ...
├── images/                      # Test images
│   ├── test_colors.jpg
│   ├── test_checkerboard.jpg
│   └── test_gradient_genai.jpg
├── qwen3-vl-text.py            # Text-only inference script
└── qwen3-vl.py                 # Multimodal inference script
```

## Script 1: Text-Only Inference (`qwen3-vl-text.py`)

Use this script for pure text generation (no vision). It's simpler and faster for text-only tasks.

### Quick Start

**Single prompt** (FP32 text model):
```bash
python qwen3-vl-text.py \
  --text_precision fp32 \
  --prompt "What is the capital of France?" \
  --max_new_tokens 50
```

**Single prompt** (INT4 text model):
```bash
python qwen3-vl-text.py \
  --text_precision int4 \
  --prompt "Explain quantum computing in simple terms." \
  --max_new_tokens 100
```

**Interactive mode**:
```bash
python qwen3-vl-text.py --text_precision fp32 --interactive
```

### Command-Line Arguments

```bash
python qwen3-vl-text.py --help

Arguments:
  --model_dir PATH              Model directory (default: current directory)
  --text_precision {fp32,int4}  Text model precision (default: fp32)
  --prompt TEXT                 Input prompt (if not provided, uses default or interactive)
  --max_new_tokens INT          Maximum tokens to generate (default: 100)
  --temperature FLOAT           Sampling temperature, 0.0=greedy (default: 0.7)
  --top_k INT                   Top-k sampling (default: 20)
  --top_p FLOAT                 Top-p (nucleus) sampling (default: 0.8)
  --no_sample                   Use greedy decoding instead of sampling
  --interactive                 Interactive mode (keep prompting)
```

### Usage Examples

#### Example 1: Question Answering

```bash
python qwen3-vl-text.py \
  --text_precision fp32 \
  --prompt "Who wrote Romeo and Juliet?" \
  --max_new_tokens 30 \
  --temperature 0.1
```

**Expected behavior**: Low temperature produces focused, deterministic answers.

#### Example 2: Creative Writing

```bash
python qwen3-vl-text.py \
  --text_precision fp32 \
  --prompt "Write a short story about a robot learning to paint." \
  --max_new_tokens 200 \
  --temperature 0.9 \
  --top_p 0.95
```

**Expected behavior**: Higher temperature and top_p enable more creative, diverse outputs.

#### Example 3: Code Generation

```bash
python qwen3-vl-text.py \
  --text_precision fp32 \
  --prompt "Write a Python function to calculate fibonacci numbers:" \
  --max_new_tokens 150 \
  --temperature 0.2
```

#### Example 4: Interactive Chat

```bash
python qwen3-vl-text.py --text_precision fp32 --interactive

# Then type prompts interactively:
Prompt: Hello, how are you?
Prompt: Tell me a joke about programming.
Prompt: quit
```

### Sampling Parameters Explained

**Temperature** (`--temperature`):
- `0.0`: Greedy decoding (deterministic, picks most likely token)
- `0.1-0.3`: Focused, factual responses
- `0.7`: Balanced (default)
- `0.9-1.0`: Creative, diverse outputs
- `>1.0`: Very random, potentially incoherent

**Top-k** (`--top_k`):
- Limits vocabulary to k most likely tokens
- `20`: Conservative (default)
- `50`: Moderate diversity
- `0`: Disabled (use all tokens)

**Top-p** (`--top_p`):
- Nucleus sampling, selects tokens with cumulative probability p
- `0.8`: Focused (default)
- `0.9-0.95`: More diverse
- `1.0`: Disabled (use all tokens)

**Best practices**:
- Factual Q&A: `temperature=0.1, top_k=10, top_p=0.9`
- Creative writing: `temperature=0.9, top_k=50, top_p=0.95`
- Code generation: `temperature=0.2, top_k=20, top_p=0.9`

## FP32 vs INT4 Comparison

### FP32 Text Model

**Pros**:
- Higher quality outputs (especially for complex tasks)
- No quantization artifacts
- Better numerical precision

**Cons**:
- Larger file size: ~9.5 GB
- Slower inference
- Higher memory usage

**Use cases**: Quality-critical applications, research, benchmarking

### INT4 Text Model

**Pros**:
- Much smaller: ~2.5 GB (4x reduction)
- Faster inference
- Lower memory footprint

**Cons**:
- Slight quality degradation (may produce gibberish in some cases)
- Requires tuning for stability
- Not recommended for production without thorough testing

**Use cases**: Resource-constrained environments, mobile deployment, rapid prototyping

**Current Status**: INT4 model exports successfully but generates low-quality output. FP32 is recommended for testing.

## Performance Benchmarks

Approximate generation speeds (CPU, Intel i7):

| Model | Tokens/sec | Memory (GB) | Quality |
|-------|-----------|-------------|---------|
| FP32  | 2-3       | ~12         | Good    |
| INT4  | 4-6       | ~6          | Poor*   |

*INT4 quality issues under investigation

## Troubleshooting

### Issue: Gibberish Output

**Symptoms**: Model generates random tokens, special characters, or nonsense.

**Possible causes**:
1. Using INT4 model (known issue)
2. Wrong position IDs encoding
3. KV cache corruption

**Solutions**:
- Use FP32 model instead: `--text_precision fp32`
- Verify exported models are valid (re-export if needed)
- Check `genai_config.json` has correct model configuration

### Issue: UnicodeEncodeError on Windows

**Symptom**: Crash when printing non-ASCII characters.

**Fix**: Script automatically handles UTF-8 encoding, but if issues persist:
```bash
# Set console encoding before running
chcp 65001
python qwen3-vl-text.py --text_precision fp32 --prompt "Hello"
```

### Issue: Model Not Found

**Symptom**: `FileNotFoundError` for `model.onnx` or `qwen3vl-embedding.onnx`.

**Fix**: Ensure models are exported correctly:
```bash
ls cpu-fp32/qwen3vl-embedding.onnx
ls cpu-fp32/model.onnx  # or cpu-int4/model.onnx
```

If missing, re-run the builder (see `MODEL_BUILDER.md`).

### Issue: Slow Generation

**Possible causes**:
- Large model on CPU
- First token prefill is slower (normal)
- Insufficient RAM causing swapping

**Solutions**:
- Use INT4 model for faster inference
- Reduce `--max_new_tokens`
- Close other applications
- Consider GPU inference (if available)

## Vision + Text Inference (TBD)

### Current Status

Vision models are exported successfully:
- `qwen3vl-vision.onnx` (1.9 GB)
- Fixed 384×384 input processing

### Planned Features

1. **Image preprocessing**: Resize and normalize images to 384×384
2. **Vision embeddings**: Extract features from images
3. **Multimodal fusion**: Merge vision and text embeddings
4. **Position encoding**: 3D position IDs for image tokens
5. **Full pipeline**: End-to-end image + text generation

### Why Not Implemented Yet?

The multimodal pipeline requires:
- Proper vision token injection into text embeddings
- Grid-based 3D position IDs for image patches
- Vision-text alignment in embedding space

Current focus: Stabilize text-only pipeline before adding vision.

### Workaround

For now, use PyTorch model for vision + text:
```bash
python test_pytorch_pipeline.py
```

## Technical Details

### Pipeline Architecture

```
Input Text → Tokenizer → Token IDs → Embedding Model → Text Embeddings
                                                              ↓
Text Embeddings → Text Model (Autoregressive) → Logits → Sampling → Output Tokens
    ↑                                                                      ↓
KV Cache ←────────────────────────────────────────────────────────── Tokenizer → Text
```

### 3D Position IDs (MRoPE)

Qwen3-VL uses Multimodal Rotary Position Embedding (MRoPE):
- **Dimension 0**: Temporal (video frames)
- **Dimension 1**: Height (image rows)
- **Dimension 2**: Width (image columns)

For text-only, all dimensions use sequential positions:
```python
position_ids = [
    [0, 1, 2, 3, ...],  # Temporal
    [0, 1, 2, 3, ...],  # Height
    [0, 1, 2, 3, ...]   # Width
]
```

### KV Cache Management

The text model uses key-value caching for efficient generation:
- **Initial forward pass**: Process full prompt, populate cache
- **Subsequent steps**: Process one token at a time, reuse cache
- **Cache shape**: `[batch, num_kv_heads, seq_len, head_dim]`
- **Updates**: Concatenate new keys/values to existing cache

### Model Inputs/Outputs

**Vision Model** (`qwen3vl-vision.onnx`):
- Inputs:
  - `pixel_values` [num_patches, 1536] (float32)
  - `image_grid_thw` [num_images, 3] (int64) - temporal/height/width grid
- Output:
  - `pooled_embeds` [num_merged_patches, hidden_size] (float32)

**Embedding Model** (`qwen3vl-embedding.onnx`):
- Input: `input_ids` [batch, seq_len] (int64)
- Output: `inputs_embeds` [batch, seq_len, hidden_size] (fp32)

**Text Model** (`model.onnx`):
- Inputs:
  - `inputs_embeds` [batch, seq_len, hidden_size] (fp32)
  - `attention_mask` [batch, total_seq_len] (int64)
  - `past_key_values.{i}.key` [batch, num_kv_heads, past_len, head_dim] (fp32)
  - `past_key_values.{i}.value` [batch, num_kv_heads, past_len, head_dim] (fp32)
  - **Note**: No `position_ids` input - Qwen3 computes positions internally with standard RoPE
- Outputs:
  - `logits` [batch, seq_len, vocab_size] (fp32)
  - `present.{i}.key` [batch, num_kv_heads, total_len, head_dim] (fp32)
  - `present.{i}.value` [batch, num_kv_heads, total_len, head_dim] (fp32)

## Advanced Usage

### Batch Processing

Process multiple prompts efficiently:

```python
from qwen3_vl_text import Qwen3VLTextOnlyPipeline

pipeline = Qwen3VLTextOnlyPipeline(text_precision="fp32")

prompts = [
    "What is AI?",
    "Explain machine learning.",
    "Define neural networks."
]

for prompt in prompts:
    output = pipeline.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7
    )
    print(f"Q: {prompt}")
    print(f"A: {output}\n")
```

### Custom Sampling

Implement custom sampling strategies:

```python
# Edit qwen3-vl-text.py and modify sample_token() method
# Examples: beam search, constrained decoding, etc.
```

### Profiling

Measure generation performance:

```bash
# Windows
python -m cProfile -o profile.stats qwen3-vl-text.py --prompt "Test" --max_new_tokens 50
python -m pstats profile.stats

# Linux/Mac
time python qwen3-vl-text.py --prompt "Test" --max_new_tokens 50
```

## Next Steps

1. **Experiment**: Try different prompts and parameters
2. **Fine-tune**: Adjust sampling parameters for your use case
3. **Integrate**: Build applications with the ONNX pipeline
4. **Contribute**: Help improve the vision + text pipeline

## Testing Both Scripts

### Test 1: Text-Only Generation

```bash
# Using qwen3-vl-text.py (simpler, text-only)
python qwen3-vl-text.py \
  --text_precision fp32 \
  --prompt "Explain quantum entanglement in simple terms." \
  --max_new_tokens 100

# Using qwen3-vl.py (multimodal, but works for text too)
python qwen3-vl.py \
  --text_precision fp32 \
  --text "Explain quantum entanglement in simple terms." \
  --max_new_tokens 100
```

Both should produce similar outputs.

### Test 2: Multimodal Generation

```bash
# Using qwen3-vl.py with test images
python qwen3-vl.py \
  --image images/test_colors.jpg \
  --text "What colors are in this image?" \
  --max_new_tokens 50 \
  --text_precision fp32

python qwen3-vl.py \
  --image images/test_checkerboard.jpg \
  --text "Describe the pattern." \
  --max_new_tokens 80 \
  --text_precision fp32

python qwen3-vl.py \
  --image images/test_gradient_genai.jpg \
  --text "What do you see?" \
  --max_new_tokens 100 \
  --text_precision fp32
```

### Expected Results

**Text-only**: Coherent, factual responses about the topic
**Vision+text**: Accurate descriptions of image content (colors, patterns, structures)

## Reference Files

- `qwen3-vl-text.py` - Text-only inference script (408 lines)
- `qwen3-vl.py` - Full multimodal inference script (563 lines)
- `MODEL_BUILDER.md` - Step-by-step export guide
- `builder_simple.py` - Model export script
- `test_pytorch_pipeline.py` - PyTorch reference implementation
- `images/` - Test images for vision pipeline

## Known Limitations

1. **INT4 quality**: FP32 recommended for production (INT4 under investigation)
2. **Context length**: Limited testing beyond 2048 tokens
3. **Performance**: CPU-only, no GPU optimization yet
4. **Batch size**: Fixed to 1, no dynamic batching
5. **Video support**: Not yet implemented (images only)

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review `MODEL_BUILDER.md` for export issues
3. Examine `builder_simple.py` and `qwen3-vl-text.py` source code
4. Test with PyTorch model first (`test_pytorch_pipeline.py`)
