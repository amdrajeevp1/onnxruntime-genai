# Qwen3-VL-4B Vision Encoder ONNX Export - SUCCESS! 🎉

## Overview

Successfully exported the **Qwen3-VL-4B vision encoder** to ONNX format for CPU execution!

**Date**: January 29, 2026  
**Model**: Qwen/Qwen3-VL-4B-Instruct (Vision component)  
**Export Time**: 85 seconds  
**Status**: ✅ COMPLETE

---

## Exported Files

```
cpu/
├── qwen3-vl-vision.onnx (900 KB)
└── qwen3-vl-vision.onnx.data (1.66 GB)
```

**Total Size**: ~1.66 GB

---

## Model Architecture

### Qwen3-VL Vision Encoder

**Structure**:
- **Type**: Qwen3VLVisionModel
- **Base**: Vision Transformer (ViT) with DeepStack
- **Layers**: 24 transformer blocks
- **Hidden Size**: 1024
- **Attention Heads**: 16
- **Patch Size**: 16x16
- **Output Size**: 2560 (matches text model hidden size)

**Key Components**:
1. `patch_embed`: Qwen3VLVisionPatchEmbed - Converts images to patches
2. `pos_embed`: Embedding - Positional embeddings
3. `rotary_pos_emb`: Qwen3VLVisionRotaryEmbedding - RoPE for vision
4. `blocks`: 24 transformer blocks with attention + MLP
5. `merger`: Qwen3VLVisionPatchMerger - Merges patches for text model
6. `deepstack_merger_list`: 3 mergers at layers [5, 11, 17] - Multi-level feature extraction

**DeepStack Innovation**:
Extracts features at multiple depths (layers 5, 11, 17) to capture both:
- **Low-level details** (early layers): Edges, textures, colors
- **High-level semantics** (deep layers): Objects, scenes, concepts

---

## Export Process

### Prerequisites

```bash
cd c:\Users\rajeevp\Documents\onnxruntime-genai-1\examples\python\qwen3-vl-4b
conda activate onnxruntime-genai
```

### Download Model

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./pytorch
```

**Downloaded**: ~8.8 GB in 2 parts (6 minutes)

### Export Command

```bash
python builder_vision.py \
  --input ./pytorch \
  --output ./cpu \
  --precision fp32 \
  --execution_provider cpu \
  --device cpu
```

**Process**:
1. Load model with `attn_implementation="eager"` (disable SDPA)
2. Process test images through Qwen3VL processor
3. Extract vision encoder module
4. Export to ONNX with TorchScript mode (`dynamo=False`)
5. Validate and save with external data

---

## Key Technical Fixes

### 1. **Dual Input Architecture**

**Discovery**: Qwen3-VL vision encoder requires **two inputs**:
```python
vision_encoder(pixel_values, grid_thw)
```

- `pixel_values`: Flattened image patches (num_patches, hidden_size)  
- `grid_thw`: Grid dimensions (num_images, 3) - temporal, height, width

### 2. **SDPA Compatibility Issue**

**Problem**: `scaled_dot_product_attention` not supported with GQA in ONNX export

**Solution**: Load model with eager attention:
```python
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation="eager"  # Disable SDPA for ONNX
)
```

### 3. **TorchScript Export Mode**

Used `dynamo=False` for better compatibility with vision transformers:
```python
torch.onnx.export(
    vision_encoder,
    args=(pixel_values, grid_thw),
    dynamo=False,  # TorchScript mode
    opset_version=14
)
```

---

## ONNX Model Specification

**Inputs**:
1. `pixel_values`: shape=[num_patches, 1536], dtype=float32
   - Flattened image patches after preprocessing
2. `grid_thw`: shape=[num_images, 3], dtype=int64
   - Grid dimensions: [temporal, height, width] for each image

**Output**:
1. `vision_features`: shape=[num_patches, 2560], dtype=float32
   - Vision features ready for text model consumption

**Dynamic Axes**:
- `pixel_values`: dim 0 (num_patches)
- `grid_thw`: dim 0 (num_images)
- `vision_features`: dim 0 (num_patches)

---

## Comparison with Phi-4 Multimodal

| Aspect | Phi-4 Vision | Qwen3-VL Vision |
|--------|-------------|-----------------|
| **Architecture** | SigLIP-based | ViT + DeepStack |
| **Hidden Size** | Variable | 1024 |
| **Layers** | Unknown | 24 |
| **Special Feature** | HD transform | DeepStack multi-level |
| **Inputs** | 3 (pixel_values, attention_mask, sizes) | 2 (pixel_values, grid_thw) |
| **Output Size** | Variable | 2560 |
| **Export Size** | ~614 MB | ~1.66 GB |
| **Export Time** | Part of 30min build | 85 seconds standalone |

---

## Next Steps

### Immediate
- ✅ Vision encoder exported
- [ ] Export embedding layer (merges vision + text)
- [ ] Export text decoder (36-layer Qwen3 model)
- [ ] Create configuration files
- [ ] Test full multimodal pipeline

### Technical Challenges Ahead
1. **Embedding Layer**: Must handle vision token insertion into text sequence
2. **Text Decoder**: 36 layers with MRoPE (multimodal RoPE)
   - mrope_section: [24, 20, 20] - splits attention across modalities
3. **Configuration**: Vision processor config + genai_config.json
4. **Testing**: Vision + text multimodal inference

---

## Files Structure

```
qwen3-vl-4b/
├── pytorch/                  # Downloaded model (8.8 GB)
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── config.json
│   └── ...
│
├── cpu/                      # ONNX exports
│   ├── qwen3-vl-vision.onnx (900 KB)
│   └── qwen3-vl-vision.onnx.data (1.66 GB)
│
├── builder.py                # Phi-4 baseline (copied)
├── builder_vision.py         # Vision-only builder ✅
└── inspect_model.py          # Model structure inspector
```

---

## Learnings

1. **Qwen3-VL has cleaner separation** than Phi-4
   - Dedicated `visual` module (vs embedded in language model)
   - Clear input/output interfaces

2. **DeepStack requires special handling**
   - Multi-level feature extraction at layers 5, 11, 17
   - More complex than single-path vision encoders

3. **SDPA must be disabled** for ONNX export
   - GQA + SDPA not supported in torch.onnx opset 14
   - `attn_implementation="eager"` is the solution

4. **Grid-based processing** is unique to Qwen3-VL
   - `grid_thw` parameter tracks temporal/spatial dimensions
   - Enables video understanding capabilities

---

## Performance Expectations

### Vision Encoder Only
- **Size**: 1.66 GB
- **Inference**: Should be fast (single forward pass through 24 layers)
- **Output**: 2560-dim features per image patch

### Full Model (After Complete Export)
- **Expected Size**: ~4-5 GB (vision + embedding + text INT4)
- **CPU Performance**: 10-20 tokens/sec (similar to Qwen3-4B text-only)
- **Use Cases**: Image captioning, VQA, visual reasoning

---

## References

- **Model Card**: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- **Qwen3-VL Docs**: https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- **Phi-4 Multimodal Pattern**: `../phi4-multi-modal/builder.py`
- **Export Script**: `builder_vision.py`

---

**Status**: ✅ Vision Encoder Export Complete  
**Next**: Embedding + Text Decoder Export  
**Timeline**: Vision done in 1.4 minutes, full export estimated 15-20 minutes total

---

**Created**: January 29, 2026  
**Last Updated**: January 29, 2026  
**Export Tool**: `builder_vision.py`
