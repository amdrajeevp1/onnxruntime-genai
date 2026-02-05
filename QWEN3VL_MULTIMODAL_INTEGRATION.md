# Qwen3-VL Multimodal Integration for ONNX Runtime GenAI

## Overview

This document describes the complete integration of Qwen3-VL (Qwen3 Vision-Language Model) into ONNX Runtime GenAI, following the Phi-4-MM multimodal architecture pattern. The integration enables native multimodal inference using OGA's `create_multimodal_processor()` API.

## Architecture

Qwen3-VL uses a three-model pipeline architecture:

1. **Vision Encoder** (`qwen3vl-vision.onnx`): Processes image patches and outputs vision embeddings
2. **Embedding Model** (`qwen3vl-embedding.onnx`): Merges text embeddings with vision features
3. **Text Decoder** (`model.onnx`): Standard language model decoder

### Key Specifications

- **Image Size**: 384×384 pixels (fixed)
- **Patch Size**: 16×16 pixels
- **Temporal Patch Size**: 2 (for video support)
- **Merge Size**: 2 (spatial merging of patches)
- **Patches**: 24×24 = 576 patches per image
- **Patch Features**: 3 channels × 2 temporal × 16 × 16 = 1536 features per patch
- **Vision Tokens**: 144 tokens per image after merging (576 / 4)

## Changes Made

### 1. C++ Processor Implementation

#### Files Added:
- `src/models/qwen3vl_image_processor.h`
- `src/models/qwen3vl_image_processor.cpp`

#### Key Features:

**Qwen3VLImageProcessor Class**:
- Integrates with `ort-extensions` for image preprocessing
- Converts CHW format `[3, H, W]` to patch format `[num_patches, patch_features]`
- Handles `image_grid_thw` tensor generation (patch grid dimensions)
- Processes vision-start tokens in prompts
- Supports dynamic image sizes with validation

**Patch Conversion Logic**:
```cpp
// Converts [C, H, W] → [num_patches, patch_features]
// For 384×384 image: [3, 384, 384] → [576, 1536]
std::unique_ptr<OrtValue> ConvertToPatch(
    const float* image_data,
    int64_t channels, int64_t height, int64_t width,
    int64_t patch_size, int64_t temporal_patch_size,
    Ort::Allocator& allocator)
```

**Prompt Processing**:
- Counts `<|vision_start|>` tokens (ID: 151652)
- Validates image count matches vision tokens
- Calculates total image tokens based on patch grid
- Supports `num_image_tokens` metadata

### 2. Model Type Registration

#### File Modified:
- `src/models/model_type.h`

#### Changes:
Added `qwen3_vl` to the VLM (Vision-Language Model) array:

```cpp
static constexpr std::array<std::string_view, 5> VLM = {
    "fara", "gemma3", "phi3v", "qwen2_5_vl", "qwen3_vl"
};
```

**Note**: Qwen3-VL uses standard RoPE (not 3D position IDs like Qwen2.5-VL), so it's classified as VLM, not Qwen25VL.

### 3. Model Creation Logic

#### File Modified:
- `src/models/model.cpp`

#### Changes:
Added Qwen3-VL to the model factory:

```cpp
std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config) {
  // ...
  if (ModelType::IsVLM(config->model.type))
    return std::make_shared<MultiModalLanguageModel>(
        std::move(config), ort_env, 
        true,   // vision = true
        false   // speech = false
    );
  // ...
}
```

This uses the existing `MultiModalLanguageModel` class, which handles the three-stage pipeline (vision → embedding → decoder).

### 4. Processor Registration

#### File Modified:
- `src/models/processor.cpp`

#### Changes:
Registered the Qwen3VL processor factory:

```cpp
#include "qwen3vl_image_processor.h"

std::map<std::string, ProcessorFactory> processor_type_to_processor_creator_ = {
    // ... existing processors ...
    {"qwen3_vl", Processor::Create<Qwen3VLImageProcessor>}
};
```

### 5. Model Export Scripts

#### Files Added:
- `examples/python/qwen3-vl-4b/export_for_oga.py`
- `examples/python/qwen3-vl-4b/export_embedding_fixed.py`

#### Export Process:

**Vision Model Export**:
- Input format: `[num_patches, patch_features]` where patch_features = 1536
- Exports with `attn_implementation="eager"` to avoid GQA ONNX issues
- Dynamic axes for patches

**Embedding Model Export**:
- **Critical**: Includes `vision_hidden_states` input for multimodal support
- Implements vision token replacement logic
- Must include vision_start token (151652) in dummy input to ensure ONNX traces the branch

**Text Decoder Export**:
- Uses standard `onnxruntime_genai.models.builder.create_model`
- Supports fp32, fp16, int4, int8 precision

### 6. Configuration Files

#### genai_config.json Structure:

```json
{
  "model": {
    "type": "qwen3_vl",
    "vision": {
      "filename": "qwen3vl-vision.onnx",
      "config_filename": "vision_processor.json",
      "inputs": {
        "pixel_values": "pixel_values",
        "image_grid_thw": "image_grid_thw"
      },
      "outputs": {
        "image_features": "pooled_embeds"
      },
      "spatial_merge_size": 2
    },
    "embedding": {
      "filename": "qwen3vl-embedding.onnx",
      "inputs": {
        "input_ids": "input_ids",
        "image_features": "vision_hidden_states"
      },
      "outputs": {
        "inputs_embeds": "inputs_embeds"
      }
    },
    "decoder": {
      "filename": "model.onnx",
      "inputs": {
        "inputs_embeds": "inputs_embeds",
        "attention_mask": "attention_mask",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value"
      },
      "outputs": {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value"
      }
    }
  }
}
```

#### vision_processor.json Structure:

```json
{
  "processor": {
    "name": "qwen3vl_image_processor",
    "transforms": [
      {
        "operation": {
          "name": "decode_image",
          "type": "DecodeImage",
          "attrs": {
            "color_space": "RGB"
          }
        }
      },
      {
        "operation": {
          "name": "rescale",
          "type": "Rescale"
        }
      },
      {
        "operation": {
          "name": "normalize",
          "type": "Normalize",
          "attrs": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711]
          }
        }
      }
    ]
  }
}
```

**Note**: Images must be 384×384. Resize operation should be added for dynamic sizing.

### 7. Python Example

#### File Added:
- `examples/python/qwen3vl-oga.py`

#### Usage:

```python
import onnxruntime_genai as og
from onnxruntime_genai import onnxruntime_genai as og

# Load model
model = og.Config(model_path)
model = og.Model(config)
processor = model.create_multimodal_processor()

# Load images
images = og.Images.open(*image_paths)

# Construct prompt
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
    f"{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
)

# Process
inputs = processor(prompt, images=images)

# Generate
params = og.GeneratorParams(model)
params.set_search_options(max_length=4096)
generator = og.Generator(model, params)
generator.set_inputs(inputs)

while not generator.is_done():
    generator.generate_next_token()
    token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(token), end="", flush=True)
```

## Special Tokens

Qwen3-VL uses specific tokens for vision:

- `<|vision_start|>` (ID: 151652): Marks start of vision sequence
- `<|vision_end|>` (ID: 151653): Marks end of vision sequence
- `<|image_pad|>` (ID: 151655): Placeholder for image tokens
- `<|im_start|>` (ID: 151644): Instruction/message start
- `<|im_end|>` (ID: 151645): Instruction/message end

## Technical Challenges Resolved

### 1. GQA (Grouped Query Attention) Export Issues

**Problem**: PyTorch's ONNX exporter doesn't support GQA's `scaled_dot_product_attention` with `enable_gqa=True`.

**Solution**: Export with `attn_implementation="eager"` to use standard attention:

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    input_dir,
    trust_remote_code=True,
    attn_implementation="eager"  # Critical for ONNX export
)
```

### 2. Embedding Model Multimodal Support

**Problem**: ONNX export optimizes away unused branches if dummy inputs don't exercise all code paths.

**Solution**: Include vision_start token in dummy inputs:

```python
input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
input_ids[0, 2] = 151652  # Include vision_start token
vision_hidden_states = torch.randn(144, 2560)
```

### 3. Image Format Conversion

**Problem**: Vision model expects patches `[num_patches, patch_features]`, but ort-extensions outputs CHW `[C, H, W]`.

**Solution**: Implement C++ patch conversion in `Qwen3VLImageProcessor::Process`:
- Extract patches from CHW image
- Duplicate for temporal dimension
- Flatten to required format

### 4. image_grid_thw Generation

**Problem**: ort-extensions doesn't output `image_grid_thw` tensor (patch grid dimensions).

**Solution**: Generate in C++ processor:
```cpp
// For 384×384 image with 16×16 patches
auto grid_tensor = OrtValue::CreateTensor<int64_t>(allocator, {1, 3});
grid_data[0] = 1;   // temporal (T)
grid_data[1] = 24;  // height in patches (H)
grid_data[2] = 24;  // width in patches (W)
```

## Build Instructions

1. **Rebuild ONNX Runtime GenAI**:
```bash
python build.py --config Release
```

2. **Install Python Package**:
```bash
pip install --force-reinstall build/Windows/Release/wheel/onnxruntime_genai-*.whl
```

3. **Copy DLL** (Windows only):
```bash
cp build/Windows/Release/Release/onnxruntime.dll <python-site-packages>/onnxruntime_genai/
```

## Model Export Instructions

1. **Download Qwen3-VL PyTorch Model**:
```bash
huggingface-cli download Qwen/Qwen3-VL-4B --local-dir pytorch/
```

2. **Export All Models**:
```bash
cd examples/python/qwen3-vl-4b
python export_for_oga.py --input pytorch --output oga-multimodal-fp32 --precision fp32
```

3. **Test**:
```bash
python qwen3vl-oga.py -m oga-multimodal-fp32 --image_paths test.jpg --prompt "Describe this image"
```

## Testing

### Test Images
- `examples/python/qwen3-vl-4b/images/test_colors.jpg`
- `examples/python/qwen3-vl-4b/images/test_checkerboard.jpg`
- `examples/python/qwen3-vl-4b/images/test_gradient_genai.jpg`

### Sample Commands

**Interactive Mode**:
```bash
python qwen3vl-oga.py -m oga-multimodal-fp32 --image_paths test.jpg
```

**Non-Interactive Mode**:
```bash
python qwen3vl-oga.py -m oga-multimodal-fp32 \
    --image_paths test.jpg \
    --prompt "What objects do you see?" \
    --non-interactive
```

## Comparison with Phi-4-MM

| Feature | Qwen3-VL | Phi-4-MM |
|---------|----------|----------|
| Image Size | 384×384 (fixed) | Dynamic (up to 448×448) |
| Patch Size | 16×16 | 14×14 |
| Temporal Patches | 2 | N/A |
| Merge Size | 2×2 | Variable |
| Position IDs | Standard RoPE | Standard |
| Audio Support | No | Yes |
| Model Type | VLM | MMM (Multimodal) |

## Future Improvements

1. **Dynamic Image Sizing**: Add proper Resize operation to vision_processor.json
2. **Video Support**: Extend temporal patch handling for multi-frame inputs
3. **Batch Processing**: Support multiple images in single inference
4. **INT4/INT8 Quantization**: Test quantized models for deployment
5. **C++ Example**: Create native C++ inference example

## References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-4B)
- [ONNX Runtime GenAI Multimodal API](https://github.com/microsoft/onnxruntime-genai)
- [ort-extensions Documentation](https://github.com/microsoft/onnxruntime-extensions)
- Phi-4-MM integration (reference implementation)

## Contributors

This integration follows the established patterns from Phi-4-MM, Qwen2.5-VL, and Gemma3 multimodal integrations in ONNX Runtime GenAI.

## License

Same as ONNX Runtime GenAI (MIT License)
