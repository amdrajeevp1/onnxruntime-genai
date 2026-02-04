# Vision Token Injection - PyTorch Reference Implementation

## 🎯 **Overview**

Based on the HuggingFace Transformers reference implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

This document explains exactly how Qwen3-VL injects vision features into text embeddings.

---

## 📚 **Key Components from Reference**

### 1. Vision Encoder (`Qwen3VLVisionModel`)

Located in `modeling_qwen3_vl.py` lines 748-924

```python
class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    def forward(self, hidden_states, grid_thw, **kwargs):
        # Process vision patches
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds
        
        # Run through transformer blocks
        for blk in self.blocks:
            hidden_states = blk(hidden_states, ...)
        
        # CRITICAL: Merge patches spatially (2x2 -> 1)
        merged_hidden_states = self.merger(hidden_states)
        
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,      # [432, 2560] - raw patches
            pooler_output=merged_hidden_states,   # [108, 2560] - MERGED for injection
            deepstack_features=deepstack_feature_lists,
        )
```

**Key point**: `pooler_output` contains the spatially merged vision features that are ready for injection into the text sequence.

### 2. Main Model (`Qwen3VLModel`)

Located in `modeling_qwen3_vl.py` lines 1350-1600

#### Step 1: Get Vision Features (lines 1404-1415)

```python
@can_return_tuple
@auto_docstring
def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
    pixel_values = pixel_values.type(self.visual.dtype)
    
    # Get vision features
    vision_output = self.visual(
        pixel_values,
        grid_thw=image_grid_thw,
        return_dict=True,
        **kwargs
    )
    
    # Extract MERGED features (ready for injection)
    image_embeds = vision_output.pooler_output
    
    # Split by image (if multiple images)
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    vision_output.pooler_output = image_embeds
    
    return vision_output
```

#### Step 2: Find Vision Token Positions (lines 1450-1476)

```python
def get_placeholder_mask(
    self,
    input_ids,
    inputs_embeds,
    image_features=None,
    video_features=None,
):
    """Find positions where vision features should be injected"""
    
    if input_ids is None:
        # If using embeddings directly, find positions by comparing embeddings
        special_image_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, ...)
        )
        special_image_mask = special_image_mask.all(-1)
    else:
        # If using token IDs, find positions directly
        special_image_mask = input_ids == self.config.image_token_id
    
    # Expand to embedding dimensions
    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    
    # Verify counts match
    if image_features is not None:
        assert inputs_embeds[special_image_mask].numel() == image_features.numel()
    
    return special_image_mask, special_video_mask
```

#### Step 3: Inject Vision into Text Embeddings (lines 1537-1547)

```python
@auto_docstring
@check_model_inputs
def forward(
    self,
    input_ids=None,
    pixel_values=None,
    image_grid_thw=None,
    ...
):
    # Get text embeddings
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    
    # Get vision features
    if pixel_values is not None:
        image_outputs = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        
        # Concatenate all images
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        
        # Find injection positions
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        
        # INJECT vision features at masked positions
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    
    # Continue to language model with merged embeddings
    outputs = self.language_model(
        input_ids=None,  # No token IDs!
        inputs_embeds=inputs_embeds,  # Use merged embeddings
        ...
    )
    
    return outputs
```

---

## 🔑 **Key Insights**

### 1. Vision Token ID

```python
config.image_token_id = 151655  # <|image_pad|> token
```

The text sequence contains multiple `<|image_pad|>` tokens (one for each merged vision patch). For example, with grid `[1, 18, 24]`:
- Original patches: 18×24 = 432
- After spatial merge (2×2): 9×12 = 108 patches
- Token sequence will have 108 `<|image_pad|>` tokens

### 2. The Injection Process

```python
# Before injection:
input_ids = [1, 2, 3, 151655, 151655, 151655, ..., 4, 5]
#                    ^^^^^^  ^^^^^^  ^^^^^^
#                    These are vision token placeholders

# Step 1: Convert to embeddings
text_embeds = embedding_layer(input_ids)  # [batch, seq_len, hidden_size]

# Step 2: Find vision token positions
mask = (input_ids == 151655)  # [batch, seq_len]
mask_expanded = mask.unsqueeze(-1).expand_as(text_embeds)  # [batch, seq_len, hidden_size]

# Step 3: Inject vision features
merged_embeds = text_embeds.masked_scatter(mask_expanded, vision_features)
#                           ^^^^^^^^^^^^^^
#                           Replace vision token embeddings with actual vision features!

# After injection:
# merged_embeds now has real vision features instead of generic <|image_pad|> embeddings
```

### 3. Why ONNX Runtime GenAI Can't Do This

```python
# ONNX Runtime GenAI API:
generator = og.Generator(model, params)
generator.append_tokens(input_ids)  # ❌ Only accepts token IDs

# Internally it does:
# embeddings = embedding_layer(input_ids)
# output = decoder(embeddings)

# We CAN'T inject vision features because:
# 1. The embedding layer is INSIDE the ONNX model
# 2. The API only accepts token IDs, not embeddings
# 3. There's no way to modify embeddings before they reach the decoder
```

---

## 🛠️ **Implementation Approaches**

### Approach A: Full PyTorch (WORKING)

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained("./pytorch")
processor = AutoProcessor.from_pretrained("./pytorch")

# Prepare inputs
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]
}]

text = processor.apply_chat_template(messages, ...)
inputs = processor(text=[text], images=[image], return_tensors="pt")

# Generate (vision injection happens automatically inside model.generate())
generated_ids = model.generate(**inputs, max_new_tokens=150)
```

**Status**: ✅ Works perfectly  
**Performance**: ~5-8 tokens/s on CPU  
**Use case**: When you need vision context

### Approach B: Hybrid with Manual Embedding Export (COMPLEX)

```python
# Step 1: Extract embedding layer
embedding_layer = model.model.language_model.embed_tokens

# Step 2: Export to ONNX
class VisionTextEmbedding(nn.Module):
    def forward(self, input_ids, vision_features, vision_positions):
        # Get text embeddings
        text_embeds = self.embed_tokens(input_ids)
        
        # Inject vision
        for i, pos in enumerate(vision_positions):
            text_embeds[0, pos] = vision_features[i]
        
        return text_embeds

torch.onnx.export(VisionTextEmbedding(...), "embedding.onnx")

# Step 3: Use in pipeline
vision_features = pytorch_vision_encoder(image)
input_embeds = onnx_embedding_layer(input_ids, vision_features, positions)
output = onnx_text_decoder(input_embeds)  # Would need custom export!
```

**Status**: ⚠️ Requires significant work  
**Challenges**:
1. Export embedding layer with vision injection logic
2. Export text decoder that accepts embeddings (not GenAI format)
3. Implement custom generation loop (no KV cache API)
4. Handle sampling, stopping criteria manually

### Approach C: Hybrid WITHOUT Vision Injection (CURRENT)

```python
# Extract vision features (PyTorch)
vision_features = pytorch_model.visual(pixel_values, grid_thw)  # [108, 2560]

# Generate text (ONNX) - but WITHOUT vision features!
generator = og.Generator(onnx_text_model, params)
generator.append_tokens(input_ids)  # Vision tokens become generic embeddings
output = generator.get_next_tokens()  # No vision context ❌
```

**Status**: ✅ Components work separately  
**Performance**: Vision 140 patches/s, Text 14-19 tok/s  
**Limitation**: Vision features are NOT used in generation!

---

## 📊 **Diagram: Vision Injection Flow**

```
                    Qwen3-VL Vision Injection Pipeline
                    =====================================

INPUT:
  Image (400×300) ─────┐
  Prompt: "Describe..."│
                       │
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Tokenize with Image Placeholders                     │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Input text with <|image_pad|> tokens:                       │
│  [1, 2, 3, 151655, 151655, ..., 151655, 4, 5, 6]             │
│            ↑ 108 vision token placeholders ↑                  │
│                                                               │
└───────────────────────────────────┬───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ↓                               ↓
┌───────────────────────────────┐   ┌───────────────────────────┐
│ STEP 2A: Vision Encoding      │   │ STEP 2B: Text Embedding   │
│──────────────────────────────│   │───────────────────────────│
│                               │   │                            │
│ Image → Vision Encoder        │   │ input_ids → embed_tokens   │
│         (PyTorch)             │   │                            │
│                               │   │ Output:                    │
│ Output:                       │   │ text_embeds: [1, 126, 2560]│
│ vision_features: [108, 2560]  │   │                            │
│                               │   │ (Vision tokens have        │
│ (Spatially merged patches)    │   │  generic embeddings)       │
└───────────────┬───────────────┘   └─────────────┬──────────────┘
                │                                 │
                └──────────┬──────────────────────┘
                           │
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Vision Token Injection                               │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Find positions: mask = (input_ids == 151655)                │
│  Inject:         text_embeds.masked_scatter(mask, vision)    │
│                                                               │
│  Result:                                                      │
│  merged_embeds: [1, 126, 2560]                               │
│                                                               │
│  Position: 0   1   2   3   4   5  ... 124 125                │
│  Before:  [emb][emb][emb][pad][pad][pad]...[emb][emb]        │
│  After:   [emb][emb][emb][VIS][VIS][VIS]...[emb][emb]        │
│                           ↑ Real vision features! ↑           │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Text Generation                                       │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  language_model(                                              │
│      inputs_embeds=merged_embeds,  ← Contains vision!        │
│      attention_mask=...,                                      │
│      ...                                                      │
│  )                                                            │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓
                    Generated Text
            "This image shows a blue rectangle
             with text 'Hello Qwen3-VL!' ..."
```

---

## ✅ **Summary**

### What We Learned from the Reference

1. **Vision features are injected at the embedding level**, not the token level
2. **The injection uses `masked_scatter`** to replace vision token embeddings with real vision features
3. **`pooler_output`** from the vision encoder contains the spatially merged features ready for injection
4. **The text decoder receives `inputs_embeds`** (not `input_ids`), which already contain vision features

### Why Hybrid is Hard

ONNX Runtime GenAI operates on **token IDs**, not **embeddings**:
- ✅ PyTorch: Has full access to embedding layer for injection
- ❌ ONNX RT GenAI: API only accepts token IDs, embedding layer is internal

### Recommendations

| Use Case | Solution | Why |
|----------|----------|-----|
| **Need vision** | Full PyTorch | Only option that works today |
| **Text-only speed** | ONNX RT GenAI | 3× faster, no vision needed |
| **Research project** | Export custom embedding | Complex but possible |

---

## 🔗 **References**

- **Source code**: [transformers/models/qwen3_vl/modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Key methods**:
  - `Qwen3VLVisionModel.forward()` - lines 838-924 (vision encoding + merging)
  - `Qwen3VLModel.get_image_features()` - lines 1404-1415 (get pooler_output)
  - `Qwen3VLModel.get_placeholder_mask()` - lines 1450-1476 (find injection positions)
  - `Qwen3VLModel.forward()` - lines 1537-1547 (vision injection with masked_scatter)

---

## 📝 **Code Examples**

See these files for working implementations:
- `multimodal_inference.py` - Full PyTorch with vision injection ✅
- `hybrid_inference_v2.py` - Hybrid without vision injection (components only)
- `VISION_INJECTION_GUIDE.md` - Complete comparison of all approaches
