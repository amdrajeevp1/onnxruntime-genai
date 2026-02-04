# Vision Token Injection - Complete Guide

## 🎯 **The Challenge**

To achieve TRUE multimodal inference, we need to **inject vision features into the text generation sequence**. This requires access to the embedding layer, which poses different challenges depending on the runtime.

---

## 📊 **Three Approaches Compared**

| Approach | Vision | Text | Vision Injection | Speed | Status |
|----------|--------|------|------------------|-------|--------|
| **1. Full PyTorch** | PyTorch | PyTorch | ✅ Native | ~5-8 tok/s | ✅ Works |
| **2. Hybrid (Current)** | PyTorch | ONNX INT4 | ❌ Not possible | Vision: 140 patches/s<br>Text: 14-19 tok/s | ✅ Components work |
| **3. Hybrid + Embedding** | PyTorch | ONNX INT4 | ✅ Via ONNX embedding | TBD | ⏳ Needs implementation |

---

## 🔍 **Why Vision Injection is Hard with ONNX Runtime GenAI**

### ONNX Runtime GenAI API Limitation

```python
# What ONNX Runtime GenAI API provides:
generator = og.Generator(model, params)
generator.append_tokens(input_tokens)  # ❌ Only accepts token IDs, not embeddings!

while not generator.is_done():
    generator.generate_next_token()
```

**Problem**: The API only accepts **token IDs**, not **embeddings**. We can't inject vision features directly.

### What We Need

```python
# What we need for vision injection:
input_embeds = merge_vision_and_text(
    text_tokens=[1, 2, 3, 151859, 151859, ..., 4, 5],  # 151859 = <|image_pad|>
    vision_features=[108, 2560],  # From PyTorch vision encoder
    positions=[3, 4, ...]  # Where to inject vision
)

# Then generate from embeddings:
generator.append_embeddings(input_embeds)  # ❌ Not supported by ONNX RT GenAI API!
```

---

## ✅ **Approach 1: Full PyTorch (WORKING)**

This is what `multimodal_inference.py` demonstrates.

### Architecture

```
Image → [PyTorch Vision Encoder] → Vision Features
                                         ↓
Text Prompt → [Tokenizer] → Tokens → [Embedding Layer] → Merged Embeddings
                                                               ↓
                                                    [PyTorch Text Decoder]
                                                               ↓
                                                        Generated Text
```

### Code

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load model with generation capability
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./pytorch",
    trust_remote_code=True
).to("cpu")

# Process image and text together
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")

# Generate with vision features automatically injected
generated_ids = model.generate(
    **inputs,
    max_new_tokens=150
)
```

### Pros & Cons

✅ **Pros**:
- Works out of the box
- Official HuggingFace implementation
- Vision features properly injected
- Handles any image size

❌ **Cons**:
- Text generation is slower (~5-8 tok/s vs 14-19 tok/s with ONNX INT4)
- Higher memory usage (8.8 GB)
- No INT4 quantization benefits

---

## ⚡ **Approach 2: Hybrid WITHOUT Vision Injection (CURRENT)**

This is what `hybrid_inference_v2.py` demonstrates.

### Architecture

```
Image → [PyTorch Vision Encoder] → Vision Features (extracted but not used)
                                         ↓
                                    [Save to disk / Log]

Text Prompt → [ONNX Runtime GenAI Text Decoder] → Generated Text
               (No vision context - text only!)
```

### What Works

```python
# 1. Vision encoding (PyTorch)
vision_features = pytorch_model.visual(pixel_values, grid_thw)
# Output: [108, 2560] tensor ✅

# 2. Text generation (ONNX INT4)
generator = og.Generator(onnx_model, params)
generator.append_tokens(input_tokens)  # Text-only tokens
generated_text = ...  # ✅ Works, but NO vision context
```

### Performance

| Component | Speed | Status |
|-----------|-------|--------|
| Vision Encoding | 140 patches/s | ✅ Excellent |
| Text Generation | 14-19 tok/s | ✅ Excellent |
| Vision Injection | N/A | ❌ Not implemented |

### Limitation

**The vision features are extracted but NOT used in text generation!** The text decoder generates based only on the text prompt, without any image context.

---

## 🛠️ **Approach 3: Hybrid WITH Vision Injection (PROPOSED)**

To get the best of both worlds, we need to export an embedding layer.

### Architecture

```
Image → [PyTorch Vision Encoder] → Vision Features
                                         ↓
Text Prompt → [Tokenizer] → Tokens → [ONNX Embedding Layer] → Merged Embeddings
                                            ↑                        ↓
                                     Vision Features         [Custom ONNX]
                                                                    ↓
                                                      [ONNX Text Decoder INT4]
                                                                    ↓
                                                            Generated Text
```

### Implementation Steps

#### Step 1: Export Embedding Layer

```python
# extract_embeddings.py
import torch
from transformers import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained("./pytorch")

# Extract embedding layer
embedding_layer = model.language_model.embed_tokens

# Create wrapper that handles vision injection
class VisionTextEmbedding(torch.nn.Module):
    def __init__(self, embed_layer):
        super().__init__()
        self.embed_tokens = embed_layer
        
    def forward(self, input_ids, vision_features, vision_positions):
        """
        input_ids: [batch, seq_len] - token IDs
        vision_features: [num_patches, hidden_size] - vision embeddings
        vision_positions: [num_patches] - positions to inject vision
        """
        # Get text embeddings
        text_embeds = self.embed_tokens(input_ids)
        
        # Inject vision features at specified positions
        for i, pos in enumerate(vision_positions):
            text_embeds[0, pos] = vision_features[i]
        
        return text_embeds

# Export to ONNX
wrapper = VisionTextEmbedding(embedding_layer)
dummy_input_ids = torch.tensor([[1, 2, 151859, 151859, 3]])  # Including vision tokens
dummy_vision_features = torch.randn(2, 2560)
dummy_vision_positions = torch.tensor([2, 3])

torch.onnx.export(
    wrapper,
    (dummy_input_ids, dummy_vision_features, dummy_vision_positions),
    "embedding_layer.onnx",
    input_names=["input_ids", "vision_features", "vision_positions"],
    output_names=["input_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "vision_features": {0: "num_patches"},
        "vision_positions": {0: "num_patches"},
        "input_embeds": {0: "batch", 1: "seq_len"}
    }
)
```

#### Step 2: Use Custom Generation Loop

Since ONNX Runtime GenAI doesn't support embedding inputs, we'd need to:

1. **Export the text decoder as a raw ONNX model** (not GenAI format)
2. **Implement custom generation loop** that:
   - Uses ONNX embedding layer to get merged embeddings
   - Feeds embeddings to ONNX text decoder
   - Implements autoregressive generation manually

This is **complex** and loses many GenAI optimizations (KV cache management, etc.).

---

## 🎯 **Recommended Solutions**

### For Production Use

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| **Need vision + text** | Full PyTorch | Simplest, works out of box |
| **Speed critical** | Hybrid (text-only) | 2-3× faster text generation |
| **Text-only with occasional images** | Hybrid + fallback to PyTorch | Best of both worlds |

### Implementation Recommendations

#### Option A: Use Full PyTorch for Multimodal

```bash
python multimodal_inference.py \
  --image test_image.jpg \
  --prompt "What is in this image?"
```

**Best for**: Applications that always need vision context

#### Option B: Hybrid with Conditional Routing

```python
def generate(image=None, prompt=""):
    if image is not None:
        # Use full PyTorch for multimodal
        return generate_with_pytorch(image, prompt)
    else:
        # Use fast ONNX INT4 for text-only
        return generate_with_onnx(prompt)
```

**Best for**: Mixed workloads (some text-only, some multimodal)

#### Option C: Export Embedding Layer (Advanced)

Only if you need:
- Maximum speed (INT4 text decoder)
- Always using images
- Can invest time in custom generation loop

---

## 📈 **Performance Comparison**

### Test: 400×300 image + "Describe this image" prompt

| Approach | Vision | Text | Total | Tok/s | Quality |
|----------|--------|------|-------|-------|---------|
| **Full PyTorch** | 0.8s | ~15-20s | ~16-21s | 5-8 | ✅ Full multimodal |
| **Hybrid (no vision injection)** | 0.8s | 8-10s | ~9-11s | 14-19 | ❌ Text-only (no vision context) |
| **Full PyTorch (GPU)** | 0.1s | 2-3s | ~2-3s | 30-50 | ✅ Full multimodal |

---

## 🔬 **Technical Deep Dive**

### How Vision Injection Works in PyTorch

```python
# In Qwen3VLForConditionalGeneration:

# 1. Vision encoder processes image
vision_features = self.visual(pixel_values, grid_thw)
# Output: [num_patches, hidden_size]

# 2. Text embeddings are created
text_embeds = self.language_model.embed_tokens(input_ids)
# Output: [batch, seq_len, hidden_size]

# 3. Vision features REPLACE image token embeddings
image_token_id = 151859  # <|image_pad|>
mask = (input_ids == image_token_id)
text_embeds[mask] = vision_features  # Direct replacement!

# 4. Generation proceeds with merged embeddings
output = self.language_model(inputs_embeds=text_embeds, ...)
```

### Why ONNX Runtime GenAI Can't Do This

```python
# ONNX Runtime GenAI hides the embedding layer:
generator = og.Generator(model, params)
generator.append_tokens([1, 2, 3, 151859, 151859, 4])
#                                 ↑       ↑
#                         These vision tokens stay as token IDs
#                         We CAN'T inject vision features!

# The API doesn't expose:
# - embedding layer access
# - inputs_embeds parameter
# - token replacement before embedding
```

---

## ✅ **Current Status**

### What Works ✅

1. **PyTorch Vision Encoding**: 140 patches/s, any image size
2. **ONNX Text Generation**: 14-19 tok/s, INT4 optimized
3. **Full PyTorch Multimodal**: Complete vision + text integration

### What's Missing ❌

1. **Hybrid with Vision Injection**: Can't inject vision into ONNX text decoder
   - Reason: ONNX RT GenAI API doesn't support embedding inputs
   - Solution: Use full PyTorch OR export custom embedding layer

---

## 📝 **Conclusion**

### For Your Use Case:

**If you need vision features in text generation RIGHT NOW:**
→ Use `multimodal_inference.py` (Full PyTorch)

**If you want maximum speed and only need text:**
→ Use `hybrid_inference_v2.py` (Hybrid PyTorch vision + ONNX text)

**If you want to experiment with hybrid + vision injection:**
→ Follow "Approach 3" to export embedding layer and implement custom loop

### Bottom Line:

The hybrid approach (PyTorch vision + ONNX text) **works great for the components separately**, but **true vision injection requires either**:
1. Full PyTorch (simplest, works now)
2. Custom ONNX embedding layer + manual generation loop (complex, advanced)

**Recommendation**: Start with full PyTorch (`multimodal_inference.py`) for multimodal tasks. It "just works" and gives you proper vision-conditioned generation.

---

## 🚀 **Quick Start**

### Run Full Multimodal (with vision injection):

```bash
python multimodal_inference.py \
  --image test_image.jpg \
  --prompt "Describe this image in detail" \
  --max_new_tokens 150
```

### Run Hybrid (fast but no vision injection):

```bash
python hybrid_inference_v2.py \
  --image test_image.jpg \
  --prompt "Describe this image"
```

---

**Files**:
- `multimodal_inference.py` - Full PyTorch with vision injection ✅
- `hybrid_inference_v2.py` - Hybrid (vision extracted but not used in text)
- `VISION_INJECTION_GUIDE.md` - This file
