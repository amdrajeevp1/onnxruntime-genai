"""
Qwen3-VL-4B Multimodal Inference Test

Tests the exported ONNX models:
- Vision Encoder: cpu/qwen3-vl-vision.onnx
- Text Decoder: cpu-text/model.onnx

This script:
1. Loads an image
2. Processes it through vision encoder
3. Generates text description using text decoder
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Fix Windows console encoding for Unicode
import locale
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import onnxruntime as ort
import onnxruntime_genai as og

print("="*80)
print("QWEN3-VL-4B MULTIMODAL INFERENCE TEST")
print("="*80)
print(f"ONNX Runtime version: {ort.__version__}")
print(f"ONNX Runtime GenAI version: {og.__version__}")
print()

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def smart_resize(image, min_pixels=256*28*28, max_pixels=1280*28*28, size_factor=28):
    """
    Qwen3-VL image resizing logic - FIXED SIZE MODE
    
    FIXED SIZE: 336x336 pixels = 21x21 patches = 441 patches
    All images are resized to this exact size to match the ONNX export.
    
    Args:
        image: PIL Image
        (other args ignored in fixed mode)
    
    Returns:
        Resized PIL Image at FIXED 336x336 size
    """
    h, w = image.height, image.width
    pixels = h * w
    
    # FIXED SIZE: 336x336 (most common size, 21x21 patches)
    FIXED_SIZE = 336  # Must be divisible by 16 (patch size)
    
    print(f"  Original size: {w}x{h} ({pixels:,} pixels)")
    print(f"  FIXED SIZE MODE: Resizing to {FIXED_SIZE}x{FIXED_SIZE}")
    
    return image.resize((FIXED_SIZE, FIXED_SIZE), Image.BICUBIC)


def preprocess_image(image_path, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Preprocess image for Qwen3-VL vision encoder
    
    Args:
        image_path: Path to image file
        mean: Normalization mean [R, G, B]
        std: Normalization std [R, G, B]
    
    Returns:
        pixel_values: [num_patches, 1536] numpy array
        grid_thw: [1, 3] numpy array (temporal=1, height, width in patches)
    """
    print(f"\n[1/5] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    print(f"[2/5] Resizing image")
    image = smart_resize(image)
    
    print(f"[3/5] Converting to patches (16x16)")
    # Convert to numpy
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    
    # Patchify (16x16 patches)
    patch_size = 16
    H, W, C = img_array.shape
    
    # Ensure dimensions are divisible by patch_size
    H = (H // patch_size) * patch_size
    W = (W // patch_size) * patch_size
    img_array = img_array[:H, :W, :]
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    print(f"  Final size: {W}x{H}")
    print(f"  Grid: {num_patches_h}x{num_patches_w} = {num_patches} patches")
    
    # Reshape to patches: [H, W, C] -> [num_patches, patch_size*patch_size*C]
    patches = img_array.reshape(
        num_patches_h, patch_size,
        num_patches_w, patch_size,
        C
    ).transpose(0, 2, 1, 3, 4).reshape(num_patches, -1)
    
    # pixel_values: [num_patches, 1536] where 1536 = 16*16*3*2 (with temporal)
    # For images (not video), we duplicate temporally
    pixel_values = np.concatenate([patches, patches], axis=1).astype(np.float32)
    
    # grid_thw: [num_images, 3] = [temporal, height_patches, width_patches]
    # For single image: temporal=1
    grid_thw = np.array([[1, num_patches_h, num_patches_w]], dtype=np.int64)
    
    print(f"[4/5] Output shapes:")
    print(f"  pixel_values: {pixel_values.shape} {pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw.shape} {grid_thw.dtype}")
    
    return pixel_values, grid_thw


# ============================================================================
# VISION ENCODER INFERENCE
# ============================================================================

def load_vision_encoder(model_path):
    """Load ONNX vision encoder"""
    print(f"\n{'='*80}")
    print("LOADING VISION ENCODER")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    
    # Create session
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # ERROR only
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    # Print model info
    print(f"\nInputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type} {inp.shape}")
    
    print(f"\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type} {out.shape}")
    
    return session


def run_vision_encoder(session, pixel_values, grid_thw):
    """Run vision encoder inference"""
    print(f"\n[5/5] Running vision encoder...")
    
    # WORKAROUND: The exported model has a hardcoded Split operation that expects batch_size >= 2
    # The model expects flattened patches: [total_patches, 1536] not [batch, patches, 1536]
    # Duplicate the patches to create a batch of 2, then extract only the first image's features
    
    num_patches = pixel_values.shape[0]
    print(f"  Original: pixel_values={pixel_values.shape}, grid_thw={grid_thw.shape}")
    
    # Concatenate patches (flatten batch dimension)
    pixel_values_batched = np.concatenate([pixel_values, pixel_values], axis=0)  # [2*num_patches, 1536]
    grid_thw_batched = np.concatenate([grid_thw, grid_thw], axis=0)  # [2, 3]
    
    print(f"  Batched: pixel_values={pixel_values_batched.shape}, grid_thw={grid_thw_batched.shape}")
    
    start_time = time.time()
    
    outputs = session.run(
        None,  # All outputs
        {
            "pixel_values": pixel_values_batched,
            "grid_thw": grid_thw_batched
        }
    )
    
    elapsed = time.time() - start_time
    
    # Extract only the first image's features (first num_patches)
    vision_features = outputs[0][:num_patches]
    
    print(f"  Vision features: {vision_features.shape} {vision_features.dtype}")
    print(f"  Inference time: {elapsed:.3f}s")
    print(f"  Tokens per second: {vision_features.shape[0] / elapsed:.1f}")
    
    return vision_features


# ============================================================================
# TEXT DECODER (ONNX RUNTIME GENAI)
# ============================================================================

def test_text_only(text_model_path, prompt):
    """Test text-only generation first"""
    print(f"\n{'='*80}")
    print("TEXT-ONLY GENERATION TEST")
    print(f"{'='*80}")
    print(f"Model: {text_model_path}")
    print(f"Prompt: {prompt}")
    
    # Load model
    print("\nLoading text decoder...")
    model = og.Model(text_model_path)
    tokenizer = og.Tokenizer(model)
    print("Model loaded!")
    
    # Tokenize
    print("\nTokenizing...")
    input_tokens = tokenizer.encode(prompt)
    print(f"Input tokens: {len(input_tokens)} tokens")
    
    # Generate
    print("\nGenerating...")
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=200,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    # Create tokenizer stream for decoding
    tokenizer_stream = tokenizer.create_stream()
    
    print("\nGenerated text:")
    print("-" * 80)
    
    generated_tokens = []
    start_time = time.time()
    
    while not generator.is_done():
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        generated_tokens.append(new_token)
        
        # Decode and print
        token_text = tokenizer_stream.decode(new_token)
        print(token_text, end='', flush=True)
    
    elapsed = time.time() - start_time
    
    print("\n" + "-" * 80)
    print(f"\nGeneration stats:")
    print(f"  Generated tokens: {len(generated_tokens)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {len(generated_tokens) / elapsed:.1f} tokens/sec")
    
    del generator
    del model
    print("\nText-only test complete!")


def run_multimodal_inference(text_model_path, vision_features, prompt):
    """
    Run multimodal inference
    
    NOTE: This is a simplified version. Full integration would require:
    1. Vision feature injection into text embeddings
    2. Custom attention mask for vision tokens
    3. Position encoding for vision tokens
    
    For now, we'll just test text generation with context about vision.
    """
    print(f"\n{'='*80}")
    print("MULTIMODAL GENERATION (Simplified)")
    print(f"{'='*80}")
    
    # For now, describe the vision features in the prompt
    num_vision_tokens = vision_features.shape[0]
    vision_summary = f"[Image processed: {num_vision_tokens} vision tokens extracted]"
    
    full_prompt = f"{vision_summary}\n\nUser: {prompt}\n\nAssistant:"
    
    print(f"Model: {text_model_path}")
    print(f"Vision tokens: {num_vision_tokens}")
    print(f"Prompt: {prompt}")
    
    # Load model
    print("\nLoading text decoder...")
    model = og.Model(text_model_path)
    tokenizer = og.Tokenizer(model)
    print("Model loaded!")
    
    # Tokenize
    print("\nTokenizing...")
    input_tokens = tokenizer.encode(full_prompt)
    print(f"Input tokens: {len(input_tokens)} tokens")
    
    # Generate
    print("\nGenerating...")
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=200,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    # Create tokenizer stream for decoding
    tokenizer_stream = tokenizer.create_stream()
    
    print("\nGenerated text:")
    print("-" * 80)
    
    generated_tokens = []
    start_time = time.time()
    
    while not generator.is_done():
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        generated_tokens.append(new_token)
        
        # Decode and print
        token_text = tokenizer_stream.decode(new_token)
        print(token_text, end='', flush=True)
    
    elapsed = time.time() - start_time
    
    print("\n" + "-" * 80)
    print(f"\nGeneration stats:")
    print(f"  Generated tokens: {len(generated_tokens)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {len(generated_tokens) / elapsed:.1f} tokens/sec")
    
    del generator
    del model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B Multimodal Inference")
    parser.add_argument(
        "--vision_model",
        type=str,
        default="cpu/qwen3-vl-vision.onnx",
        help="Path to vision encoder ONNX model"
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="cpu-text",
        help="Path to text decoder model directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt"
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Test text-only generation (no vision)"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.text_only:
        # Text-only test
        test_prompt = "Hello! Please introduce yourself and tell me what you can do."
        test_text_only(args.text_model, test_prompt)
    else:
        # Multimodal test
        if not args.image:
            print("ERROR: --image required for multimodal test")
            print("Use --text_only flag to test text generation without image")
            return
        
        # 1. Load and process image
        pixel_values, grid_thw = preprocess_image(args.image)
        
        # 2. Load vision encoder
        vision_session = load_vision_encoder(args.vision_model)
        
        # 3. Run vision encoder
        vision_features = run_vision_encoder(vision_session, pixel_values, grid_thw)
        
        # 4. Run text generation with vision context
        run_multimodal_inference(args.text_model, vision_features, args.prompt)
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
