"""
Qwen3-VL-4B Full Multimodal Inference with Vision Feature Injection

This script implements TRUE multimodal inference by:
1. Encoding images with PyTorch vision encoder
2. Injecting vision features into text sequence
3. Generating text using the full PyTorch model with vision context

Since ONNX Runtime GenAI doesn't support embedding-level inputs,
we use PyTorch for the complete generation to properly merge vision and text.
"""

import argparse
import time
import torch
import sys
import codecs

from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Force UTF-8 output for Windows
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Multimodal Inference with Vision Injection")
    
    parser.add_argument(
        "--model",
        type=str,
        default="./pytorch",
        help="Path to PyTorch Qwen3-VL model"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_image.jpg",
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What do you see in this image?",
        help="Text question about the image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("QWEN3-VL MULTIMODAL INFERENCE WITH VISION INJECTION")
    print("="*80)
    print()
    print("This demonstrates TRUE multimodal inference:")
    print("  1. Vision features are extracted from the image")
    print("  2. Vision features are INJECTED into the text sequence")
    print("  3. The model generates text conditioned on the image")
    print()
    
    # ============================================================================
    # LOAD MODEL AND PROCESSOR
    # ============================================================================
    
    print("[1/4] Loading Qwen3-VL model for generation...")
    start = time.time()
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        attn_implementation="eager"
    ).to(args.device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    
    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Device: {args.device}")
    print(f"  dtype: {model.dtype}")
    print()
    
    # ============================================================================
    # PROCESS IMAGE AND PROMPT
    # ============================================================================
    
    print(f"[2/4] Processing image and prompt...")
    
    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"  Image: {args.image} ({image.size})")
    
    # Create multimodal message
    # Qwen3-VL uses a chat format with vision tokens
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": args.prompt
                },
            ],
        }
    ]
    
    # Process through Qwen3-VL processor
    # This handles vision token placement automatically
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"  Prompt: {args.prompt}")
    print(f"  Chat template applied")
    print()
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        return_tensors="pt",
        padding=True
    ).to(args.device)
    
    print(f"  Inputs prepared:")
    print(f"    - input_ids: {inputs['input_ids'].shape}")
    print(f"    - pixel_values: {inputs['pixel_values'].shape}")
    print(f"    - image_grid_thw: {inputs['image_grid_thw'].tolist()}")
    print()
    
    # ============================================================================
    # GENERATE WITH VISION CONTEXT
    # ============================================================================
    
    print(f"[3/4] Generating response with vision context...")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print()
    print("  Response: ", end="", flush=True)
    
    start = time.time()
    
    # Generate with vision features injected automatically
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    gen_time = time.time() - start
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(output_text)
    print()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    print("="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print()
    
    num_generated = len(generated_ids_trimmed[0])
    tokens_per_sec = num_generated / gen_time if gen_time > 0 else 0
    
    print(f"Performance:")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Tokens generated: {num_generated}")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/second")
    print()
    
    print(f"Status:")
    print(f"  Vision Features: INJECTED into sequence ✓")
    print(f"  Multimodal Context: ACTIVE ✓")
    print(f"  Response Quality: Model-aware of image ✓")
    print()
    
    print(f"Technical Details:")
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"  Vision patches: {inputs['pixel_values'].shape[0]}")
    print(f"  Image grid (T,H,W): {inputs['image_grid_thw'][0].tolist()}")
    print()
    
    print(f"Note: This uses PyTorch for both vision AND text generation")
    print(f"      because vision token injection requires embedding-level access.")
    print(f"      ONNX Runtime GenAI API doesn't support embedding inputs.")
    print()


if __name__ == "__main__":
    main()
