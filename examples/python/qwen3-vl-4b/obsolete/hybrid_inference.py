"""
Qwen3-VL-4B Hybrid Multimodal Inference Pipeline

Architecture:
- Vision Encoder: PyTorch (native, supports dynamic shapes)
- Text Decoder: ONNX Runtime GenAI (INT4 quantized, 19.3 tok/s)

This hybrid approach leverages:
- PyTorch's flexibility for complex vision operations
- ONNX Runtime's optimization for fast text generation
"""

import argparse
import numpy as np
import torch
import onnxruntime_genai as og
import sys
import codecs

from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig

# Force UTF-8 output for Windows
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


class HybridQwen3VL:
    """
    Hybrid Qwen3-VL pipeline with PyTorch vision and ONNX text.
    """
    
    def __init__(self, pytorch_model_path, onnx_text_path, device="cpu"):
        """
        Initialize hybrid pipeline.
        
        Args:
            pytorch_model_path: Path to PyTorch model (for vision encoder)
            onnx_text_path: Path to ONNX text decoder
            device: Device for PyTorch model (cpu/cuda)
        """
        print("="*80)
        print("INITIALIZING HYBRID QWEN3-VL PIPELINE")
        print("="*80)
        print()
        
        self.device = device
        
        # Load PyTorch model (for vision encoder)
        print("[1/3] Loading PyTorch vision encoder...")
        self.pytorch_model = AutoModel.from_pretrained(
            pytorch_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        ).to(device)
        self.pytorch_model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            pytorch_model_path,
            trust_remote_code=True
        )
        
        self.config = AutoConfig.from_pretrained(
            pytorch_model_path,
            trust_remote_code=True
        )
        
        print(f"  Vision encoder loaded: {type(self.pytorch_model.visual).__name__}")
        print(f"  Device: {device}")
        
        # Load ONNX text decoder
        print("\n[2/3] Loading ONNX text decoder...")
        self.onnx_model = og.Model(onnx_text_path)
        self.tokenizer = og.Tokenizer(self.onnx_model)
        
        print(f"  Text decoder loaded from: {onnx_text_path}")
        print(f"  Vocab size: {self.config.text_config.vocab_size}")
        
        print("\n[3/3] Pipeline ready!")
        print()
        print("Architecture:")
        print("  Vision: PyTorch (dynamic shapes, full functionality)")
        print("  Text:   ONNX Runtime GenAI (INT4 quantized, optimized)")
        print()
    
    def encode_image(self, image_path):
        """
        Encode image using PyTorch vision encoder.
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            vision_features: [num_patches, hidden_size] numpy array
        """
        print(f"[Vision Encoding]")
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
            print(f"  Loaded image: {image.size}")
        else:
            image = image_path
            print(f"  Using provided image: {image.size}")
        
        # Process through Qwen3-VL processor
        prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(
            text=[prompt],
            images=[image],
            videos=None,
            return_tensors="pt"
        ).to(self.device)
        
        # Extract vision inputs
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]
        
        print(f"  Pixel values: {pixel_values.shape}")
        print(f"  Grid (T,H,W): {grid_thw.tolist()}")
        
        # Run through PyTorch vision encoder
        with torch.no_grad():
            vision_features = self.pytorch_model.visual(pixel_values, grid_thw)
        
        print(f"  Vision features: {vision_features.shape} (patches x hidden_dim)")
        
        # Convert to numpy
        vision_features_np = vision_features.cpu().numpy()
        
        return vision_features_np
    
    def generate_text(self, prompt, vision_features=None, max_length=50, temperature=0.7, top_p=0.9):
        """
        Generate text using ONNX text decoder.
        
        Args:
            prompt: Text prompt
            vision_features: Optional vision features from encode_image()
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text string
        """
        print(f"\n[Text Generation]")
        
        if vision_features is not None:
            # Multimodal: vision + text
            print(f"  Mode: Multimodal (vision + text)")
            print(f"  Vision features: {vision_features.shape}")
            print(f"  Text prompt: {prompt}")
            
            # For now, just use text (vision feature injection needs embedding layer)
            # TODO: Implement vision token injection into text sequence
            print(f"  WARNING: Vision feature injection not yet implemented")
            print(f"  Falling back to text-only mode")
            full_prompt = f"Image description: [vision features would go here]\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            # Text-only
            print(f"  Mode: Text-only")
            full_prompt = prompt
        
        print(f"  Prompt: {full_prompt[:100]}...")
        
        # Tokenize
        input_tokens = self.tokenizer.encode(full_prompt)
        print(f"  Input tokens: {len(input_tokens)}")
        
        # Set generation parameters
        params = og.GeneratorParams(self.onnx_model)
        params.set_search_options(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Generate
        generator = og.Generator(self.onnx_model, params)
        generator.append_tokens(input_tokens)
        
        # Stream output
        print(f"  Generating (max {max_length} tokens)...")
        print()
        print("  Response: ", end="", flush=True)
        
        tokenizer_stream = self.tokenizer.create_stream()
        generated_text = ""
        
        while not generator.is_done():
            generator.generate_next_token()
            
            new_token = generator.get_next_tokens()[0]
            token_text = tokenizer_stream.decode(new_token)
            print(token_text, end="", flush=True)
            generated_text += token_text
        
        print()
        
        return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Hybrid Qwen3-VL Inference (PyTorch Vision + ONNX Text)")
    
    parser.add_argument(
        "--pytorch_model",
        type=str,
        default="./pytorch",
        help="Path to PyTorch model (for vision encoder)"
    )
    parser.add_argument(
        "--onnx_text",
        type=str,
        default="./cpu-text",
        help="Path to ONNX text decoder"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (optional)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What do you see in this image?",
        help="Text prompt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PyTorch model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize hybrid pipeline
    pipeline = HybridQwen3VL(
        pytorch_model_path=args.pytorch_model,
        onnx_text_path=args.onnx_text,
        device=args.device
    )
    
    # Process image if provided
    vision_features = None
    if args.image:
        print("="*80)
        print("MULTIMODAL INFERENCE (Vision + Text)")
        print("="*80)
        print()
        
        vision_features = pipeline.encode_image(args.image)
    else:
        print("="*80)
        print("TEXT-ONLY INFERENCE")
        print("="*80)
        print()
    
    # Generate response
    response = pipeline.generate_text(
        prompt=args.prompt,
        vision_features=vision_features,
        max_length=args.max_length
    )
    
    # Summary
    print()
    print("="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print()
    print(f"Generated {len(response.split())} words")
    print()


if __name__ == "__main__":
    main()
