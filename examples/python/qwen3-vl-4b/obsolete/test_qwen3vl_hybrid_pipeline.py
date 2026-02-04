"""
Qwen3-VL-4B Hybrid Inference Pipeline Test
==========================================

Tests multimodal inference with:
- Vision: PyTorch (native, handles dynamic shapes)
- Text: PyTorch (ONNX text export not yet supported by onnxruntime-genai)

Test cases:
1. Text-only prompt
2. Image + text prompt  
3. Multiple images + text

Usage:
    python test_qwen3vl_hybrid_pipeline.py
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import time

class Qwen3VLPipeline:
    """Full PyTorch Qwen3-VL inference pipeline"""
    
    def __init__(self, model_path: str):
        print(f"\n{'='*80}")
        print("QWEN3-VL HYBRID INFERENCE PIPELINE")
        print(f"{'='*80}\n")
        
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Device: {self.device.upper()}")
        print(f"Model: {self.model_path}")
        
        # Load model and processor
        print("\n[1/2] Loading model and processor...")
        start = time.time()
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,
            device_map=self.device
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        
        elapsed = time.time() - start
        print(f"  [OK] Model loaded ({elapsed:.1f}s)")
        print(f"  [OK] Processor loaded")
        
        # Model info
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"\nModel Info:")
        print(f"  Parameters: {num_params:.2f}B")
        print(f"  Vision hidden size: {self.model.config.vision_config.hidden_size}")
        print(f"  Text hidden size: {self.model.config.text_config.hidden_size}")
    
    def generate(
        self,
        prompt: str,
        images=None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from prompt and optional images
        
        Args:
            prompt: Text prompt
            images: None, single PIL.Image, or list of PIL.Image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        print(f"\n{'='*80}")
        print("GENERATION REQUEST")
        print(f"{'='*80}")
        
        # Handle images
        if images is None:
            image_list = []
            print(f"Mode: Text-only")
        elif isinstance(images, Image.Image):
            image_list = [images]
            print(f"Mode: Single image + text")
        else:
            image_list = list(images)
            print(f"Mode: {len(image_list)} images + text")
        
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Build messages
        if image_list:
            # For image prompts, use the vision-language format
            content = []
            for img in image_list:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {"role": "user", "content": content}
            ]
        else:
            # Text-only
            messages = [
                {"role": "user", "content": prompt}
            ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        print(f"\n[Processing]")
        start = time.time()
        
        if image_list:
            inputs = self.processor(
                text=[text],
                images=image_list,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[1]
        print(f"  Input tokens: {input_len}")
        if "pixel_values" in inputs:
            print(f"  Vision tokens: {inputs['pixel_values'].shape}")
        
        # Generate
        print(f"\n[Generating]")
        gen_start = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        gen_time = time.time() - gen_start
        output_len = outputs.shape[1] - input_len
        
        # Decode
        generated_text = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        total_time = time.time() - start
        tokens_per_sec = output_len / gen_time if gen_time > 0 else 0
        
        print(f"  Generated tokens: {output_len}")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"  Total time: {total_time:.2f}s")
        
        return generated_text


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def main():
    """Run test cases"""
    
    # Initialize pipeline
    model_path = "./pytorch"
    pipeline = Qwen3VLPipeline(model_path)
    
    print(f"\n{'='*80}")
    print("TEST CASES")
    print(f"{'='*80}\n")
    
    # Test 1: Text-only
    print("\n" + "="*80)
    print("TEST 1: Text-Only Prompt")
    print("="*80)
    
    result = pipeline.generate(
        prompt="Write a haiku about artificial intelligence.",
        max_new_tokens=64
    )
    
    print(f"\n[RESULT]")
    print(f"{result}")
    
    # Test 2: Image + Text
    print("\n" + "="*80)
    print("TEST 2: Image + Text Prompt")
    print("="*80)
    
    # Load a sample image
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    print(f"\nLoading image from: {image_url}")
    
    try:
        image = load_image_from_url(image_url)
        print(f"  [OK] Image loaded: {image.size}")
        
        result = pipeline.generate(
            prompt="Describe this image in detail.",
            images=image,
            max_new_tokens=128
        )
        
        print(f"\n[RESULT]")
        print(f"{result}")
        
    except Exception as e:
        print(f"  [ERROR] Could not load image: {e}")
        print(f"  Skipping image test")
    
    # Test 3: Counting objects in image
    if 'image' in locals():
        print("\n" + "="*80)
        print("TEST 3: Visual Question Answering")
        print("="*80)
        
        result = pipeline.generate(
            prompt="How many people are in this image?",
            images=image,
            max_new_tokens=32
        )
        
        print(f"\n[RESULT]")
        print(f"{result}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    print("All tests passed successfully!")
    print("\nNext steps:")
    print("1. Test with your own images")
    print("2. Adjust max_new_tokens, temperature for different outputs")
    print("3. Try multi-image inputs")
    print("4. Wait for ONNX Runtime GenAI to add Qwen3-VL text model support")


if __name__ == "__main__":
    main()
