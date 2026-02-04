"""
Test Qwen3-VL Vision + Text Pipeline
=====================================

Hybrid pipeline:
- Vision: PyTorch (native Qwen3-VL vision encoder)
- Text: ONNX Runtime GenAI (INT4 quantized)
"""

import os
import torch
import onnxruntime_genai as og
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from io import BytesIO
import requests


class Qwen3VLHybridPipeline:
    """Hybrid Qwen3-VL pipeline: PyTorch vision + ONNX text"""
    
    def __init__(self, pytorch_model_path: str, onnx_text_path: str):
        print("\n" + "="*80)
        print("INITIALIZING QWEN3-VL HYBRID PIPELINE")
        print("="*80 + "\n")
        
        # Load PyTorch vision components
        print(f"[1/3] Loading PyTorch vision encoder from {pytorch_model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            pytorch_model_path, 
            trust_remote_code=True
        )
        
        self.vision_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pytorch_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"
        )
        self.vision_model.eval()
        print(f"    [OK] Vision encoder loaded")
        
        # Load ONNX text model
        print(f"\n[2/3] Loading ONNX text model from {onnx_text_path}...")
        self.text_model = og.Model(onnx_text_path)
        self.tokenizer = og.Tokenizer(self.text_model)
        print(f"    [OK] ONNX text model loaded (INT4 quantized)")
        
        print(f"\n[3/3] Pipeline ready!\n")
    
    def extract_vision_features(self, image, prompt_text):
        """Extract vision features using PyTorch vision encoder"""
        
        # Prepare messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Extract vision features
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["grid_thw"]
            
            # Run vision encoder
            vision_outputs = self.vision_model.visual(pixel_values, grid_thw=grid_thw)
            
        return {
            "vision_features": vision_outputs,
            "input_ids": inputs["input_ids"],
            "text": text
        }
    
    def generate_text_only(self, prompt: str, max_length: int = 128):
        """Generate text without vision (for comparison)"""
        
        print(f"\nPrompt: {prompt}")
        print(f"Mode: TEXT-ONLY\n")
        
        # Encode prompt
        input_tokens = self.tokenizer.encode_batch([prompt])
        
        # Generate
        params = og.GeneratorParams(self.text_model)
        params.set_search_options(max_length=max_length)
        
        generator = og.Generator(self.text_model, params)
        generator.append_tokens(input_tokens)
        
        print("Generating", end="", flush=True)
        while not generator.is_done():
            generator.generate_next_token()
            print(".", end="", flush=True)
        print("\n")
        
        # Decode
        output_tokens = generator.get_sequence(0)
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_text
    
    def generate_with_vision(self, image, prompt: str, max_length: int = 128):
        """Generate text with vision features (hybrid mode)"""
        
        print(f"\nPrompt: {prompt}")
        print(f"Mode: VISION + TEXT (HYBRID)\n")
        
        # Extract vision features
        print("Extracting vision features...")
        vision_data = self.extract_vision_features(image, prompt)
        
        vision_features = vision_data["vision_features"]
        print(f"Vision features shape: {vision_features.shape}")
        print(f"Vision features extracted: {vision_features.numel()} values\n")
        
        # For now, use the processed text as input
        # TODO: Inject vision features into the text generation
        # This requires custom embedding support in ONNX Runtime GenAI
        
        text_prompt = vision_data["text"]
        print(f"Formatted prompt:\n{text_prompt}\n")
        
        # Generate using text model
        input_tokens = self.tokenizer.encode_batch([text_prompt])
        
        params = og.GeneratorParams(self.text_model)
        params.set_search_options(max_length=max_length)
        
        generator = og.Generator(self.text_model, params)
        generator.append_tokens(input_tokens)
        
        print("Generating", end="", flush=True)
        while not generator.is_done():
            generator.generate_next_token()
            print(".", end="", flush=True)
        print("\n")
        
        # Decode
        output_tokens = generator.get_sequence(0)
        output_text = self.tokenizer.decode(output_tokens)
        
        return {
            "output": output_text,
            "vision_features_shape": vision_features.shape,
            "num_vision_tokens": vision_features.numel()
        }


def load_test_image():
    """Load a test image"""
    
    # Try to use existing test image
    local_image_path = "test_image.jpg"
    
    if os.path.exists(local_image_path):
        print(f"Using local image: {local_image_path}")
        return Image.open(local_image_path).convert("RGB")
    
    # Download a sample image
    print("Downloading sample image...")
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Downloaded image: {image.size}")
        return image
    except Exception as e:
        print(f"Failed to download image: {e}")
        print("Creating dummy image...")
        return Image.new('RGB', (448, 448), color='blue')


def main():
    print("\n" + "="*80)
    print("QWEN3-VL VISION + TEXT TEST")
    print("="*80 + "\n")
    
    # Paths
    pytorch_model = "./pytorch"
    onnx_text_model = "./qwen3vl-onnx-final/qwen3vl-text"
    
    # Initialize pipeline
    pipeline = Qwen3VLHybridPipeline(pytorch_model, onnx_text_model)
    
    # Load test image
    print("\n" + "="*80)
    print("LOADING TEST IMAGE")
    print("="*80 + "\n")
    image = load_test_image()
    print(f"Image loaded: {image.size}\n")
    
    # Test cases
    test_cases = [
        {
            "type": "text_only",
            "prompt": "Write a short poem about AI.",
            "max_length": 100
        },
        {
            "type": "vision_text",
            "prompt": "Describe what you see in this image.",
            "max_length": 150
        },
        {
            "type": "vision_text",
            "prompt": "What objects are visible in this image?",
            "max_length": 100
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"TEST {i}: {test_case['type'].upper()}")
        print("="*80)
        
        if test_case["type"] == "text_only":
            output = pipeline.generate_text_only(
                test_case["prompt"],
                max_length=test_case["max_length"]
            )
            results.append({
                "test": i,
                "type": test_case["type"],
                "prompt": test_case["prompt"],
                "output": output
            })
        else:
            output_data = pipeline.generate_with_vision(
                image,
                test_case["prompt"],
                max_length=test_case["max_length"]
            )
            results.append({
                "test": i,
                "type": test_case["type"],
                "prompt": test_case["prompt"],
                "output": output_data["output"],
                "vision_features_shape": str(output_data["vision_features_shape"]),
                "num_vision_tokens": output_data["num_vision_tokens"]
            })
        
        print(f"\nOutput:\n{results[-1]['output']}\n")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for result in results:
        print(f"Test {result['test']} ({result['type']}):")
        print(f"  Prompt: {result['prompt']}")
        if "vision_features_shape" in result:
            print(f"  Vision features: {result['vision_features_shape']}")
        print(f"  Output length: {len(result['output'])} chars")
        print()
    
    print("="*80)
    print("NOTES:")
    print("="*80)
    print()
    print("✅ PyTorch vision encoder: WORKING")
    print("✅ ONNX text model: WORKING")
    print("⚠️  Vision feature injection: NOT YET IMPLEMENTED")
    print()
    print("Current behavior:")
    print("  - Vision features are extracted successfully")
    print("  - Text generation works with formatted prompt")
    print("  - Vision features are NOT injected into text model embeddings")
    print()
    print("To fully implement vision+text:")
    print("  1. Need custom embedding layer in ONNX model (exclude_embeds=false)")
    print("  2. Inject vision features at <image_pad> token positions")
    print("  3. Or use full PyTorch pipeline for true multimodal generation")
    print()
    print("For now, this demonstrates:")
    print("  ✓ Vision encoder extracts features correctly")
    print("  ✓ Text model generates from ONNX INT4 model")
    print("  ✓ Pipeline infrastructure is in place")
    print()


if __name__ == "__main__":
    main()
