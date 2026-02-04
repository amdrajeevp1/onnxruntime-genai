"""
Test script for Qwen3-VL ONNX multimodal inference
"""

import sys
from pathlib import Path
import importlib.util

# Load module from file with hyphen
spec = importlib.util.spec_from_file_location("qwen3vl_mm", "qwen3vl-mm.py")
qwen3vl_mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3vl_mm)

# Test with a simple text-only first
def test_text_only():
    print("="*70)
    print("TEST 1: Text-only inference")
    print("="*70)
    
    Qwen3VLONNXPipeline = qwen3vl_mm.Qwen3VLONNXPipeline
    
    pipeline = Qwen3VLONNXPipeline(
        model_dir=".",
        execution_provider="CPUExecutionProvider"
    )
    
    text = "Hello, how are you?"
    
    try:
        output = pipeline.generate(
            text=text,
            image_paths=None,
            max_new_tokens=20,
            temperature=0.0,  # Greedy for consistent testing
            do_sample=False,
            stream=True
        )
        print(f"\nSuccess! Output: {output}")
        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test with image
def test_with_image():
    print("\n" + "="*70)
    print("TEST 2: Image + text inference")
    print("="*70)
    
    # Create a dummy test image
    from PIL import Image
    import numpy as np
    
    # Create 224x224 RGB image
    test_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    test_img.save("test_image.jpg")
    print("Created test_image.jpg")
    
    Qwen3VLONNXPipeline = qwen3vl_mm.Qwen3VLONNXPipeline
    
    pipeline = Qwen3VLONNXPipeline(
        model_dir=".",
        execution_provider="CPUExecutionProvider"
    )
    
    text = f"Describe this image.\n{pipeline.image_token}"
    
    try:
        output = pipeline.generate(
            text=text,
            image_paths=["test_image.jpg"],
            max_new_tokens=30,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            stream=True
        )
        print(f"\nSuccess! Output: {output}")
        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Qwen3-VL ONNX Pipeline\n")
    
    # Test 1: Text only
    result1 = test_text_only()
    
    # Test 2: Image + text
    result2 = test_with_image()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Text-only: {'PASS' if result1 else 'FAIL'}")
    print(f"Image + text: {'PASS' if result2 else 'FAIL'}")
