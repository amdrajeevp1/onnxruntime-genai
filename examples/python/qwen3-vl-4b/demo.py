"""
Qwen3-VL ONNX Demo with Real Images
Test the complete pipeline with actual photos
"""

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Import our pipeline
import importlib.util
spec = importlib.util.spec_from_file_location("qwen3vl_mm", "qwen3vl-mm.py")
qwen3vl_mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3vl_mm)

Qwen3VLONNXPipeline = qwen3vl_mm.Qwen3VLONNXPipeline


def create_test_images():
    """Create sample test images"""
    print("Creating test images...")
    
    # Image 1: Gradient image (landscape)
    img1 = Image.new('RGB', (512, 384))
    pixels1 = img1.load()
    for y in range(384):
        for x in range(512):
            r = int((x / 512) * 255)
            g = int((y / 384) * 255)
            b = 128
            pixels1[x, y] = (r, g, b)
    img1.save("test_gradient.jpg")
    print("  [OK] Created test_gradient.jpg (512x384)")
    
    # Image 2: Color blocks
    img2 = Image.new('RGB', (400, 400))
    pixels2 = img2.load()
    for y in range(400):
        for x in range(400):
            if x < 200 and y < 200:
                pixels2[x, y] = (255, 0, 0)  # Red
            elif x >= 200 and y < 200:
                pixels2[x, y] = (0, 255, 0)  # Green
            elif x < 200 and y >= 200:
                pixels2[x, y] = (0, 0, 255)  # Blue
            else:
                pixels2[x, y] = (255, 255, 0)  # Yellow
    img2.save("test_colors.jpg")
    print("  [OK] Created test_colors.jpg (400x400)")
    
    # Image 3: Checkerboard pattern
    img3 = Image.new('RGB', (320, 320))
    pixels3 = img3.load()
    square_size = 40
    for y in range(320):
        for x in range(320):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                pixels3[x, y] = (255, 255, 255)  # White
            else:
                pixels3[x, y] = (0, 0, 0)  # Black
    img3.save("test_checkerboard.jpg")
    print("  [OK] Created test_checkerboard.jpg (320x320)")
    
    return ["test_gradient.jpg", "test_colors.jpg", "test_checkerboard.jpg"]


def run_demo():
    """Run complete demo"""
    print("="*70)
    print("Qwen3-VL ONNX Pipeline Demo")
    print("="*70)
    print()
    
    # Create test images
    image_paths = create_test_images()
    print()
    
    # Initialize pipeline
    print("Loading Qwen3-VL ONNX pipeline...")
    pipeline = Qwen3VLONNXPipeline(
        model_dir=".",
        execution_provider="CPUExecutionProvider"
    )
    print()
    
    # Test configurations
    tests = [
        {
            "name": "Test 1: Text-only (Greedy)",
            "image": None,
            "prompt": "What is the capital of France?",
            "params": {
                "max_new_tokens": 20,
                "temperature": 0.0,
                "do_sample": False,
                "stream": True
            }
        },
        {
            "name": "Test 2: Image Description (Sampling)",
            "image": image_paths[0],
            "prompt": "Describe this image in detail.",
            "params": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "do_sample": True,
                "stream": True
            }
        },
        {
            "name": "Test 3: Image Colors (Low Temperature)",
            "image": image_paths[1],
            "prompt": "What colors do you see in this image?",
            "params": {
                "max_new_tokens": 30,
                "temperature": 0.3,
                "top_k": 40,
                "top_p": 0.95,
                "do_sample": True,
                "stream": True
            }
        },
        {
            "name": "Test 4: Pattern Recognition (High Temperature)",
            "image": image_paths[2],
            "prompt": "What pattern do you see?",
            "params": {
                "max_new_tokens": 40,
                "temperature": 0.9,
                "top_k": 100,
                "top_p": 0.85,
                "do_sample": True,
                "stream": True
            }
        }
    ]
    
    results = []
    
    for test in tests:
        print("\n" + "="*70)
        print(test["name"])
        print("="*70)
        print(f"Prompt: {test['prompt']}")
        if test["image"]:
            print(f"Image: {test['image']}")
        print(f"Parameters: temp={test['params']['temperature']}, "
              f"top_k={test['params'].get('top_k', 'N/A')}, "
              f"top_p={test['params'].get('top_p', 'N/A')}")
        print()
        
        # Prepare prompt
        if test["image"]:
            prompt = f"{test['prompt']}\n{pipeline.image_token}"
            image_list = [test["image"]]
        else:
            prompt = test['prompt']
            image_list = None
        
        # Generate
        start_time = time.time()
        try:
            output = pipeline.generate(
                text=prompt,
                image_paths=image_list,
                **test["params"]
            )
            elapsed = time.time() - start_time
            
            results.append({
                "test": test["name"],
                "success": True,
                "output": output,
                "time": elapsed
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] {e}")
            results.append({
                "test": test["name"],
                "success": False,
                "error": str(e),
                "time": elapsed
            })
        
        print()
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for i, result in enumerate(results, 1):
        status = "[PASS]" if result["success"] else "[FAIL]"
        print(f"{i}. {result['test']}: {status} ({result['time']:.2f}s)")
        if result["success"]:
            preview = result["output"][:60] + "..." if len(result["output"]) > 60 else result["output"]
            print(f"   Output: {preview}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    return results


if __name__ == "__main__":
    try:
        results = run_demo()
        
        # Exit code
        all_passed = all(r["success"] for r in results)
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
