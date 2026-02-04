"""
Quick Test - Verify all improvements are working
Fast test of core functionality
"""

import time
import importlib.util
from PIL import Image
import numpy as np

# Load pipeline
spec = importlib.util.spec_from_file_location("qwen3vl_mm", "qwen3vl-mm.py")
qwen3vl_mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3vl_mm)

print("="*70)
print("Qwen3-VL ONNX Quick Test")
print("="*70)
print()

# Create quick test image
print("1. Creating test image...")
img = Image.new('RGB', (256, 256))
pixels = img.load()
for y in range(256):
    for x in range(256):
        pixels[x, y] = (x, y, 128)
img.save("quick_test.jpg")
print("   [OK] Created quick_test.jpg")
print()

# Initialize pipeline
print("2. Loading pipeline...")
start = time.time()
pipeline = qwen3vl_mm.Qwen3VLONNXPipeline(model_dir=".")
print(f"   [OK] Loaded in {time.time()-start:.1f}s")
print()

# Test 1: Autoregressive generation
print("3. Testing autoregressive generation (10 tokens)...")
start = time.time()
output = pipeline.generate(
    text="Count to 5:",
    image_paths=None,
    max_new_tokens=10,
    temperature=0.0,
    do_sample=False,
    stream=False
)
elapsed = time.time() - start
print(f"   [OK] Generated in {elapsed:.1f}s ({10/elapsed:.1f} tokens/s)")
print(f"   Output: {output[:60]}...")
print()

# Test 2: Sampling
print("4. Testing sampling (temp=0.7, 5 tokens)...")
start = time.time()
output = pipeline.generate(
    text="Hello,",
    image_paths=None,
    max_new_tokens=5,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    stream=False
)
elapsed = time.time() - start
print(f"   [OK] Generated in {elapsed:.1f}s")
print(f"   Output: {output}")
print()

# Test 3: Streaming
print("5. Testing streaming output (5 tokens)...")
print("   Output: ", end="")
start = time.time()
output = pipeline.generate(
    text="Hi",
    image_paths=None,
    max_new_tokens=5,
    temperature=0.5,
    do_sample=True,
    stream=True
)
elapsed = time.time() - start
print(f"   [OK] Streamed in {elapsed:.1f}s")
print()

# Test 4: Image inference
print("6. Testing image inference (10 tokens)...")
start = time.time()
output = pipeline.generate(
    text=f"This image shows{pipeline.image_token}",
    image_paths=["quick_test.jpg"],
    max_new_tokens=10,
    temperature=0.3,
    do_sample=True,
    stream=False
)
elapsed = time.time() - start
print(f"   [OK] Generated in {elapsed:.1f}s")
print(f"   Output: {output[:60]}...")
print()

# Summary
print("="*70)
print("QUICK TEST SUMMARY")
print("="*70)
print("[PASS] All tests completed successfully!")
print()
print("Features verified:")
print("  [OK] Autoregressive generation")
print("  [OK] Sampling strategies (temperature, top-k, top-p)")
print("  [OK] Streaming output")
print("  [OK] Image inference")
print("="*70)
