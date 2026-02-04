"""
Qwen3-VL-4B Builder
===================

Exports Qwen3-VL model components to ONNX for onnxruntime-genai.

Components:
- Vision: PyTorch vision encoder (ONNX export for reference)
- Text: INT4 quantized text decoder via onnxruntime-genai builder

Usage:
    python builder_qwen3vl.py -i ./pytorch -o ./qwen3vl-onnx -p fp32 -e cpu
"""

import argparse
import onnx
import os
import shutil
import subprocess
import sys
import time
import torch

from pathlib import Path
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


def safe_rmtree(path, retries=3, delay=1):
    """Safely remove directory tree with retry logic for Windows file locks."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as e:
            if attempt < retries - 1:
                print(f"Warning: Failed to remove {path}, retrying in {delay}s... ({e})")
                time.sleep(delay)
            else:
                print(f"Warning: Could not remove {path}, continuing anyway... ({e})")
        except Exception as e:
            print(f"Warning: Error removing {path}: {e}")


def build_vision(args, model, processor):
    """
    Export Qwen3-VL vision model to ONNX
    
    Note: This is mainly for reference. The hybrid pipeline uses PyTorch vision
    due to dynamic shape handling issues in ONNX Runtime.
    """
    print(f"\n{'='*80}")
    print("BUILDING VISION MODEL")
    print(f"{'='*80}\n")
    
    # Prepare sample image
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    try:
        import requests
        from io import BytesIO
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        # Fallback to a dummy image
        image = Image.new('RGB', (448, 448), color='white')
    
    # Process image
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    # Extract vision model
    vision_model = model.visual
    vision_model.eval()
    
    # Get dummy inputs
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["grid_thw"]
    
    print(f"Vision model input shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw.shape}")
    
    # Export to ONNX
    vision_onnx_path = os.path.join(args.output, "qwen3vl-vision.onnx")
    
    print(f"\nExporting vision model to {vision_onnx_path}...")
    
    torch.onnx.export(
        vision_model,
        (pixel_values, grid_thw),
        vision_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values", "grid_thw"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "grid_thw": {0: "num_images"}
        }
    )
    
    print(f"[OK] Vision model exported: {os.path.getsize(vision_onnx_path) / (1024**3):.2f} GB")
    
    # Validate
    onnx.checker.check_model(vision_onnx_path)
    print("[OK] ONNX model validation passed")
    
    return vision_onnx_path


def build_text(args):
    """
    Export Qwen3-VL text model using onnxruntime-genai builder
    
    This uses the Qwen3VLTextModel builder class we added to the genai package.
    """
    print(f"\n{'='*80}")
    print("BUILDING TEXT MODEL")
    print(f"{'='*80}\n")
    
    text_output = os.path.join(args.output, "qwen3vl-text")
    
    # Build command for onnxruntime-genai builder
    builder_cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", args.input,
        "-o", text_output,
        "-p", "int4",
        "-e", args.execution_provider,
    ]
    
    if args.cache_dir:
        builder_cmd.extend(["-c", args.cache_dir])
    
    print(f"Running command:")
    print(f"  {' '.join(builder_cmd)}\n")
    
    # Run builder
    result = subprocess.run(
        builder_cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Builder failed:")
        print(result.stderr)
        raise RuntimeError(f"Text model export failed with return code {result.returncode}")
    
    print(result.stdout)
    
    if os.path.exists(text_output):
        print(f"[OK] Text model exported to {text_output}")
        
        # List exported files
        print(f"\nExported files:")
        for file in sorted(os.listdir(text_output)):
            fpath = os.path.join(text_output, file)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024**2)
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        return text_output
    else:
        raise RuntimeError(f"Text model directory not found: {text_output}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Build Qwen3-VL ONNX models for onnxruntime-genai"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to Qwen3-VL PyTorch model directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for ONNX models"
    )
    
    parser.add_argument(
        "-p", "--precision",
        required=True,
        choices=["fp16", "fp32"],
        help="Precision for PyTorch components (vision model)"
    )
    
    parser.add_argument(
        "-e", "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml"],
        help="Execution provider"
    )
    
    parser.add_argument(
        "-c", "--cache_dir",
        default=None,
        help="Cache directory for temporary files"
    )
    
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision model export (for testing text-only)"
    )
    
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text model export (for testing vision-only)"
    )
    
    args = parser.parse_args()
    
    # Convert precision string to torch dtype
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Set device
    args.device = "cuda" if args.execution_provider == "cuda" else "cpu"
    
    return args


def main():
    print(f"\n{'='*80}")
    print("QWEN3-VL ONNX MODEL BUILDER")
    print(f"{'='*80}\n")
    
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Precision: {args.precision}")
    print(f"  Execution Provider: {args.execution_provider}")
    print(f"  Device: {args.device}")
    
    # Load model and processor (for vision export)
    if not args.skip_vision:
        print(f"\nLoading Qwen3-VL model...")
        config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.input,
            torch_dtype=args.precision,
            trust_remote_code=True
        ).to(args.device)
        model.eval()
        print(f"[OK] Model loaded")
    
    # Build components
    vision_path = None
    text_path = None
    
    try:
        if not args.skip_vision:
            vision_path = build_vision(args, model, processor)
        
        if not args.skip_text:
            text_path = build_text(args)
        
        # Summary
        print(f"\n{'='*80}")
        print("BUILD COMPLETE")
        print(f"{'='*80}\n")
        
        if vision_path:
            print(f"Vision Model: {vision_path}")
        if text_path:
            print(f"Text Model:   {text_path}")
        
        print(f"\nOutput directory: {args.output}")
        print(f"\nTo use these models:")
        print(f"  - Vision: Use PyTorch for best compatibility (see run_qwen3vl_onnx_pipeline.py)")
        print(f"  - Text: Use onnxruntime-genai (INT4 quantized)")
        
    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
