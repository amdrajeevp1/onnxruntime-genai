"""
Qwen3-VL-4B Vision Encoder - FIXED SHAPE Export

This exports the vision encoder with concrete, fixed shapes (no dynamic axes).
Supports a single image size: 336x336 pixels = 21x21 patches.

This is a practical solution for production use where you can standardize
all input images to a specific size.
"""

import argparse
import numpy as np
import onnx
import os
import shutil
import time
import torch

from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModel


def safe_rmtree(path, retries=3, delay=1):
    """Safely remove directory tree with retry logic for Windows file locks."""
    for attempt in range(retries):
        try:
            if os.path.exists(path):
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


def build_vision_fixed(args):
    """
    Export Qwen3-VL vision encoder with FIXED SHAPES.
    
    FIXED SIZE: 336x336 pixels = 21x21 patches = 441 patches
    """
    print("="*80)
    print("QWEN3-VL VISION ENCODER - FIXED SHAPE EXPORT")
    print("="*80)
    print()
    print("Fixed Input Size: 336x336 pixels (21x21 patches)")
    print("No dynamic shape support - all images must be resized to 336x336")
    print()
    
    # Load model
    print("[1/6] Loading Qwen3-VL model...")
    model_path = args.input
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=args.precision,
        attn_implementation="eager"  # Required for ONNX export
    ).to(args.device)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Device: {args.device}")
    print(f"  Precision: {args.precision}")
    
    # Extract vision encoder
    print("\n[2/6] Extracting vision encoder...")
    vision_encoder = model.visual
    vision_encoder.eval()
    
    print(f"  Vision encoder type: {type(vision_encoder).__name__}")
    print(f"  Hidden size: {config.vision_config.hidden_size}")
    print(f"  Num heads: {config.vision_config.num_heads}")
    print(f"  Depth: {config.vision_config.depth}")
    
    # Prepare test inputs with FIXED SIZE
    print("\n[3/6] Preparing fixed-size test inputs...")
    
    # Create a 336x336 test image
    FIXED_SIZE = 336  # Must be divisible by 16 (patch size)
    test_image = Image.new('RGB', (FIXED_SIZE, FIXED_SIZE), color='white')
    
    # Process through Qwen3-VL processor
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image."
    inputs = processor(
        text=[prompt],
        images=[test_image],
        videos=None,
        return_tensors="pt"
    ).to(args.device)
    
    print(f"  Test image size: {FIXED_SIZE}x{FIXED_SIZE}")
    print(f"  Available keys: {list(inputs.keys())}")
    
    # Extract vision inputs
    if 'pixel_values' not in inputs or 'image_grid_thw' not in inputs:
        print(f"  ERROR: Required keys not found!")
        print(f"  Available: {list(inputs.keys())}")
        raise KeyError("Missing pixel_values or image_grid_thw")
    
    pixel_values = inputs["pixel_values"].to(args.precision)
    grid_thw = inputs["image_grid_thw"]
    
    print(f"  pixel_values: {pixel_values.shape} {pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw.shape} {grid_thw.dtype} = {grid_thw.tolist()}")
    
    num_patches = pixel_values.shape[0]
    print(f"  Total patches: {num_patches}")
    
    # Export with TorchScript (NO dynamic axes)
    print("\n[4/6] Exporting to ONNX with FIXED shapes...")
    
    filename = "qwen3-vl-vision-fixed.onnx"
    temp_folder_1 = os.path.join(args.output, "vision_fixed_init_export")
    os.makedirs(temp_folder_1, exist_ok=True)
    fpath_1 = os.path.join(temp_folder_1, filename)
    
    print(f"  Output: {fpath_1}")
    print("  Exporting with TorchScript (no dynamic axes)...")
    start_time = time.time()
    
    # NO DYNAMIC AXES - export with concrete shapes
    torch.onnx.export(
        vision_encoder,
        args=(pixel_values, grid_thw),
        f=fpath_1,
        export_params=True,
        input_names=["pixel_values", "grid_thw"],
        output_names=["vision_features"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False  # TorchScript
        # NO dynamic_axes parameter!
    )
    
    elapsed = time.time() - start_time
    print(f"  Export completed in {elapsed:.1f}s")
    
    # Validate exported model
    print("\n[5/6] Validating exported model...")
    try:
        onnx.checker.check_model(fpath_1)
        print("  Model is valid")
        
        # Shape inference
        onnx.shape_inference.infer_shapes_path(fpath_1)
        print("  Shape inference complete")
        
    except Exception as e:
        print(f"  Warning: Validation issue: {e}")
        print("  Continuing anyway...")
    
    # Load and inspect shapes
    print("\n[6/6] Finalizing model...")
    onnx_model = onnx.load_model(fpath_1, load_external_data=True)
    
    # Print actual shapes
    print(f"\n  Input shapes (FIXED):")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    
    print(f"\n  Output shapes (FIXED):")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")
    
    # Save to final location
    fpath_2 = os.path.join(args.output, filename)
    print(f"\n  Saving to {fpath_2}...")
    
    onnx.save_model(
        onnx_model,
        fpath_2,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False
    )
    
    # Clean up temp folder
    safe_rmtree(temp_folder_1)
    
    # Get file sizes
    model_size = os.path.getsize(fpath_2) / (1024**2)
    data_size = os.path.getsize(fpath_2 + ".data") / (1024**3)
    
    print()
    print("="*80)
    print("EXPORT SUCCESSFUL!")
    print("="*80)
    print()
    print(f"Model: {fpath_2}")
    print(f"  - Model file: {model_size:.2f} MB")
    print(f"  - Data file: {data_size:.2f} GB")
    print()
    print("FIXED INPUT SIZE: 336x336 pixels")
    print(f"  - Patches: 21x21 = {num_patches} patches")
    print(f"  - pixel_values shape: [{num_patches}, 1536]")
    print(f"  - grid_thw shape: [1, 3] (always [1, 21, 21])")
    print()
    print("IMPORTANT: All input images MUST be resized to 336x336!")
    print()
    print("Next step: Test with test_qwen3vl.py")
    print()


def get_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL Vision Encoder ONNX Export (Fixed Shape)")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input PyTorch model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for ONNX models"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision for export (fp32 or fp16)"
    )
    parser.add_argument(
        "--execution_provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "dml"],
        help="Execution provider (cpu, cuda, or dml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model export (cpu or cuda)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model downloads"
    )
    
    args = parser.parse_args()
    
    # Convert precision string to torch dtype
    if args.precision == "fp16":
        args.precision = torch.float16
    else:
        args.precision = torch.float32
    
    return args


if __name__ == "__main__":
    args = get_args()
    build_vision_fixed(args)
