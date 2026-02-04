"""
Qwen3-VL-4B Vision Encoder ONNX Export Builder

This builder exports the vision component of Qwen3-VL-4B to ONNX format.

Based on Phi-4 multimodal builder pattern but adapted for Qwen3-VL architecture:
- ViT-based vision encoder with DeepStack (multi-level feature extraction)
- Rotary position embeddings for vision
- Spatial merging and temporal processing

Architecture details:
- Vision encoder: 24 layers, hidden_size 1024, 16 attention heads
- Patch size: 16x16
- DeepStack indexes: [5, 11, 17] - extracts features at these layer depths
- Output size: 2560 (matches text model hidden size)
"""
import argparse
import numpy as np
import onnx
import os
import requests
import shutil
import time
import torch

from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModel

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


def build_vision(args):
    """
    Export Qwen3-VL vision encoder to ONNX.
    
    The vision encoder:
    - Takes pixel_values as input (images)
    - Processes through ViT with DeepStack
    - Outputs vision features for text model
    """
    print("\n" + "="*80)
    print("BUILDING QWEN3-VL VISION ENCODER")
    print("="*80)
    
    # Load model and processor
    print("\n[1/6] Loading model and processor...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.input, 
        trust_remote_code=True, 
        torch_dtype=args.precision,
        attn_implementation="eager"  # Disable SDPA for ONNX export compatibility
    ).to(args.device)
    
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Vision module: {type(model.visual).__name__}")
    
    # Prepare test images
    print("\n[2/6] Preparing test images...")
    test_image_urls = [
        "https://www.ilankelman.org/stopsigns/australia.jpg",
        "https://th.bing.com/th/id/OIP.gCvQ1vmPVJmrq1nnzM3ZHQHaEo?rs=1&pid=ImgDetMain"
    ]
    
    images = []
    for i, url in enumerate(test_image_urls, 1):
        try:
            print(f"  Downloading test image {i}...", end=" ")
            img = Image.open(requests.get(url, stream=True, verify=False).raw)
            images.append(img)
            print(f"✓ ({img.size})")
        except Exception as e:
            print(f"✗ Failed: {e}")
            # Create a dummy image if download fails
            print(f"  Creating dummy image {i}...")
            img = Image.new('RGB', (224, 224), color=(100+i*50, 150, 200))
            images.append(img)
    
    # Process images with the processor
    print("\n[3/6] Processing images through Qwen3VL processor...")
    
    # Create a simple prompt for processing
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe these images."
    
    inputs = processor(
        text=[prompt],
        images=images,
        videos=None,
        return_tensors="pt"
    ).to(args.device)
    
    print(f"✓ Inputs processed")
    print(f"  Keys: {list(inputs.keys())}")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Extract vision module
    print("\n[4/6] Extracting vision encoder...")
    vision_encoder = model.visual
    vision_encoder.eval()
    
    # Prepare dummy inputs for export
    # Qwen3-VL vision encoder needs both pixel_values and grid_thw
    if 'pixel_values' not in inputs or 'image_grid_thw' not in inputs:
        print(f"✗ Error: Required inputs not found!")
        print(f"  Available keys: {list(inputs.keys())}")
        return
    
    dummy_pixel_values = inputs['pixel_values'].to(args.precision)
    dummy_grid_thw = inputs['image_grid_thw']
    
    print(f"✓ pixel_values ready: {dummy_pixel_values.shape}")
    print(f"✓ image_grid_thw ready: {dummy_grid_thw.shape}")
    
    # Define dynamic axes for variable batch/image sizes
    dynamic_axes = {
        "pixel_values": {0: "num_patches"},
        "grid_thw": {0: "num_images"},
        "vision_features": {0: "num_patches"}
    }
    
    filename = "qwen3-vl-vision.onnx"
    
    # Create temporary export directory
    print("\n[5/6] Exporting to ONNX...")
    temp_folder_1 = os.path.join(args.output, "vision_init_export")
    os.makedirs(temp_folder_1, exist_ok=True)
    
    fpath_1 = os.path.join(temp_folder_1, filename)
    
    print(f"  Using TorchScript export mode (dynamo=False)")
    print(f"  Output file: {fpath_1}")
    print(f"  Input 1: pixel_values {dummy_pixel_values.shape}")
    print(f"  Input 2: grid_thw {dummy_grid_thw.shape}")
    
    try:
        torch.onnx.export(
            vision_encoder,
            args=(dummy_pixel_values, dummy_grid_thw),
            f=fpath_1,
            export_params=True,
            input_names=["pixel_values", "grid_thw"],
            output_names=["vision_features"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,  # Use TorchScript for compatibility
        )
        print(f"✓ ONNX export successful!")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify and save
    print("\n[6/6] Verifying and saving ONNX model...")
    try:
        onnx.checker.check_model(fpath_1)
        print(f"✓ Model validation passed")
        
        onnx.shape_inference.infer_shapes_path(fpath_1)
        print(f"✓ Shape inference completed")
        
        onnx_model = onnx.load_model(fpath_1, load_external_data=True)
        
        # Save to final location
        temp_folder_2 = os.path.join(args.output, "vision_final")
        os.makedirs(temp_folder_2, exist_ok=True)
        
        fpath_2 = os.path.join(temp_folder_2, filename)
        onnx.save_model(
            onnx_model,
            fpath_2,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{filename}.data",
            size_threshold=0,
            convert_attribute=False,
        )
        print(f"✓ Model saved to {fpath_2}")
        
        # Copy to output directory
        final_path = os.path.join(args.output, filename)
        final_data_path = os.path.join(args.output, f"{filename}.data")
        
        shutil.copy(fpath_2, final_path)
        if os.path.exists(os.path.join(temp_folder_2, f"{filename}.data")):
            shutil.copy(os.path.join(temp_folder_2, f"{filename}.data"), final_data_path)
        
        print(f"✓ Final model: {final_path}")
        
        # Cleanup
        print(f"\n[CLEANUP] Removing temporary directories...")
        safe_rmtree(temp_folder_1)
        safe_rmtree(temp_folder_2)
        print(f"✓ Cleanup complete")
        
    except Exception as e:
        print(f"✗ Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ VISION ENCODER EXPORT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-4B vision encoder to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Path to PyTorch model directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory for ONNX model")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Model precision")
    parser.add_argument("--execution_provider", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution provider")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model loading")
    
    args = parser.parse_args()
    
    # Convert precision string to torch dtype
    if args.precision == "fp32":
        args.precision = torch.float32
    elif args.precision == "fp16":
        args.precision = torch.float16
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*80)
    print("QWEN3-VL-4B VISION ENCODER EXPORT")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Precision: {args.precision}")
    print(f"Device: {args.device}")
    print(f"Execution Provider: {args.execution_provider}")
    
    # Export vision encoder
    build_vision(args)
    
    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
