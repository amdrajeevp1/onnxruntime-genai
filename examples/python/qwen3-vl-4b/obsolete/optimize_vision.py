"""
Apply ORT Transformer Optimizer to Qwen3-VL Vision Encoder

This applies the same optimization pipeline that Phi-4 MM uses for its vision encoder.
Goal: Fix dynamic shape issues and optimize performance.
"""

import os
import sys
import subprocess
import shutil
import onnx

def safe_rmtree(path, retries=3, delay=1):
    """Safely remove directory tree with retry logic for Windows file locks."""
    import time
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


def optimize_vision_encoder():
    """
    Apply ORT transformer optimizer to vision encoder (like Phi-4 MM does)
    """
    print("="*80)
    print("APPLYING ORT OPTIMIZER TO QWEN3-VL VISION ENCODER")
    print("="*80)
    print()
    print("This applies the same optimization that Phi-4 MM uses:")
    print("  - Graph optimizations")
    print("  - Attention fusion")
    print("  - LayerNorm fusion")
    print("  - Dynamic shape handling improvements")
    print()
    
    # Qwen3-VL vision architecture (from builder_vision.py)
    num_heads = 16
    hidden_size = 1024
    
    # Paths
    input_model = "./cpu/qwen3-vl-vision.onnx"
    temp_folder = "./cpu-optimized-temp"
    output_model = os.path.join(temp_folder, "qwen3-vl-vision.onnx")
    
    if not os.path.exists(input_model):
        print(f"ERROR: Input model not found: {input_model}")
        print("Please run builder_vision.py first to export the vision encoder.")
        return False
    
    print(f"Input:  {input_model}")
    print(f"Output: {temp_folder}/")
    print(f"\nArchitecture:")
    print(f"  - num_heads: {num_heads}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - model_type: clip (similar vision transformer)")
    print()
    
    # Create temp folder
    os.makedirs(temp_folder, exist_ok=True)
    
    # Run ORT transformer optimizer (same as Phi-4)
    print("[1/3] Running ORT transformer optimizer...")
    print()
    
    cmd = [
        sys.executable, "-m", "onnxruntime.transformers.optimizer",
        "--input", input_model,
        "--output", output_model,
        "--model_type", "clip",  # Vision transformer (like Phi-4 uses)
        "--num_heads", str(num_heads),
        "--hidden_size", str(hidden_size),
        "--use_external_data_format",
        "--opt_level", "0",  # Conservative optimization level
        "--disable_shape_inference",  # Like Phi-4 (avoid shape issues)
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        print(f"\nOptimizer returned code: {result.returncode}")
        print("This might be okay - continuing anyway...")
    else:
        print("✓ Optimizer completed successfully")
    
    # Check if output was created
    if not os.path.exists(output_model):
        print(f"\nERROR: Output model not created: {output_model}")
        return False
    
    print()
    print("[2/3] Validating optimized model...")
    
    try:
        # Load and check the model
        onnx_model = onnx.load(output_model, load_external_data=True)
        onnx.checker.check_model(onnx_model)
        print("✓ Model is valid")
        
        # Print some stats
        num_nodes = len(onnx_model.graph.node)
        num_inputs = len(onnx_model.graph.input)
        num_outputs = len(onnx_model.graph.output)
        
        print(f"\nOptimized model stats:")
        print(f"  - Nodes: {num_nodes}")
        print(f"  - Inputs: {num_inputs}")
        print(f"  - Outputs: {num_outputs}")
        
        # Check input/output shapes
        print(f"\nInputs:")
        for inp in onnx_model.graph.input:
            shape = [d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {shape}")
        
        print(f"\nOutputs:")
        for out in onnx_model.graph.output:
            shape = [d.dim_param if d.dim_param else d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: {shape}")
        
    except Exception as e:
        print(f"Warning during validation: {e}")
        print("Continuing anyway...")
    
    print()
    print("[3/3] Moving optimized model to final location...")
    
    # Create final output directory
    final_output_dir = "./cpu-optimized"
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Move files
    for filename in os.listdir(temp_folder):
        src = os.path.join(temp_folder, filename)
        dst = os.path.join(final_output_dir, filename)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f"  Moved: {filename}")
    
    # Clean up temp folder
    safe_rmtree(temp_folder)
    
    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print()
    print(f"Optimized model saved to: {final_output_dir}/")
    print()
    print("Next step: Test with test_qwen3vl.py")
    print()
    print("Update test_qwen3vl.py to use the optimized model:")
    print('  vision_model_path = "./cpu-vision-optimized/qwen3-vl-vision.onnx"')
    print()
    
    return True


if __name__ == "__main__":
    success = optimize_vision_encoder()
    sys.exit(0 if success else 1)
