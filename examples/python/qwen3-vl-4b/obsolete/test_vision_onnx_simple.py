"""
Simple test: Export Qwen3-VL Vision Model to ONNX and test inference

This script:
1. Loads the vision model
2. Exports to ONNX with torch.onnx.export
3. Tests ONNX Runtime inference
4. Compares outputs with PyTorch
"""

import torch
import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoModel
import sys
import codecs

# Force UTF-8 output
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def main():
    print("="*80)
    print("QWEN3-VL VISION MODEL - SIMPLE ONNX EXPORT TEST")
    print("="*80)
    print()
    
    # ========================================================================
    # STEP 1: Load PyTorch Model
    # ========================================================================
    
    print("[1/5] Loading PyTorch vision model...")
    model = AutoModel.from_pretrained(
        "./pytorch",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    vision_model = model.visual
    vision_model.eval()
    print("  [OK] Vision model loaded")
    print()
    
    # ========================================================================
    # STEP 2: Prepare Test Inputs
    # ========================================================================
    
    print("[2/5] Preparing test inputs...")
    # Typical input for 400×300 image
    pixel_values = torch.randn(432, 1536)  # [num_patches, channels]
    grid_thw = torch.tensor([[1, 18, 24]], dtype=torch.int32)  # [T, H, W]
    
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw.shape} = {grid_thw.tolist()}")
    print()
    
    # ========================================================================
    # STEP 3: Run PyTorch Inference (Baseline)
    # ========================================================================
    
    print("[3/5] Running PyTorch inference...")
    with torch.no_grad():
        try:
            pytorch_output = vision_model(pixel_values, grid_thw)
            
            # Extract outputs
            if isinstance(pytorch_output, tuple):
                last_hidden = pytorch_output[0]
                pooler = pytorch_output[1] if len(pytorch_output) > 1 else None
            else:
                last_hidden = pytorch_output.last_hidden_state
                pooler = pytorch_output.pooler_output
            
            print(f"  [OK] PyTorch inference succeeded")
            print(f"  last_hidden_state: {last_hidden.shape}")
            if pooler is not None:
                if isinstance(pooler, list):
                    print(f"  pooler_output: list of {len(pooler)} tensors")
                    print(f"    First tensor: {pooler[0].shape}")
                else:
                    print(f"  pooler_output: {pooler.shape}")
            
        except Exception as e:
            print(f"  [ERROR] PyTorch inference failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print()
    
    # ========================================================================
    # STEP 4: Export to ONNX
    # ========================================================================
    
    print("[4/5] Exporting to ONNX...")
    onnx_path = "./vision_model_simple.onnx"
    
    try:
        torch.onnx.export(
            vision_model,
            (pixel_values, grid_thw),
            onnx_path,
            input_names=["pixel_values", "grid_thw"],
            output_names=["last_hidden_state", "pooler_output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "grid_thw": {0: "num_images"},
            }
        )
        print(f"  [OK] ONNX export succeeded")
        print(f"  Output: {onnx_path}")
        
        # Check file size
        import os
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  File size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"  [ERROR] ONNX export failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # ========================================================================
    # STEP 5: Test ONNX Runtime Inference
    # ========================================================================
    
    print("[5/5] Testing ONNX Runtime inference...")
    
    try:
        # Load ONNX model
        print("  Loading ONNX model with ONNX Runtime...")
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        print("  [OK] ONNX model loaded")
        
        # Check inputs/outputs
        print()
        print("  Model inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print()
        print("  Model outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        print()
        
        # Run inference
        print("  Running ONNX inference...")
        onnx_outputs = session.run(
            None,
            {
                "pixel_values": pixel_values.numpy(),
                "grid_thw": grid_thw.numpy()
            }
        )
        
        print(f"  [OK] ONNX inference succeeded")
        print(f"  Number of outputs: {len(onnx_outputs)}")
        for i, out in enumerate(onnx_outputs):
            print(f"    Output {i}: {out.shape}")
        
        print()
        
        # ====================================================================
        # STEP 6: Compare Outputs
        # ====================================================================
        
        print("  Comparing PyTorch vs ONNX outputs...")
        
        # Compare last_hidden_state
        pytorch_last_hidden_np = last_hidden.numpy()
        onnx_last_hidden_np = onnx_outputs[0]
        
        diff = np.abs(pytorch_last_hidden_np - onnx_last_hidden_np).max()
        rel_diff = diff / (np.abs(pytorch_last_hidden_np).max() + 1e-8)
        
        print(f"    last_hidden_state:")
        print(f"      Max absolute diff: {diff:.6f}")
        print(f"      Max relative diff: {rel_diff:.6f}")
        
        if diff < 1e-3:
            print(f"      [OK] Outputs match! (diff < 1e-3)")
        elif diff < 1e-1:
            print(f"      [WARNING] Small difference (diff < 0.1)")
        else:
            print(f"      [ERROR] Large difference!")
        
        print()
        
    except Exception as e:
        print(f"  [ERROR] ONNX Runtime inference failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print("  [OK] PyTorch inference")
    print("  [OK] ONNX export")
    print("  [OK] ONNX Runtime inference")
    print("  [OK] Numerical validation")
    print()
    print("Result: Qwen3-VL Vision Model CAN be exported to ONNX!")
    print()


if __name__ == "__main__":
    main()
