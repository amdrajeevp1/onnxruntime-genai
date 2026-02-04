"""
Test ONNX Export for Every Module in Qwen3-VL Vision Model

This script systematically tests ONNX export capabilities for each
nn.Module in the Qwen3-VL vision encoder.

Tests both:
1. Traditional torch.onnx.export
2. torch.export + dynamo export

Reports which modules can/cannot be exported.
"""

import torch
import torch.nn as nn
import os
import sys
import codecs
from pathlib import Path
from transformers import AutoModel, AutoConfig
import traceback

# Force UTF-8 output for Windows
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Create output directory
OUTPUT_DIR = Path("./module_export_tests")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_dummy_inputs(module, module_name):
    """
    Create appropriate dummy inputs for each module type
    """
    if "PatchEmbed" in module_name:
        # Input: [432, 1536]
        return (torch.randn(432, 1536),)
    
    elif "RotaryEmbedding" in module_name:
        # Input: sequence length
        return (100,)
    
    elif "PatchMerger" in module_name:
        # Input: [432, 1024]
        return (torch.randn(432, 1024),)
    
    elif "VisionMLP" in module_name:
        # Input: [432, 1024]
        return (torch.randn(432, 1024),)
    
    elif "VisionAttention" in module_name:
        # Input: hidden_states, cu_seqlens, position_embeddings
        hidden_states = torch.randn(432, 1024)
        cu_seqlens = torch.tensor([0, 432], dtype=torch.int32)
        cos = torch.randn(432, 128)
        sin = torch.randn(432, 128)
        return (hidden_states, cu_seqlens, None, (cos, sin))
    
    elif "VisionBlock" in module_name:
        # Input: hidden_states, cu_seqlens, position_embeddings
        hidden_states = torch.randn(432, 1024)
        cu_seqlens = torch.tensor([0, 432], dtype=torch.int32)
        cos = torch.randn(432, 128)
        sin = torch.randn(432, 128)
        return (hidden_states, cu_seqlens, None, (cos, sin))
    
    elif "VisionModel" in module_name:
        # Input: pixel_values, grid_thw
        pixel_values = torch.randn(432, 1536)
        grid_thw = torch.tensor([[1, 18, 24]], dtype=torch.int32)
        return (pixel_values, grid_thw)
    
    elif isinstance(module, nn.LayerNorm):
        # Get input size from normalized_shape
        if hasattr(module, 'normalized_shape'):
            size = module.normalized_shape[0]
            return (torch.randn(1, 432, size),)
        return (torch.randn(1, 432, 1024),)
    
    elif isinstance(module, nn.Linear):
        # Get input size from in_features
        in_features = module.in_features
        return (torch.randn(1, 432, in_features),)
    
    elif isinstance(module, nn.Embedding):
        # Get vocab size
        num_embeddings = module.num_embeddings
        return (torch.randint(0, num_embeddings, (1, 10)),)
    
    elif isinstance(module, nn.Conv3d):
        # Get input shape from module
        in_channels = module.in_channels
        return (torch.randn(1, in_channels, 2, 16, 16),)
    
    elif isinstance(module, nn.GELU):
        return (torch.randn(1, 432, 1024),)
    
    else:
        # Default
        return (torch.randn(1, 432, 1024),)


def test_torch_onnx_export(module, module_name, dummy_inputs):
    """
    Test traditional torch.onnx.export
    """
    output_path = OUTPUT_DIR / f"{module_name}_torch_onnx.onnx"
    
    try:
        module.eval()
        with torch.no_grad():
            torch.onnx.export(
                module,
                dummy_inputs,
                output_path,
                input_names=[f"input_{i}" for i in range(len(dummy_inputs))],
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
            )
        return True, None, output_path
    except Exception as e:
        return False, str(e), None


def test_dynamo_export(module, module_name, dummy_inputs):
    """
    Test torch.export + dynamo ONNX export
    """
    output_path = OUTPUT_DIR / f"{module_name}_dynamo.onnx"
    
    try:
        module.eval()
        
        # First export to ExportedProgram
        exported_program = torch.export.export(module, dummy_inputs)
        
        # Then export to ONNX using dynamo
        import torch.onnx
        torch.onnx.dynamo_export(
            exported_program,
            *dummy_inputs,
        ).save(str(output_path))
        
        return True, None, output_path
    except Exception as e:
        return False, str(e), None


def extract_all_modules(model, prefix=""):
    """
    Recursively extract all nn.Module instances from the model
    """
    modules = {}
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Add this module
        modules[full_name] = module
        
        # Recursively get child modules
        child_modules = extract_all_modules(module, full_name)
        modules.update(child_modules)
    
    return modules


def main():
    print("="*80)
    print("QWEN3-VL VISION MODEL - MODULE ONNX EXPORT TEST")
    print("="*80)
    print()
    
    # Load model
    print("[1/4] Loading Qwen3-VL model...")
    model_path = "./pytorch"
    
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        )
        model.eval()
        print(f"  [OK] Model loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return
    
    # Extract vision model
    vision_model = model.visual
    print(f"  [OK] Extracted vision model")
    print()
    
    # Extract all modules
    print("[2/4] Extracting all nn.Module instances...")
    all_modules = extract_all_modules(vision_model)
    
    # Also add the full vision model
    all_modules["VisionModel"] = vision_model
    
    print(f"  Found {len(all_modules)} modules")
    print()
    
    # Test each module
    print("[3/4] Testing ONNX export for each module...")
    print()
    
    results = []
    
    for i, (module_name, module) in enumerate(all_modules.items(), 1):
        print(f"[{i}/{len(all_modules)}] Testing: {module_name}")
        print(f"  Module type: {type(module).__name__}")
        
        # Get dummy inputs
        try:
            dummy_inputs = get_dummy_inputs(module, module_name)
            print(f"  Dummy inputs: {len(dummy_inputs)} tensors")
        except Exception as e:
            print(f"  [ERROR] Failed to create dummy inputs: {e}")
            results.append({
                "module_name": module_name,
                "module_type": type(module).__name__,
                "torch_onnx": "SKIP",
                "dynamo": "SKIP",
                "error": "Failed to create dummy inputs"
            })
            print()
            continue
        
        # Test torch.onnx.export
        print("  Testing torch.onnx.export...", end=" ")
        torch_success, torch_error, torch_path = test_torch_onnx_export(
            module, module_name.replace(".", "_"), dummy_inputs
        )
        if torch_success:
            print(f"[SUCCESS]")
            torch_status = "SUCCESS"
        else:
            print(f"[FAILED]")
            print(f"    Error: {torch_error[:100]}...")
            torch_status = "FAILED"
        
        # Test dynamo export
        print("  Testing torch.export + dynamo...", end=" ")
        dynamo_success, dynamo_error, dynamo_path = test_dynamo_export(
            module, module_name.replace(".", "_"), dummy_inputs
        )
        if dynamo_success:
            print(f"[SUCCESS]")
            dynamo_status = "SUCCESS"
        else:
            print(f"[FAILED]")
            print(f"    Error: {dynamo_error[:100]}...")
            dynamo_status = "FAILED"
        
        # Store result
        results.append({
            "module_name": module_name,
            "module_type": type(module).__name__,
            "torch_onnx": torch_status,
            "torch_onnx_error": torch_error if not torch_success else None,
            "dynamo": dynamo_status,
            "dynamo_error": dynamo_error if not dynamo_success else None,
        })
        
        print()
    
    # Generate report
    print("[4/4] Generating report...")
    print()
    print("="*80)
    print("EXPORT TEST RESULTS")
    print("="*80)
    print()
    
    # Count successes
    torch_successes = sum(1 for r in results if r["torch_onnx"] == "SUCCESS")
    dynamo_successes = sum(1 for r in results if r["dynamo"] == "SUCCESS")
    
    print(f"Total modules tested: {len(results)}")
    print(f"torch.onnx.export successes: {torch_successes}/{len(results)}")
    print(f"torch.export + dynamo successes: {dynamo_successes}/{len(results)}")
    print()
    
    # Modules that failed both
    print("-"*80)
    print("MODULES THAT CANNOT BE EXPORTED TO ONNX (Failed both methods):")
    print("-"*80)
    
    failed_both = [r for r in results if r["torch_onnx"] == "FAILED" and r["dynamo"] == "FAILED"]
    
    if failed_both:
        for result in failed_both:
            print(f"\n[X] {result['module_name']}")
            print(f"  Type: {result['module_type']}")
            print(f"  torch.onnx.export error:")
            print(f"    {result['torch_onnx_error'][:200]}")
            print(f"  dynamo export error:")
            print(f"    {result['dynamo_error'][:200]}")
    else:
        print("  (None - all modules can be exported with at least one method)")
    
    print()
    
    # Modules that succeeded with torch.onnx but failed with dynamo
    print("-"*80)
    print("MODULES: torch.onnx.export [OK], dynamo [FAIL]")
    print("-"*80)
    
    torch_only = [r for r in results if r["torch_onnx"] == "SUCCESS" and r["dynamo"] == "FAILED"]
    
    if torch_only:
        for result in torch_only:
            print(f"\n  {result['module_name']} ({result['module_type']})")
    else:
        print("  (None)")
    
    print()
    
    # Modules that succeeded with dynamo but failed with torch.onnx
    print("-"*80)
    print("MODULES: torch.onnx.export [FAIL], dynamo [OK]")
    print("-"*80)
    
    dynamo_only = [r for r in results if r["torch_onnx"] == "FAILED" and r["dynamo"] == "SUCCESS"]
    
    if dynamo_only:
        for result in dynamo_only:
            print(f"\n  {result['module_name']} ({result['module_type']})")
    else:
        print("  (None)")
    
    print()
    
    # Save detailed report
    report_path = OUTPUT_DIR / "export_test_report.txt"
    with open(report_path, "w") as f:
        f.write("QWEN3-VL VISION MODEL - ONNX EXPORT TEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total modules tested: {len(results)}\n")
        f.write(f"torch.onnx.export successes: {torch_successes}/{len(results)}\n")
        f.write(f"torch.export + dynamo successes: {dynamo_successes}/{len(results)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for result in results:
            f.write(f"Module: {result['module_name']}\n")
            f.write(f"Type: {result['module_type']}\n")
            f.write(f"torch.onnx.export: {result['torch_onnx']}\n")
            if result['torch_onnx_error']:
                f.write(f"  Error: {result['torch_onnx_error']}\n")
            f.write(f"dynamo export: {result['dynamo']}\n")
            if result['dynamo_error']:
                f.write(f"  Error: {result['dynamo_error']}\n")
            f.write("\n")
    
    print(f"Detailed report saved to: {report_path}")
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
