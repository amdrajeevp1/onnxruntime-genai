"""
Verify all three ONNX models were exported correctly
"""
import onnx
from pathlib import Path

def verify_model(model_path, model_name):
    """Verify an ONNX model"""
    print(f"\n{'='*70}")
    print(f"Verifying {model_name}")
    print(f"{'='*70}")
    
    if not Path(model_path).exists():
        print(f"  [ERROR] File not found: {model_path}")
        return False
    
    try:
        model = onnx.load(model_path)
        print(f"  [OK] Model loaded successfully")
        print(f"  File size: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")
        
        print(f"\n  Inputs:")
        for inp in model.graph.input[:5]:  # Show first 5
            dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {dims}")
        if len(model.graph.input) > 5:
            print(f"    ... and {len(model.graph.input) - 5} more")
        
        print(f"\n  Outputs:")
        for out in model.graph.output[:5]:  # Show first 5
            dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {dims}")
        if len(model.graph.output) > 5:
            print(f"    ... and {len(model.graph.output) - 5} more")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return False

def main():
    print("="*70)
    print("Qwen3-VL ONNX Export Verification")
    print("="*70)
    
    models = [
        ("cpu/vision_encoder.onnx", "Vision Encoder"),
        ("cpu/embeddings.onnx", "Embeddings"),
        ("cpu-text/model.onnx", "Text Decoder"),
    ]
    
    results = []
    for model_path, model_name in models:
        success = verify_model(model_path, model_name)
        results.append((model_name, success))
    
    # Summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    all_success = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_success:
        print("[SUCCESS] All models exported correctly! ✓✓✓")
        print("="*70)
        print("\nReady for:")
        print("  1. Integration pipeline")
        print("  2. End-to-end testing")
        print("  3. Optimization (INT4, etc.)")
    else:
        print("[WARNING] Some models missing or invalid")
        print("="*70)
        print("\nRe-run the export for failed models")

if __name__ == "__main__":
    main()
