"""
Qwen3-VL-4B Text Decoder ONNX Export Builder

This builder exports the text component (language_model) of Qwen3-VL-4B to ONNX format
using the generic onnxruntime-genai builder.

The text decoder:
- 36 transformer layers
- MRoPE (Multimodal Rotary Position Embeddings)
- Hidden size: 2560
- Vocabulary: 151,936 tokens
- Supports vision token integration

This is simpler than Phi-4 because we can use the generic builder directly!
"""
import argparse
import os
import time
from onnxruntime_genai.models.builder import create_model


def build_text_decoder(args):
    """
    Export Qwen3-VL text decoder using generic builder.
    
    This handles:
    - 36-layer transformer
    - MRoPE with mrope_section [24, 20, 20]
    - INT4 quantization
    - KV cache optimization
    - Vision token embeddings
    """
    print("\n" + "="*80)
    print("BUILDING QWEN3-VL TEXT DECODER")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Precision: {args.precision}")
    print(f"Execution Provider: {args.execution_provider}")
    
    start_time = time.time()
    
    try:
        print("\n" + "="*80)
        print("USING GENERIC ONNX RUNTIME GENAI BUILDER")
        print("="*80)
        print("This builder handles:")
        print("  - Qwen3 architecture")
        print("  - MRoPE (multimodal positional embeddings)")
        print("  - INT4 quantization")
        print("  - KV cache optimization")
        print("  - Vision token integration")
        print()
        
        # Use the generic builder - it knows how to handle Qwen3-VL!
        create_model(
            model_name="Qwen/Qwen3-VL-4B-Instruct",  # Model identifier
            input_path=args.input,                    # PyTorch model directory
            output_dir=args.output,                   # ONNX output directory
            precision=args.precision,                 # int4, fp32, fp16
            execution_provider=args.execution_provider,  # cpu, cuda, etc.
            cache_dir=os.path.join(args.output, ".cache")  # Cache directory
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("[SUCCESS] TEXT DECODER EXPORT COMPLETE")
        print("="*80)
        print(f"Export time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # List exported files
        print("\nExported files:")
        for root, dirs, files in os.walk(args.output):
            for file in files:
                if file.endswith(('.onnx', '.onnx.data', '.json')):
                    filepath = os.path.join(root, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  - {file} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print("\n" + "="*80)
        print("[ERROR] TEXT DECODER EXPORT FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL-4B text decoder to ONNX using generic builder"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to PyTorch model directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to output directory for ONNX model"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="int4", 
        choices=["fp32", "fp16", "int4", "int8"],
        help="Model precision (int4 recommended for CPU)"
    )
    parser.add_argument(
        "--execution_provider", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda", "dml"],
        help="Execution provider"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*80)
    print("QWEN3-VL-4B TEXT DECODER ONNX EXPORT")
    print("="*80)
    print("Using Generic ONNX Runtime GenAI Builder")
    print()
    print("Why generic builder?")
    print("  - Built-in Qwen3 architecture support")
    print("  - MRoPE (multimodal RoPE) handling")
    print("  - INT4 quantization optimized")
    print("  - Battle-tested and maintained")
    print()
    print("This is MUCH simpler than custom Phi-4 builder!")
    
    # Export text decoder
    build_text_decoder(args)
    
    print("\n" + "="*80)
    print("[DONE] ALL COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Verify text decoder: load the ONNX model")
    print("  2. Test text-only generation")
    print("  3. Integrate with vision encoder")
    print("  4. Test full multimodal pipeline")
    print()
    print("Text decoder location:")
    print(f"  {args.output}/")
