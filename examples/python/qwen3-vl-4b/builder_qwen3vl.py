"""
Builder script for Qwen3-VL ONNX export
Exports vision encoder, embeddings, and text decoder separately

Based on Phi4-MM builder approach
"""
import argparse
import os
import sys
import torch
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

def export_vision_encoder(model, output_dir, precision="fp32"):
    """
    Export the vision encoder to ONNX
    
    Input: pixel_values [num_patches, patch_features]
    Input: image_grid_thw [num_images, 3]
    Output: pooled_embeds [num_patches_merged, hidden_dim]
    """
    print("\n" + "=" * 70)
    print("Exporting Vision Encoder")
    print("=" * 70)
    
    vision_model = model.visual
    vision_model.eval()
    
    # CRITICAL: Force eager attention instead of SDPA
    # SDPA with GQA doesn't export to ONNX
    vision_model.config._attn_implementation = "eager"
    print(f"  Forced attention implementation: {vision_model.config._attn_implementation}")
    
    # Create wrapper to extract just pooler_output
    class VisionModelWrapper(torch.nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model
        
        def forward(self, pixel_values, image_grid_thw):
            outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw, return_dict=True)
            # Return pooler_output (merged patches for LLM)
            # Handle both dict and object return types
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif isinstance(outputs, dict):
                return outputs['pooler_output']
            else:
                # If tuple, pooler_output is typically index 1
                return outputs[1]
    
    wrapped_model = VisionModelWrapper(vision_model)
    wrapped_model.eval()
    
    # Qwen3-VL Vision Model Input Format:
    # - pixel_values: [num_patches, patch_features]
    # - image_grid_thw: [num_images, 3] where 3 = [temporal, height, width]
    #
    # Key constraint: num_patches MUST equal T * H * W from grid_thw
    # Otherwise pos_embed interpolation will fail with shape mismatch
    #
    # Example: For an image resized to 384x384 pixels:
    #   - Spatial patches: 384/16 = 24 patches per side
    #   - Temporal: 1 (single image, no video)
    #   - Total patches: 1 * 24 * 24 = 576
    #
    # Patch features: channels * temporal_patch_size * patch_size^2
    #                 = 3 * 2 * 16 * 16 = 1536
    
    grid_t, grid_h, grid_w = 1, 24, 24  # 384x384 image → 24x24 patches
    num_patches = grid_t * grid_h * grid_w  # Must match!
    patch_features = 3 * 2 * 16 * 16  # RGB * temporal_patch * spatial_patch^2
    
    dummy_pixel_values = torch.randn(num_patches, patch_features)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    
    print(f"  Vision input shape: {dummy_pixel_values.shape}")
    print(f"  Grid THW: {grid_thw.shape} = {grid_thw}")
    print(f"  Patches: T={grid_t} × H={grid_h} × W={grid_w} = {num_patches}")
    print(f"  Patch features: {patch_features} (3×2×16×16)")
    
    # Export to ONNX
    output_path = Path(output_dir) / "vision_encoder.onnx"
    
    print(f"  Exporting to {output_path}...")
    
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_pixel_values, grid_thw),
            str(output_path),
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["pooled_embeds"],  # Merged patches for LLM
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "image_grid_thw": {0: "num_images"},
                "pooled_embeds": {0: "num_merged_patches"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [SUCCESS] Vision encoder exported successfully!")
        return True
    except Exception as e:
        print(f"  [ERROR] exporting vision encoder: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_embeddings(model, output_dir, precision="fp32"):
    """
    Export the embedding layer to ONNX
    
    Input: input_ids [batch, seq_len]
    Output: inputs_embeds [batch, seq_len, hidden_dim]
    """
    print("\n" + "=" * 70)
    print("Exporting Embeddings")
    print("=" * 70)
    
    # Fix: Qwen3VL uses language_model, not model
    embeddings = model.model.language_model.embed_tokens
    embeddings.eval()
    print(f"  Embedding layer: {type(embeddings)}")
    print(f"  Vocab size: {embeddings.num_embeddings}")
    print(f"  Embedding dim: {embeddings.embedding_dim}")
    
    # Create dummy input
    batch_size = 1
    seq_len = 10
    vocab_size = model.config.text_config.vocab_size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    
    # Export to ONNX
    output_path = Path(output_dir) / "embeddings.onnx"
    
    print(f"  Exporting to {output_path}...")
    
    try:
        torch.onnx.export(
            embeddings,
            dummy_input_ids,
            str(output_path),
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [SUCCESS] Embeddings exported successfully!")
        return True
    except Exception as e:
        print(f"  [ERROR] exporting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_text_decoder(input_dir, output_dir, precision="fp32", execution_provider="cpu"):
    """
    Export the text decoder using the existing builder
    """
    print("\n" + "=" * 70)
    print("Exporting Text Decoder")
    print("=" * 70)
    
    # Use the existing builder from onnxruntime-genai
    from py.models.builder import create_model
    
    print(f"  Using existing builder for Qwen3VLForConditionalGeneration...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Precision: {precision}")
    print(f"  EP: {execution_provider}")
    
    try:
        create_model(
            model_name=None,
            input_path=str(input_dir),
            output_dir=str(output_dir),
            precision=precision,
            execution_provider=execution_provider,
            cache_dir=str(Path(output_dir).parent / "cache_dir"),
            exclude_embeds=True,  # We export embeddings separately
        )
        print(f"  [SUCCESS] Text decoder exported successfully!")
        return True
    except Exception as e:
        print(f"  [ERROR] exporting text decoder: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_processor_config(output_dir):
    """
    Create vision_processor.json for image preprocessing
    """
    print("\n" + "=" * 70)
    print("Creating Processor Config")
    print("=" * 70)
    
    config = {
        "processor_type": "Qwen3VLImageProcessor",
        "min_pixels": 256 * 28 * 28,  # 200704
        "max_pixels": 1280 * 28 * 28,  # 1003520
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "spatial_factor": 32,
        "temporal_factor": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_resize": True,
        "do_normalize": True,
        "do_convert_rgb": True
    }
    
    output_path = Path(output_dir) / "vision_processor.json"
    print(f"  Writing to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  [SUCCESS] Processor config created!")

def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL to ONNX")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to PyTorch model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory for ONNX models")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "int4"],
                        help="Model precision")
    parser.add_argument("--execution_provider", type=str, default="cpu",
                        choices=["cpu", "cuda", "dml"],
                        help="Execution provider")
    parser.add_argument("--skip_vision", action="store_true",
                        help="Skip vision encoder export")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip embeddings export")
    parser.add_argument("--skip_decoder", action="store_true",
                        help="Skip text decoder export")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Qwen3-VL ONNX Export")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Precision: {args.precision}")
    print(f"Execution Provider: {args.execution_provider}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load the model
    print("\nLoading Qwen3-VL model...")
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.input,
        torch_dtype=torch.float32 if args.precision == "fp32" else torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"  # Force eager attention for ONNX export
    )
    model.eval()
    print("[SUCCESS] Model loaded!")
    print(f"  Model attention implementation: {model.config._attn_implementation}")
    
    # Export components
    success = True
    
    if not args.skip_vision:
        if not export_vision_encoder(model, args.output, args.precision):
            success = False
    
    if not args.skip_embeddings:
        if not export_embeddings(model, args.output, args.precision):
            success = False
    
    if not args.skip_decoder:
        if not export_text_decoder(args.input, args.output, args.precision, args.execution_provider):
            success = False
    
    # Create processor config
    create_processor_config(args.output)
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("[SUCCESS] Export Complete!")
        print("=" * 70)
        print("\nExported files:")
        output_path = Path(args.output)
        for file in output_path.glob("*.onnx"):
            print(f"  - {file.name}")
        for file in output_path.glob("*.json"):
            print(f"  - {file.name}")
        
        print("\nNext steps:")
        print(f"  1. Test inference: python test_qwen3vl_inference.py --model_path {args.output}")
        print(f"  2. Optimize models: python optimize_onnx.py --input {args.output}")
    else:
        print("[ERROR] Export Failed!")
        print("=" * 70)
        print("\nSome components failed to export. Check the error messages above.")

if __name__ == "__main__":
    main()
