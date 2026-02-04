"""
Simplified Qwen3-VL ONNX Model Builder

Exports three separate ONNX models:
1. Vision encoder (FP32) - fixed 384×384 input
2. Embeddings (FP32) - simple text embedding layer
3. Text decoder (INT4-RTN) - using onnxruntime_genai builder
"""

import argparse
import os
import shutil
import torch
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from onnxruntime_genai.models.builder import create_model


def prepare_model(args):
    """Copy modified files and load model."""
    print("=" * 80)
    print("Preparing Qwen3-VL Model for ONNX Export")
    print("=" * 80)
    
    # Copy modified files to input directory
    print("\n[1/3] Copying modified files to model directory...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_modified_dir = os.path.join(script_dir, 'pytorch_modified')
    
    modified_files = [
        'modeling_qwen3_vl.py',
        'modular_qwen3_vl.py',
        'processing_qwen3_vl.py',
        'video_processing_qwen3_vl.py',
        'configuration_qwen3_vl.py'
    ]
    
    for fname in modified_files:
        src = os.path.join(pytorch_modified_dir, fname)
        dst = os.path.join(args.input, fname)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  - Copied: {fname}")
        else:
            print(f"  [WARNING] Not found: {fname}")
    
    print("  [OK] Modified files copied")
    
    # Load config
    print("\n[2/3] Loading configuration...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    print(f"  - Model type: {config.model_type}")
    print(f"  - Vision hidden size: {config.vision_config.hidden_size}")
    print(f"  - Text hidden size: {config.text_config.hidden_size}")
    
    # Load processor
    print("\n[3/3] Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
    
    # Import after copying modified files
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.input,
        trust_remote_code=True,
        torch_dtype=args.precision,
    )
    model.to(args.device)
    model.eval()
    
    print(f"  [OK] Model loaded")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return config, processor, model


def build_vision(args, config, processor, model, output_dir):
    """
    Export vision encoder to ONNX - simplified with fixed dimensions.
    
    Fixed image size: 384×384 pixels
    Patch size: 16×16 → 24×24 = 576 patches
    Patch features: 3×2×16×16 = 1536
    """
    print("\n" + "=" * 80)
    print("Building Vision Model (Simplified)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print("\n[1/3] Preparing fixed-size dummy inputs...")
    
    # Fixed dimensions for 384×384 image
    grid_t, grid_h, grid_w = 1, 24, 24  # temporal=1, spatial=24×24
    num_patches = grid_t * grid_h * grid_w  # 576 patches
    patch_features = 3 * 2 * 16 * 16  # RGB × temporal_patch × spatial_patch²
    
    dummy_pixel_values = torch.randn(num_patches, patch_features)
    dummy_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    
    print(f"  - Image size: 384×384 (fixed)")
    print(f"  - Patches: {grid_t}×{grid_h}×{grid_w} = {num_patches}")
    print(f"  - Patch features: {patch_features}")
    print(f"  - pixel_values: {dummy_pixel_values.shape}")
    print(f"  - grid_thw: {dummy_grid_thw.shape}")
    
    print("\n[2/3] Creating vision wrapper...")
    
    # Wrapper to extract only pooler_output
    class VisionWrapper(torch.nn.Module):
        def __init__(self, visual_model):
            super().__init__()
            self.visual = visual_model
        
        def forward(self, pixel_values, image_grid_thw):
            outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
            # Return pooler_output (merged patches for LLM)
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif isinstance(outputs, dict):
                return outputs['pooler_output']
            else:
                return outputs[1]  # Tuple: (last_hidden_state, pooler_output)
    
    wrapped_model = VisionWrapper(model.model.visual)
    wrapped_model.eval()
    
    # Force eager attention (workaround for GQA/SDPA export issues)
    original_attn = getattr(model.config, '_attn_implementation', None)
    model.config._attn_implementation = 'eager'
    
    if hasattr(model.model.visual.config, '_attn_implementation'):
        original_vision_attn = model.model.visual.config._attn_implementation
        model.model.visual.config._attn_implementation = 'eager'
    
    print(f"  - Forced attention: eager (for ONNX compatibility)")
    
    print("\n[3/3] Exporting to ONNX...")
    
    output_path = os.path.join(output_dir, "qwen3vl-vision.onnx")
    
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_pixel_values, dummy_grid_thw),
            output_path,
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["pooled_embeds"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "image_grid_thw": {0: "num_images"},
                "pooled_embeds": {0: "num_merged_patches"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [OK] Vision model exported: {output_path}")
        
    except Exception as e:
        print(f"  [FAIL] Vision export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Restore original attention
        if original_attn:
            model.config._attn_implementation = original_attn
        if 'original_vision_attn' in locals():
            model.model.visual.config._attn_implementation = original_vision_attn


def build_embedding(args, config, processor, model, output_dir):
    """
    Export embedding layer to ONNX - simplified direct export.
    
    Just the text embedding layer, no vision merging.
    """
    print("\n" + "=" * 80)
    print("Building Embedding Model (Simplified)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print("\n[1/2] Preparing embedding layer...")
    
    # Get the text embedding layer directly
    embeddings = model.model.language_model.embed_tokens
    embeddings.eval()
    
    print(f"  - Embedding layer: {type(embeddings)}")
    print(f"  - Vocab size: {embeddings.num_embeddings}")
    print(f"  - Embedding dim: {embeddings.embedding_dim}")
    
    # Create dummy input
    batch_size, seq_len = 1, 10
    vocab_size = config.text_config.vocab_size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)
    
    print(f"  - Input IDs shape: {dummy_input_ids.shape}")
    
    print("\n[2/2] Exporting to ONNX...")
    
    output_path = os.path.join(output_dir, "qwen3vl-embedding.onnx")
    
    try:
        torch.onnx.export(
            embeddings,
            dummy_input_ids,
            output_path,
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [OK] Embedding model exported: {output_path}")
        
    except Exception as e:
        print(f"  [FAIL] Embedding export failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_text(args, config, processor, model, output_dir, text_precision="int4"):
    """Export text model using onnxruntime_genai builder."""
    print("\n" + "=" * 80)
    print(f"Building Text Model ({text_precision.upper()})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print(f"\nUsing onnxruntime_genai builder for text model (precision={text_precision})...")
    
    model_name = None
    extra_options = {
        "exclude_embeds": "true",
        "filename": "model.onnx",  # Standard GenAI filename
    }
    
    # Only add INT4 options if using INT4
    if text_precision == "int4":
        extra_options["int4_algo_config"] = "rtn"
        if args.precision == torch.float32:
            extra_options["int4_accuracy_level"] = 4
    
    try:
        create_model(
            model_name, 
            args.input, 
            output_dir, 
            text_precision,  # Use the parameter instead of hardcoded "int4"
            args.execution_provider, 
            args.cache_dir, 
            **extra_options
        )
        print(f"\n[OK] Text model exported ({text_precision})")
    except Exception as e:
        print(f"\n[FAIL] Text model export failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_args():
    parser = argparse.ArgumentParser(description="Build Qwen3-VL ONNX models (simplified)")
    
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to folder containing the Hugging Face model files",
    )
    
    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Precision for vision/embedding models",
    )
    
    parser.add_argument(
        "-t",
        "--text_precision",
        required=False,
        default="int4",
        choices=["fp32", "int4"],
        help="Precision for text model (default: int4)",
    )
    
    parser.add_argument(
        "-e",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "dml"],
        help="Execution provider for ONNX models",
    )
    
    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        default=os.path.join('.', 'cache_dir'),
        help="Cache directory for temporary files",
    )
    
    args = parser.parse_args()
    
    # Convert precision string to torch dtype
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Set device based on execution provider
    if args.execution_provider == "cuda":
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    # Set output directories based on model type
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precision_str = "fp32" if args.precision == torch.float32 else "fp16"
    
    args.vision_output = os.path.join(script_dir, f"cpu-{precision_str}")
    args.embedding_output = args.vision_output  # Same directory
    args.text_output = os.path.join(script_dir, f"cpu-{args.text_precision}")
    
    # Create output directories
    os.makedirs(args.vision_output, exist_ok=True)
    os.makedirs(args.text_output, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    return args


def main():
    args = get_args()
    
    print("\n" + "=" * 80)
    print("Qwen3-VL ONNX Model Builder (Simplified)")
    print("=" * 80)
    print(f"\nInput directory: {args.input}")
    print(f"Vision/Embedding output: {args.vision_output}")
    print(f"Text output: {args.text_output}")
    print(f"Precision: {args.precision}")
    print(f"Text precision: {args.text_precision}")
    print(f"Execution provider: {args.execution_provider}")
    print(f"Device: {args.device}")
    print("\nSimplifications:")
    print("  - Fixed image size: 384×384 pixels")
    print("  - No post-processing (optimization/quantization for vision)")
    print("  - Direct embedding export (no vision merging)")
    print(f"  - Text model uses {args.text_precision.upper()} precision")
    
    # Prepare model
    config, processor, model = prepare_model(args)
    
    # Build components
    #build_vision(args, config, processor, model, args.vision_output)
    build_embedding(args, config, processor, model, args.embedding_output)
    build_text(args, config, processor, model, args.text_output, args.text_precision)
    
    print("\n" + "=" * 80)
    print("[OK] All ONNX models exported successfully!")
    print("=" * 80)
    precision_str = "FP32" if args.precision == torch.float32 else "FP16"
    print(f"\nVision/Embedding models: {args.vision_output}")
    print(f"  - qwen3vl-vision.onnx ({precision_str}, fixed 384×384 input)")
    print(f"  - qwen3vl-embedding.onnx ({precision_str})")
    print(f"\nText model: {args.text_output}")
    print(f"  - model.onnx ({args.text_precision.upper()})")
    print(f"  - genai_config.json")
    print(f"  - tokenizer files")
    print("\nNext steps:")
    print("  1. Test with: python qwen3-vl.py --text 'Hello' --max_new_tokens 20")
    print("  2. For images, resize to 384×384 before processing")
    print()


if __name__ == "__main__":
    main()
