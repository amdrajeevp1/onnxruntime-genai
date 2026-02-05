"""
Export Qwen3-VL ONNX Models for OGA Integration

Exports all three models to a single directory with proper configuration.
"""

import argparse
import os
import shutil
import torch
import json
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from onnxruntime_genai.models.builder import create_model


def prepare_model(input_dir):
    """Copy modified files and load model."""
    print("\n[1/4] Preparing model...")
    
    # Copy modified files to input directory
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
        dst = os.path.join(input_dir, fname)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  [OK] Copied: {fname}")
    
    # Load model
    print("  [OK] Loading model...")
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(input_dir, trust_remote_code=True)
    
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"  # Use eager attention to avoid GQA ONNX export issues
    )
    model = model.to("cpu")
    model.eval()
    
    print("  [OK] Model loaded")
    return config, processor, model


def export_vision_model(model, output_dir):
    """Export vision encoder."""
    print("\n[2/4] Exporting vision encoder...")
    
    import torch.nn as nn
    
    class VisionWrapper(nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual
        
        def forward(self, pixel_values, image_grid_thw):
            outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif isinstance(outputs, dict):
                return outputs['pooler_output']
            else:
                return outputs[1]
    
    wrapper = VisionWrapper(model.model.visual)
    
    # Vision model expects patches: [num_patches, patch_dim]
    # For 384x384 image with 16x16 patches: 24x24=576 patches
    # Each patch: 3 channels * 2 temporal * 16 * 16 = 1536 features
    num_patches = 576
    patch_dim = 1536
    
    pixel_values = torch.randn(num_patches, patch_dim)
    image_grid_thw = torch.tensor([[1, 24, 24]], dtype=torch.int64)
    
    output_path = os.path.join(output_dir, "qwen3vl-vision.onnx")
    
    torch.onnx.export(
        wrapper,
        (pixel_values, image_grid_thw),
        output_path,
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["pooled_embeds"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
            "pooled_embeds": {0: "sequence"}
        },
        opset_version=18
    )
    
    print(f"  [OK] Vision model: {output_path}")


def export_embedding_model(model, output_dir):
    """Export embedding model."""
    print("\n[3/4] Exporting embedding model...")
    
    import torch.nn as nn
    
    class EmbeddingWrapper(nn.Module):
        def __init__(self, embed_tokens, merge_length=2):
            super().__init__()
            self.embed_tokens = embed_tokens
            self.merge_length = merge_length
        
        def forward(self, input_ids, vision_hidden_states, position_ids):
            inputs_embeds = self.embed_tokens(input_ids)
            
            vision_start_token_id = 151652
            num_vision_tokens = (vision_hidden_states.size(0) + self.merge_length - 1) // self.merge_length
            
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            
            vision_mask = (input_ids.view(-1) == vision_start_token_id)
            vision_indices = vision_mask.nonzero(as_tuple=True)[0]
            
            if len(vision_indices) > 0:
                for idx in range(0, min(len(vision_indices), num_vision_tokens)):
                    if idx < vision_hidden_states.size(0):
                        pos = vision_indices[idx]
                        inputs_embeds[pos] = vision_hidden_states[idx]
            
            inputs_embeds = inputs_embeds.reshape(B, N, C)
            return inputs_embeds
    
    wrapper = EmbeddingWrapper(model.model.language_model.embed_tokens)
    
    input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
    # Include vision_start token to ensure the branch is traced
    input_ids[0, 2] = 151652  # vision_start_token_id
    vision_hidden_states = torch.randn(144, 2560)
    position_ids = torch.arange(10, dtype=torch.long).unsqueeze(0)
    
    output_path = os.path.join(output_dir, "qwen3vl-embedding.onnx")
    
    torch.onnx.export(
        wrapper,
        (input_ids, vision_hidden_states, position_ids),
        output_path,
        input_names=["input_ids", "vision_hidden_states", "position_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "vision_hidden_states": {0: "vision_sequence"},
            "position_ids": {0: "batch", 1: "sequence"},
            "inputs_embeds": {0: "batch", 1: "sequence"}
        },
        opset_version=18
    )
    
    print(f"  [OK] Embedding model: {output_path}")


def export_text_model(input_dir, output_dir, precision="int4"):
    """Export text decoder using OGA builder."""
    print(f"\n[4/4] Exporting text decoder ({precision.upper()})...")
    
    create_model(
        "qwen3-vl",  # model_name
        input_dir,   # input_path
        output_dir,  # output_dir
        precision,   # precision
        "cpu",       # execution_provider
        os.path.join(output_dir, ".cache")  # cache_dir
    )
    
    print(f"  [OK] Text model: {os.path.join(output_dir, 'model.onnx')}")


def create_vision_processor_config(output_dir):
    """Create vision_processor.json in ort-extensions format."""
    config = {
        "processor": {
            "name": "qwen3vl_image_processor",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {
                            "color_space": "RGB"
                        }
                    }
                },
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "size": [384, 384],
                            "interpolation": "BICUBIC"
                        }
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale"
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711]
                        }
                    }
                },
                {
                    "operation": {
                        "name": "qwen3vl_vision_processor",
                        "type": "Qwen3VLVisionProcessor",
                        "attrs": {
                            "patch_size": 16,
                            "temporal_patch_size": 2,
                            "merge_size": 2
                        }
                    }
                }
            ]
        }
    }
    
    config_path = os.path.join(output_dir, "vision_processor.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  [OK] Created: vision_processor.json")


def update_genai_config(output_dir):
    """Update genai_config.json with vision and embedding sections."""
    config_path = os.path.join(output_dir, "genai_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add type
    config["model"]["type"] = "qwen3_vl"
    
    # Add vision configuration
    config["model"]["vision"] = {
        "filename": "qwen3vl-vision.onnx",
        "config_filename": "vision_processor.json",
        "inputs": {
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw"
        },
        "outputs": {
            "image_features": "pooled_embeds"
        },
        "spatial_merge_size": 2
    }
    
    # Add embedding configuration
    config["model"]["embedding"] = {
        "filename": "qwen3vl-embedding.onnx",
        "inputs": {
            "input_ids": "input_ids",
            "image_features": "vision_hidden_states"
        },
        "outputs": {
            "inputs_embeds": "inputs_embeds"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  [OK] Updated: genai_config.json")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL for OGA Integration")
    parser.add_argument(
        "--input",
        type=str,
        default="./pytorch",
        help="Input PyTorch model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./qwen3vl-oga-fp32-int4",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int4",
        choices=["fp32", "fp16", "int4", "int8"],
        help="Text model precision"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle relative paths
    if not os.path.isabs(args.input):
        input_dir = os.path.normpath(os.path.join(script_dir, args.input))
    else:
        input_dir = args.input
    
    if not os.path.isabs(args.output):
        output_dir = os.path.normpath(os.path.join(script_dir, args.output))
    else:
        output_dir = args.output
    
    print("=" * 80)
    print("Qwen3-VL ONNX Export for OGA Integration")
    print("=" * 80)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Text precision: {args.precision.upper()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare model
    config, processor, model = prepare_model(input_dir)
    
    # Export models
    export_vision_model(model, output_dir)
    export_embedding_model(model, output_dir)
    export_text_model(input_dir, output_dir, args.precision)
    
    # Create configuration files
    print("\n[5/4] Creating configuration files...")
    create_vision_processor_config(output_dir)
    update_genai_config(output_dir)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Export completed successfully!")
    print("=" * 80)
    print(f"\nAll files in: {output_dir}")
    print("\nExported models:")
    print("  - qwen3vl-vision.onnx       (FP32, vision encoder)")
    print("  - qwen3vl-embedding.onnx    (FP32, embedding injector)")
    print(f"  - model.onnx                ({args.precision.upper()}, text decoder)")
    print("\nConfiguration files:")
    print("  - genai_config.json         (OGA configuration)")
    print("  - vision_processor.json     (vision preprocessing)")
    print("  - tokenizer.json            (tokenizer)")
    print("\nNext steps:")
    print(f"  python qwen3vl-oga.py -m {output_dir}")
    print()


if __name__ == "__main__":
    main()
