"""
Test Qwen3-VL ONNX inference pipeline

This script demonstrates how to run inference with exported Qwen3-VL models:
1. Vision Encoder: Processes images
2. Embeddings: Converts tokens to embeddings  
3. Text Decoder: Generates text with merged image+text embeddings

Based on Phi4-MM inference approach
"""
import argparse
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import torch

class Qwen3VLImageProcessor:
    """Image processor for Qwen3-VL"""
    
    def __init__(self, config):
        self.min_pixels = config.get("min_pixels", 256 * 28 * 28)
        self.max_pixels = config.get("max_pixels", 1280 * 28 * 28)
        self.patch_size = config.get("patch_size", 16)
        self.temporal_patch_size = config.get("temporal_patch_size", 2)
        self.merge_size = config.get("merge_size", 2)
        self.image_mean = config.get("image_mean", [0.5, 0.5, 0.5])
        self.image_std = config.get("image_std", [0.5, 0.5, 0.5])
        
    def smart_resize(self, height, width):
        """
        Calculate target size based on min/max pixels constraint
        Similar to Qwen3-VL smart_resize logic
        """
        factor = self.patch_size * self.merge_size  # 32
        
        # Round to nearest factor
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        
        # Check pixel count
        num_pixels = h_bar * w_bar
        
        if num_pixels > self.max_pixels:
            # Scale down
            scale = (self.max_pixels / num_pixels) ** 0.5
            h_bar = max(factor, int(height * scale / factor) * factor)
            w_bar = max(factor, int(width * scale / factor) * factor)
        elif num_pixels < self.min_pixels:
            # Scale up
            scale = (self.min_pixels / num_pixels) ** 0.5
            h_bar = int(height * scale / factor) * factor
            w_bar = int(width * scale / factor) * factor
        
        return h_bar, w_bar
    
    def preprocess(self, image_path):
        """
        Preprocess image to format expected by vision encoder
        
        Returns:
            pixel_values: [num_patches, channels * temporal_patch * patch_size^2]
            image_grid_thw: [num_images, 3] - temporal, height, width in patches
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Smart resize
        target_h, target_w = self.smart_resize(height, width)
        image = image.resize((target_w, target_h), Image.BICUBIC)
        
        print(f"  Original size: {width}x{height}")
        print(f"  Resized to: {target_w}x{target_h}")
        
        # Convert to numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        for c in range(3):
            img_array[:, :, c] = (img_array[:, :, c] - self.image_mean[c]) / self.image_std[c]
        
        # Convert to CHW format
        img_array = img_array.transpose(2, 0, 1)  # [C, H, W]
        
        # Reshape to patches
        # For simplicity, we'll create the flattened patch representation
        # In practice, this involves Conv3D patch embedding
        # For now, let's create a simplified version
        
        # Calculate grid dimensions
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size
        grid_t = 1  # Single image, no temporal dimension
        
        # Create patches by reshaping
        # [C, H, W] -> [C, grid_h, patch_size, grid_w, patch_size]
        patches = img_array.reshape(
            3, 
            grid_h, self.patch_size,
            grid_w, self.patch_size
        )
        
        # Transpose to [grid_h, grid_w, C, patch_size, patch_size]
        patches = patches.transpose(1, 3, 0, 2, 4)
        
        # Flatten patches: [grid_h, grid_w, C * patch_size * patch_size]
        patches = patches.reshape(grid_h, grid_w, -1)
        
        # Further flatten to [num_patches, feature_dim]
        pixel_values = patches.reshape(-1, patches.shape[-1])
        
        # Add temporal dimension (for Conv3D compatibility)
        # Reshape to account for temporal_patch_size=2
        # We need to duplicate or pad to make it compatible
        # For single image, we'll duplicate the patches
        feature_dim = pixel_values.shape[1]
        num_patches = pixel_values.shape[0]
        
        # Expand for temporal: duplicate to create temporal_patch_size frames
        # [num_patches, feature_dim] -> [num_patches, temporal_patch_size, feature_dim]
        # Then flatten: [num_patches, temporal_patch_size * feature_dim]
        pixel_values = np.repeat(pixel_values[:, np.newaxis, :], self.temporal_patch_size, axis=1)
        pixel_values = pixel_values.reshape(num_patches, -1)
        
        # Grid THW
        image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
        
        print(f"  Pixel values shape: {pixel_values.shape}")
        print(f"  Grid THW: {image_grid_thw}")
        
        return pixel_values.astype(np.float32), image_grid_thw

class Qwen3VLPipeline:
    """ONNX inference pipeline for Qwen3-VL"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # Load ONNX sessions
        print("Loading ONNX models...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        vision_path = self.model_path / "vision_encoder.onnx"
        embeddings_path = self.model_path / "embeddings.onnx"
        decoder_path = self.model_path / "model.onnx"
        
        if vision_path.exists():
            self.vision_session = ort.InferenceSession(str(vision_path), sess_options)
            print(f"  ✓ Loaded vision encoder")
        else:
            self.vision_session = None
            print(f"  ⚠ Vision encoder not found, skipping")
        
        if embeddings_path.exists():
            self.embeddings_session = ort.InferenceSession(str(embeddings_path), sess_options)
            print(f"  ✓ Loaded embeddings")
        else:
            self.embeddings_session = None
            print(f"  ⚠ Embeddings not found")
        
        if decoder_path.exists():
            self.decoder_session = ort.InferenceSession(str(decoder_path), sess_options)
            print(f"  ✓ Loaded text decoder")
        else:
            raise FileNotFoundError(f"Text decoder not found at {decoder_path}")
        
        # Load processor config
        processor_config_path = self.model_path / "vision_processor.json"
        if processor_config_path.exists():
            with open(processor_config_path) as f:
                config = json.load(f)
            self.image_processor = Qwen3VLImageProcessor(config)
            print(f"  ✓ Loaded image processor config")
        else:
            print(f"  ⚠ Image processor config not found, using defaults")
            self.image_processor = Qwen3VLImageProcessor({})
        
        # Load tokenizer
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            print(f"  ✓ Loaded tokenizer")
        except Exception as e:
            print(f"  ✗ Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def process_image(self, image_path):
        """Process image through vision encoder"""
        if self.vision_session is None:
            print("Vision encoder not available")
            return None
        
        print("\nProcessing image...")
        pixel_values, image_grid_thw = self.image_processor.preprocess(image_path)
        
        print("Running vision encoder...")
        outputs = self.vision_session.run(
            None,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw
            }
        )
        
        image_embeds = outputs[0]
        print(f"  Image embeddings shape: {image_embeds.shape}")
        
        return image_embeds
    
    def tokenize_text(self, prompt, add_vision_token=True):
        """Tokenize text prompt"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        
        # Add special vision tokens if needed
        if add_vision_token:
            # Qwen3-VL uses <|vision_start|><|image_pad|><|vision_end|>
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        return inputs["input_ids"]
    
    def embed_tokens(self, input_ids):
        """Convert token IDs to embeddings"""
        if self.embeddings_session is None:
            raise RuntimeError("Embeddings model not available")
        
        print("Converting tokens to embeddings...")
        outputs = self.embeddings_session.run(
            None,
            {"input_ids": input_ids}
        )
        
        text_embeds = outputs[0]
        print(f"  Text embeddings shape: {text_embeds.shape}")
        
        return text_embeds
    
    def merge_embeddings(self, text_embeds, image_embeds, image_token_id=None):
        """
        Merge text and image embeddings
        
        Replace <|image_pad|> tokens with image embeddings
        """
        if image_embeds is None:
            return text_embeds
        
        # Find image token positions
        # This is a simplified version - actual implementation needs to handle
        # the exact token sequence
        
        # For now, we'll assume image embeddings go at the beginning
        # Shape: [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = text_embeds.shape
        num_image_tokens = image_embeds.shape[0]
        
        # Create merged embeddings
        # [batch, num_image_tokens + seq_len, hidden_dim]
        merged_embeds = np.concatenate([
            image_embeds[np.newaxis, :, :],  # Add batch dim
            text_embeds
        ], axis=1)
        
        print(f"  Merged embeddings shape: {merged_embeds.shape}")
        
        return merged_embeds
    
    def generate(self, inputs_embeds, max_length=100):
        """
        Generate text using the decoder
        
        Note: This is a simplified version. Full implementation needs:
        - KV cache management
        - Position IDs generation (3D for Qwen3-VL)
        - Attention mask handling
        """
        print(f"\nGenerating (max_length={max_length})...")
        
        # For now, just run one forward pass
        # Full implementation would need autoregressive generation
        
        # Create dummy position_ids (3D: [3, batch, seq_len])
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = np.arange(seq_len)[np.newaxis, :].repeat(batch_size, axis=0)
        position_ids = np.stack([position_ids, position_ids, position_ids], axis=0)
        
        print(f"  Position IDs shape: {position_ids.shape}")
        
        # Run decoder
        outputs = self.decoder_session.run(
            None,
            {
                "inputs_embeds": inputs_embeds.astype(np.float32),
                "position_ids": position_ids.astype(np.int64)
            }
        )
        
        logits = outputs[0]
        print(f"  Logits shape: {logits.shape}")
        
        # Get predicted tokens (argmax)
        predicted_ids = np.argmax(logits[0], axis=-1)
        
        return predicted_ids
    
    def __call__(self, prompt, image_path=None):
        """Run full inference pipeline"""
        print("=" * 70)
        print("Running Qwen3-VL Inference")
        print("=" * 70)
        
        # Process image
        image_embeds = None
        if image_path:
            image_embeds = self.process_image(image_path)
        
        # Tokenize text
        print(f"\nTokenizing prompt: {prompt}")
        input_ids = self.tokenize_text(prompt, add_vision_token=(image_path is not None))
        print(f"  Input IDs shape: {input_ids.shape}")
        
        # Embed tokens
        text_embeds = self.embed_tokens(input_ids)
        
        # Merge embeddings
        inputs_embeds = self.merge_embeddings(text_embeds, image_embeds)
        
        # Generate
        output_ids = self.generate(inputs_embeds)
        
        # Decode
        if self.tokenizer:
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"\nGenerated text: {output_text}")
        else:
            print(f"\nGenerated token IDs: {output_ids}")
        
        print("=" * 70)
        
        return output_ids

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL ONNX inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to ONNX models directory")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="Text prompt")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Qwen3VLPipeline(args.model_path)
    
    # Run inference
    pipeline(args.prompt, args.image_path)

if __name__ == "__main__":
    main()
