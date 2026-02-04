"""
Qwen3-VL Multimodal Inference with ONNX Runtime
Hybrid approach: HuggingFace preprocessing + ONNX inference

Components:
1. HuggingFace ImageProcessor - Image preprocessing (lightweight)
2. vision_encoder.onnx - Image → Vision embeddings (ONNX)
3. embeddings.onnx - Token IDs → Text embeddings (ONNX)
4. model.onnx - Merged embeddings → Logits (ONNX with 3D MRoPE)
"""

import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import json

# Import HuggingFace components for preprocessing
try:
    from transformers import AutoTokenizer, AutoProcessor
    import torch  # Needed for image processor tensor conversion
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Install with: pip install transformers torch")
    raise


# Note: We use HuggingFace's image processor for correct preprocessing,
# but all inference runs on ONNX models


class Qwen3VLONNXPipeline:
    """
    Complete ONNX-based inference pipeline for Qwen3-VL
    """
    
    def __init__(
        self,
        model_dir: str,
        execution_provider: str = "CPUExecutionProvider"
    ):
        self.model_dir = Path(model_dir)
        self.execution_provider = execution_provider
        
        print(f"Loading Qwen3-VL ONNX models from {model_dir}...")
        
        # Load ONNX models
        self.vision_session = self._load_model("cpu/vision_encoder.onnx")
        self.embeddings_session = self._load_model("cpu/embeddings.onnx")
        self.text_session = self._load_model("cpu-text/model.onnx")
        
        print("  [OK] All ONNX models loaded")
        
        # Load config
        with open(self.model_dir / "cpu-text" / "genai_config.json") as f:
            self.config = json.load(f)
        
        # Load HuggingFace processor (for image preprocessing and tokenization)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir / "pytorch"),
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        
        # OPTION B QUICK FIX: Force 384×384 to match vision encoder export dimensions
        # Vision encoder was exported with 24×24 patches = 384×384 pixels
        # This ensures runtime dimensions match export dimensions
        target_pixels = 384 * 384
        self.image_processor.min_pixels = target_pixels
        self.image_processor.max_pixels = target_pixels
        
        print("  [OK] Processor (tokenizer + image processor) loaded")
        print(f"  [FIX] Forcing image size to {int(target_pixels**0.5)}×{int(target_pixels**0.5)} to match export")
        
        # Model config
        self.vocab_size = 151936
        self.hidden_size = 2560
        self.num_layers = 36
        self.num_kv_heads = 8
        self.head_dim = 128
        
        # Special tokens
        self.image_token = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        
        print("Model ready for inference!\n")
    
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX model"""
        full_path = self.model_dir / model_path
        providers = [self.execution_provider]
        session = ort.InferenceSession(str(full_path), providers=providers)
        return session
    
    def process_images(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process images and get vision embeddings
        Uses HuggingFace processor for preprocessing, ONNX for inference
        """
        # Load images
        images = [Image.open(path) for path in image_paths]
        
        # Preprocess using HuggingFace processor
        # Qwen3VL processor returns PyTorch tensors by default, we'll convert to NumPy
        try:
            # Try with return_tensors="np" first
            inputs = self.image_processor(images=images, return_tensors="np")
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
        except (ValueError, NotImplementedError):
            # Fallback: Get PyTorch tensors and convert to NumPy
            inputs = self.image_processor(images=images, return_tensors="pt")
            # Convert PyTorch tensors to NumPy
            pixel_values = inputs["pixel_values"].cpu().numpy()  # [num_patches, 1536]
            grid_thw = inputs["image_grid_thw"].cpu().numpy()  # [num_images, 3]
        
        print(f"  Image preprocessing:")
        print(f"    pixel_values: {pixel_values.shape}")
        print(f"    grid_thw: {grid_thw.shape} = {grid_thw.tolist()}")
        
        # Run vision encoder ONNX model
        vision_outputs = self.vision_session.run(
            None,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
            }
        )
        
        # Get pooled embeddings (merged patches for LLM)
        vision_embeds = vision_outputs[0]  # [num_merged_patches, hidden_size]
        
        print(f"    vision_embeds: {vision_embeds.shape}")
        
        return vision_embeds, grid_thw
    
    def prepare_text_with_images(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> dict:
        """
        Prepare text with image placeholders
        Returns input_ids with <|image_pad|> tokens
        """
        # If images provided, add image tokens to prompt
        if image_paths:
            # Count merged patches per image
            vision_embeds, grid_thw = self.process_images(image_paths)
            
            # Calculate number of image tokens per image
            merge_size = self.image_processor.merge_size
            num_images = len(grid_thw)
            
            image_token_counts = []
            merge_size = 2  # Qwen3-VL merge_size
            for i in range(num_images):
                t, h, w = grid_thw[i]
                num_tokens = (t * h * w) // (merge_size ** 2)
                image_token_counts.append(num_tokens)
            
            print(f"  Text processing:")
            print(f"    Image token counts: {image_token_counts}")
            
            # Replace <|image_pad|> in text with correct number of tokens
            for num_tokens in image_token_counts:
                if self.image_token in text:
                    # Replace first occurrence
                    placeholder = self.image_token * num_tokens
                    text = text.replace(self.image_token, placeholder, 1)
        else:
            vision_embeds = None
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            add_special_tokens=True
        )
        
        # Ensure int64 for ONNX compatibility
        input_ids = inputs["input_ids"][0].astype(np.int64)  # [seq_len]
        
        print(f"    input_ids: {input_ids.shape}")
        print(f"    Text: {text[:100]}...")
        
        return {
            "input_ids": input_ids,
            "vision_embeds": vision_embeds,
        }
    
    def merge_embeddings(
        self,
        input_ids: np.ndarray,
        vision_embeds: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Merge text and vision embeddings
        
        Args:
            input_ids: [seq_len] (int64)
            vision_embeds: [num_vision_tokens, hidden_size]
        
        Returns:
            merged_embeds: [1, total_seq_len, hidden_size]
        """
        # Ensure int64 dtype
        input_ids = input_ids.astype(np.int64)
        
        # Get text embeddings from ONNX model
        text_embeds = self.embeddings_session.run(
            None,
            {"input_ids": input_ids[None, :]}  # Add batch dim [1, seq_len]
        )[0]  # [1, seq_len, hidden_size]
        
        print(f"  Embedding merge:")
        print(f"    text_embeds: {text_embeds.shape}")
        
        if vision_embeds is None:
            return text_embeds
        
        # Find image token positions
        image_positions = np.where(input_ids == self.image_token_id)[0]
        
        print(f"    Image token positions: {len(image_positions)} tokens")
        print(f"    vision_embeds: {vision_embeds.shape}")
        
        # Replace image tokens with vision embeddings
        # Create output tensor
        merged_embeds = text_embeds[0].copy()  # [seq_len, hidden_size]
        
        # Inject vision embeddings at image token positions
        vision_idx = 0
        for pos in image_positions:
            if vision_idx < len(vision_embeds):
                merged_embeds[pos] = vision_embeds[vision_idx]
                vision_idx += 1
        
        print(f"    merged_embeds: {merged_embeds.shape}")
        print(f"    Injected {vision_idx} vision tokens")
        
        return merged_embeds[None, :]  # Add batch dim
    
    def create_position_ids_3d(self, seq_len: int, past_seq_len: int = 0) -> np.ndarray:
        """
        Create 3D position IDs for MRoPE
        
        Args:
            seq_len: Current sequence length
            past_seq_len: Length of past sequence (for incremental decoding)
        
        Returns:
            position_ids: [3, 1, seq_len]
        """
        # MRoPE sections: [24, 20, 20] for temporal/height/width
        # For text-only or mixed, we use default positions
        
        # Create position IDs starting from past_seq_len
        pos_ids = np.arange(past_seq_len, past_seq_len + seq_len, dtype=np.int64)
        
        # Stack for 3 dimensions: [3, seq_len]
        position_ids = np.stack([pos_ids, pos_ids, pos_ids], axis=0)
        
        # Add batch dimension: [3, 1, seq_len]
        position_ids = position_ids[:, None, :]
        
        return position_ids
    
    def apply_temperature(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply temperature scaling to logits"""
        if temperature == 0.0:
            # Greedy - return as is for argmax
            return logits
        return logits / temperature
    
    def top_k_filtering(self, logits: np.ndarray, top_k: int = 50) -> np.ndarray:
        """Filter logits to keep only top-k tokens"""
        if top_k <= 0:
            return logits
        
        # Get indices of top-k logits
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def top_p_filtering(self, logits: np.ndarray, top_p: float = 0.9) -> np.ndarray:
        """Filter logits using nucleus (top-p) sampling"""
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Compute softmax
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        # Compute cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep at least one token
        sorted_indices_to_remove[0] = False
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def sample_token(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> int:
        """
        Sample next token from logits
        
        Args:
            logits: [vocab_size] logits for next token
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            do_sample: If False, use greedy decoding
        
        Returns:
            token_id: Sampled token ID
        """
        # Apply temperature
        logits = self.apply_temperature(logits, temperature)
        
        if not do_sample or temperature == 0.0:
            # Greedy decoding
            return int(np.argmax(logits))
        
        # Apply top-k filtering
        if top_k > 0:
            logits = self.top_k_filtering(logits, top_k)
        
        # Apply top-p filtering
        if top_p < 1.0:
            logits = self.top_p_filtering(logits, top_p)
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        
        # Sample from distribution
        token_id = np.random.choice(len(probs), p=probs)
        
        return int(token_id)
    
    def generate(
        self,
        text: str,
        image_paths: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> str:
        """
        Generate text response with autoregressive decoding
        
        Args:
            text: Input prompt
            image_paths: Optional list of image paths
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Keep only top k tokens (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            do_sample: If False, use greedy decoding
            stream: If True, print tokens as they're generated
            eos_token_id: End-of-sequence token ID (stops generation)
        
        Returns:
            generated_text: Complete generated text
        """
        if stream:
            print("="*70)
            print("Starting generation...")
            print("="*70)
        
        # Get EOS token if not provided
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Prepare inputs
        inputs = self.prepare_text_with_images(text, image_paths)
        input_ids = inputs["input_ids"]
        vision_embeds = inputs["vision_embeds"]
        
        # Merge embeddings for prompt
        merged_embeds = self.merge_embeddings(input_ids, vision_embeds)
        
        batch_size = 1
        prompt_len = merged_embeds.shape[1]
        
        # Create position IDs (3D for MRoPE)
        position_ids = self.create_position_ids_3d(prompt_len, past_seq_len=0)
        
        # Create attention mask
        attention_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
        
        # Create empty KV caches
        kv_cache_shape = (batch_size, self.num_kv_heads, 0, self.head_dim)
        
        inputs_dict = {
            "inputs_embeds": merged_embeds.astype(np.float32),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        
        # Add empty KV caches
        for layer_idx in range(self.num_layers):
            inputs_dict[f"past_key_values.{layer_idx}.key"] = np.zeros(kv_cache_shape, dtype=np.float32)
            inputs_dict[f"past_key_values.{layer_idx}.value"] = np.zeros(kv_cache_shape, dtype=np.float32)
        
        if stream:
            print(f"\n  Initial forward pass:")
            print(f"    inputs_embeds: {merged_embeds.shape}")
            print(f"    position_ids: {position_ids.shape}")
            print(f"    attention_mask: {attention_mask.shape}")
            print(f"\n  Generating (max {max_new_tokens} tokens)...")
            print(f"  ", end="", flush=True)
        
        # First forward pass with prompt
        outputs = self.text_session.run(None, inputs_dict)
        logits = outputs[0]  # [batch_size, seq_len, vocab_size]
        
        # Extract updated KV caches
        kv_caches = {}
        output_idx = 1
        for layer_idx in range(self.num_layers):
            kv_caches[f"past_key_values.{layer_idx}.key"] = outputs[output_idx]
            kv_caches[f"past_key_values.{layer_idx}.value"] = outputs[output_idx + 1]
            output_idx += 2
        
        # Sample first token
        next_token_logits = logits[0, -1, :]  # Last position
        next_token_id = self.sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        
        generated_ids = [next_token_id]
        
        # Decode and print first token
        if stream:
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(token_text, end="", flush=True)
        
        # Autoregressive generation loop
        current_seq_len = prompt_len
        
        for step in range(1, max_new_tokens):
            # Check for EOS
            if next_token_id == eos_token_id:
                if stream:
                    print()  # Newline after generation
                break
            
            # Get embedding for next token
            next_token_embed = self.embeddings_session.run(
                None,
                {"input_ids": np.array([[next_token_id]], dtype=np.int64)}
            )[0]  # [1, 1, hidden_size]
            
            # Create position IDs for new token
            position_ids = self.create_position_ids_3d(1, past_seq_len=current_seq_len)
            
            # Expand attention mask
            attention_mask = np.ones((batch_size, current_seq_len + 1), dtype=np.int64)
            
            # Prepare inputs for incremental decoding
            inputs_dict = {
                "inputs_embeds": next_token_embed.astype(np.float32),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
            
            # Add KV caches from previous step
            for layer_idx in range(self.num_layers):
                inputs_dict[f"past_key_values.{layer_idx}.key"] = kv_caches[f"past_key_values.{layer_idx}.key"]
                inputs_dict[f"past_key_values.{layer_idx}.value"] = kv_caches[f"past_key_values.{layer_idx}.value"]
            
            # Forward pass
            outputs = self.text_session.run(None, inputs_dict)
            logits = outputs[0]  # [batch_size, 1, vocab_size]
            
            # Update KV caches
            output_idx = 1
            for layer_idx in range(self.num_layers):
                kv_caches[f"past_key_values.{layer_idx}.key"] = outputs[output_idx]
                kv_caches[f"past_key_values.{layer_idx}.value"] = outputs[output_idx + 1]
                output_idx += 2
            
            # Sample next token
            next_token_logits = logits[0, 0, :]  # Only one position
            next_token_id = self.sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
            
            generated_ids.append(next_token_id)
            current_seq_len += 1
            
            # Decode and print token
            if stream:
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(token_text, end="", flush=True)
        
        # Final decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if stream:
            print()  # Newline
            print(f"\n  Generated {len(generated_ids)} tokens")
            print("="*70)
        
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL ONNX Inference with Streaming")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory containing cpu/ and cpu-text/ folders"
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="Path to image(s)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Describe this image in detail.",
        help="Input text/prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (0 = disabled)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (1.0 = disabled)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Qwen3VLONNXPipeline(
        model_dir=args.model_dir,
        execution_provider="CPUExecutionProvider"
    )
    
    # Prepare prompt
    if args.image:
        prompt = f"{args.text}\n{pipeline.image_token}"
    else:
        prompt = args.text
    
    # Generate
    output = pipeline.generate(
        text=prompt,
        image_paths=args.image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        stream=not args.no_stream
    )
    
    if args.no_stream:
        print(f"\nGenerated: {output}")
    
    print(f"\n[OK] Generation complete!")


if __name__ == "__main__":
    main()
