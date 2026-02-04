"""
Qwen3-VL Multimodal ONNX Inference
Full vision + text pipeline using ONNX Runtime

Components:
1. HuggingFace ImageProcessor - Image preprocessing
2. qwen3vl-vision.onnx - Image → Vision embeddings
3. qwen3vl-embedding.onnx - Token IDs → Text embeddings
4. model.onnx - Merged embeddings → Text generation
"""

import sys
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from typing import List, Optional
import json

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

try:
    from transformers import AutoProcessor
    import torch  # Needed for image processor tensor conversion
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Install with: pip install transformers torch pillow")
    raise


class Qwen3VLMultimodalPipeline:
    """
    Complete ONNX-based multimodal inference pipeline for Qwen3-VL
    Supports both text-only and vision+text inputs
    """
    
    def __init__(
        self,
        model_dir: str = ".",
        text_precision: str = "fp32",
        execution_provider: str = "CPUExecutionProvider"
    ):
        self.model_dir = Path(model_dir)
        self.text_precision = text_precision
        self.execution_provider = execution_provider
        
        print(f"Loading Qwen3-VL Multimodal ONNX Pipeline...")
        print(f"  Vision: FP32")
        print(f"  Embeddings: FP32")
        print(f"  Text: {text_precision.upper()}")
        
        # Load ONNX models
        vision_path = self.model_dir / "cpu-fp32" / "qwen3vl-vision.onnx"
        embedding_path = self.model_dir / "cpu-fp32" / "qwen3vl-embedding.onnx"
        text_path = self.model_dir / f"cpu-{text_precision}" / "model.onnx"
        
        print(f"\n  Loading models...")
        providers = [self.execution_provider]
        
        self.vision_session = ort.InferenceSession(str(vision_path), providers=providers)
        self.embeddings_session = ort.InferenceSession(str(embedding_path), providers=providers)
        self.text_session = ort.InferenceSession(str(text_path), providers=providers)
        
        print(f"    Vision: {vision_path.name}")
        print(f"    Embeddings: {embedding_path.name}")
        print(f"    Text: {text_path.name}")
        print("  [OK] All ONNX models loaded")
        
        # Load HuggingFace processor (for image preprocessing and tokenization)
        pytorch_dir = self.model_dir / "pytorch"
        self.processor = AutoProcessor.from_pretrained(
            str(pytorch_dir),
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        
        print("  [OK] Processor (tokenizer + image processor) loaded")
        
        # Load model config
        config_path = self.model_dir / f"cpu-{text_precision}" / "genai_config.json"
        with open(config_path) as f:
            config = json.load(f)["model"]
        
        decoder_config = config["decoder"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = decoder_config["hidden_size"]
        self.num_layers = decoder_config["num_hidden_layers"]
        self.num_kv_heads = decoder_config["num_key_value_heads"]
        self.head_dim = decoder_config["head_size"]
        self.eos_token_id = config["eos_token_id"]
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        
        # Special tokens
        self.image_token = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        
        print(f"\n  Model config:")
        print(f"    Layers: {self.num_layers}")
        print(f"    Hidden size: {self.hidden_size}")
        print(f"    Image token: '{self.image_token}' (id={self.image_token_id})")
        print("\nPipeline ready for multimodal inference!\n")
    
    def process_images(self, image_paths: List[str]) -> tuple:
        """
        Process images and get vision embeddings from ONNX model
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            vision_embeds: [num_merged_patches, hidden_size]
            grid_thw: [num_images, 3] (temporal, height, width grid)
        """
        print(f"  Processing {len(image_paths)} image(s)...")
        
        # Load images
        images = [Image.open(path).convert('RGB') for path in image_paths]
        
        # Preprocess using HuggingFace processor
        try:
            # Try NumPy directly
            inputs = self.image_processor(images=images, return_tensors="np")
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
        except (ValueError, NotImplementedError):
            # Fallback: Get PyTorch tensors and convert to NumPy
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].cpu().numpy()
            grid_thw = inputs["image_grid_thw"].cpu().numpy()
        
        print(f"    Preprocessed: pixel_values={pixel_values.shape}, grid_thw={grid_thw.shape}")
        
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
        
        print(f"    Vision embeddings: {vision_embeds.shape}")
        
        return vision_embeds, grid_thw
    
    def prepare_multimodal_prompt(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> dict:
        """
        Prepare multimodal prompt with image placeholders
        
        Args:
            text: Text prompt (should contain <|image_pad|> for each image)
            image_paths: Optional list of image paths
        
        Returns:
            dict with input_ids, vision_embeds, image_token_positions
        """
        if image_paths:
            # Process images through vision model
            vision_embeds, grid_thw = self.process_images(image_paths)
            
            # Calculate number of image tokens per image
            # Qwen3-VL merges patches with merge_size=2
            merge_size = 2
            num_images = len(grid_thw)
            
            image_token_counts = []
            for i in range(num_images):
                t, h, w = grid_thw[i]
                num_tokens = (t * h * w) // (merge_size ** 2)
                image_token_counts.append(num_tokens)
            
            print(f"    Image tokens per image: {image_token_counts}")
            
            # Replace <|image_pad|> placeholders with correct number of tokens
            prompt_text = text
            for num_tokens in image_token_counts:
                if self.image_token in prompt_text:
                    # Replace first occurrence with repeated tokens
                    placeholder = self.image_token * num_tokens
                    prompt_text = prompt_text.replace(self.image_token, placeholder, 1)
        else:
            vision_embeds = None
            prompt_text = text
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids = np.array(input_ids, dtype=np.int64)
        
        print(f"  Text preparation:")
        print(f"    Input tokens: {len(input_ids)}")
        
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
            vision_embeds: [num_vision_tokens, hidden_size] or None
        
        Returns:
            merged_embeds: [1, total_seq_len, hidden_size]
        """
        # Get text embeddings from ONNX model
        text_embeds = self.embeddings_session.run(
            None,
            {"input_ids": input_ids[None, :]}  # [1, seq_len]
        )[0]  # [1, seq_len, hidden_size]
        
        print(f"    Text embeddings: {text_embeds.shape}")
        
        if vision_embeds is None:
            # Text-only mode
            return text_embeds
        
        # Find image token positions
        image_positions = np.where(input_ids == self.image_token_id)[0]
        
        print(f"    Image token positions: {len(image_positions)} tokens at indices {image_positions[:5].tolist()}...")
        print(f"    Vision embeddings to inject: {vision_embeds.shape}")
        
        # Replace image tokens with vision embeddings
        merged_embeds = text_embeds[0].copy()  # [seq_len, hidden_size]
        
        # Inject vision embeddings at image token positions
        vision_idx = 0
        for pos in image_positions:
            if vision_idx < len(vision_embeds):
                merged_embeds[pos] = vision_embeds[vision_idx]
                vision_idx += 1
        
        print(f"    Merged embeddings: {merged_embeds.shape} (injected {vision_idx} vision tokens)")
        
        return merged_embeds[None, :]  # Add batch dim
    
    def sample_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> int:
        """Sample next token from logits with temperature/top-k/top-p"""
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        if not do_sample or temperature == 0.0:
            return int(np.argmax(logits))
        
        # Apply top-k
        if top_k > 0:
            top_k = min(top_k, len(logits))
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -float('inf')
        
        # Apply top-p
        if top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
            sorted_probs = sorted_probs / sorted_probs.sum()
            cumulative_probs = np.cumsum(sorted_probs)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('inf')
        
        # Sample
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
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
    ) -> str:
        """
        Generate text response with optional vision input
        
        Args:
            text: Input prompt (use <|image_pad|> for image placeholders)
            image_paths: Optional list of image paths
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p sampling (1.0 = disabled)
            do_sample: If False, use greedy decoding
        
        Returns:
            generated_text: Complete generated text
        """
        print("=" * 70)
        if image_paths:
            print(f"Multimodal Generation ({len(image_paths)} image(s))")
        else:
            print("Text-Only Generation")
        print("=" * 70)
        print(f"Prompt: {text}")
        print("=" * 70)
        
        # Prepare inputs
        inputs = self.prepare_multimodal_prompt(text, image_paths)
        input_ids = inputs["input_ids"]
        vision_embeds = inputs["vision_embeds"]
        
        # Merge embeddings
        merged_embeds = self.merge_embeddings(input_ids, vision_embeds)
        
        batch_size = 1
        prompt_len = merged_embeds.shape[1]
        
        # Create attention mask
        # NOTE: Qwen3 does NOT use position_ids input - it computes positions internally
        attention_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
        
        # Empty KV caches
        kv_cache_shape = (batch_size, self.num_kv_heads, 0, self.head_dim)
        inputs_dict = {
            "inputs_embeds": merged_embeds.astype(np.float32),
            "attention_mask": attention_mask,
        }
        
        # Add empty KV caches
        for layer_idx in range(self.num_layers):
            inputs_dict[f"past_key_values.{layer_idx}.key"] = np.zeros(kv_cache_shape, dtype=np.float32)
            inputs_dict[f"past_key_values.{layer_idx}.value"] = np.zeros(kv_cache_shape, dtype=np.float32)
        
        print(f"\n  Initial forward pass...")
        print(f"    Input embeddings: {merged_embeds.shape}")
        print(f"    Attention mask: {attention_mask.shape}")
        print(f"\n  Generating (max {max_new_tokens} tokens)...\n  ", end="", flush=True)
        
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
        next_token_id = self.sample_token(
            logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        
        generated_ids = [next_token_id]
        
        # Print first token
        token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
        print(token_text, end="", flush=True)
        
        # Autoregressive generation loop
        current_seq_len = prompt_len
        
        for step in range(1, max_new_tokens):
            # Check for EOS
            if next_token_id == self.eos_token_id:
                break
            
            # Get embedding for next token
            next_token_embed = self.embeddings_session.run(
                None,
                {"input_ids": np.array([[next_token_id]], dtype=np.int64)}
            )[0]  # [1, 1, hidden_size]
            
            # Expand attention mask
            attention_mask = np.ones((batch_size, current_seq_len + 1), dtype=np.int64)
            
            # Prepare inputs for incremental decoding
            inputs_dict = {
                "inputs_embeds": next_token_embed.astype(np.float32),
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
            next_token_id = self.sample_token(
                logits[0, 0, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
            
            generated_ids.append(next_token_id)
            current_seq_len += 1
            
            # Print token
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(token_text, end="", flush=True)
        
        # Final decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\n\n  Generated {len(generated_ids)} tokens")
        print("=" * 70)
        
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Multimodal ONNX Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=".",
        help="Path to model directory"
    )
    parser.add_argument(
        "--text_precision",
        type=str,
        default="fp32",
        choices=["fp32", "int4"],
        help="Text model precision"
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="Path to image(s) - use <|image_pad|> in prompt for each image"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Describe this image.",
        help="Input text/prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Qwen3VLMultimodalPipeline(
        model_dir=args.model_dir,
        text_precision=args.text_precision,
        execution_provider="CPUExecutionProvider"
    )
    
    # Prepare prompt with image placeholders
    if args.image:
        # Add <|image_pad|> for each image if not already in prompt
        prompt = args.text
        if "<|image_pad|>" not in prompt:
            # Add image token for each image
            image_tokens = "".join([f"<|image_pad|> " for _ in args.image])
            prompt = f"{image_tokens}\n{prompt}"
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
    )
    
    print(f"\n[OK] Generation complete!")
    print(f"\nFull output:\n{output}")


if __name__ == "__main__":
    main()
