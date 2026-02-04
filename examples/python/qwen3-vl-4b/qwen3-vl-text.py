"""
Qwen3-VL Text-Only ONNX Inference
Uses embeddings.onnx + model.onnx for text generation

No vision models - pure text generation
"""

import sys
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional
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
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error: transformers not found: {e}")
    print("Install with: pip install transformers")
    raise


class Qwen3VLTextOnlyPipeline:
    """Text-only ONNX inference pipeline for Qwen3-VL"""
    
    def __init__(self, model_dir: str = ".", text_precision: str = "fp32"):
        self.model_dir = Path(model_dir)
        self.text_precision = text_precision
        
        print(f"Loading Qwen3-VL Text-Only ONNX Pipeline (text={text_precision.upper()})...")
        
        # Load ONNX models - always use fp32 for embeddings, configurable for text
        embedding_path = self.model_dir / "cpu-fp32" / "qwen3vl-embedding.onnx"
        text_path = self.model_dir / f"cpu-{text_precision}" / "model.onnx"
        
        print(f"  Embedding model: {embedding_path}")
        print(f"  Text model: {text_path}")
        
        providers = ["CPUExecutionProvider"]
        self.embeddings_session = ort.InferenceSession(str(embedding_path), providers=providers)
        self.text_session = ort.InferenceSession(str(text_path), providers=providers)
        
        print("  [OK] ONNX models loaded")
        
        # Load tokenizer from pytorch directory
        pytorch_dir = self.model_dir / "pytorch"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(pytorch_dir),
            trust_remote_code=True
        )
        
        print("  [OK] Tokenizer loaded")
        
        # Model config (from genai_config.json)
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
        
        print(f"  Model: {self.num_layers} layers, hidden_size={self.hidden_size}")
        print("Pipeline ready!\n")
    
    def create_position_ids_3d(self, seq_len: int, past_seq_len: int = 0) -> np.ndarray:
        """
        Create 3D position IDs for MRoPE
        Returns: [3, 1, seq_len]
        """
        pos_ids = np.arange(past_seq_len, past_seq_len + seq_len, dtype=np.int64)
        position_ids_3d = np.stack([pos_ids, pos_ids, pos_ids], axis=0)  # [3, seq_len]
        return position_ids_3d[:, None, :]  # [3, 1, seq_len]
    
    def sample_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.8,
        do_sample: bool = True
    ) -> int:
        """Sample next token from logits"""
        
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
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.8,
        do_sample: bool = True,
        use_chat_template: bool = True,
    ) -> str:
        """Generate text from prompt"""
        
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print("=" * 70)
        
        # Apply chat template if requested (default for Qwen3-VL)
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"  Using chat template: Yes")
        else:
            formatted_prompt = prompt
            print(f"  Using chat template: No")
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids = np.array(input_ids, dtype=np.int64)
        
        print(f"  Input tokens: {len(input_ids)}")
        
        # Get embeddings
        text_embeds = self.embeddings_session.run(
            None,
            {"input_ids": input_ids[None, :]}
        )[0]  # [1, seq_len, hidden_size]
        
        print(f"  Embeddings: {text_embeds.shape}")
        
        batch_size = 1
        prompt_len = text_embeds.shape[1]
        
        # Create inputs
        # NOTE: Qwen3 (unlike Qwen2.5-VL) does NOT use position_ids input
        # It computes positions internally using standard RoPE
        attention_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
        
        # Empty KV caches
        kv_cache_shape = (batch_size, self.num_kv_heads, 0, self.head_dim)
        inputs_dict = {
            "inputs_embeds": text_embeds.astype(np.float32),
            "attention_mask": attention_mask,
        }
        
        for layer_idx in range(self.num_layers):
            inputs_dict[f"past_key_values.{layer_idx}.key"] = np.zeros(kv_cache_shape, dtype=np.float32)
            inputs_dict[f"past_key_values.{layer_idx}.value"] = np.zeros(kv_cache_shape, dtype=np.float32)
        
        print(f"\nGenerating (max {max_new_tokens} tokens)...\n  ", end="", flush=True)
        
        # Initial forward pass
        outputs = self.text_session.run(None, inputs_dict)
        logits = outputs[0]  # [batch_size, seq_len, vocab_size]
        
        # Extract KV caches
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
        
        # Generation loop
        current_seq_len = prompt_len
        
        for step in range(1, max_new_tokens):
            if next_token_id == self.eos_token_id:
                break
            
            # Get embedding for next token
            next_token_embed = self.embeddings_session.run(
                None,
                {"input_ids": np.array([[next_token_id]], dtype=np.int64)}
            )[0]  # [1, 1, hidden_size]
            
            # Create inputs for incremental decoding (no position_ids for Qwen3)
            attention_mask = np.ones((batch_size, current_seq_len + 1), dtype=np.int64)
            
            inputs_dict = {
                "inputs_embeds": next_token_embed.astype(np.float32),
                "attention_mask": attention_mask,
            }
            
            # Add KV caches
            for layer_idx in range(self.num_layers):
                inputs_dict[f"past_key_values.{layer_idx}.key"] = kv_caches[f"past_key_values.{layer_idx}.key"]
                inputs_dict[f"past_key_values.{layer_idx}.value"] = kv_caches[f"past_key_values.{layer_idx}.value"]
            
            # Forward pass
            outputs = self.text_session.run(None, inputs_dict)
            logits = outputs[0]
            
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
    parser = argparse.ArgumentParser(description="Qwen3-VL Text-Only ONNX Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=".",
        help="Path to model directory (containing cpu-fp32/ or cpu-int4/)"
    )
    parser.add_argument(
        "--text_precision",
        type=str,
        default="fp32",
        choices=["fp32", "int4"],
        help="Text model precision (default: fp32)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt (if not provided, will prompt interactively)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
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
        default=20,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (keep prompting)"
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template (not recommended for Qwen3-VL)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Qwen3VLTextOnlyPipeline(
        model_dir=args.model_dir,
        text_precision=args.text_precision
    )
    
    # Interactive or single prompt mode
    if args.interactive:
        print("\n" + "=" * 70)
        print("Interactive Mode (type 'quit' to exit)")
        print("=" * 70)
        
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                
                if not prompt.strip():
                    continue
                
                pipeline.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    use_chat_template=not args.no_chat_template,
                )
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
    
    else:
        # Single prompt
        prompt = args.prompt if args.prompt else "What is the capital of France?"
        
        pipeline.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.no_sample,
            use_chat_template=not args.no_chat_template,
        )
        
        print("\n[OK] Generation complete!")


if __name__ == "__main__":
    main()
