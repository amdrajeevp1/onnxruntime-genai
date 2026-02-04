"""
Test Qwen3-VL Exported Text Model
==================================

Tests the INT4 quantized text model exported by builder_qwen3vl.py
"""

import onnxruntime_genai as og

def test_text_generation():
    print("\n" + "="*80)
    print("TESTING QWEN3-VL EXPORTED TEXT MODEL")
    print("="*80 + "\n")
    
    model_path = "./qwen3vl-onnx-final/qwen3vl-text"
    
    print(f"Loading model from: {model_path}")
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    
    print(f"[OK] Model and tokenizer loaded\n")
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "Write a haiku about AI.",
        "What is 2+2?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {prompt}")
        print(f"{'='*80}\n")
        
        # Encode
        input_tokens = tokenizer.encode(prompt)
        print(f"Input tokens: {len(input_tokens)}")
        
        # Generate
        params = og.GeneratorParams(model)
        params.input_ids = input_tokens
        params.set_search_options(max_length=128)
        
        generator = og.Generator(model, params)
        
        print("Generating", end="", flush=True)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            print(".", end="", flush=True)
        print("\n")
        
        # Decode
        output_tokens = generator.get_sequence(0)
        output_text = tokenizer.decode(output_tokens)
        
        print(f"Output ({len(output_tokens)} tokens):")
        print(f"{output_text}\n")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_text_generation()
