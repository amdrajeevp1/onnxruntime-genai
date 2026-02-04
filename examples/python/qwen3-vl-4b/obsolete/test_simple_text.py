"""
Simple Qwen3-VL Text Generation Test
=====================================

Tests the exported ONNX text model with a simple text-only prompt.
"""

import onnxruntime_genai as og

def test_simple_generation():
    print("\n" + "="*80)
    print("SIMPLE QWEN3-VL TEXT GENERATION TEST")
    print("="*80 + "\n")
    
    model_path = "./qwen3vl-onnx-final/qwen3vl-text"
    
    print(f"Loading model from: {model_path}")
    model = og.Model(model_path)
    tokenizer = model.create_tokenizer()
    
    print(f"[OK] Model and tokenizer loaded\n")
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me a joke."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {prompt}")
        print(f"{'='*80}\n")
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        
        # Create params
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=100)
        params.input_ids = tokens
        
        # Generate
        print("Generating", end="", flush=True)
        output_tokens = model.generate(params)
        print("done!\n")
        
        # Decode
        output_text = tokenizer.decode(output_tokens[0])
        
        print(f"Output:")
        print(f"{output_text}\n")
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_simple_generation()
