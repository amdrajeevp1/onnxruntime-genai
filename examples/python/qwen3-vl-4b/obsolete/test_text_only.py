"""
Simple test of Qwen3-VL text decoder (text-only mode)
"""

import onnxruntime_genai as og
import sys
import codecs

# Force UTF-8 output for Windows console
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

print("="*80)
print("QWEN3-VL TEXT DECODER TEST")
print("="*80)
print()

# Load model
model_path = "./cpu-text"
print(f"Loading model from: {model_path}")

model = og.Model(model_path)
tokenizer = og.Tokenizer(model)

print(f"Model loaded successfully!")
print()

# Test prompt
prompt = "What is the capital of France? Answer concisely."

print(f"Prompt: {prompt}")
print()
print("Response: ", end="", flush=True)

# Tokenize
input_tokens = tokenizer.encode(prompt)

# Set generation parameters
params = og.GeneratorParams(model)
params.set_search_options(max_length=100, temperature=0.7, top_p=0.9)

# Generate
generator = og.Generator(model, params)
generator.append_tokens(input_tokens)

# Stream output
tokenizer_stream = tokenizer.create_stream()

while not generator.is_done():
    generator.generate_next_token()
    
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end="", flush=True)

print()
print()
print("="*80)
print("TEXT DECODER TEST COMPLETE!")
print("="*80)
