"""
Extract language_model from Qwen3-VL and save as standalone Qwen3 model.

The language_model component is essentially a Qwen3 text decoder,
so we can export it using the generic Qwen3 builder!
"""
import os
import torch
from transformers import AutoModel, AutoConfig
import shutil
import json

print("="*80)
print("EXTRACTING LANGUAGE MODEL FROM QWEN3-VL")
print("="*80)

# Load full Qwen3-VL model
print("\n[1/5] Loading Qwen3-VL model...")
model_path = "./pytorch"
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32
)
print(f"  Model loaded: {type(model).__name__}")

# Extract language model
print("\n[2/5] Extracting language_model component...")
language_model = model.language_model
print(f"  Language model type: {type(language_model).__name__}")
print(f"  Layers: {len(language_model.layers)}")

# Load config and extract text config
print("\n[3/5] Creating text-only configuration...")
full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
text_config = full_config.text_config

# Create a Qwen3 config from the text_config
text_config_dict = {
    "architectures": ["Qwen3ForCausalLM"],  # Change to Qwen3 (not Qwen3-VL!)
    "model_type": "qwen3",  # Standard Qwen3, not qwen3_vl
    "torch_dtype": "bfloat16",
    
    # Copy all text config attributes
    "attention_bias": text_config.attention_bias,
    "attention_dropout": text_config.attention_dropout,
    "bos_token_id": text_config.bos_token_id,
    "eos_token_id": text_config.eos_token_id,
    "head_dim": text_config.head_dim,
    "hidden_act": text_config.hidden_act,
    "hidden_size": text_config.hidden_size,
    "initializer_range": text_config.initializer_range,
    "intermediate_size": text_config.intermediate_size,
    "max_position_embeddings": text_config.max_position_embeddings,
    "num_attention_heads": text_config.num_attention_heads,
    "num_hidden_layers": text_config.num_hidden_layers,
    "num_key_value_heads": text_config.num_key_value_heads,
    "rms_norm_eps": text_config.rms_norm_eps,
    "rope_theta": text_config.rope_theta,
    "tie_word_embeddings": text_config.tie_word_embeddings,
    "use_cache": text_config.use_cache,
    "vocab_size": text_config.vocab_size,
    
    # Important: Include rope_scaling for MRoPE
    "rope_scaling": {
        "mrope_interleaved": text_config.rope_scaling.get("mrope_interleaved", True),
        "mrope_section": text_config.rope_scaling.get("mrope_section", [24, 20, 20]),
        "rope_type": text_config.rope_scaling.get("rope_type", "default")
    }
}

# Save to output directory
output_path = "./pytorch-text-only"
os.makedirs(output_path, exist_ok=True)

print(f"  Saving config to {output_path}/config.json")
with open(os.path.join(output_path, "config.json"), "w") as f:
    json.dump(text_config_dict, f, indent=2)

# Save the language model weights
print("\n[4/5] Saving language_model weights...")
try:
    language_model.save_pretrained(output_path)
    print(f"  Weights saved successfully")
except Exception as e:
    print(f"  Warning: save_pretrained failed: {e}")
    print(f"  Trying manual state_dict save...")
    
    # Manual save
    state_dict = language_model.state_dict()
    torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
    print(f"  State dict saved to pytorch_model.bin")

# Copy tokenizer files
print("\n[5/5] Copying tokenizer files...")
tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.json"
]

for filename in tokenizer_files:
    src = os.path.join(model_path, filename)
    dst = os.path.join(output_path, filename)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  Copied {filename}")

print("\n" + "="*80)
print("[SUCCESS] LANGUAGE MODEL EXTRACTED")
print("="*80)
print(f"\nExtracted model location: {output_path}/")
print(f"Architecture: Qwen3ForCausalLM (text-only)")
print(f"Layers: {text_config_dict['num_hidden_layers']}")
print(f"Hidden size: {text_config_dict['hidden_size']}")
print(f"\nNext step:")
print(f"  python builder_text.py --input {output_path} --output ./cpu-text")
