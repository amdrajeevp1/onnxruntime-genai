"""
Inspect Qwen3-VL-4B model structure to understand vision encoder
"""
import torch
from transformers import AutoConfig, AutoProcessor, AutoModel

print("Loading Qwen3-VL-4B model...")
model_path = "./pytorch"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print("\n" + "="*80)
print("MODEL CONFIG")
print("="*80)
print(f"Architecture: {config.architectures}")
print(f"Model Type: {config.model_type}")

print("\n" + "="*80)
print("VISION CONFIG")
print("="*80)
if hasattr(config, 'vision_config'):
    vision_config = config.vision_config
    for key in dir(vision_config):
        if not key.startswith('_'):
            try:
                value = getattr(vision_config, key)
                if not callable(value):
                    print(f"{key}: {value}")
            except:
                pass

print("\n" + "="*80)
print("TEXT CONFIG")
print("="*80)
if hasattr(config, 'text_config'):
    text_config = config.text_config
    for key in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'rope_scaling']:
        if hasattr(text_config, key):
            print(f"{key}: {getattr(text_config, key)}")

print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)
model = AutoModel.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float32
).cpu()

print(f"✓ Model loaded: {type(model).__name__}")

print("\n" + "="*80)
print("MODEL STRUCTURE")
print("="*80)

# Print top-level modules
print("\nTop-level modules:")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

# Check for vision encoder
if hasattr(model, 'visual'):
    print("\n✓ Found 'visual' module")
    print(f"  Type: {type(model.visual).__name__}")
    for name, module in model.visual.named_children():
        print(f"    - visual.{name}: {type(module).__name__}")

if hasattr(model, 'model'):
    print("\n✓ Found 'model' module")
    print(f"  Type: {type(model.model).__name__}")
    for name, module in model.model.named_children():
        print(f"    - model.{name}: {type(module).__name__}")
        
        # Look for vision-related submodules
        if 'vision' in name.lower() or 'image' in name.lower() or 'visual' in name.lower():
            print(f"      ^^^ VISION-RELATED MODULE ^^^")
            for subname, submodule in module.named_children():
                print(f"        - model.{name}.{subname}: {type(submodule).__name__}")

print("\n" + "="*80)
print("PROCESSOR")
print("="*80)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(f"✓ Processor loaded: {type(processor).__name__}")
if hasattr(processor, 'image_processor'):
    print(f"  Image Processor: {type(processor.image_processor).__name__}")

print("\n" + "="*80)
print("MODEL READY FOR EXPORT")
print("="*80)
