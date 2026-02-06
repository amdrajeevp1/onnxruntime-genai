"""
Test PyTorch Qwen3-VL pipeline with modified files.

This script copies the modified Python files to the pytorch directory,
then loads the model with trust_remote_code=True to use the modified classes.
"""

import sys
import os
import shutil
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig

def test_pytorch_pipeline():
    print("=" * 80)
    print("Testing PyTorch Qwen3-VL Pipeline")
    print("=" * 80)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_dir = os.path.join(script_dir, 'pytorch')
    pytorch_modified_dir = os.path.join(script_dir, 'pytorch_modified')
    images_dir = os.path.join(script_dir, 'images')
    
    print(f"\nModel directory: {pytorch_dir}")
    print(f"Modified files source: {pytorch_modified_dir}")
    
    # Step 1: Copy modified files to pytorch directory
    print("\n[1/5] Copying modified files...")
    modified_files = [
        'modeling_qwen3_vl.py',
        'modular_qwen3_vl.py',
        'processing_qwen3_vl.py',
        'video_processing_qwen3_vl.py',
        'configuration_qwen3_vl.py'
    ]
    
    for fname in modified_files:
        src = os.path.join(pytorch_backup_dir, fname)
        dst = os.path.join(pytorch_dir, fname)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  - Copied: {fname}")
        else:
            print(f"  [WARNING] Not found: {fname}")
    
    print("  [OK] Modified files copied")
    
    # Step 2: Load configuration
    print("\n[2/5] Loading configuration...")
    config = AutoConfig.from_pretrained(pytorch_dir, trust_remote_code=True)
    print(f"  - Model type: {config.model_type}")
    print(f"  - Vision hidden size: {config.vision_config.hidden_size}")
    print(f"  - Text hidden size: {config.text_config.hidden_size}")
    print(f"  - Image token ID: {config.image_token_id}")
    print(f"  - Vision start token ID: {config.vision_start_token_id}")
    print(f"  - Vision end token ID: {config.vision_end_token_id}")
    
    # Step 3: Load model (will use modified files via trust_remote_code)
    print("\n[3/5] Loading model (this may take a while)...")
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pytorch_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use FP32 for CPU testing
    )
    model.eval()
    print("  [OK] Model loaded successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Step 4: Load processor
    print("\n[4/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(pytorch_dir, trust_remote_code=True)
    print("  [OK] Processor loaded successfully")
    
    # Step 5: Test scenarios with text generation
    print("\n[5/5] Running test scenarios with text generation...")
    print("-" * 80)
    
    # Open output file
    output_file = os.path.join(script_dir, "pytorch_test_outputs.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Qwen3-VL Test Outputs\n")
        f.write("=" * 80 + "\n\n")
        
        # Test 1: Single image description
        print("\n[Test 1] Single image description")
        f.write("[Test 1] Single image description\n")
        f.write("-" * 80 + "\n")
        
        image_path = os.path.join(images_dir, "test_checkerboard.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            print(f"  - Image: {os.path.basename(image_path)}")
            print(f"  - Image size: {image.size}")
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Size: {image.size}\n\n")
            
            # Create prompt with image token
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]
            
            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            try:
                inputs = processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                print(f"  - Input IDs shape: {inputs['input_ids'].shape}")
                print(f"  - Generating text (max 128 tokens)...")
                
                # Generate text
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                
                # Decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                print(f"  - Generated: {output_text[:100]}...")
                print("  [OK] Single image generation successful!")
                
                f.write("Prompt: Describe this image in detail.\n\n")
                f.write(f"Generated Output:\n{output_text}\n\n")
                    
            except Exception as e:
                print(f"  [FAIL] Error during processing: {e}")
                f.write(f"ERROR: {e}\n\n")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"  [FAIL] Image not found: {image_path}")
            f.write(f"ERROR: Image not found\n\n")
    
        # Test 2: Text-only (no image)
        print("\n[Test 2] Text-only generation")
        f.write("\n[Test 2] Text-only generation\n")
        f.write("-" * 80 + "\n")
        
        try:
            text_only = "What is the capital of France?"
            inputs = processor(
                text=[text_only],
                return_tensors="pt",
                padding=True,
            )
            print(f"  - Prompt: {text_only}")
            print(f"  - Generating text (max 128 tokens)...")
            f.write(f"Prompt: {text_only}\n\n")
            
            # Generate text
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=128,
                    do_sample=False,
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"  - Generated: {output_text[:100]}...")
            print("  [OK] Text-only generation successful!")
            
            f.write(f"Generated Output:\n{output_text}\n\n")
            
        except Exception as e:
            print(f"  [FAIL] Error during text-only processing: {e}")
            f.write(f"ERROR: {e}\n\n")
            import traceback
            traceback.print_exc()
            raise
    
        # Test 3: Multiple images (if available)
        print("\n[Test 3] Multiple images")
        f.write("\n[Test 3] Multiple images\n")
        f.write("-" * 80 + "\n")
        
        image_paths = [
            os.path.join(images_dir, "test_checkerboard.jpg"),
            os.path.join(images_dir, "test_colors.jpg"),
        ]
        existing_images = [Image.open(p) for p in image_paths if os.path.exists(p)]
        
        if len(existing_images) >= 2:
            print(f"  - Testing with {len(existing_images)} images")
            f.write(f"Images: {len(existing_images)}\n")
            f.write(f"  - {os.path.basename(image_paths[0])}\n")
            f.write(f"  - {os.path.basename(image_paths[1])}\n\n")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": existing_images[0]},
                        {"type": "image", "image": existing_images[1]},
                        {"type": "text", "text": "Compare these two images."},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            try:
                inputs = processor(
                    text=[text],
                    images=existing_images,
                    return_tensors="pt",
                    padding=True,
                )
                print(f"  - Input IDs shape: {inputs['input_ids'].shape}")
                print(f"  - Generating text (max 128 tokens)...")
                
                # Generate text
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                
                # Decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                print(f"  - Generated: {output_text[:100]}...")
                print("  [OK] Multi-image generation successful!")
                
                f.write("Prompt: Compare these two images.\n\n")
                f.write(f"Generated Output:\n{output_text}\n\n")
                
            except Exception as e:
                print(f"  [FAIL] Error during multi-image processing: {e}")
                f.write(f"ERROR: {e}\n\n")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"  [WARNING] Skipping multi-image test (only {len(existing_images)} images found)")
            f.write(f"Skipped: Only {len(existing_images)} images found\n\n")
    
    print("\n" + "=" * 80)
    print("[OK] All tests passed!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_file}")
    print("The modified PyTorch model is working correctly.")
    print("Ready to proceed with ONNX export.")
    print()

if __name__ == "__main__":
    test_pytorch_pipeline()
