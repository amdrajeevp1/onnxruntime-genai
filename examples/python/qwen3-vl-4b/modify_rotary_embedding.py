"""
Modify Qwen3-VL rotary embedding to remove dynamic decisions for ONNX export
"""
import re
from pathlib import Path

def modify_rotary_embedding(file_path):
    """
    Modify the Qwen3VLTextRotaryEmbedding.forward() method to remove dynamic decisions
    """
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find the forward method
    # We'll look for the @dynamic_rope_update decorator and the following forward method
    pattern = r'(@torch\.no_grad\(\)\s+@dynamic_rope_update[^\n]*\s+def forward\(self, x, position_ids\):.*?return cos\.to\(dtype=x\.dtype\), sin\.to\(dtype=x\.dtype\))'
    
    # Find all matches
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        print("  Warning: Could not find @dynamic_rope_update decorated forward method")
        print("  Trying alternative pattern...")
        
        # Try finding just by method signature
        pattern2 = r'(@torch\.no_grad\(\).*?def forward\(self, x, position_ids\):.*?# In contrast to other models.*?return cos\.to\(dtype=x\.dtype\), sin\.to\(dtype=x\.dtype\))'
        matches = list(re.finditer(pattern2, content, re.DOTALL))
    
    if not matches:
        print("  [ERROR] Could not find the forward method to modify")
        return False
    
    print(f"  Found {len(matches)} forward method(s) to modify")
    
    # Prepare the modified version
    modified_content = content
    
    for match in matches:
        original_method = match.group(0)
        
        # Create modified version
        modified_method = original_method
        
        # 1. Remove @dynamic_rope_update decorator
        modified_method = re.sub(r'@dynamic_rope_update[^\n]*\n\s*', '', modified_method)
        
        # 2. Replace the conditional expansion with assertion
        expansion_pattern = r'if position_ids\.ndim == 2:\s+position_ids = position_ids\[None, \.\.\.\]\.expand\(3, position_ids\.shape\[0\], -1\)'
        replacement = '''# ONNX Export Modification: Assume position_ids is always 3D
        assert position_ids.ndim == 3, "position_ids must be 3D [3, batch, seq_len] for ONNX export"'''
        
        if re.search(expansion_pattern, modified_method):
            modified_method = re.sub(expansion_pattern, replacement, modified_method)
            print("  [OK] Removed conditional position_ids expansion")
        else:
            # Try to add assertion before the inv_freq_expanded line
            inv_freq_line = r'inv_freq_expanded = self\.inv_freq'
            if re.search(inv_freq_line, modified_method):
                modified_method = re.sub(
                    inv_freq_line,
                    f'{replacement}\n        inv_freq_expanded = self.inv_freq',
                    modified_method
                )
                print("  [OK] Added assertion for 3D position_ids")
        
        # 3. Add comment explaining the modification
        comment = '''    # ONNX Export Modification (onnxruntime-genai):
    # - Removed @dynamic_rope_update decorator
    # - Assume position_ids is always 3D [3, batch_size, seq_len]
    # - Removed dynamic expansion logic for ONNX static graph
    '''
        modified_method = modified_method.replace('def forward(self, x, position_ids):', 
                                                   f'def forward(self, x, position_ids):\n{comment}')
        
        # Replace in content
        modified_content = modified_content.replace(original_method, modified_method)
    
    # Write back
    backup_path = file_path.with_suffix('.py.backup')
    print(f"  Creating backup at {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Writing modified file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("  [SUCCESS] Modification complete!")
    return True

def main():
    script_dir = Path(__file__).parent
    
    # Files to modify
    files_to_modify = [
        script_dir / "pytorch_modified" / "modular_qwen3_vl.py",
        script_dir / "pytorch_modified" / "modeling_qwen3_vl.py",
    ]
    
    print("=" * 70)
    print("Modifying Qwen3-VL Rotary Embedding for ONNX Export")
    print("=" * 70)
    print()
    
    success_count = 0
    for file_path in files_to_modify:
        if not file_path.exists():
            print(f"Skipping {file_path} (not found)")
            continue
        
        print(f"\nProcessing: {file_path.name}")
        if modify_rotary_embedding(file_path):
            success_count += 1
    
    print()
    print("=" * 70)
    if success_count > 0:
        print(f"[SUCCESS] Modified {success_count} file(s) successfully!")
    else:
        print("[ERROR] No files were modified")
    print("=" * 70)
    print()
    
    if success_count > 0:
        print("Next steps:")
        print("1. Copy modified files to pytorch/ directory:")
        print("   cp pytorch_modified/*.py pytorch/")
        print("2. Run the builder:")
        print("   python builder_qwen3vl.py")

if __name__ == "__main__":
    main()
