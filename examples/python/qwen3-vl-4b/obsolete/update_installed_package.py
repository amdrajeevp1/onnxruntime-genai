"""
Update installed onnxruntime-genai package with Qwen3-VL support
"""

import shutil
import os

# Source files
src_root = r"c:\Users\rajeevp\Documents\onnxruntime-genai-1\src\python\py\models"
src_files = {
    "builder.py": os.path.join(src_root, "builder.py"),
    "builders/__init__.py": os.path.join(src_root, "builders\__init__.py"),
    "builders/qwen.py": os.path.join(src_root, "builders\qwen.py"),
}

# Target directory (installed package)
target_root = r"C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\Lib\site-packages\onnxruntime_genai\models"

print("Updating installed onnxruntime-genai package with Qwen3-VL support...")
print(f"Source: {src_root}")
print(f"Target: {target_root}\n")

for rel_path, src_file in src_files.items():
    target_file = os.path.join(target_root, rel_path)
    
    if not os.path.exists(src_file):
        print(f"[SKIP] Source file not found: {src_file}")
        continue
    
    try:
        # Create target directory if needed
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Copy file
        shutil.copy2(src_file, target_file)
        print(f"[OK] Copied: {rel_path}")
    except Exception as e:
        print(f"[ERROR] Failed to copy {rel_path}: {e}")

print("\nDone! The installed package now has Qwen3-VL support.")
print("You can now run the builder.")
