"""
Master setup script for Qwen3-VL ONNX export

This script automates the entire setup process:
1. Download HuggingFace model files
2. Modify rotary embedding for ONNX export
3. Copy files to pytorch directory
4. Export ONNX models
5. Create processor configurations
"""
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print results"""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return False
    
    print(f"[SUCCESS] {description} completed successfully!")
    return True

def main():
    print("=" * 70)
    print("Qwen3-VL ONNX Export - Master Setup Script")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Download HuggingFace model files")
    print("  2. Modify rotary embedding for ONNX export")
    print("  3. Copy modified files to pytorch directory")
    print("  4. Export ONNX models (vision + embeddings + text)")
    print()
    # Auto-continue in non-interactive mode
    # input("Press Enter to continue...")
    print("Starting setup...")
    print()
    
    script_dir = Path(__file__).parent
    
    # Step 1: Download HF files
    if not run_command(
        f"{sys.executable} {script_dir / 'copy_hf_files.py'}",
        "Step 1: Downloading HuggingFace model files"
    ):
        print("\n[ERROR] Setup failed at step 1")
        return
    
    # Step 2: Modify rotary embedding
    if not run_command(
        f"{sys.executable} {script_dir / 'modify_rotary_embedding.py'}",
        "Step 2: Modifying rotary embedding for ONNX"
    ):
        print("\n[ERROR] Setup failed at step 2")
        return
    
    # Step 3: Copy modified files
    print("\n" + "=" * 70)
    print("Step 3: Copying modified files to pytorch directory")
    print("=" * 70)
    
    pytorch_dir = script_dir / "pytorch"
    modified_dir = script_dir / "pytorch_modified"
    
    files_to_copy = [
        "modeling_qwen3_vl.py",
        "modular_qwen3_vl.py",
        "processing_qwen3_vl.py",
        "configuration_qwen3_vl.py",
        "video_processing_qwen3_vl.py",
    ]
    
    for filename in files_to_copy:
        src = modified_dir / filename
        dst = pytorch_dir / filename
        
        if not src.exists():
            print(f"  [SKIP] {filename} (not found)")
            continue
        
        try:
            shutil.copy2(src, dst)
            print(f"  [OK] Copied {filename}")
        except Exception as e:
            print(f"  [ERROR] copying {filename}: {e}")
    
    print("[SUCCESS] File copying completed!")
    
    # Step 4: Export ONNX models
    print("\n" + "=" * 70)
    print("Step 4: Export ONNX models")
    print("=" * 70)
    print()
    print("Export options:")
    print("  1. FP32 (CPU)")
    print("  2. FP16 (CUDA)")
    print("  3. FP16 (DirectML)")
    
    # Auto-select FP32/CPU for non-interactive mode
    choice = "1"
    print(f"\nAuto-selecting option {choice}: FP32 (CPU)")
    
    if choice == "1":
        precision = "fp32"
        ep = "cpu"
        output_dir = "cpu"
    elif choice == "2":
        precision = "fp16"
        ep = "cuda"
        output_dir = "cuda"
    elif choice == "3":
        precision = "fp16"
        ep = "dml"
        output_dir = "dml"
    else:
        print(f"Invalid choice: {choice}")
        return
    
    if not run_command(
        f"{sys.executable} {script_dir / 'builder_qwen3vl.py'} "
        f"--input {pytorch_dir} "
        f"--output {script_dir / output_dir} "
        f"--precision {precision} "
        f"--execution_provider {ep}",
        f"Step 4: Exporting ONNX models ({precision} / {ep})"
    ):
        print("\n[ERROR] Setup failed at step 4")
        return
    
    # Done!
    print("\n" + "=" * 70)
    print("[SUCCESS] Setup Complete!")
    print("=" * 70)
    print()
    print("Exported models are in:")
    print(f"  {script_dir / output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Test inference:")
    print(f"     python test_qwen3vl_inference.py --model_path {output_dir} --image_path <image.jpg>")
    print()
    print(f"  2. Optimize models:")
    print(f"     Use ONNX Runtime optimization tools")
    print()
    print(f"  3. Deploy with ONNX Runtime GenAI")

if __name__ == "__main__":
    main()
