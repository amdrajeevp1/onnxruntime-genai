"""
Copy HuggingFace Qwen3-VL model files locally for modification
"""
import os
import requests
from pathlib import Path

# Base URL for HuggingFace transformers repository
HF_BASE_URL = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_vl"

# Files to download
FILES_TO_DOWNLOAD = [
    "modeling_qwen3_vl.py",
    "modular_qwen3_vl.py",  # This is the actual source
    "processing_qwen3_vl.py",
    "configuration_qwen3_vl.py",
    "video_processing_qwen3_vl.py",
]

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"  [OK] Saved to {output_path}")

def main():
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "pytorch_modified"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Copying HuggingFace Qwen3-VL Files Locally")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Download each file
    for filename in FILES_TO_DOWNLOAD:
        url = f"{HF_BASE_URL}/{filename}"
        output_path = output_dir / filename
        
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  [ERROR] downloading {filename}: {e}")
            continue
    
    print()
    print("=" * 70)
    print("[SUCCESS] Download complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run: python modify_rotary_embedding.py")
    print("2. Run: python builder_qwen3vl.py")

if __name__ == "__main__":
    main()
