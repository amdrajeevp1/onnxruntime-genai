# Qwen3-VL-4B (OGA) Quick Start

Minimal steps to export and run Qwen3-VL with dynamic image size support.

## 1) Export ONNX package

From `examples/python/qwen3-vl-4b`:

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "builder.py" `
  --input "./pytorch" `
  --reference "./pytorch_reference" `
  --output "./oga-dynamic-vision-fp32-llm-fp32" `
  --precision fp32
```

## 2) Sanity test: chat only

From repo root:

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "examples/python/qwen3-vl-4b/qwen3vl-oga.py" `
  -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-fp32" `
  -e follow_config `
  --non-interactive `
  -pr "Say hello in one short sentence."
```

## 3) Sanity test: image + chat

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "examples/python/qwen3-vl-4b/qwen3vl-oga.py" `
  -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-fp32" `
  -e follow_config `
  --non-interactive `
  --image_paths "examples/python/qwen3-vl-4b/test_images/img_50.jpg" `
  -pr "Describe this image in one sentence."
```
