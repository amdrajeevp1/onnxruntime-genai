# Qwen3-VL-4B (OGA) Quick Start

Minimal steps to export and run Qwen3-VL with dynamic image size support.

## 1) Download Qwen3-VL-4B into local `pytorch` folder

From `examples/python/qwen3-vl-4b`:

```powershell
huggingface-cli download Qwen/Qwen3-VL-4B --local-dir "./pytorch"
```

## 2) Export ONNX package

From `examples/python/qwen3-vl-4b`:

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "builder.py" `
  --input "./pytorch" `
  --reference "./pytorch_reference" `
  --output "./oga-dynamic-vision-fp32-llm-fp32" `
  --precision fp32
```

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "builder.py" `
  --input "./pytorch" `
  --reference "./pytorch_reference" `
  --output "./oga-dynamic-vision-fp32-llm-int4" `
  --precision int4
```


## 3) Sanity test: chat only

From repo root:

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "examples/python/qwen3-vl-4b/qwen3vl-oga.py" `
  -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-fp32" `
  -e follow_config `
  --non-interactive `
  -pr "Say hello in one short sentence."
```
```
$ & "C:/Users/rajeevp/AppData/Local/miniconda3/envs/onnxruntime-genai/python.exe" "examples/python/qwen3-vl-4b/qwen3vl-oga.py" -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-int4" -e follow_config --non-interactive -pr "Say hello in one short sentence."
Loading model...
Model loaded
No image provided
Processing inputs...
Processor complete. Output keys: ['input_ids', 'num_image_tokens']
Generating response...
Hello!
Total Time : 0.10
```

## 4) Sanity test: image + chat

```powershell
& "C:\Users\rajeevp\AppData\Local\miniconda3\envs\onnxruntime-genai\python.exe" `
  "examples/python/qwen3-vl-4b/qwen3vl-oga.py" `
  -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-fp32" `
  -e follow_config `
  --non-interactive `
  --image_paths "examples/python/qwen3-vl-4b/test_images/img_50.jpg" `
  -pr "Describe this image in one sentence."
```

```
$ & "C:/Users/rajeevp/AppData/Local/miniconda3/envs/onnxruntime-genai/python.exe" "examples/python/qwen3-vl-4b/qwen3vl-oga.py" -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-int4" -e follow_config --non-interactive --image_paths "examples/python/qwen3-vl-4b/test_images/img_10.jpg" -pr "Describe this image in one sentence."; & "C:/Users/rajeevp/AppData/Local/miniconda3/envs/onnxruntime-genai/python.exe" "examples/python/qwen3-vl-4b/qwen3vl-oga.py" -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-int4" -e follow_config --non-interactive --image_paths "examples/python/qwen3-vl-4b/test_images/img_50.jpg" -pr "Describe this image in one sentence."; & "C:/Users/rajeevp/AppData/Local/miniconda3/envs/onnxruntime-genai/python.exe" "examples/python/qwen3-vl-4b/qwen3vl-oga.py" -m "examples/python/qwen3-vl-4b/oga-dynamic-vision-fp32-llm-int4" -e follow_config --non-interactive --image_paths "examples/python/qwen3-vl-4b/test_images/img_100.jpg" -pr "Describe this image in one sentence."

Loading model...
Model loaded
Using image: examples/python/qwen3-vl-4b/test_images/img_10.jpg
Processing inputs...
Processor complete. Output keys: ['pixel_values', 'input_ids', 'image_grid_thw', 'num_image_tokens']
Generating response...
A serene coastal landscape with a dense green forest in the foreground, a vast turquoise sea stretching to distant, hazy mountains under a vast, clear blue sky.
Total Time : 1.96

Loading model...
Model loaded
Using image: examples/python/qwen3-vl-4b/test_images/img_50.jpg
Processing inputs...
Processor complete. Output keys: ['pixel_values', 'input_ids', 'image_grid_thw', 'num_image_tokens']
Generating response...
A solitary cormorant with wings spread stands on a rope railing of a weathered pier, silhouetted against the calm, sun-dappled water of a vast, serene harbor.
Total Time : 2.40

Loading model...
Model loaded
Using image: examples/python/qwen3-vl-4b/test_images/img_100.jpg
Processing inputs...
Processor complete. Output keys: ['pixel_values', 'input_ids', 'image_grid_thw', 'num_image_tokens']
Generating response...
This is a sepia-toned, high-angle photograph capturing a bustling beach scene with numerous people enjoying the water and sand under a hazy, overcast sky, with distant city buildings and mountains visible on the horizon.
Total Time : 2.60
```