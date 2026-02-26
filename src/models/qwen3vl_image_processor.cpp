// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "qwen3vl_image_processor.h"
#include <numeric>
#include <regex>

namespace Generators {

namespace {

// Convert CHW image [3, H, W] to patch format [num_patches, patch_features]
// For Qwen3-VL: patch_size=16, temporal_patch_size=2
// patch_features = channels * temporal_patch_size * patch_size * patch_size = 3 * 2 * 16 * 16 = 1536
std::unique_ptr<OrtValue> ConvertToPatch(const float* image_data,
                                          int64_t channels, int64_t height, int64_t width,
                                          int64_t patch_size, int64_t temporal_patch_size,
                                          Ort::Allocator& allocator) {
  int64_t patches_h = height / patch_size;
  int64_t patches_w = width / patch_size;
  int64_t num_patches = patches_h * patches_w;
  int64_t patch_features = channels * temporal_patch_size * patch_size * patch_size;
  
  auto patch_tensor = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{num_patches, patch_features});
  float* patch_data = patch_tensor->GetTensorMutableData<float>();
  
  // Extract patches
  for (int64_t ph = 0; ph < patches_h; ++ph) {
    for (int64_t pw = 0; pw < patches_w; ++pw) {
      int64_t patch_idx = ph * patches_w + pw;
      float* patch_ptr = patch_data + patch_idx * patch_features;
      
      // For each channel
      for (int64_t c = 0; c < channels; ++c) {
        // Duplicate for temporal dimension (temporal_patch_size times)
        for (int64_t t = 0; t < temporal_patch_size; ++t) {
          // Copy patch_size x patch_size region
          for (int64_t h = 0; h < patch_size; ++h) {
            for (int64_t w = 0; w < patch_size; ++w) {
              int64_t img_h = ph * patch_size + h;
              int64_t img_w = pw * patch_size + w;
              int64_t img_idx = c * (height * width) + img_h * width + img_w;
              *patch_ptr++ = image_data[img_idx];
            }
          }
        }
      }
    }
  }
  
  return patch_tensor;
}

// Convert HWC image [H, W, 3] to patch format [num_patches, patch_features]
std::unique_ptr<OrtValue> ConvertToPatchHWC(const float* image_data,
                                            int64_t channels, int64_t height, int64_t width,
                                            int64_t patch_size, int64_t temporal_patch_size,
                                            Ort::Allocator& allocator) {
  int64_t patches_h = height / patch_size;
  int64_t patches_w = width / patch_size;
  int64_t num_patches = patches_h * patches_w;
  int64_t patch_features = channels * temporal_patch_size * patch_size * patch_size;

  auto patch_tensor = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{num_patches, patch_features});
  float* patch_data = patch_tensor->GetTensorMutableData<float>();

  for (int64_t ph = 0; ph < patches_h; ++ph) {
    for (int64_t pw = 0; pw < patches_w; ++pw) {
      int64_t patch_idx = ph * patches_w + pw;
      float* patch_ptr = patch_data + patch_idx * patch_features;

      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < temporal_patch_size; ++t) {
          for (int64_t h = 0; h < patch_size; ++h) {
            for (int64_t w = 0; w < patch_size; ++w) {
              int64_t img_h = ph * patch_size + h;
              int64_t img_w = pw * patch_size + w;
              int64_t img_idx = img_h * width * channels + img_w * channels + c;
              *patch_ptr++ = image_data[img_idx];
            }
          }
        }
      }
    }
  }

  return patch_tensor;
}

// Helper to convert float32 tensor to target type (float16 or bfloat16)
std::unique_ptr<OrtValue> ConvertPixelValues(const OrtValue& float_tensor,
                                             ONNXTensorElementDataType target_type,
                                             Ort::Allocator& allocator) {
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    // No conversion needed, return a copy
    auto shape = float_tensor.GetTensorTypeAndShapeInfo()->GetShape();
    auto result = OrtValue::CreateTensor<float>(allocator, shape);
    const float* src = float_tensor.GetTensorData<float>();
    float* dst = result->GetTensorMutableData<float>();
    size_t count = float_tensor.GetTensorTypeAndShapeInfo()->GetElementCount();
    std::copy(src, src + count, dst);
    return result;
  }

  auto shape = float_tensor.GetTensorTypeAndShapeInfo()->GetShape();
  size_t count = float_tensor.GetTensorTypeAndShapeInfo()->GetElementCount();

  std::unique_ptr<OrtValue> result;
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    result = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, shape);
  } else if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    result = OrtValue::CreateTensor<Ort::Float16_t>(allocator, shape);
  } else {
    throw std::runtime_error("Unsupported target type for pixel values conversion");
  }

  // Use CPU device Cast method for optimized conversion
  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);
  void* input_data = const_cast<void*>(static_cast<const void*>(float_tensor.GetTensorData<float>()));
  void* output_data = result->GetTensorMutableRawData();
  cpu_device->Cast(input_data, output_data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, target_type, count);

  return result;
}

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* pixel_values, OrtxTensor* image_grid_thw,
                   Ort::Allocator& allocator, int64_t spatial_merge_size) {
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char vision_end_token[] = "<|vision_end|>";
  constexpr char image_pad_token[] = "<|image_pad|>";

  int64_t num_images = 0;
  int64_t total_image_tokens = 0;
  const int64_t* image_grid_thw_data = nullptr;
  std::vector<int64_t> computed_grid_thw;

  if (pixel_values) {
    // Get image_grid_thw data
    if (image_grid_thw) {
      const int64_t* image_grid_thw_shape{};
      size_t image_grid_thw_num_dims;
      CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&image_grid_thw_data),
                                    &image_grid_thw_shape, &image_grid_thw_num_dims));
      num_images = image_grid_thw_shape[0];

      // Calculate total image tokens based on grid dimensions
      // For each image: (temporal * height * width) / (merge_size^2)
      for (int64_t i = 0; i < num_images; ++i) {
        int64_t t = image_grid_thw_data[i * 3 + 0];
        int64_t h = image_grid_thw_data[i * 3 + 1];
        int64_t w = image_grid_thw_data[i * 3 + 2];
        int64_t tokens = (t * h * w) / (spatial_merge_size * spatial_merge_size);
        total_image_tokens += tokens;
      }
    } else {
      // image_grid_thw not provided - compute from fixed 384x384 with 16x16 patches
      // For Qwen3-VL: 384/16 = 24 patches per side
      num_images = 1;  // Assuming single image
      computed_grid_thw = {1, 24, 24};  // [temporal, height_patches, width_patches]
      image_grid_thw_data = computed_grid_thw.data();
      
      int64_t t = 1, h = 24, w = 24;
      total_image_tokens = (t * h * w) / (spatial_merge_size * spatial_merge_size);
    }
  }

  // Generate input_ids with vision tokens
  std::string text = prompt;

  // If prompt is empty, add vision markers for each image
  if (text.empty()) {
    for (int64_t i = 0; i < num_images; ++i) {
      text += std::string(vision_start_token) + " " + std::string(vision_end_token);
      if (i < num_images - 1) {
        text += " ";
      }
    }
  }

  // Count the number of vision_start tokens and make sure it matches the number of images
  const std::regex vision_start_regex{R"(<\|vision_start\|>)"};
  const auto vision_start_begin = std::sregex_iterator(text.begin(), text.end(), vision_start_regex);
  const auto vision_start_end = std::sregex_iterator();
  const auto vision_start_tokens = std::distance(vision_start_begin, vision_start_end);

  if (num_images != vision_start_tokens) {
    throw std::runtime_error("Prompt contained " + std::to_string(vision_start_tokens) +
                             " vision_start tokens but received " + std::to_string(num_images) + " images.");
  }

  // For Qwen3-VL, we need to replace vision markers with image_pad tokens
  // The number of image_pad tokens for each image depends on the image dimensions
  if (num_images > 0 && image_grid_thw_data) {
    std::string modified_text;
    size_t last_pos = 0;
    size_t image_idx = 0;

    std::smatch match;
    std::string temp_text = text;
    while (std::regex_search(temp_text, match, vision_start_regex)) {
      // Add text before the vision_start token
      modified_text += text.substr(last_pos, match.position() - (last_pos - (text.size() - temp_text.size())));

      // Calculate number of image_pad tokens for this image
      int64_t t = image_grid_thw_data[image_idx * 3 + 0];
      int64_t h = image_grid_thw_data[image_idx * 3 + 1];
      int64_t w = image_grid_thw_data[image_idx * 3 + 2];
      int64_t num_pads = (t * h * w) / (spatial_merge_size * spatial_merge_size);

      // Add vision_start, image_pad tokens, and vision_end
      modified_text += vision_start_token;
      for (int64_t i = 0; i < num_pads; ++i) {
        modified_text += image_pad_token;
      }
      modified_text += vision_end_token;

      last_pos = match.position() + match.length() + (text.size() - temp_text.size());

      // Find and skip vision_end token
      size_t vision_end_pos = text.find(vision_end_token, last_pos);
      if (vision_end_pos != std::string::npos) {
        last_pos = vision_end_pos + strlen(vision_end_token);
      }

      temp_text = match.suffix();
      image_idx++;
    }
    modified_text += text.substr(last_pos);
    text = modified_text;
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  std::unique_ptr<OrtValue> num_img_tokens = OrtValue::CreateTensor<int64_t>(
      allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int64_t>()[0] = total_image_tokens;

  return {std::move(input_ids_value), std::move(num_img_tokens)};
}

}  // namespace

Qwen3VLImageProcessor::Qwen3VLImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},  // Default to float, will be determined at runtime if vision session exists
      spatial_merge_size_{2} {  // Qwen3-VL uses merge_size=2
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  // Try to get pixel values type from session info if vision session exists
  try {
    pixel_values_type_ = session_info.GetInputDataType(config.model.vision.inputs.pixel_values);
  } catch (...) {
    // Vision session not in session_info (e.g., for pipeline models), keep default float type
  }

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> Qwen3VLImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images || images->num_images_ == 0) {
    auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, nullptr, allocator, spatial_merge_size_);
    named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  OrtxTensor* image_grid_thw = nullptr;
  // Try to get image_grid_thw from processor (second output)
  OrtxTensorResultGetAt(result.get(), 1, &image_grid_thw);

  auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, pixel_values, image_grid_thw, allocator, spatial_merge_size_);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));

  // Process pixel_values - convert from CHW to patch format
  const void* pixel_data{};
  const int64_t* pixel_shape{};
  size_t pixel_num_dims;
  CheckResult(OrtxGetTensorData(pixel_values, &pixel_data, &pixel_shape, &pixel_num_dims));

  // Squeeze out leading dimension of size 1 if present
  std::vector<int64_t> pixel_target_shape;
  size_t squeeze_offset = 0;
  if (pixel_num_dims >= 4 && pixel_shape[0] == 1) {
    // Skip the batch dimension
    squeeze_offset = 1;
  }
  for (size_t i = squeeze_offset; i < pixel_num_dims; ++i) {
    pixel_target_shape.push_back(pixel_shape[i]);
  }

  // Accept CHW ([3, H, W]) and HWC ([H, W, 3]) from different ort-extensions pipelines.
  if (pixel_target_shape.size() != 3) {
    throw std::runtime_error("Expected pixel_values in CHW format [C, H, W], got rank " + 
                             std::to_string(pixel_target_shape.size()));
  }

  bool is_chw = false;
  bool is_hwc = false;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;

  if (pixel_target_shape[0] == 3) {
    // CHW: [3, H, W]
    is_chw = true;
    channels = pixel_target_shape[0];
    height = pixel_target_shape[1];
    width = pixel_target_shape[2];
  } else if (pixel_target_shape[2] == 3) {
    // HWC: [H, W, 3]
    is_hwc = true;
    channels = pixel_target_shape[2];
    height = pixel_target_shape[0];
    width = pixel_target_shape[1];
  } else {
    throw std::runtime_error("Expected pixel_values in CHW [3,H,W] or HWC [H,W,3] layout.");
  }

  // Qwen3-VL expects 384x384 images
  if (height != 384 || width != 384) {
    throw std::runtime_error("Qwen3-VL expects 384x384 images, got " + 
                             std::to_string(height) + "x" + std::to_string(width) + 
                             ". Please ensure vision_processor.json includes a Resize operation to 384x384.");
  }

  // Convert to patch format for Qwen3-VL vision model.
  const int64_t patch_size = 16;
  const int64_t temporal_patch_size = 2;
  std::unique_ptr<OrtValue> patch_tensor;
  if (is_chw) {
    patch_tensor = ConvertToPatch(static_cast<const float*>(pixel_data),
                                  channels, height, width,
                                  patch_size, temporal_patch_size, allocator);
  } else {
    patch_tensor = ConvertToPatchHWC(static_cast<const float*>(pixel_data),
                                     channels, height, width,
                                     patch_size, temporal_patch_size, allocator);
  }

  // Convert to target type if needed
  auto converted_tensor = ConvertPixelValues(*patch_tensor, pixel_values_type_, allocator);
  named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                         std::make_shared<Tensor>(std::move(converted_tensor)));

  // Add image_grid_thw tensor
  if (image_grid_thw) {
    // Get the tensor data and shape from OrtxTensor
    const int64_t* grid_data{};
    const int64_t* grid_shape{};
    size_t grid_num_dims;
    CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&grid_data),
                                  &grid_shape, &grid_num_dims));

    // Squeeze out leading dimension of size 1
    std::vector<int64_t> grid_target_shape;
    size_t grid_squeeze_offset = 0;
    if (grid_num_dims >= 3 && grid_shape[0] == 1) {
      // Skip the batch dimension
      grid_squeeze_offset = 1;
    }
    for (size_t i = grid_squeeze_offset; i < grid_num_dims; ++i) {
      grid_target_shape.push_back(grid_shape[i]);
    }

    // Ensure we have rank 2 [num_images, 3]
    if (grid_target_shape.size() != 2 || grid_target_shape[1] != 3) {
      throw std::runtime_error("image_grid_thw must have shape [num_images, 3], got shape with " +
                               std::to_string(grid_target_shape.size()) + " dimensions");
    }

    int64_t num_grid_elements = std::accumulate(grid_target_shape.begin(), grid_target_shape.end(), 1LL, std::multiplies<int64_t>());
    auto grid_tensor = OrtValue::CreateTensor<int64_t>(allocator, grid_target_shape);
    std::copy(grid_data, grid_data + num_grid_elements, grid_tensor->GetTensorMutableData<int64_t>());

    named_tensors->emplace("image_grid_thw", std::make_shared<Tensor>(std::move(grid_tensor)));
  } else {
    // Create computed grid_thw for fixed 384x384 images
    auto grid_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{1, 3});
    auto* grid_data = grid_tensor->GetTensorMutableData<int64_t>();
    grid_data[0] = 1;   // temporal
    grid_data[1] = 24;  // height in patches (384/16)
    grid_data[2] = 24;  // width in patches (384/16)
    named_tensors->emplace("image_grid_thw", std::make_shared<Tensor>(std::move(grid_tensor)));
  }

  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

  return named_tensors;
}

}  // namespace Generators
