/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/kernels/conv.h"

#include "data_capture.h"  // ADDED FOR DATA CAPTURE
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  // ========================================================================
  // CORRECTED DEBUG BLOCK STARTS HERE
  // ========================================================================
  static int conv_bn_counter = 0;
  const int input_depth = tflite::micro::GetTensorShape(input).Dims(3);
  const int output_depth = tflite::micro::GetTensorShape(output).Dims(3);
  const int filter_height = tflite::micro::GetTensorShape(filter).Dims(1);
  const int filter_width = tflite::micro::GetTensorShape(filter).Dims(2);
  const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1);
  const bool is_expansion = is_1x1_kernel && (output_depth > input_depth);
  const bool is_projection = is_1x1_kernel && (output_depth < input_depth);

  static bool has_printed_debug_data = false;
  const bool is_target_layer = is_expansion && (conv_bn_counter == 4);

  if (is_target_layer && !has_printed_debug_data) {
    printf("\n\n--- DEBUG DUMP: EXPANSION STAGE, TOP-LEFT PIXEL, CHANNEL 0 ---\n\n");
    
    // 1. Print Input Data (first 16 values for the top-left pixel)
    printf("// 1. IFMAP Data (Top-Left Pixel, 1x1x16):\n");
    printf("const int8_t debug_ifmap_pixel[] = {");
    for (int i = 0; i < 16; ++i) {
      printf(" %d,", tflite::micro::GetTensorData<int8_t>(input)[i]);
    }
    printf(" };\n\n");

    // 2. Print Filter Data (first filter, 1x1x16)
    printf("// 2. Filter Data (First Filter, Channel 0, 1x1x16):\n");
    printf("const int8_t debug_filter_ch0[] = {");
    for (int i = 0; i < 16; ++i) {
      printf(" %d,", tflite::micro::GetTensorData<int8_t>(filter)[i]);
    }
    printf(" };\n\n");

    // 3. Print Bias Data (for Channel 0)
    if (bias) {
      printf("// 3. Bias Data (Channel 0):\n");
      printf("const int32_t debug_bias_ch0 = %ld; // (0x%08lx)\n\n",
             tflite::micro::GetOptionalTensorData<int32_t>(bias)[0],
             tflite::micro::GetOptionalTensorData<int32_t>(bias)[0]);
    }

    // 4. Print Quantization Parameters
    printf("// 4. Quantization Parameters:\n");
    printf("const int32_t debug_input_offset = %ld;\n", data.input_zero_point);
    printf("const int32_t debug_output_offset = %ld;\n", data.output_zero_point);
    printf("const int32_t debug_multiplier_ch0 = 0x%08lx;\n", data.per_channel_output_multiplier[0]);
    printf("const int32_t debug_shift_ch0 = %ld;\n\n", data.per_channel_output_shift[0]);
  }
  // --- End of pre-computation debug block ---

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 && filter->type == kTfLiteInt4),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      switch (bias->type) {
        case kTfLiteInt32: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        case kTfLiteInt64: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default:
          MicroPrintf("Bias type %s (%d) not supported.",
                      TfLiteTypeGetName(bias->type), bias->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          reference_integer_ops::ConvPerChannelWithPackedInt4Weights(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              unpacked_filter_data, tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  
  // --- Start of post-computation debug block ---
  if (is_target_layer && !has_printed_debug_data) {
    // 5. Print Final Result (first channel of the first output pixel)
    printf("// 5. Final Result (Output Pixel (0,0), Channel 0):\n");
    printf("const int8_t debug_output_result = %d;\n\n",
           tflite::micro::GetTensorData<int8_t>(output)[0]);
    printf("--- END DEBUG DUMP ---\n\n");
    has_printed_debug_data = true; // Set flag so we don't print again
  }
  
  // ALWAYS increment the counter after a projection layer is found
  if (is_projection) {
    conv_bn_counter++;
  }
  // ========================================================================

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, Eval);
}

}  // namespace tflite
