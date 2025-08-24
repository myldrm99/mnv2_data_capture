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

// Helper function to print all quantization parameters for a layer
void PrintQuantParams(const char* layer_name, const OpDataConv& data, int num_channels) {
    printf("\n// --- %s: REQUANTIZATION PARAMS ---\n", layer_name);
    printf("const int32_t %s_input_offset = %ld;\n", layer_name, data.input_zero_point);
    printf("const int32_t %s_output_offset = %ld;\n\n", layer_name, data.output_zero_point);

    printf("// Per-channel output multipliers:\n");
    printf("const int32_t %s_output_multiplier[] = {\n    ", layer_name);
    for (int i = 0; i < num_channels; ++i) {
        printf("0x%08lx, ", data.per_channel_output_multiplier[i]);
        if ((i + 1) % 8 == 0 && (i + 1) < num_channels) printf("\n    ");
    }
    printf("\n};\n\n");
    
    printf("// Per-channel output shifts:\n");
    printf("const int32_t %s_output_shift[] = {\n    ", layer_name);
    for (int i = 0; i < num_channels; ++i) {
        printf("%ld, ", data.per_channel_output_shift[i]);
        if ((i + 1) % 16 == 0 && (i + 1) < num_channels) printf("\n    ");
    }
    printf("\n};\n");
}


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
  // DATA CAPTURE BLOCK
  // ========================================================================
  static int conv_bn_counter = 0;
  const int input_depth = tflite::micro::GetTensorShape(input).Dims(3);
  const int output_depth = tflite::micro::GetTensorShape(output).Dims(3);
  const bool is_1x1_kernel = (tflite::micro::GetTensorShape(filter).Dims(1) == 1 && tflite::micro::GetTensorShape(filter).Dims(2) == 1);
  const bool is_expansion = is_1x1_kernel && (output_depth > input_depth);
  const bool is_projection = is_1x1_kernel && (output_depth < input_depth);

  // --- Pre-computation data dump ---
  if (is_expansion && conv_bn_counter == 4) {
    printf("\n// ======================================================================");
    printf("\n// BN 5: EXPANSION LAYER DATA");
    printf("\n// ======================================================================\n");
    print_tensor_as_h("bn5_ex_ifmap", input);
    print_tensor_as_h("bn5_ex_filter", filter);
    if (bias) print_tensor_as_h("bn5_ex_bias", bias, true);
    PrintQuantParams("bn5_ex", data, output_depth);
  }
  if (is_projection && conv_bn_counter == 4) {
    printf("\n// ======================================================================");
    printf("\n// BN 5: PROJECTION LAYER DATA");
    printf("\n// ======================================================================\n");
    print_tensor_as_h("bn5_pr_ifmap", input);
    print_tensor_as_h("bn5_pr_filter", filter);
    if (bias) print_tensor_as_h("bn5_pr_bias", bias, true);
    PrintQuantParams("bn5_pr", data, output_depth);
  }

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
      // ... (code unchanged)
      break;
    }
    case kTfLiteInt8: {
      // ... (code unchanged)
      switch (filter->type) {
        case kTfLiteInt4: {
          // ...
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
  
  // --- Post-computation data dump ---
  if (is_projection && conv_bn_counter == 4) {
    printf("\n// --- BN 5: FINAL OUTPUT DATA ---\n");
    print_tensor_as_h("bn5_final_output", output);
  }

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
