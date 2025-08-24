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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "data_capture.h" // ADDED FOR DATA CAPTURE
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
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
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;
          
  // ========================================================================
  // DATA CAPTURE BLOCK
  // ========================================================================
  static int dw_bn_counter = 0;
  static bool has_printed_dw_debug = false;

  if (dw_bn_counter == 4) {
      printf("\n// ======================================================================");
      printf("\n// BN 5: DEPTHWISE LAYER DATA");
      printf("\n// ======================================================================\n");
      print_tensor_as_h("bn5_dw_ifmap", input);
      print_tensor_as_h("bn5_dw_filter", filter);
      if (bias) print_tensor_as_h("bn5_dw_bias", bias, true);
      PrintQuantParams("bn5_dw", data, tflite::micro::GetTensorShape(output).Dims(3));

      if (!has_printed_dw_debug) {
          printf("\n\n--- DEBUG DUMP: DEPTHWISE STAGE, TOP-LEFT 3x3 WINDOW, CHANNEL 0 ---\n\n");
          const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
          const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
          const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
          
          printf("// 1. DW Input Data (Top-Left 3x3 Window, Channel 0):\n");
          printf("const int8_t debug_dw_window_ch0[] = {");
          for (int y = 0; y < 3; ++y) {
              for (int x = 0; x < 3; ++x) {
                  printf(" %d,", input_data[Offset(input_shape, 0, y, x, 0)]);
              }
          }
          printf(" };\n\n");

          printf("// 2. DW Filter Data (First Filter, Channel 0, 3x3):\n");
          printf("const int8_t debug_dw_filter_ch0[] = {");
          for (int i = 0; i < 9; ++i) {
              printf(" %d,", filter_data[i]);
          }
          printf(" };\n\n");
          has_printed_dw_debug = true;
      }
  }

  switch (input->type) {
    case kTfLiteFloat32: {
      // ... (code unchanged)
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt8: {
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, data),
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
        case kTfLiteInt4: {
          // ... (code unchanged)
          break;
        }
        default:
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }

  // --- Post-computation data dump ---
  if (dw_bn_counter == 4) {
      printf("\n// --- BN 5: DEPTHWISE LAYER OUTPUT (PROJECTION INPUT) ---\n");
      print_tensor_as_h("bn5_dw_output_pr_ifmap", output);
  }
  
  dw_bn_counter++;
  // ========================================================================

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return tflite::micro::RegisterOp(Init, DepthwiseConvPrepare, Eval);
}

}  // namespace tflite
