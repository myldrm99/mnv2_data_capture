#pragma once
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/kernels/kernel_util.h>
#include <cstdio>
#include <cinttypes>

using namespace tflite;

// Helper to print a tensor's data as a C-style array
inline void print_tensor_as_h(const char* name, const TfLiteEvalTensor* tensor) {
    // Print shape as a comment
    printf("// Tensor '%s', Shape: [", name);
    for (int i = 0; i < tflite::micro::GetTensorShape(tensor).DimensionsCount(); ++i) {
        printf("%" PRId32, tflite::micro::GetTensorShape(tensor).Dims(i));
        if (i < tflite::micro::GetTensorShape(tensor).DimensionsCount() - 1) printf(", ");
    }
    printf("]\n");

    // Print data as a C array
    const int8_t* data = tflite::micro::GetTensorData<int8_t>(tensor);
    int flat_size = tflite::micro::GetTensorShape(tensor).FlatSize();
    
    printf("const int8_t %s[] = {", name);
    for (int i = 0; i < flat_size; ++i) {
        if (i % 16 == 0) printf("\n    ");
        printf("0x%02x, ", data[i]);
    }
    printf("\n};\n\n");
}

// Overloaded version for bias data (int32_t)
inline void print_tensor_as_h(const char* name, const TfLiteEvalTensor* tensor, bool is_bias) {
    printf("// Tensor '%s', Shape: [%" PRId32 "]\n", name, static_cast<int32_t>(tflite::micro::GetTensorShape(tensor).FlatSize()));
    const int32_t* data = tflite::micro::GetTensorData<int32_t>(tensor);
    printf("const int32_t %s[] = {", name);
    for (int i = 0; i < tflite::micro::GetTensorShape(tensor).FlatSize(); ++i) {
        if (i % 8 == 0) printf("\n    ");
        printf("0x%08" PRIx32 ", ", data[i]);
    }
    printf("\n};\n\n");
}