#pragma once

#include "NvInfer.h"

#include <memory>
#include <string>
#include <vector>

template <typename T>
struct TrtDeleter {
    void operator()(T* object) const noexcept {
        delete object;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

std::vector<char> ReadBinaryFile(const std::string& path);
void LoadTensorRTEngine(const std::string& engine_path, nvinfer1::ILogger& logger,
                        nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine,
                        nvinfer1::IExecutionContext*& context);
void BuildTensorRTFromOnnx(const std::string& onnx_path, nvinfer1::ILogger& logger, bool use_fp16,
                           nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine,
                           nvinfer1::IExecutionContext*& context);
bool SaveTensorRTEngine(nvinfer1::ICudaEngine* engine, const std::string& onnx_path);
