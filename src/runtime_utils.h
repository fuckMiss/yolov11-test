#pragma once

#include "NvInfer.h"
#include "preprocess.h"

#include <cuda_runtime_api.h>

inline void SafeDestroyCudaStream(cudaStream_t& stream) noexcept
{
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

template <typename T>
inline void SafeCudaFree(T*& buffer) noexcept
{
    if (buffer) {
        cudaFree(reinterpret_cast<void*>(buffer));
        buffer = nullptr;
    }
}

template <typename T>
inline void SafeDeleteArray(T*& buffer) noexcept
{
    delete[] buffer;
    buffer = nullptr;
}

template <typename T>
inline void SafeDelete(T*& object) noexcept
{
    delete object;
    object = nullptr;
}

inline void SafeFreeHostBuffer(void*& buffer) noexcept
{
    free(buffer);
    buffer = nullptr;
}

inline void SafeDestroyPreprocess(bool& preprocess_initialized) noexcept
{
    if (preprocess_initialized) {
        cuda_preprocess_destroy();
        preprocess_initialized = false;
    }
}

inline void SafeDestroyTensorRT(nvinfer1::IRuntime*& runtime,
                                nvinfer1::ICudaEngine*& engine,
                                nvinfer1::IExecutionContext*& context) noexcept
{
    SafeDelete(context);
    SafeDelete(engine);
    SafeDelete(runtime);
}
