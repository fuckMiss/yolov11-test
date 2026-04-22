#include "trt_utils.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;

vector<char> ReadBinaryFile(const string& path)
{
    ifstream file(path, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open file: " + path);
    }

    file.seekg(0, ios::end);
    const streampos end_pos = file.tellg();
    if (end_pos <= 0) {
        throw runtime_error("File is empty or invalid: " + path);
    }
    file.seekg(0, ios::beg);

    vector<char> data(static_cast<size_t>(end_pos));
    if (!file.read(data.data(), static_cast<streamsize>(data.size()))) {
        throw runtime_error("Failed to read file: " + path);
    }
    return data;
}

void LoadTensorRTEngine(const string& engine_path, nvinfer1::ILogger& logger,
                        nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine,
                        nvinfer1::IExecutionContext*& context)
{
    vector<char> engine_data = ReadBinaryFile(engine_path);

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw runtime_error("Failed to create TensorRT runtime: " + engine_path);
    }

    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        throw runtime_error("Failed to deserialize TensorRT engine: " + engine_path);
    }

    context = engine->createExecutionContext();
    if (!context) {
        throw runtime_error("Failed to create TensorRT execution context: " + engine_path);
    }
}

void BuildTensorRTFromOnnx(const string& onnx_path, nvinfer1::ILogger& logger, bool use_fp16,
                           nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine,
                           nvinfer1::IExecutionContext*& context)
{
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw runtime_error("Failed to create TensorRT builder");
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config) {
        throw runtime_error("Failed to create TensorRT network or builder config");
    }

    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw runtime_error("Failed to create ONNX parser: " + onnx_path);
    }
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        throw runtime_error("Failed to parse ONNX: " + onnx_path);
    }

    TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        throw runtime_error("Failed to build TensorRT engine from ONNX: " + onnx_path);
    }

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw runtime_error("Failed to create TensorRT runtime from ONNX: " + onnx_path);
    }

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine) {
        throw runtime_error("Failed to deserialize built TensorRT engine: " + onnx_path);
    }

    context = engine->createExecutionContext();
    if (!context) {
        throw runtime_error("Failed to create TensorRT execution context from ONNX: " + onnx_path);
    }
}

bool SaveTensorRTEngine(nvinfer1::ICudaEngine* engine, const string& onnx_path)
{
    const size_t dot_index = onnx_path.find_last_of(".");
    if (!engine || dot_index == string::npos) {
        return false;
    }

    const string engine_path = onnx_path.substr(0, dot_index) + ".engine";
    nvinfer1::IHostMemory* data = engine->serialize();
    if (!data) {
        return false;
    }

    ofstream file(engine_path, ios::binary | ios::out);
    if (!file.is_open()) {
        cout << "Create engine file " << engine_path << " failed" << endl;
        delete data;
        return false;
    }

    file.write(static_cast<const char*>(data->data()), data->size());
    file.close();
    delete data;
    return true;
}
