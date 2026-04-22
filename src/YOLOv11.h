#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{
public:
    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);
    void initializeEngineState();
    void bindBuffers();
    void cleanup() noexcept;

    float* gpu_buffers[2]{};               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer = nullptr;

    cudaStream_t stream{};
    IRuntime* runtime = nullptr;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine = nullptr;               //!< The TensorRT engine used to run the network
    IExecutionContext* context = nullptr;        //!< The context for executing inference using an ICudaEngine
    bool preprocess_initialized_ = false;

    // Model parameters
    int input_w = 0;
    int input_h = 0;
    int num_detections = 0;
    int detection_attribute_size = 0;
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;
    std::string input_tensor_name;
    std::string output_tensor_name;
    int input_binding_index = 0;
    int output_binding_index = 1;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};
