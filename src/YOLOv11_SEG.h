#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct SegDetection
{
    float conf;
    int class_id;
    Rect bbox;
    Mat mask;
};

struct SEGConfig
{
    // <=0 表示从引擎输出自动推断
    int expected_num_classes = -1;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;
    float mask_threshold = 0.5f;
    float mask_alpha = 0.45f;
    bool use_fp16 = false;
    bool enable_warmup = true;
    // 可选：命令行传 --labels=PATH 时会覆盖默认类别名
    vector<string> class_names;
};

struct SEGRuntimeState
{
    // 以下字段都是运行时自动推断/计算出来的状态，不是手动配置项。
    int input_w = 0;
    int input_h = 0;

    int num_detections = 0;
    int detection_attribute_size = 0;
    int num_classes = 0;

    int mask_dim = 0;
    int mask_h = 0;
    int mask_w = 0;

    nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType det_dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType mask_dtype = nvinfer1::DataType::kFLOAT;

    float scale_ratio = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;

    size_t det_output_numel = 0;
    size_t mask_output_numel = 0;
};

class YOLOv11_SEG
{
public:
    YOLOv11_SEG(string model_path, nvinfer1::ILogger& logger, const SEGConfig& config = {});
    ~YOLOv11_SEG();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<SegDetection>& output, int img_w, int img_h);
    void draw(Mat& image, const vector<SegDetection>& output, const string& output_path = "seg_result.jpg");

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);
    void initializeEngineState();
    void bindBuffers();
    void cleanup() noexcept;
    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);

    Mat decodeMask(const vector<float>& coeffs, const Rect& box, int img_w, int img_h) const;

    float* gpu_buffers[3]{};
    float* cpu_det_output_buffer = nullptr;
    float* cpu_mask_output_buffer = nullptr;
    void* gpu_output_buffers[2]{};
    void* cpu_det_raw_buffer = nullptr;
    void* cpu_mask_raw_buffer = nullptr;

    cudaStream_t stream{};
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    bool preprocess_initialized_ = false;

    SEGConfig config_;
    SEGRuntimeState runtime_;

    const int MAX_IMAGE_SIZE = 4096 * 4096;

    std::string input_tensor_name;
    std::string det_tensor_name;
    std::string mask_tensor_name;
    int input_binding_index = 0;
    int det_binding_index = 1;
    int mask_binding_index = 2;
};
