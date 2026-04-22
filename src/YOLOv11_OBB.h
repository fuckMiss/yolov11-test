#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

// 【注意】头文件中不建议using namespace，但保持与原代码风格一致
using namespace nvinfer1;
using namespace std;
using namespace cv;

// ========== 旋转框检测结果结构体 ==========
// 【修改1】与Detection完全不同：使用RotatedRect替代Rect
struct OBBDetection
{
    float conf;                 // 置信度分数 0~1
    int class_id;               // 类别索引
    RotatedRect rotated_rect;   // 【核心】OpenCV旋转矩形 (中心点cx,cy, 宽高w,h, 角度angle)
    vector<Point2f> corners;    // 四个角点坐标（可选，方便绘制和计算）
};

struct OBBConfig
{
    int expected_num_classes = -1;      // <=0 表示从引擎输出自动推断
    float conf_threshold = 0.5f;
    float nms_threshold = 0.3f;
    bool use_fp16 = false;
    bool enable_warmup = true;
    vector<string> class_names;         // 可选，自定义类别名
};

struct OBBRuntimeState
{
    // 以下字段都不是配置项，运行时会根据引擎输出和预处理过程自动更新。
    int input_w = 0;
    int input_h = 0;

    int num_detections = 0;
    int detection_attribute_size = 0;
    int num_classes = 0;

    float scale_ratio = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    size_t output_numel = 0;
};

// ========== YOLOv11 OBB 推理类 ==========
class YOLOv11_OBB
{
public:
    // 构造函数：加载.engine或从.onnx构建
    YOLOv11_OBB(string model_path, nvinfer1::ILogger& logger, const OBBConfig& config = {});
    
    // 析构函数：释放所有资源
    ~YOLOv11_OBB();

    // 预处理：图片→GPU输入缓冲区
    void preprocess(Mat& image);
    
    // 推理：执行TensorRT前向传播
    void infer();
    
    // 【修改2】后处理需要原图尺寸来做坐标逆映射
    void postprocess(vector<OBBDetection>& output, int img_w, int img_h);
    
    // 【修改3】绘制增加output_path参数，支持自定义保存路径
    void draw(Mat& image, const vector<OBBDetection>& output, const string& output_path = "obb_result.jpg");

private:
    // 从.engine文件加载引擎
    void init(std::string engine_path, nvinfer1::ILogger& logger);
    void initializeEngineState();
    void bindBuffers();
    void cleanup() noexcept;
    
    // 【修改4】计算两个旋转矩形的IoU（替代标准IoU）
    float computeRotatedIoU(const OBBDetection& det1, const OBBDetection& det2) const;
    
    // 【修改5】旋转框NMS（替代标准NMS）
    void nmsRotated(vector<OBBDetection>& detections, float nms_threshold) const;

    // GPU缓冲区：0号输入，1号输出
    float* gpu_buffers[2]{};
    float* cpu_output_buffer = nullptr;

    cudaStream_t stream{};       // CUDA异步流
    IRuntime* runtime = nullptr;           // TensorRT运行时
    ICudaEngine* engine = nullptr;         // TensorRT引擎
    IExecutionContext* context = nullptr;  // 执行上下文
    bool preprocess_initialized_ = false;

    // 运行参数和中间状态分开存，避免把状态字段误当成可配置项。
    OBBConfig config_;
    OBBRuntimeState runtime_;

    const int MAX_IMAGE_SIZE = 4096 * 4096;  // 预处理最大像素数

    std::string input_tensor_name;
    std::string output_tensor_name;
    int input_binding_index = 0;
    int output_binding_index = 1;

    // 从ONNX构建引擎
    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    
    // 保存序列化引擎
    bool saveEngine(const std::string& filename);
};
