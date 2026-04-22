#include "YOLOv11.h"

#include "cuda_utils.h"
#include "logging.h"
#include "macros.h"
#include "model_utils.h"
#include "preprocess.h"
#include "runtime_utils.h"
#include "trt_utils.h"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>


static Logger logger;
static std::mutex g_preprocess_mutex_det;

namespace {

constexpr bool kUseFp16 = false;
constexpr bool kEnableWarmup = true;

}  // namespace

YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    try {
        if (model_path.find(".onnx") == std::string::npos) {
            init(model_path, logger);
        } else {
            build(model_path, logger);
            saveEngine(model_path);
            initializeEngineState();
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

YOLOv11::~YOLOv11()
{
    cleanup();
}

void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    LoadTensorRTEngine(engine_path, logger, runtime, engine, context);
    initializeEngineState();
}

void YOLOv11::initializeEngineState()
{
    nvinfer1::Dims input_dims{};
    nvinfer1::Dims output_dims{};

#if NV_TENSORRT_MAJOR < 10
    input_binding_index = 0;
    output_binding_index = 1;

    const auto binding1_dims = engine->getBindingDimensions(1);
    if (binding1_dims.nbDims == 3) {
        output_binding_index = 1;
    } else if (engine->getNbBindings() > 2 && engine->getBindingDimensions(2).nbDims == 3) {
        output_binding_index = 2;
    } else {
        throw std::runtime_error("Unable to identify DET output binding");
    }

    input_dims = engine->getBindingDimensions(input_binding_index);
    output_dims = engine->getBindingDimensions(output_binding_index);
#else
    const int io_count = engine->getNbIOTensors();
    for (int i = 0; i < io_count; ++i) {
        const char* tensor_name = engine->getIOTensorName(i);
        if (!tensor_name) {
            continue;
        }

        if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name = tensor_name;
        } else if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
            output_tensor_name = tensor_name;
        }
    }

    if (input_tensor_name.empty() || output_tensor_name.empty()) {
        throw std::runtime_error("Failed to identify TensorRT input/output tensors for DET engine");
    }

    input_dims = engine->getTensorShape(input_tensor_name.c_str());
    output_dims = engine->getTensorShape(output_tensor_name.c_str());
#endif

    if (input_dims.nbDims != 4 || output_dims.nbDims != 3) {
        throw std::runtime_error(
            "Unsupported DET tensor layout: input=" + DimsToString(input_dims) +
            ", output=" + DimsToString(output_dims));
    }

    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    const int candidate_a = output_dims.d[1];
    const int candidate_b = output_dims.d[2];
    if (candidate_a > 4 && candidate_b > 4) {
        if (candidate_a < candidate_b) {
            detection_attribute_size = candidate_a;
            num_detections = candidate_b;
        } else {
            detection_attribute_size = candidate_b;
            num_detections = candidate_a;
        }
    } else if (candidate_a > 4) {
        detection_attribute_size = candidate_a;
        num_detections = candidate_b;
    } else if (candidate_b > 4) {
        detection_attribute_size = candidate_b;
        num_detections = candidate_a;
    } else {
        throw std::runtime_error("Invalid DET output layout: " + DimsToString(output_dims));
    }

    num_classes = detection_attribute_size - 4;
    if (num_classes <= 0) {
        throw std::runtime_error("Invalid DET output layout: detection_attribute_size must be >= 5");
    }

    cpu_output_buffer = new float[static_cast<size_t>(detection_attribute_size) * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], static_cast<size_t>(detection_attribute_size) * num_detections * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&stream));
    bindBuffers();

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    preprocess_initialized_ = true;

    CUDA_CHECK(cudaMemset(gpu_buffers[0], 0, 3 * input_w * input_h * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_buffers[1], 0, static_cast<size_t>(detection_attribute_size) * num_detections * sizeof(float)));

    if (kEnableWarmup) {
        for (int i = 0; i < 10; ++i) {
            infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv11::bindBuffers()
{
#if NV_TENSORRT_MAJOR >= 10
    if (!input_tensor_name.empty()) {
        context->setTensorAddress(input_tensor_name.c_str(), gpu_buffers[0]);
    }
    if (!output_tensor_name.empty()) {
        context->setTensorAddress(output_tensor_name.c_str(), gpu_buffers[1]);
    }
#endif
}

void YOLOv11::cleanup() noexcept
{
    SafeDestroyCudaStream(stream);

    for (float*& buffer : gpu_buffers) {
        SafeCudaFree(buffer);
    }

    SafeDeleteArray(cpu_output_buffer);
    SafeDestroyPreprocess(preprocess_initialized_);
    SafeDestroyTensorRT(runtime, engine, context);
}

void YOLOv11::preprocess(Mat& image)
{
    ValidateInputImage(image, MAX_IMAGE_SIZE);

    std::lock_guard<std::mutex> lock(g_preprocess_mutex_det);
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    void* bindings[2]{};
    bindings[input_binding_index] = gpu_buffers[0];
    bindings[output_binding_index] = gpu_buffers[1];
    context->enqueueV2(bindings, stream, nullptr);
#else
    context->enqueueV3(stream);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1],
                               static_cast<size_t>(num_detections) * detection_attribute_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    output.clear();

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);

            Rect box;
            box.x = static_cast<int>(cx - 0.5f * ow);
            box.y = static_cast<int>(cy - 0.5f * oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            if (box.width <= 1 || box.height <= 1) {
                continue;
            }

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(static_cast<float>(score));
        }
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    BuildTensorRTFromOnnx(onnxPath, logger, kUseFp16, runtime, engine, context);
}

bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    return SaveTensorRTEngine(engine, onnxpath);
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = input_h / static_cast<float>(image.rows);
    const float ratio_w = input_w / static_cast<float>(image.cols);

    for (const auto& detection : output) {
        auto box = detection.bbox;
        const int class_id = detection.class_id;
        const float conf = detection.conf;
        const cv::Scalar color = GetClassColor(class_id);

        if (ratio_h > ratio_w) {
            box.x = static_cast<int>(box.x / ratio_w);
            box.y = static_cast<int>((box.y - (input_h - ratio_w * image.rows) / 2.0f) / ratio_w);
            box.width = static_cast<int>(box.width / ratio_w);
            box.height = static_cast<int>(box.height / ratio_w);
        } else {
            box.x = static_cast<int>((box.x - (input_w - ratio_h * image.cols) / 2.0f) / ratio_h);
            box.y = static_cast<int>(box.y / ratio_h);
            box.width = static_cast<int>(box.width / ratio_h);
            box.height = static_cast<int>(box.height / ratio_h);
        }

        box &= Rect(0, 0, image.cols, image.rows);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }

        rectangle(image, box, color, 3);

        const string class_name = GetClassName(CLASS_NAMES, class_id);
        string class_string = class_name + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, max(0, box.y - 40), text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, text_rect.y + text_rect.height - 10),
                FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
