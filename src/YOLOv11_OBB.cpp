#include "YOLOv11_OBB.h"
#include "cuda_utils.h"
#include "logging.h"
#include "macros.h"
#include "model_utils.h"
#include "preprocess.h"
#include "runtime_utils.h"
#include "trt_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <mutex>


static Logger logger;
static std::mutex g_preprocess_mutex;

// DOTA dataset class names (15 classes for OBB)
// static const std::vector<std::string> OBB_CLASS_NAMES = {
//     "plane",           "ship",              "storage tank",       "baseball diamond",
//     "tennis court",    "swimming pool",     "ground track field", "harbor",
//     "bridge",          "large vehicle",     "small vehicle",      "helicopter",
//     "roundabout",      "soccer ball field", "basketball court"
// };
static const std::vector<std::string> OBB_CLASS_NAMES = {
    "border"
};

static std::vector<std::string> buildDefaultClassNames(int num_classes) {
    return BuildDefaultClassNames(OBB_CLASS_NAMES, num_classes);
}

static float normalizeAnglePi(float angle) {
    float normalized = std::fmod(angle, static_cast<float>(CV_PI));
    if (normalized < 0.0f) {
        normalized += static_cast<float>(CV_PI);
    }
    return normalized;
}

static std::vector<Point2f> xywhrToCorners(float cx, float cy, float w, float h, float angle) {
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);
    const Point2f vx((w * 0.5f) * cos_a, (w * 0.5f) * sin_a);
    const Point2f vy(-(h * 0.5f) * sin_a, (h * 0.5f) * cos_a);

    return {
        Point2f(cx, cy) + vx + vy,
        Point2f(cx, cy) + vx - vy,
        Point2f(cx, cy) - vx - vy,
        Point2f(cx, cy) - vx + vy
    };
}

static float polygonArea(const std::vector<Point2f>& polygon) {
    if (polygon.size() < 3) {
        return 0.0f;
    }
    return std::fabs(static_cast<float>(cv::contourArea(polygon)));
}


YOLOv11_OBB::YOLOv11_OBB(string model_path, nvinfer1::ILogger& logger, const OBBConfig& config)
{
    config_ = config;

    try {
        // Deserialize an engine
        if (model_path.find(".onnx") == std::string::npos)
        {
            init(model_path, logger);
        }
        // Build an engine from an onnx model
        else
        {
            build(model_path, logger);
            saveEngine(model_path);
            initializeEngineState();
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

void YOLOv11_OBB::bindBuffers()
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

void YOLOv11_OBB::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    LoadTensorRTEngine(engine_path, logger, runtime, engine, context);
    initializeEngineState();
}

void YOLOv11_OBB::initializeEngineState()
{
    // Get input and output sizes of the model
#if NV_TENSORRT_MAJOR < 10
    nvinfer1::Dims input_dims{};
    nvinfer1::Dims output_dims{};
    input_tensor_name.clear();
    output_tensor_name.clear();

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        const auto dims = engine->getBindingDimensions(i);
        if (dims.nbDims == 4 && input_dims.nbDims == 0) {
            input_binding_index = i;
            input_dims = dims;
        } else if (dims.nbDims == 3 && output_dims.nbDims == 0) {
            output_binding_index = i;
            output_dims = dims;
        }
    }

    if (input_dims.nbDims != 4 || output_dims.nbDims != 3) {
        throw std::runtime_error("Unsupported OBB binding layout");
    }

    runtime_.input_h = input_dims.d[2];
    runtime_.input_w = input_dims.d[3];
    runtime_.detection_attribute_size = output_dims.d[1];
    runtime_.num_detections = output_dims.d[2];
#else
    input_tensor_name.clear();
    output_tensor_name.clear();
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
        throw std::runtime_error("Failed to identify TensorRT OBB input/output tensors");
    }
    auto input_dims = engine->getTensorShape(input_tensor_name.c_str());
    auto output_dims = engine->getTensorShape(output_tensor_name.c_str());
    runtime_.input_h = input_dims.d[2];
    runtime_.input_w = input_dims.d[3];
    runtime_.detection_attribute_size = output_dims.d[1];
    runtime_.num_detections = output_dims.d[2];
#endif

    // Ultralytics OBB export layout: cx, cy, w, h, class_probs..., angle
    runtime_.num_classes = runtime_.detection_attribute_size - 5;
    if (runtime_.num_classes <= 0) {
        throw std::runtime_error("Invalid OBB output layout: detection_attribute_size must be >= 6");
    }
    if (config_.expected_num_classes > 0 && config_.expected_num_classes != runtime_.num_classes) {
        std::ostringstream oss;
        oss << "Configured num_classes (" << config_.expected_num_classes
            << ") does not match engine output (" << runtime_.num_classes << ")";
        throw std::runtime_error(oss.str());
    }
    if (config_.class_names.empty()) {
        config_.class_names = buildDefaultClassNames(runtime_.num_classes);
    } else if (static_cast<int>(config_.class_names.size()) != runtime_.num_classes) {
        std::ostringstream oss;
        oss << "Configured class_names size (" << config_.class_names.size()
            << ") does not match num_classes (" << runtime_.num_classes << ")";
        throw std::runtime_error(oss.str());
    }
    runtime_.output_numel = static_cast<size_t>(runtime_.detection_attribute_size) * runtime_.num_detections;

    printf("YOLOv11_OBB Engine Info:\n");
    printf("  Input: %dx%d\n", runtime_.input_w, runtime_.input_h);
    printf("  Output: %d x %d\n", runtime_.detection_attribute_size, runtime_.num_detections);
    printf("  Num classes: %d\n", runtime_.num_classes);

    // Initialize input buffers
    cpu_output_buffer = new float[runtime_.output_numel];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * runtime_.input_w * runtime_.input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], runtime_.output_numel * sizeof(float)));
    bindBuffers();
    CUDA_CHECK(cudaMemset(gpu_buffers[0], 0, 3 * runtime_.input_w * runtime_.input_h * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_buffers[1], 0, runtime_.output_numel * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    preprocess_initialized_ = true;

    CUDA_CHECK(cudaStreamCreate(&stream));

    if (config_.enable_warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

YOLOv11_OBB::~YOLOv11_OBB()
{
    cleanup();
}

void YOLOv11_OBB::preprocess(Mat& image) {
    ValidateInputImage(image, MAX_IMAGE_SIZE);
    runtime_.scale_ratio = std::min(runtime_.input_h / static_cast<float>(image.rows),
                                    runtime_.input_w / static_cast<float>(image.cols));
    runtime_.pad_x = (runtime_.input_w - runtime_.scale_ratio * image.cols) * 0.5f;
    runtime_.pad_y = (runtime_.input_h - runtime_.scale_ratio * image.rows) * 0.5f;

    // preprocess.cu uses shared staging buffers, so guard the full copy+kernel+sync section.
    std::lock_guard<std::mutex> lock(g_preprocess_mutex);
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], runtime_.input_w, runtime_.input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11_OBB::infer()
{
#if NV_TENSORRT_MAJOR < 10
    void* bindings[2]{};
    bindings[input_binding_index] = gpu_buffers[0];
    bindings[output_binding_index] = gpu_buffers[1];
    context->enqueueV2(bindings, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
}

void YOLOv11_OBB::postprocess(vector<OBBDetection>& output, int img_w, int img_h)
{
    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], runtime_.output_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    output.clear();

    const Mat det_output(runtime_.detection_attribute_size, runtime_.num_detections, CV_32F, cpu_output_buffer);
    const float inv_scale = runtime_.scale_ratio > 0.0f ? (1.0f / runtime_.scale_ratio) : 0.0f;

    for (int i = 0; i < det_output.cols; ++i) {
        const int angle_channel = 4 + runtime_.num_classes;
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + runtime_.num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > config_.conf_threshold) {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);
            float angle = det_output.at<float>(angle_channel, i);

            // Map coordinates back to original image
            // Inverse of letterbox: subtract padding, then divide by scale
            cx = (cx - runtime_.pad_x) * inv_scale;
            cy = (cy - runtime_.pad_y) * inv_scale;
            ow *= inv_scale;
            oh *= inv_scale;

            if (ow <= 1.0f || oh <= 1.0f) {
                continue;
            }

            // Regularize to the long-edge representation used by Ultralytics OBB.
            angle = normalizeAnglePi(angle);
            if (ow < oh) {
                std::swap(ow, oh);
                angle = normalizeAnglePi(angle + static_cast<float>(CV_PI) * 0.5f);
            }

            std::vector<Point2f> corners = xywhrToCorners(cx, cy, ow, oh, angle);
            for (Point2f& point : corners) {
                point.x = std::clamp(point.x, 0.0f, static_cast<float>(img_w - 1));
                point.y = std::clamp(point.y, 0.0f, static_cast<float>(img_h - 1));
            }

            OBBDetection det;
            det.conf = static_cast<float>(score);
            det.class_id = class_id_point.y;
            det.corners = corners;
            det.rotated_rect = RotatedRect(
                Point2f(std::clamp(cx, 0.0f, static_cast<float>(img_w - 1)),
                        std::clamp(cy, 0.0f, static_cast<float>(img_h - 1))),
                Size2f(ow, oh),
                angle * 180.0f / static_cast<float>(CV_PI));
            output.push_back(std::move(det));
        }
    }
    // Apply rotated NMS
    nmsRotated(output, config_.nms_threshold);
}

float YOLOv11_OBB::computeRotatedIoU(const OBBDetection& det1, const OBBDetection& det2) const
{
    std::vector<Point2f> intersection;
    const float inter_area = cv::intersectConvexConvex(det1.corners, det2.corners, intersection, true);
    if (inter_area <= 1e-6f) {
        return 0.0f;
    }

    const float area1 = polygonArea(det1.corners);
    const float area2 = polygonArea(det2.corners);
    float union_area = area1 + area2 - inter_area;

    if (union_area < 1e-6f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

void YOLOv11_OBB::nmsRotated(vector<OBBDetection>& detections, float nms_threshold) const
{
    if (detections.empty()) {
        return;
    }

    std::stable_sort(detections.begin(), detections.end(), [](const OBBDetection& a, const OBBDetection& b) {
        return a.conf > b.conf;
    });

    std::vector<OBBDetection> result;
    for (int class_id = 0; class_id < runtime_.num_classes; ++class_id) {
        std::vector<bool> suppressed(detections.size(), false);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed[i] || detections[i].class_id != class_id) {
                continue;
            }
            result.push_back(detections[i]);
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (suppressed[j] || detections[j].class_id != class_id) {
                    continue;
                }
                if (computeRotatedIoU(detections[i], detections[j]) > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    std::stable_sort(result.begin(), result.end(), [](const OBBDetection& a, const OBBDetection& b) {
        return a.conf > b.conf;
    });
    detections = result;
}

void YOLOv11_OBB::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    BuildTensorRTFromOnnx(onnxPath, logger, config_.use_fp16, runtime, engine, context);
}

bool YOLOv11_OBB::saveEngine(const std::string& onnxpath)
{
    return SaveTensorRTEngine(engine, onnxpath);
}

void YOLOv11_OBB::cleanup() noexcept
{
    SafeDestroyCudaStream(stream);

    for (float*& buffer : gpu_buffers) {
        SafeCudaFree(buffer);
    }

    SafeDeleteArray(cpu_output_buffer);
    SafeDestroyPreprocess(preprocess_initialized_);
    SafeDestroyTensorRT(runtime, engine, context);
}

void YOLOv11_OBB::draw(Mat& image, const vector<OBBDetection>& output, const string& output_path)
{
    Mat display = image.clone();

    for (size_t i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto rrect = detection.rotated_rect;
        int class_id = detection.class_id;
        float conf = detection.conf;

        // Get class name and color
        const string class_name = GetClassName(config_.class_names, class_id);
        const cv::Scalar color = GetClassColor(class_id);

        // Get corner points of rotated rectangle
        Point2f corners[4];
        if (detection.corners.size() == 4) {
            for (int j = 0; j < 4; ++j) corners[j] = detection.corners[j];
        } else {
            rrect.points(corners);
        }

        // Draw rotated rectangle
        for (int j = 0; j < 4; j++) {
            line(display, corners[j], corners[(j + 1) % 4], color, 2);
        }

        // Draw class label and confidence
        string label = class_name + " " + to_string(conf).substr(0, 4);
        int baseLine = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        Point labelOrigin(static_cast<int>(corners[0].x), static_cast<int>(corners[0].y) - 5);
        // Ensure label doesn't go off-screen
        if (labelOrigin.y < labelSize.height + 5) {
            labelOrigin.y = static_cast<int>(corners[0].y) + labelSize.height + 10;
        }
        if (labelOrigin.x < 0) {
            labelOrigin.x = 0;
        }
        if (labelOrigin.x + labelSize.width > display.cols) {
            labelOrigin.x = std::max(0, display.cols - labelSize.width);
        }
        if (labelOrigin.y >= display.rows) {
            labelOrigin.y = std::max(labelSize.height + 5, display.rows - 1);
        }

        rectangle(display,
                  Point(labelOrigin.x, labelOrigin.y - labelSize.height - 5),
                  Point(labelOrigin.x + labelSize.width, labelOrigin.y + baseLine),
                  color, -1);
        putText(display, label, Point(labelOrigin.x, labelOrigin.y),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    }

    image = display;
    if (!output_path.empty()) {
        imwrite(output_path, display);
        printf("Saved result to: %s\n", output_path.c_str());
    }
}
