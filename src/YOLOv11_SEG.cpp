#include "YOLOv11_SEG.h"

#include "cuda_utils.h"
#include "logging.h"
#include "macros.h"
#include "model_utils.h"
#include "preprocess.h"
#include "runtime_utils.h"
#include "trt_utils.h"

#include <algorithm>
#include <cuda_fp16.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>


static Logger logger;
static std::mutex g_preprocess_mutex_seg;

// 在这里直接改成你自己数据集的类别名。
// 例如：{"crack", "scratch", "dent"}
// 如果这里留空，程序会对 COCO 80 类模型继续使用默认 COCO 名称；
// 非 COCO 类别数时才退回到 class_0/class_1/... 这类默认名称。
static const std::vector<std::string> SEG_CLASS_NAMES = {
};

static size_t dataTypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(__half);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    case nvinfer1::DataType::kUINT8:
        return sizeof(uint8_t);
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kINT64:
    case nvinfer1::DataType::kBF16:
        break;
#endif
    }
    throw runtime_error("Unsupported TensorRT tensor data type");
}

static const char* dataTypeName(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return "float32";
    case nvinfer1::DataType::kHALF: return "float16";
    case nvinfer1::DataType::kINT8: return "int8";
    case nvinfer1::DataType::kINT32: return "int32";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kBOOL: return "bool";
    case nvinfer1::DataType::kUINT8: return "uint8";
    case nvinfer1::DataType::kFP8: return "fp8";
    case nvinfer1::DataType::kINT64: return "int64";
    case nvinfer1::DataType::kBF16: return "bf16";
#endif
    }
    return "unknown";
}

static void convertBufferToFloat(const void* src, nvinfer1::DataType dtype, float* dst, size_t count) {
    if (dtype == nvinfer1::DataType::kFLOAT) {
        memcpy(dst, src, count * sizeof(float));
        return;
    }
    if (dtype == nvinfer1::DataType::kHALF) {
        const __half* half_src = static_cast<const __half*>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = __half2float(half_src[i]);
        }
        return;
    }
    throw runtime_error("Unsupported TensorRT output dtype for float conversion");
}

static vector<string> buildDefaultSegClassNames(int num_classes) {
    if (!SEG_CLASS_NAMES.empty()) {
        return BuildDefaultClassNames(SEG_CLASS_NAMES, num_classes);
    }
    return BuildCocoClassNames(num_classes);
}

YOLOv11_SEG::YOLOv11_SEG(string model_path, nvinfer1::ILogger& logger, const SEGConfig& config)
{
    config_ = config;

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

YOLOv11_SEG::~YOLOv11_SEG()
{
    cleanup();
}

void YOLOv11_SEG::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    LoadTensorRTEngine(engine_path, logger, runtime, engine, context);
    initializeEngineState();
}

void YOLOv11_SEG::initializeEngineState()
{
    nvinfer1::Dims det_dims{};
    nvinfer1::Dims mask_dims{};

#if NV_TENSORRT_MAJOR < 10
    input_binding_index = 0;
    runtime_.input_h = engine->getBindingDimensions(input_binding_index).d[2];
    runtime_.input_w = engine->getBindingDimensions(input_binding_index).d[3];
    runtime_.input_dtype = engine->getBindingDataType(input_binding_index);

    auto output_dims_a = engine->getBindingDimensions(1);
    auto output_dims_b = engine->getBindingDimensions(2);
    const auto output_dtype_a = engine->getBindingDataType(1);
    const auto output_dtype_b = engine->getBindingDataType(2);
    if (output_dims_a.nbDims == 3 && output_dims_b.nbDims == 4) {
        det_dims = output_dims_a;
        mask_dims = output_dims_b;
        runtime_.det_dtype = output_dtype_a;
        runtime_.mask_dtype = output_dtype_b;
        det_binding_index = 1;
        mask_binding_index = 2;
    } else if (output_dims_a.nbDims == 4 && output_dims_b.nbDims == 3) {
        det_dims = output_dims_b;
        mask_dims = output_dims_a;
        runtime_.det_dtype = output_dtype_b;
        runtime_.mask_dtype = output_dtype_a;
        det_binding_index = 2;
        mask_binding_index = 1;
    } else {
        det_dims = output_dims_a;
        mask_dims = output_dims_b;
        runtime_.det_dtype = output_dtype_a;
        runtime_.mask_dtype = output_dtype_b;
        det_binding_index = 1;
        mask_binding_index = 2;
    }
#else
    vector<string> output_names;
    const int io_count = engine->getNbIOTensors();
    for (int i = 0; i < io_count; ++i) {
        const char* tensor_name = engine->getIOTensorName(i);
        if (!tensor_name) {
            continue;
        }
        if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name = tensor_name;
        } else if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names.emplace_back(tensor_name);
        }
    }

    if (input_tensor_name.empty() || output_names.size() != 2) {
        throw runtime_error("Failed to identify TensorRT SEG input/output tensors");
    }

    const string& output_name_a = output_names[0];
    const string& output_name_b = output_names[1];

    auto input_dims = engine->getTensorShape(input_tensor_name.c_str());
    auto output_dims_a = engine->getTensorShape(output_name_a.c_str());
    auto output_dims_b = engine->getTensorShape(output_name_b.c_str());
    runtime_.input_dtype = engine->getTensorDataType(input_tensor_name.c_str());
    const auto output_dtype_a = engine->getTensorDataType(output_name_a.c_str());
    const auto output_dtype_b = engine->getTensorDataType(output_name_b.c_str());

    if (output_dims_a.nbDims == 3 && output_dims_b.nbDims == 4) {
        det_tensor_name = output_name_a;
        mask_tensor_name = output_name_b;
        det_dims = output_dims_a;
        mask_dims = output_dims_b;
        runtime_.det_dtype = output_dtype_a;
        runtime_.mask_dtype = output_dtype_b;
    } else if (output_dims_a.nbDims == 4 && output_dims_b.nbDims == 3) {
        det_tensor_name = output_name_b;
        mask_tensor_name = output_name_a;
        det_dims = output_dims_b;
        mask_dims = output_dims_a;
        runtime_.det_dtype = output_dtype_b;
        runtime_.mask_dtype = output_dtype_a;
    } else {
        det_tensor_name = output_name_a;
        mask_tensor_name = output_name_b;
        det_dims = output_dims_a;
        mask_dims = output_dims_b;
        runtime_.det_dtype = output_dtype_a;
        runtime_.mask_dtype = output_dtype_b;
    }

    runtime_.input_h = input_dims.d[2];
    runtime_.input_w = input_dims.d[3];
#endif

    if (mask_dims.nbDims != 4 || det_dims.nbDims != 3) {
        ostringstream oss;
        oss << "Unsupported SEG output rank: det=" << DimsToString(det_dims)
            << ", proto=" << DimsToString(mask_dims);
        throw runtime_error(oss.str());
    }

    {
        vector<int> proto_axes = {mask_dims.d[1], mask_dims.d[2], mask_dims.d[3]};
        sort(proto_axes.begin(), proto_axes.end());
        runtime_.mask_dim = proto_axes[0];
        runtime_.mask_h = proto_axes[1];
        runtime_.mask_w = proto_axes[2];
    }

    if (runtime_.mask_dim <= 0 || runtime_.mask_h <= 0 || runtime_.mask_w <= 0) {
        ostringstream oss;
        oss << "Invalid SEG mask prototype output shape: " << DimsToString(mask_dims);
        throw runtime_error(oss.str());
    }

    const int candidate_a = det_dims.d[1];
    const int candidate_b = det_dims.d[2];
    const int classes_if_attr_a = candidate_a - 4 - runtime_.mask_dim;
    const int classes_if_attr_b = candidate_b - 4 - runtime_.mask_dim;

    if (classes_if_attr_a > 0 && classes_if_attr_b <= 0) {
        runtime_.detection_attribute_size = candidate_a;
        runtime_.num_detections = candidate_b;
    } else if (classes_if_attr_b > 0 && classes_if_attr_a <= 0) {
        runtime_.detection_attribute_size = candidate_b;
        runtime_.num_detections = candidate_a;
    } else if (classes_if_attr_a > 0 && classes_if_attr_b > 0) {
        if (candidate_a < candidate_b) {
            runtime_.detection_attribute_size = candidate_a;
            runtime_.num_detections = candidate_b;
        } else {
            runtime_.detection_attribute_size = candidate_b;
            runtime_.num_detections = candidate_a;
        }
    } else {
        ostringstream oss;
        oss << "Invalid SEG output layout: det=" << DimsToString(det_dims)
            << ", proto=" << DimsToString(mask_dims)
            << ", expected detection_attribute_size = 4 + num_classes + mask_dim";
        throw runtime_error(oss.str());
    }

    runtime_.num_classes = runtime_.detection_attribute_size - 4 - runtime_.mask_dim;
    if (config_.expected_num_classes > 0 && config_.expected_num_classes != runtime_.num_classes) {
        ostringstream oss;
        oss << "Configured num_classes (" << config_.expected_num_classes
            << ") does not match engine output (" << runtime_.num_classes << ")";
        throw runtime_error(oss.str());
    }

    if (config_.class_names.empty()) {
        config_.class_names = buildDefaultSegClassNames(runtime_.num_classes);
    } else if (static_cast<int>(config_.class_names.size()) != runtime_.num_classes) {
        ostringstream oss;
        oss << "Configured class_names size (" << config_.class_names.size()
            << ") does not match num_classes (" << runtime_.num_classes << ")";
        throw runtime_error(oss.str());
    }

    runtime_.det_output_numel =
        static_cast<size_t>(runtime_.detection_attribute_size) * runtime_.num_detections;
    runtime_.mask_output_numel =
        static_cast<size_t>(runtime_.mask_dim) * runtime_.mask_h * runtime_.mask_w;

    printf("YOLOv11_SEG Engine Info:\n");
    printf("  Input: %dx%d\n", runtime_.input_w, runtime_.input_h);
    printf("  Raw Detection Tensor: %s\n", DimsToString(det_dims).c_str());
    printf("  Raw Proto Tensor: %s\n", DimsToString(mask_dims).c_str());
    printf("  Input dtype: %s\n", dataTypeName(runtime_.input_dtype));
    printf("  Detection dtype: %s\n", dataTypeName(runtime_.det_dtype));
    printf("  Proto dtype: %s\n", dataTypeName(runtime_.mask_dtype));
    printf("  Detection Output: %d x %d\n", runtime_.detection_attribute_size, runtime_.num_detections);
    printf("  Proto Output: %d x %d x %d\n", runtime_.mask_dim, runtime_.mask_h, runtime_.mask_w);
    printf("  Num classes: %d\n", runtime_.num_classes);

    cpu_det_output_buffer = new float[runtime_.det_output_numel];
    cpu_mask_output_buffer = new float[runtime_.mask_output_numel];
    cpu_det_raw_buffer = malloc(runtime_.det_output_numel * dataTypeSize(runtime_.det_dtype));
    cpu_mask_raw_buffer = malloc(runtime_.mask_output_numel * dataTypeSize(runtime_.mask_dtype));
    if (!cpu_det_raw_buffer || !cpu_mask_raw_buffer) {
        throw runtime_error("Failed to allocate raw host output buffers");
    }

    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * runtime_.input_w * runtime_.input_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_output_buffers[0], runtime_.det_output_numel * dataTypeSize(runtime_.det_dtype)));
    CUDA_CHECK(cudaMalloc(&gpu_output_buffers[1], runtime_.mask_output_numel * dataTypeSize(runtime_.mask_dtype)));

    bindBuffers();

    CUDA_CHECK(cudaMemset(gpu_buffers[0], 0, 3 * runtime_.input_w * runtime_.input_h * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_output_buffers[0], 0, runtime_.det_output_numel * dataTypeSize(runtime_.det_dtype)));
    CUDA_CHECK(cudaMemset(gpu_output_buffers[1], 0, runtime_.mask_output_numel * dataTypeSize(runtime_.mask_dtype)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    preprocess_initialized_ = true;

    CUDA_CHECK(cudaStreamCreate(&stream));

    if (config_.enable_warmup) {
        for (int i = 0; i < 10; ++i) {
            infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv11_SEG::bindBuffers()
{
#if NV_TENSORRT_MAJOR >= 10
    if (!input_tensor_name.empty()) {
        context->setTensorAddress(input_tensor_name.c_str(), gpu_buffers[0]);
    }
    if (!det_tensor_name.empty()) {
        context->setTensorAddress(det_tensor_name.c_str(), gpu_output_buffers[0]);
    }
    if (!mask_tensor_name.empty()) {
        context->setTensorAddress(mask_tensor_name.c_str(), gpu_output_buffers[1]);
    }
#endif
}

void YOLOv11_SEG::preprocess(Mat& image)
{
    ValidateInputImage(image, MAX_IMAGE_SIZE);

    runtime_.scale_ratio = min(runtime_.input_h / static_cast<float>(image.rows),
                               runtime_.input_w / static_cast<float>(image.cols));
    runtime_.pad_x = (runtime_.input_w - runtime_.scale_ratio * image.cols) * 0.5f;
    runtime_.pad_y = (runtime_.input_h - runtime_.scale_ratio * image.rows) * 0.5f;

    lock_guard<mutex> lock(g_preprocess_mutex_seg);
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], runtime_.input_w, runtime_.input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11_SEG::infer()
{
#if NV_TENSORRT_MAJOR < 10
    void* bindings[3]{};
    bindings[input_binding_index] = gpu_buffers[0];
    bindings[det_binding_index] = gpu_output_buffers[0];
    bindings[mask_binding_index] = gpu_output_buffers[1];
    context->enqueueV2(bindings, stream, nullptr);
#else
    context->enqueueV3(stream);
#endif
}

Mat YOLOv11_SEG::decodeMask(const vector<float>& coeffs, const Rect& box, int img_w, int img_h) const
{
    Mat protos(runtime_.mask_dim, runtime_.mask_h * runtime_.mask_w, CV_32F, cpu_mask_output_buffer);
    Mat coeff_mat(1, runtime_.mask_dim, CV_32F, const_cast<float*>(coeffs.data()));
    Mat mask_flat = coeff_mat * protos;
    Mat mask = mask_flat.reshape(1, runtime_.mask_h);

    Mat mask_sigmoid;
    exp(-mask, mask_sigmoid);
    mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);

    const float proto_to_input_x = runtime_.input_w / static_cast<float>(runtime_.mask_w);
    const float proto_to_input_y = runtime_.input_h / static_cast<float>(runtime_.mask_h);

    const int x0 = max(0, min(runtime_.mask_w, static_cast<int>(floor((box.x * runtime_.scale_ratio + runtime_.pad_x) / proto_to_input_x))));
    const int y0 = max(0, min(runtime_.mask_h, static_cast<int>(floor((box.y * runtime_.scale_ratio + runtime_.pad_y) / proto_to_input_y))));
    const int x1 = max(0, min(runtime_.mask_w, static_cast<int>(ceil(((box.x + box.width) * runtime_.scale_ratio + runtime_.pad_x) / proto_to_input_x))));
    const int y1 = max(0, min(runtime_.mask_h, static_cast<int>(ceil(((box.y + box.height) * runtime_.scale_ratio + runtime_.pad_y) / proto_to_input_y))));

    if (x1 <= x0 || y1 <= y0) {
        return Mat::zeros(img_h, img_w, CV_8U);
    }

    Mat crop = mask_sigmoid(Range(y0, y1), Range(x0, x1)).clone();
    if (crop.empty()) {
        return Mat::zeros(img_h, img_w, CV_8U);
    }

    Rect clipped_box = box & Rect(0, 0, img_w, img_h);
    if (clipped_box.width <= 0 || clipped_box.height <= 0) {
        return Mat::zeros(img_h, img_w, CV_8U);
    }

    Mat resized_crop;
    resize(crop, resized_crop, clipped_box.size(), 0.0, 0.0, INTER_LINEAR);

    Mat binary_crop;
    threshold(resized_crop, binary_crop, config_.mask_threshold, 255.0, THRESH_BINARY);
    binary_crop.convertTo(binary_crop, CV_8U);

    Mat full_mask = Mat::zeros(img_h, img_w, CV_8U);
    binary_crop.copyTo(full_mask(clipped_box));
    return full_mask;
}

void YOLOv11_SEG::postprocess(vector<SegDetection>& output, int img_w, int img_h)
{
    CUDA_CHECK(cudaMemcpyAsync(cpu_det_raw_buffer, gpu_output_buffers[0],
                               runtime_.det_output_numel * dataTypeSize(runtime_.det_dtype), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu_mask_raw_buffer, gpu_output_buffers[1],
                               runtime_.mask_output_numel * dataTypeSize(runtime_.mask_dtype), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    convertBufferToFloat(cpu_det_raw_buffer, runtime_.det_dtype, cpu_det_output_buffer, runtime_.det_output_numel);
    convertBufferToFloat(cpu_mask_raw_buffer, runtime_.mask_dtype, cpu_mask_output_buffer, runtime_.mask_output_numel);

    output.clear();

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    vector<vector<float>> mask_coeffs;

    const Mat det_output(runtime_.detection_attribute_size, runtime_.num_detections, CV_32F, cpu_det_output_buffer);
    const float inv_scale = runtime_.scale_ratio > 0.0f ? (1.0f / runtime_.scale_ratio) : 0.0f;

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + runtime_.num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        float conf = static_cast<float>(score);

        float cx = det_output.at<float>(0, i);
        float cy = det_output.at<float>(1, i);
        float ow = det_output.at<float>(2, i);
        float oh = det_output.at<float>(3, i);

        if (conf <= config_.conf_threshold) {
            continue;
        }

        float left = (cx - 0.5f * ow - runtime_.pad_x) * inv_scale;
        float top = (cy - 0.5f * oh - runtime_.pad_y) * inv_scale;
        float right = (cx + 0.5f * ow - runtime_.pad_x) * inv_scale;
        float bottom = (cy + 0.5f * oh - runtime_.pad_y) * inv_scale;

        left = clamp(left, 0.0f, static_cast<float>(img_w - 1));
        top = clamp(top, 0.0f, static_cast<float>(img_h - 1));
        right = clamp(right, 0.0f, static_cast<float>(img_w - 1));
        bottom = clamp(bottom, 0.0f, static_cast<float>(img_h - 1));

        Rect box(static_cast<int>(left),
                 static_cast<int>(top),
                 static_cast<int>(right - left),
                 static_cast<int>(bottom - top));

        if (box.width <= 1 || box.height <= 1) {
            continue;
        }

        vector<float> coeff(runtime_.mask_dim);
        for (int j = 0; j < runtime_.mask_dim; ++j) {
            coeff[j] = det_output.at<float>(4 + runtime_.num_classes + j, i);
        }

        boxes.push_back(box);
        class_ids.push_back(class_id_point.y);
        confidences.push_back(conf);
        mask_coeffs.push_back(std::move(coeff));
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, config_.conf_threshold, config_.nms_threshold, nms_result);

    for (int idx : nms_result) {
        SegDetection result;
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        result.mask = decodeMask(mask_coeffs[idx], result.bbox, img_w, img_h);
        output.push_back(std::move(result));
    }
}

void YOLOv11_SEG::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    BuildTensorRTFromOnnx(onnxPath, logger, config_.use_fp16, runtime, engine, context);
}

bool YOLOv11_SEG::saveEngine(const std::string& onnxpath)
{
    return SaveTensorRTEngine(engine, onnxpath);
}

void YOLOv11_SEG::cleanup() noexcept
{
    SafeDestroyCudaStream(stream);

    SafeCudaFree(gpu_buffers[0]);
    for (void*& buffer : gpu_output_buffers) {
        SafeCudaFree(buffer);
    }

    SafeDeleteArray(cpu_det_output_buffer);
    SafeDeleteArray(cpu_mask_output_buffer);
    SafeFreeHostBuffer(cpu_det_raw_buffer);
    SafeFreeHostBuffer(cpu_mask_raw_buffer);
    SafeDestroyPreprocess(preprocess_initialized_);
    SafeDestroyTensorRT(runtime, engine, context);
}

void YOLOv11_SEG::draw(Mat& image, const vector<SegDetection>& output, const string& output_path)
{
    Mat overlay = image.clone();

    for (const auto& detection : output) {
        const int class_id = detection.class_id;
        const Scalar color = class_id < static_cast<int>(COLORS.size())
            ? Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2])
            : Scalar(0, 255, 0);

        if (!detection.mask.empty()) {
            overlay.setTo(color, detection.mask);
        }
    }

    addWeighted(overlay, config_.mask_alpha, image, 1.0f - config_.mask_alpha, 0.0, image);

    for (const auto& detection : output) {
        const int class_id = detection.class_id;
        const Scalar color = class_id < static_cast<int>(COLORS.size())
            ? Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2])
            : Scalar(0, 255, 0);

        rectangle(image, detection.bbox, color, 2);

        const string& class_name =
            (class_id >= 0 && class_id < static_cast<int>(config_.class_names.size()))
                ? config_.class_names[class_id]
                : string("unknown");
        string label = class_name + " " + to_string(detection.conf).substr(0, 4);
        Size text_size = getTextSize(label, FONT_HERSHEY_DUPLEX, 0.8, 1, 0);
        Rect text_rect(detection.bbox.x,
                       max(0, detection.bbox.y - text_size.height - 12),
                       text_size.width + 10,
                       text_size.height + 10);
        rectangle(image, text_rect, color, FILLED);
        putText(image, label,
                Point(text_rect.x + 5, text_rect.y + text_rect.height - 4),
                FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 0), 1, 0);
    }

    if (!output_path.empty()) {
        imwrite(output_path, image);
        printf("Seg result saved to %s\n", output_path.c_str());
    }
}
