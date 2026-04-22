#pragma once

#include "common.h"
#include "NvInfer.h"

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

inline std::string DimsToString(const nvinfer1::Dims& dims)
{
    std::string result = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(dims.d[i]);
    }
    result += "]";
    return result;
}

inline std::vector<std::string> BuildDefaultClassNames(const std::vector<std::string>& default_names,
                                                       int num_classes)
{
    if (num_classes == static_cast<int>(default_names.size())) {
        return default_names;
    }

    std::vector<std::string> names;
    names.reserve(std::max(num_classes, 0));
    for (int i = 0; i < num_classes; ++i) {
        names.push_back("class_" + std::to_string(i));
    }
    return names;
}

inline std::vector<std::string> BuildCocoClassNames(int num_classes)
{
    return BuildDefaultClassNames(CLASS_NAMES, num_classes);
}

inline void ValidateInputImage(cv::Mat& image, int max_image_size)
{
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (!image.isContinuous()) {
        image = image.clone();
    }
    if (image.cols * image.rows > max_image_size) {
        throw std::invalid_argument("Input image is larger than preprocess buffer limit");
    }
}

inline cv::Scalar GetClassColor(int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(COLORS.size())) {
        return cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);
    }
    return cv::Scalar(0, 255, 0);
}

inline std::string GetClassName(const std::vector<std::string>& class_names, int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(class_names.size())) {
        return class_names[class_id];
    }
    return "class_" + std::to_string(class_id);
}
