#pragma once

#include "YOLOv11_OBB.h"

#include <opencv2/opencv.hpp>
#include <vector>

void DrawOBBAngleOverlay(cv::Mat& image,
                         const std::vector<OBBDetection>& output,
                         const std::vector<std::string>& class_names,
                         bool print_angle = true);
