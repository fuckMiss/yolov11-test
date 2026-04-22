#include "app_postprocess.h"

#include "model_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using namespace cv;
using namespace std;

void DrawOBBAngleOverlay(Mat& image,
                         const vector<OBBDetection>& output,
                         const vector<string>& class_names,
                         bool print_angle)
{
    for (size_t i = 0; i < output.size(); ++i) {
        const auto& detection = output[i];
        const auto& rrect = detection.rotated_rect;
        const int class_id = detection.class_id;
        const float conf = detection.conf;
        const float angle_deg = rrect.angle;
        const float angle_rad = angle_deg * static_cast<float>(CV_PI) / 180.0f;
        const Scalar color = GetClassColor(class_id);
        const string class_name = GetClassName(class_names, class_id);

        const float arrow_length = std::max(rrect.size.width, rrect.size.height) * 0.5f;
        const Point2f center = rrect.center;
        const Point2f arrow_tip(center.x + std::cos(angle_rad) * arrow_length,
                                center.y + std::sin(angle_rad) * arrow_length);

        arrowedLine(image, center, arrow_tip, color, 2, LINE_AA, 0, 0.25);
        circle(image, center, 3, color, FILLED);

        char angle_buffer[32];
        snprintf(angle_buffer, sizeof(angle_buffer), "%.1fdeg", angle_deg);
        string label = class_name + " " + to_string(conf).substr(0, 4) + " " + angle_buffer;
        int baseLine = 0;
        Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        Point label_origin(static_cast<int>(center.x), static_cast<int>(center.y) - 10);

        if (label_origin.y < label_size.height + 5) {
            label_origin.y = static_cast<int>(center.y) + label_size.height + 10;
        }
        if (label_origin.x < 0) {
            label_origin.x = 0;
        }
        if (label_origin.x + label_size.width > image.cols) {
            label_origin.x = std::max(0, image.cols - label_size.width);
        }
        if (label_origin.y >= image.rows) {
            label_origin.y = std::max(label_size.height + 5, image.rows - 1);
        }

        rectangle(image,
                  Point(label_origin.x, label_origin.y - label_size.height - 5),
                  Point(label_origin.x + label_size.width, label_origin.y + baseLine),
                  color, FILLED);
        putText(image, label, Point(label_origin.x, label_origin.y),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

        if (print_angle) {
            printf("OBB[%zu] conf=%.4f angle=%.2f deg center=(%.1f, %.1f)\n",
                   i, conf, angle_deg, center.x, center.y);
        }
    }
}
