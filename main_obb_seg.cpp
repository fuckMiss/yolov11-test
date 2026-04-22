#include <chrono>
#include <iostream>
#include <string>

#include "YOLOv11_OBB.h"
#include "YOLOv11_SEG.h"
#include "app_utils.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " <obb_engine_or_onnx> <seg_engine_or_onnx> <image>"
             << " [--output=PATH]"
             << " [--obb-num-classes=N] [--obb-conf=T] [--obb-nms=T] [--obb-labels=PATH] [--obb-fp16] [--no-obb-warmup]"
             << " [--seg-num-classes=N] [--seg-conf=T] [--seg-nms=T] [--seg-mask-thres=T] [--seg-alpha=T] [--seg-labels=PATH] [--seg-fp16] [--no-seg-warmup]"
             << endl;
        return -1;
    }

    const string obb_model_path{ argv[1] };
    const string seg_model_path{ argv[2] };
    const string image_path{ argv[3] };

    OBBConfig obb_config;
    SEGConfig seg_config;
    string output_path;
    TrtLogger logger;

    for (int i = 4; i < argc; ++i) {
        const string arg = argv[i];
        if (StartsWith(arg, "--output=")) {
            output_path = arg.substr(string("--output=").size());
        } else if (StartsWith(arg, "--obb-num-classes=")) {
            obb_config.expected_num_classes = stoi(arg.substr(string("--obb-num-classes=").size()));
        } else if (StartsWith(arg, "--obb-conf=")) {
            obb_config.conf_threshold = stof(arg.substr(string("--obb-conf=").size()));
        } else if (StartsWith(arg, "--obb-nms=")) {
            obb_config.nms_threshold = stof(arg.substr(string("--obb-nms=").size()));
        } else if (StartsWith(arg, "--obb-labels=")) {
            obb_config.class_names = LoadClassNames(arg.substr(string("--obb-labels=").size()));
        } else if (arg == "--obb-fp16") {
            obb_config.use_fp16 = true;
        } else if (arg == "--no-obb-warmup") {
            obb_config.enable_warmup = false;
        } else if (StartsWith(arg, "--seg-num-classes=")) {
            seg_config.expected_num_classes = stoi(arg.substr(string("--seg-num-classes=").size()));
        } else if (StartsWith(arg, "--seg-conf=")) {
            seg_config.conf_threshold = stof(arg.substr(string("--seg-conf=").size()));
        } else if (StartsWith(arg, "--seg-nms=")) {
            seg_config.nms_threshold = stof(arg.substr(string("--seg-nms=").size()));
        } else if (StartsWith(arg, "--seg-mask-thres=")) {
            seg_config.mask_threshold = stof(arg.substr(string("--seg-mask-thres=").size()));
        } else if (StartsWith(arg, "--seg-alpha=")) {
            seg_config.mask_alpha = stof(arg.substr(string("--seg-alpha=").size()));
        } else if (StartsWith(arg, "--seg-labels=")) {
            seg_config.class_names = LoadClassNames(arg.substr(string("--seg-labels=").size()));
        } else if (arg == "--seg-fp16") {
            seg_config.use_fp16 = true;
        } else if (arg == "--no-seg-warmup") {
            seg_config.enable_warmup = false;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            return -1;
        }
    }

    if (!IsFile(image_path)) {
        cerr << "Only single-image input is supported: " << image_path << endl;
        return -1;
    }

    Mat image = imread(image_path);
    if (image.empty()) {
        cerr << "Error reading image: " << image_path << endl;
        return -1;
    }

    if (output_path.empty()) {
        output_path = "obb_seg_result_" + image_path.substr(image_path.find_last_of("/\\") + 1);
    }

    YOLOv11_OBB obb_model(obb_model_path, logger, obb_config);
    YOLOv11_SEG seg_model(seg_model_path, logger, seg_config);

    vector<OBBDetection> obb_objects;
    vector<SegDetection> seg_objects;

    obb_model.preprocess(image);
    auto obb_start = chrono::system_clock::now();
    obb_model.infer();
    auto obb_end = chrono::system_clock::now();
    obb_model.postprocess(obb_objects, image.cols, image.rows);

    seg_model.preprocess(image);
    auto seg_start = chrono::system_clock::now();
    seg_model.infer();
    auto seg_end = chrono::system_clock::now();
    seg_model.postprocess(seg_objects, image.cols, image.rows);

    Mat result = image.clone();
    obb_model.draw(result, obb_objects, "");
    seg_model.draw(result, seg_objects, output_path);

    const double obb_ms = static_cast<double>(
        chrono::duration_cast<chrono::microseconds>(obb_end - obb_start).count()) / 1000.0;
    const double seg_ms = static_cast<double>(
        chrono::duration_cast<chrono::microseconds>(seg_end - seg_start).count()) / 1000.0;

    printf("OBB detections: %zu, cost %2.4lf ms\n", obb_objects.size(), obb_ms);
    printf("SEG detections: %zu, cost %2.4lf ms\n", seg_objects.size(), seg_ms);
    printf("Combined result saved to %s\n", output_path.c_str());

    imshow("OBB + SEG Result", result);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
