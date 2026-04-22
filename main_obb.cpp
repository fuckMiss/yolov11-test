#include <chrono>
#include <iostream>
#include <string>

#include "YOLOv11_OBB.h"
#include "app_utils.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " <engine_file> <image/video/folder> [--num-classes=N] [--conf=T] [--nms=T] [--labels=PATH] [--fp16] [--no-warmup]"
             << endl;
        return -1;
    }

    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    OBBConfig config;
    TrtLogger logger;

    for (int i = 3; i < argc; ++i) {
        string arg = argv[i];
        if (StartsWith(arg, "--num-classes=")) {
            config.expected_num_classes = stoi(arg.substr(string("--num-classes=").size()));
        } else if (StartsWith(arg, "--conf=")) {
            config.conf_threshold = stof(arg.substr(string("--conf=").size()));
        } else if (StartsWith(arg, "--nms=")) {
            config.nms_threshold = stof(arg.substr(string("--nms=").size()));
        } else if (StartsWith(arg, "--labels=")) {
            config.class_names = LoadClassNames(arg.substr(string("--labels=").size()));
        } else if (arg == "--fp16") {
            config.use_fp16 = true;
        } else if (arg == "--no-warmup") {
            config.enable_warmup = false;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            return -1;
        }
    }

    vector<string> imagePathList;
    bool isVideo{ false };
    try {
        imagePathList = CollectImagePaths(path, isVideo);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    YOLOv11_OBB model(engine_file_path, logger, config);

    if (isVideo) {
        VideoCapture cap(path);
        if (!cap.isOpened()) {
            cerr << "Failed to open video: " << path << endl;
            return -1;
        }

        int frame_count = 0;
        while (1) {
            Mat image;
            cap >> image;
            if (image.empty()) {
                break;
            }

            vector<OBBDetection> objects;
            model.preprocess(image);

            auto start = chrono::system_clock::now();
            model.infer();
            auto end = chrono::system_clock::now();

            model.postprocess(objects, image.cols, image.rows);
            model.draw(image, objects, "");

            auto tc = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0;
            printf("Frame %d cost %2.4lf ms\n", frame_count++, tc);

            imshow("prediction", image);
            if (waitKey(1) == 27) {
                break;
            }
        }

        destroyAllWindows();
        cap.release();
    } else {
        if (imagePathList.empty()) {
            cerr << "No images found in: " << path << endl;
            return -1;
        }

        for (const auto& imagePath : imagePathList) {
            Mat image = imread(imagePath);
            if (image.empty()) {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            vector<OBBDetection> objects;
            model.preprocess(image);

            auto start = chrono::system_clock::now();
            model.infer();
            auto end = chrono::system_clock::now();

            model.postprocess(objects, image.cols, image.rows);

            string output_path = "obb_result_" + imagePath.substr(imagePath.find_last_of("/\\") + 1);
            model.draw(image, objects, output_path);

            auto tc = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0;
            printf("%s cost %2.4lf ms\n", imagePath.c_str(), tc);

            imshow("Result", image);
            waitKey(0);
        }
    }

    return 0;
}
