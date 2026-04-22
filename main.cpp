#include <chrono>
#include <iostream>
#include <string>

#include "YOLOv11.h"
#include "app_utils.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <engine_file> <image/video/folder>" << endl;
        return -1;
    }

    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool isVideo{ false };
    TrtLogger logger;
    try {
        imagePathList = CollectImagePaths(path, isVideo);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    YOLOv11 model(engine_file_path, logger);

    if (isVideo) {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            cerr << "Failed to open video: " << path << endl;
            return -1;
        }

        while (1) {
            Mat image;
            cap >> image;
            if (image.empty()) {
                break;
            }

            vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);

            auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
            printf("cost %2.4lf ms\n", tc);

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

            vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);

            auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
            printf("%s cost %2.4lf ms\n", imagePath.c_str(), tc);

            imshow("Result", image);
            waitKey(0);
        }
    }

    return 0;
}
