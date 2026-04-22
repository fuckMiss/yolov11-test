#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "app_utils.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

using namespace std;
using namespace cv;

bool StartsWith(const string& value, const string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

vector<string> LoadClassNames(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Failed to open labels file: " + path);
    }

    vector<string> names;
    string line;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}

bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

vector<string> CollectImagePaths(const string& path, bool& isVideo) {
    vector<string> image_paths;
    isVideo = false;

    if (IsFile(path)) {
        string suffix = path.substr(path.find_last_of('.') + 1);
        transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png" || suffix == "bmp") {
            image_paths.push_back(path);
        } else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" ||
                   suffix == "mov" || suffix == "mkv" || suffix == "webm") {
            isVideo = true;
        } else {
            throw invalid_argument("Unsupported file suffix: " + suffix);
        }
        return image_paths;
    }

    if (!IsPathExist(path)) {
        throw invalid_argument("Path does not exist: " + path);
    }

    vector<string> jpg_files;
    vector<string> jpeg_files;
    vector<string> png_files;
    vector<string> bmp_files;
    glob(path + "/*.jpg", jpg_files, false);
    glob(path + "/*.jpeg", jpeg_files, false);
    glob(path + "/*.png", png_files, false);
    glob(path + "/*.bmp", bmp_files, false);
    image_paths.insert(image_paths.end(), jpg_files.begin(), jpg_files.end());
    image_paths.insert(image_paths.end(), jpeg_files.begin(), jpeg_files.end());
    image_paths.insert(image_paths.end(), png_files.begin(), png_files.end());
    image_paths.insert(image_paths.end(), bmp_files.begin(), bmp_files.end());
    return image_paths;
}

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}
