#pragma once

#include "NvInfer.h"

#include <string>
#include <vector>

bool StartsWith(const std::string& value, const std::string& prefix);
std::vector<std::string> LoadClassNames(const std::string& path);
bool IsPathExist(const std::string& path);
bool IsFile(const std::string& path);
std::vector<std::string> CollectImagePaths(const std::string& path, bool& isVideo);

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};
