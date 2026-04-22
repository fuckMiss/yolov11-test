cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find and include OpenCV
set(OpenCV_DIR "/home/cll/下载/opencv/build" CACHE PATH "OpenCV config directory")
set(OPENCV_ROOT "/home/cll/下载/opencv")
set(OPENCV_LIB_DIR "${OpenCV_DIR}/lib")

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "/opt/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")

# Include directories
file(GLOB OPENCV_MODULE_INCLUDE_DIRS "${OPENCV_ROOT}/modules/*/include")
set(PROJECT_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_DIR}
    ${OPENCV_ROOT}/include
    ${OPENCV_MODULE_INCLUDE_DIRS}
)

set(OPENCV_MODULES
    gapi
    highgui
    ml
    objdetect
    photo
    stitching
    video
    calib3d
    features2d
    dnn
    flann
    videoio
    imgcodecs
    imgproc
    core
)
set(OpenCV_LIBS)
foreach(module IN LISTS OPENCV_MODULES)
    list(APPEND OpenCV_LIBS "${OPENCV_LIB_DIR}/libopencv_${module}.so")
endforeach()

set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Add CUDA source files
set(CUDA_SOURCES src/preprocess.cu)
set(COMMON_SOURCES
    src/app_utils.cpp
    src/app_postprocess.cpp
    src/trt_utils.cpp
)
set(COMMON_HEADERS
    src/app_utils.h
    src/app_postprocess.h
    src/macros.h
    src/logging.h
    src/model_utils.h
    src/runtime_utils.h
    src/trt_utils.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
)

function(add_yolov11_target target_name main_source model_source model_header)
    cuda_add_executable(${target_name}
        ${main_source}
        ${model_source}
        ${COMMON_SOURCES}
        ${model_header}
        ${COMMON_HEADERS}
        ${CUDA_SOURCES}
    )

    target_compile_definitions(${target_name} PRIVATE API_EXPORTS)
    target_include_directories(${target_name} PRIVATE ${PROJECT_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${TENSORRT_LIB_DIR})
    target_link_libraries(${target_name}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBS}
    )
endfunction()

# ========== Standard YOLOv11 detection executable ==========
add_yolov11_target(${PROJECT_NAME}
    main.cpp
    src/YOLOv11.cpp
    src/YOLOv11.h
)

# ========== YOLOv11 OBB detection executable ==========
add_yolov11_target(${PROJECT_NAME}_obb
    main_obb.cpp
    src/YOLOv11_OBB.cpp
    src/YOLOv11_OBB.h
)

# ========== YOLOv11 OBB angle demo executable ==========
add_yolov11_target(${PROJECT_NAME}_obb_angle
    main_obb_angle.cpp
    src/YOLOv11_OBB.cpp
    src/YOLOv11_OBB.h
)

# ========== YOLOv11 SEG detection executable ==========
add_yolov11_target(${PROJECT_NAME}_seg
    main_seg.cpp
    src/YOLOv11_SEG.cpp
    src/YOLOv11_SEG.h
)

# ========== YOLOv11 OBB + SEG combined executable ==========
cuda_add_executable(${PROJECT_NAME}_obb_seg
    main_obb_seg.cpp
    src/YOLOv11_OBB.cpp
    src/YOLOv11_OBB.h
    src/YOLOv11_SEG.cpp
    src/YOLOv11_SEG.h
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
    ${CUDA_SOURCES}
)

target_compile_definitions(${PROJECT_NAME}_obb_seg PRIVATE API_EXPORTS)
target_include_directories(${PROJECT_NAME}_obb_seg PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME}_obb_seg PRIVATE ${TENSORRT_LIB_DIR})
target_link_libraries(${PROJECT_NAME}_obb_seg
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)


# /opt/TensorRT-8.6.1.6/bin/trtexec --onnx=./weights/yolo11s-seg.onnx --saveEngine=./weights/yolo11s-seg.engine --fp16
