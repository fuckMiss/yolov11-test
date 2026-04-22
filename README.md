# YOLOv11 TensorRT

这是一个基于 TensorRT 的 YOLOv11 C++ 推理项目，目前已经拆成两层：

- 部署层：`DET / OBB / SEG` 各自独立推理
- 应用层：多模型组合调用、额外后处理、项目侧逻辑

如果后续你要继续把它做成完整项目，建议一直保持这个分层。

## 文档导航

按用途分别看：

- `README.md`
  项目总览与入口导航
- `README_OBB.md`
  `OBB` 部署与推理说明
- `README_SEG.md`
  `SEG` 部署与推理说明
- `README_APP.md`
  应用层组合调用与后处理说明

## 当前可执行入口

### 1. DET

文件：

- `main.cpp`

目标：

- `yolov11-tensorrt`

用途：

- 标准目标检测

### 2. OBB

文件：

- `main_obb.cpp`

目标：

- `yolov11-tensorrt_obb`

用途：

- 标准旋转框检测

### 3. SEG

文件：

- `main_seg.cpp`

目标：

- `yolov11-tensorrt_seg`

用途：

- 标准实例分割

### 4. OBB Angle Demo

文件：

- `main_obb_angle.cpp`

目标：

- `yolov11-tensorrt_obb_angle`

用途：

- 正常 `OBB` 推理后，额外做角度箭头和角度打印
- 这是应用层示例，不是底层部署入口

### 5. OBB + SEG Demo

文件：

- `main_obb_seg.cpp`

目标：

- `yolov11-tensorrt_obb_seg`

用途：

- 同一张图串联 `OBB` 和 `SEG`
- 用于项目层组合调用示例

## 项目结构

```text
yolov11-tensorrt/
├── asset/
├── build/
├── src/
│   ├── YOLOv11.cpp
│   ├── YOLOv11.h
│   ├── YOLOv11_OBB.cpp
│   ├── YOLOv11_OBB.h
│   ├── YOLOv11_SEG.cpp
│   ├── YOLOv11_SEG.h
│   ├── app_utils.cpp
│   ├── app_utils.h
│   ├── app_postprocess.cpp
│   └── app_postprocess.h
├── main.cpp
├── main_obb.cpp
├── main_obb_angle.cpp
├── main_obb_seg.cpp
├── main_seg.cpp
├── README.md
├── README_OBB.md
├── README_SEG.md
└── README_APP.md
```

## 推荐理解方式

### 部署层

这些文件尽量保持干净稳定：

- `src/YOLOv11.cpp`
- `src/YOLOv11_OBB.cpp`
- `src/YOLOv11_SEG.cpp`

它们主要负责：

- `preprocess`
- `infer`
- `postprocess`
- 基础结果绘制

### 应用层

这些文件用于写项目自己的业务逻辑：

- `main_obb_angle.cpp`
- `main_obb_seg.cpp`
- `src/app_postprocess.cpp`

它们主要负责：

- 额外可视化
- 多模型组合
- 业务后处理
- 项目逻辑验证

## 基本构建

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

如果只想单独编译某个目标：

```bash
cmake --build . --target yolov11-tensorrt --config Release
cmake --build . --target yolov11-tensorrt_obb --config Release
cmake --build . --target yolov11-tensorrt_seg --config Release
cmake --build . --target yolov11-tensorrt_obb_angle --config Release
cmake --build . --target yolov11-tensorrt_obb_seg --config Release
```

## 快速开始

### 1. DET

```bash
./build/yolov11-tensorrt ./weights/yolo11s_trt.engine ./asset/bus.jpg
```

### 2. OBB

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./asset/boats.jpg
```

### 3. SEG

```bash
./build/yolov11-tensorrt_seg ./weights/yolo11s-seg.engine ./asset/bus.jpg
```

### 4. OBB 角度应用层示例

```bash
./build/yolov11-tensorrt_obb_angle ./weights/best_obb.engine ./asset/boats.jpg
```

### 5. OBB + SEG 组合示例

```bash
./build/yolov11-tensorrt_obb_seg ./weights/best_obb.engine ./weights/yolo11s-seg.engine ./asset/boats.jpg
```

## 后续扩展建议

如果你后面还要继续做项目功能，推荐保持下面这个原则：

- 模型部署问题，改各自的 `YOLOv11*.cpp`
- 项目业务逻辑，写在应用层
- 不要把业务后处理继续堆进底层部署代码

比如以后继续加这些都适合放在应用层：

- `OBB` 角度业务判断
- `SEG` mask 面积统计
- `SEG` 轮廓提取
- `OBB + SEG` 多阶段逻辑
- 最终项目工作流封装

## 说明

如果你现在的目标是：

- 只跑模型
  看对应的部署 README
- 做项目级联动
  看 `README_APP.md`

这样文档之间不会互相打扰，后面维护也更清楚。
