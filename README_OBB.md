# YOLOv11 OBB TensorRT 推理指南

本文档只说明 `OBB` 模型的部署、推理和类别名配置。

如果你要看“应用层组合调用”或“额外后处理示例”，请单独查看：

- `README_APP.md`

## 项目结构

```text
yolov11-tensorrt/
├── asset/                # 测试资源与可选标签文件
├── build/                # 构建目录
├── src/
│   ├── YOLOv11_OBB.cpp   # OBB 模型实现
│   └── YOLOv11_OBB.h     # OBB 模型头文件
├── CMakeLists.txt        # CMake 配置
├── export.py             # 模型导出脚本
├── main_obb.cpp          # OBB 推理主程序
└── README_OBB.md         # 本文档
```

## 环境要求

- 操作系统：Linux 或 Windows
- CUDA：11.6 或更高版本
- TensorRT：8.6 或更高版本
- OpenCV：4.0 或更高版本
- Python：3.10 或更高版本
- `ultralytics`：用于导出 YOLOv11 OBB 模型

## 安装与构建

### 1. 克隆仓库

```bash
git clone https://github.com/spacewalk01/yolov11-tensorrt.git
cd yolov11-tensorrt
```

### 2. 安装 Python 依赖

```bash
pip install --upgrade ultralytics
```

### 3. 配置 C++ 依赖

确保 CUDA、TensorRT、OpenCV 已正确安装，并在 `CMakeLists.txt` 中设置好本机路径。

### 4. 构建 OBB 可执行文件

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
cmake --build . --target yolov11-tensorrt_obb --config Release
```

`--target yolov11-tensorrt_obb` 表示只编译 OBB 推理目标。

## 基本使用

### 1. 导出 OBB 模型

可以修改 `export.py` 导出自己的 OBB 模型，也可以直接使用：

```python
from ultralytics import YOLO

model = YOLO("weights/best_obb.pt")
model.export(format="onnx")
```

导出后通常会得到：

```text
weights/best_obb.onnx
```

### 2. 从 ONNX 现场构建 TensorRT engine

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.onnx ""
```

执行后会生成对应的 `.engine` 文件。

### 3. 运行推理

单张图片：

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./asset/boats.jpg
```

视频：

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./video.mp4
```

文件夹：

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./images/
```

图片和文件夹模式会输出 `obb_result_xxx.jpg`；视频模式实时显示结果，不逐帧写盘。

## 运行参数

当前 OBB 入口支持以下参数，这些参数都会真正参与推理流程：

```bash
./build/yolov11-tensorrt_obb <engine_or_onnx> <image/video/folder> \
  [--num-classes=N] \
  [--conf=T] \
  [--nms=T] \
  [--labels=PATH] \
  [--fp16] \
  [--no-warmup]
```

- `--num-classes=N`
  用于校验模型输出类别数是否符合预期。如果与引擎真实输出不一致，程序会直接报错。
- `--conf=T`
  设置后处理置信度阈值，例如 `--conf=0.25`。
- `--nms=T`
  设置旋转框 NMS 阈值，例如 `--nms=0.30`。
- `--labels=PATH`
  从文本文件加载类别名，每行一个类别。只要传了这个参数，就会覆盖源码里的默认类别名。
- `--fp16`
  仅在输入为 `.onnx`、需要现场构建 TensorRT 引擎时生效。
- `--no-warmup`
  关闭默认的 10 次 warmup。

示例：

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./asset/boats.jpg \
  --num-classes=1 \
  --conf=0.30 \
  --nms=0.25 \
  --labels=./asset/border.labels.txt \
  --no-warmup
```

## 换成自己的 OBB 模型

### 1. 准备自己的 `.pt` 模型

例如：

```text
weights/best_obb.pt
```

注意必须是 Ultralytics 的 `obb` 模型，不要把普通检测 `det` 或分割 `seg` 模型拿来这里用。

### 2. 导出成 ONNX

```python
from ultralytics import YOLO

model = YOLO("weights/best_obb.pt")
model.export(format="onnx")
```

### 3. 生成 engine

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.onnx ""
```

### 4. 运行自己的模型

如果你是单类模型：

```bash
./build/yolov11-tensorrt_obb ./weights/best_obb.engine ./asset/boats.jpg \
  --num-classes=1 \
  --labels=./asset/border.labels.txt
```

`asset/border.labels.txt` 例如：

```text
border
```

如果你是多类模型，例如 3 类：

```bash
./build/yolov11-tensorrt_obb ./weights/your.engine ./asset/your.jpg \
  --num-classes=3 \
  --labels=./asset/your.labels.txt
```

## 类别名配置

`OBB` 支持两种设置类别名的方式，优先级如下：

1. `--labels=PATH`
2. `src/YOLOv11_OBB.cpp` 里的 `OBB_CLASS_NAMES`
3. 自动生成 `class_0`、`class_1`、`class_2` ...

### 方式 1：命令行传 `--labels`

准备一个文本文件，每行一个类别，例如：

```text
plane
ship
harbor
```

然后运行：

```bash
./build/yolov11-tensorrt_obb ./weights/your.engine ./asset/your.jpg \
  --num-classes=3 \
  --labels=./asset/your.labels.txt
```

### 方式 2：直接在代码里写死默认类别名

如果你以后都固定使用自己的类别名，不想每次都传 `--labels`，可以直接修改：

```cpp
// src/YOLOv11_OBB.cpp
static const std::vector<std::string> OBB_CLASS_NAMES = {
    "border"
};
```

注意：

- 这里的类别数量必须和模型真实类别数一致
- 如果后面又传了 `--labels=PATH`，则以 `--labels` 为准
- 如果 `OBB_CLASS_NAMES` 数量和模型类别数不一致，程序会退回到 `class_0/class_1/...`

## OBB 输出说明

OBB 模型输出的核心信息包括：

1. 目标中心点 `cx, cy`
2. 宽高 `w, h`
3. 类别分数
4. 旋转角度

当前实现会完成以下流程：

1. 解析每个候选框的 `cx + cy + w + h + class score + angle`
2. 结合缩放和 padding 把坐标映射回原图
3. 生成旋转框四个角点
4. 通过旋转框 NMS 过滤重复目标
5. 绘制旋转框、类别名和置信度

## 代码配置说明

`src/YOLOv11_OBB.h` 里有两类字段：

- `OBBConfig`
  这是可配置参数，`--num-classes`、`--conf`、`--nms`、`--labels`、`--fp16`、`--no-warmup` 都会真正参与推理流程。
- `OBBRuntimeState`
  这是运行时状态，例如输入尺寸、输出张量尺寸、缩放比例、padding 等，程序会自动计算，不需要手工修改。

## 常见问题

### 1. 类别名显示不对

检查：

- `--labels` 文件行数是否与模型真实类别数一致
- `--num-classes` 是否与模型导出类别数匹配
- `src/YOLOv11_OBB.cpp` 中的 `OBB_CLASS_NAMES` 数量是否与模型真实类别数一致

### 2. 检测结果不准确

优先检查：

- 是否真的使用了 `obb` 模型
- `conf` 和 `nms` 阈值是否合适
- 输入图像尺寸是否适合模型

### 3. 构建失败

- 检查 `CMakeLists.txt` 中的路径设置
- 确保 OpenCV 和 TensorRT 安装正确
- 查看构建日志获取详细错误信息

### 4. 最容易出错的地方

- 用错模型类型：`obb` 模型必须用 `yolov11-tensorrt_obb`
- `--num-classes` 和真实输出不一致
- `--labels` 文件行数和真实类别数不一致
- `src/YOLOv11_OBB.cpp` 里的 `OBB_CLASS_NAMES` 数量和真实类别数不一致
- 直接拿 `det/seg` 的 engine 来跑 `obb`
