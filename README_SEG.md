# YOLOv11 SEG TensorRT 推理指南

本文档只说明 `SEG` 模型的部署、推理和类别名配置。

如果你要看“应用层组合调用”或“额外后处理示例”，请单独查看：

- `README_APP.md`

## 项目结构

```text
yolov11-tensorrt/
├── asset/                # 测试资源与可选标签文件
├── build/                # 构建目录
├── src/
│   ├── YOLOv11_SEG.cpp   # SEG 模型实现
│   └── YOLOv11_SEG.h     # SEG 模型头文件
├── CMakeLists.txt        # CMake 配置
├── export.py             # 模型导出脚本
├── main_seg.cpp          # SEG 推理主程序
└── README_SEG.md         # 本文档
```

## 环境要求

- 操作系统：Linux 或 Windows
- CUDA：11.6 或更高版本
- TensorRT：8.6 或更高版本
- OpenCV：4.0 或更高版本
- Python：3.10 或更高版本
- `ultralytics`：用于导出 YOLOv11 SEG 模型

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

### 4. 构建 SEG 可执行文件

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
cmake --build . --target yolov11-tensorrt_seg --config Release
```

`--target yolov11-tensorrt_seg` 表示只编译 SEG 推理目标。

## 基本使用

### 1. 导出 SEG 模型

可以修改 `export.py` 导出自己的 SEG 模型，也可以直接使用：

```python
from ultralytics import YOLO

model = YOLO("weights/best_seg.pt")
model.export(format="onnx")
```

导出后通常会得到：

```text
weights/best_seg.onnx
```

### 2. 从 ONNX 现场构建 TensorRT engine

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.onnx ""
```

执行后会生成对应的 `.engine` 文件。

### 3. 运行推理

单张图片：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./asset/bus.jpg
```

视频：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./video.mp4
```

文件夹：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./images/
```

图片和文件夹模式会输出 `seg_result_xxx.jpg`；视频模式实时显示结果，不逐帧写盘。

## 运行参数

当前 SEG 入口支持以下参数，这些参数都会真正参与推理流程：

```bash
./build/yolov11-tensorrt_seg <engine_or_onnx> <image/video/folder> \
  [--num-classes=N] \
  [--conf=T] \
  [--nms=T] \
  [--mask-thres=T] \
  [--alpha=T] \
  [--labels=PATH] \
  [--fp16] \
  [--no-warmup]
```

- `--num-classes=N`
  显式校验模型类别数是否与引擎输出一致。
- `--conf=T`
  设置检测置信度阈值，例如 `--conf=0.25`。
- `--nms=T`
  设置 NMS 阈值，例如 `--nms=0.45`。
- `--mask-thres=T`
  设置 mask 二值化阈值，例如 `--mask-thres=0.50`。
- `--alpha=T`
  设置 mask 覆盖透明度，例如 `--alpha=0.45`。
- `--labels=PATH`
  从文本文件加载类别名，每行一个类别。只要传了这个参数，就会覆盖源码里的默认类别名。
- `--fp16`
  仅在输入为 `.onnx`、需要现场构建 engine 时生效。
- `--no-warmup`
  关闭默认的 10 次 warmup。

示例：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./asset/bus.jpg \
  --conf=0.25 \
  --nms=0.45 \
  --mask-thres=0.50 \
  --alpha=0.45 \
  --no-warmup
```

## 换成自己的 SEG 模型

### 1. 准备自己的 `.pt` 分割模型

例如：

```text
weights/best_seg.pt
```

注意必须是 `seg` 模型，不要把 `det` 或 `obb` 模型拿来这里用。

### 2. 导出成 ONNX

```python
from ultralytics import YOLO

model = YOLO("weights/best_seg.pt")
model.export(format="onnx")
```

### 3. 生成 engine

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.onnx ""
```

### 4. 运行自己的模型

如果你有 3 个类别，并且想通过标签文件传入类别名：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./asset/bus.jpg \
  --num-classes=3 \
  --labels=./asset/seg.labels.txt
```

`asset/seg.labels.txt` 例如：

```text
crack
scratch
dent
```

## 类别名配置

`SEG` 支持两种设置类别名的方式，优先级如下：

1. `--labels=PATH`
2. `src/YOLOv11_SEG.cpp` 里的 `SEG_CLASS_NAMES`
3. 自动生成 `class_0`、`class_1`、`class_2` ...

### 方式 1：命令行传 `--labels`

准备一个文本文件，每行一个类别，例如：

```text
crack
scratch
dent
```

然后运行：

```bash
./build/yolov11-tensorrt_seg ./weights/best_seg.engine ./asset/test.jpg \
  --num-classes=3 \
  --labels=./asset/seg.labels.txt
```

### 方式 2：直接在代码里写死默认类别名

如果你以后都固定使用自己的类别名，不想每次都传 `--labels`，可以直接修改：

```cpp
// src/YOLOv11_SEG.cpp
static const std::vector<std::string> SEG_CLASS_NAMES = {
    "crack",
    "scratch",
    "dent"
};
```

注意：

- 这里的类别数量必须和模型真实类别数一致
- 如果后面又传了 `--labels=PATH`，则以 `--labels` 为准
- 如果 `SEG_CLASS_NAMES` 数量和模型真实类别数不一致，程序会退回到 `class_0/class_1/...`

## SEG 输出说明

SEG 模型输出包含两部分：

1. 检测分支：目标框、类别分数、mask 系数
2. prototype 分支：分割原型图

当前实现会完成以下流程：

1. 从检测输出中解析 `bbox + class + score + mask coeff`
2. 通过 `NMS` 过滤重复目标
3. 用 mask 系数和 prototype 特征图恢复实例 mask
4. 把 mask 映射回原图尺寸
5. 绘制框、类别名和半透明分割区域

## 代码配置说明

`src/YOLOv11_SEG.h` 里有两类字段：

- `SEGConfig`
  这是可配置参数，`--num-classes`、`--conf`、`--nms`、`--mask-thres`、`--alpha`、`--labels`、`--fp16`、`--no-warmup` 都会真正参与推理流程。
- `SEGRuntimeState`
  这是运行时状态，例如输入尺寸、输出张量形状、mask 尺寸、缩放比例等，程序会自动根据模型和图片计算，不需要手工修改。

## 常见问题

### 1. 能检测到框，但没有 mask

优先检查：

- 使用的是否真的是 `seg` 模型导出的 ONNX/engine
- `--mask-thres` 是否设置过高
- 输出张量数量是否为 2 个输出分支

### 2. 类别名显示不对

检查：

- `--labels` 文件行数是否与模型真实类别数一致
- `--num-classes` 是否与模型导出类别数匹配
- `src/YOLOv11_SEG.cpp` 中的 `SEG_CLASS_NAMES` 数量是否与模型真实类别数一致

### 3. 推理结果错位

通常与导出的模型类型不匹配有关，比如把 `det` 或 `obb` 的 engine 当作 `seg` 来跑。

### 4. 构建失败

- 检查 `CMakeLists.txt` 中的路径设置
- 确保 OpenCV 和 TensorRT 安装正确
- 查看构建日志获取详细错误信息

### 5. 最容易出错的地方

- 用错模型类型：`seg` 模型必须用 `yolov11-tensorrt_seg`
- `--num-classes` 与真实类别数不一致
- `--labels` 文件行数和真实类别数不一致
- 改了源码里的 `SEG_CLASS_NAMES`，但类别数量和模型真实类别数不一致
- 导出的 ONNX 不是实例分割模型，只有检测分支没有 prototype 分支
