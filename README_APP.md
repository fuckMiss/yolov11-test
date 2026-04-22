# 应用层调用与后处理说明

本文档只讲“项目应用层”的内容，不和 `DET / OBB / SEG` 各自的部署 README 混在一起。

适合看这份文档的场景：

- 你要把当前仓库继续往完整项目方向做
- 你想组合多个模型一起调用
- 你想在推理结果上加自己的业务后处理
- 你不想把这些业务逻辑混进底层部署代码

部署相关请分别看：

- `README.md`
- `README_OBB.md`
- `README_SEG.md`

## 目标

这里推荐的结构是：

- 部署层负责模型推理
- 应用层负责业务逻辑和结果后处理

也就是：

- `src/YOLOv11.cpp`
- `src/YOLOv11_OBB.cpp`
- `src/YOLOv11_SEG.cpp`

尽量保持干净、稳定；

而：

- 角度箭头
- 角度打印
- 多模型串联
- 结果统计
- 业务判定

这些都放在应用层。

## 当前应用层相关文件

```text
yolov11-tensorrt/
├── main_obb_angle.cpp          # OBB + 角度后处理示例
├── main_obb_seg.cpp            # OBB + SEG 联合调用示例
├── src/
│   ├── app_postprocess.cpp     # 应用层后处理
│   └── app_postprocess.h
└── README_APP.md
```

## 当前已经分好的边界

### 部署层

这些文件属于底层部署，不建议把业务逻辑继续往里塞：

- `src/YOLOv11.cpp`
- `src/YOLOv11_OBB.cpp`
- `src/YOLOv11_SEG.cpp`

这些文件主要负责：

- `preprocess`
- `infer`
- `postprocess`
- 基础结果绘制

### 应用层

这些文件用于写你自己的调用方式和后处理：

- `main_obb_angle.cpp`
- `main_obb_seg.cpp`
- `src/app_postprocess.cpp`

这些文件主要负责：

- 模型组合调用
- 额外可视化
- 业务输出
- 项目侧后处理

## 示例 1：OBB 角度后处理

入口：

- `main_obb_angle.cpp`

它的流程是：

1. 正常调用 `YOLOv11_OBB`
2. 正常执行 `preprocess -> infer -> postprocess`
3. 先画基础 OBB 结果
4. 再调用应用层角度后处理

当前角度后处理函数：

```cpp
void DrawOBBAngleOverlay(cv::Mat& image,
                         const std::vector<OBBDetection>& output,
                         const std::vector<std::string>& class_names,
                         bool print_angle = true);
```

它现在做的事：

- 计算角度显示
- 画朝向箭头
- 叠加角度文本
- 在终端打印角度

注意：

- 正常 `main_obb.cpp` 不会画箭头
- 只有 `main_obb_angle.cpp` 才会额外做这些事情

## 示例 2：OBB + SEG 联合调用

入口：

- `main_obb_seg.cpp`

它的作用是：

- 同一张图先跑 `OBB`
- 再跑 `SEG`
- 最后把两个模型结果合成一张图

适合后续扩展成：

- 先 `OBB` 定位方向
- 再 `SEG` 提取区域
- 再做你自己的业务判断

## 为什么不建议混进部署层

如果把项目侧逻辑直接写进 `YOLOv11_OBB.cpp` 或 `YOLOv11_SEG.cpp`，后面会有这些问题：

- 正常部署路径会越来越重
- 不同项目需求会互相污染
- 测试时难区分“模型问题”还是“业务后处理问题”
- 后面项目变大时，代码会越来越难维护

所以这里建议长期保持：

- 模型部署文档独立
- 项目应用文档独立
- 部署代码独立
- 项目后处理代码独立

## 后面加 SEG 后处理怎么做

继续沿用当前结构即可。

推荐做法：

1. 保持 `src/YOLOv11_SEG.cpp` 只负责 SEG 推理
2. 在 `src/app_postprocess.cpp` 里新增 SEG 后处理函数
3. 视需求再新增新的轻量入口

例如后面你可以继续加：

- `DrawSEGContourOverlay(...)`
- `ComputeMaskArea(...)`
- `ExtractMaskCenter(...)`
- `RunOBBAndSEGWorkflow(...)`

这样以后项目做大，也还是清楚的。

## 推荐维护方式

如果后续这个仓库要继续往完整项目走，建议文档长期保持下面这种分层：

- `README.md`
  只讲 DET 的基础推理和项目总览
- `README_OBB.md`
  只讲 OBB 部署
- `README_SEG.md`
  只讲 SEG 部署
- `README_APP.md`
  只讲应用层组合调用、项目后处理、业务逻辑

这样后面无论你继续加：

- `DET + OBB`
- `OBB + SEG`
- `DET + SEG`
- 多阶段项目逻辑

都不会和底层部署 README 互相打架。
