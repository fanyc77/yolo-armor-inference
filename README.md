# YOLO26n 装甲板视频推理

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的 **YOLO26n-pose** 装甲板检测与关键点推理脚本，用于 RoboMaster 视觉学习与验证。

在视频上逐帧推理，检出目标时绘制**检测框、类别、置信度，以及模型预测的 4 个关键点**，并导出可正常播放的结果视频。

## 模型说明

参考材料提供的 `best.pt` 为 **pose（姿态估计）** 模型，从模型元数据可读出：

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `task` | `pose` | 姿态估计任务 |
| `kpt_shape` | `[4, 2]` | 每个目标 **4 个关键点**，各含 (x, y) 两个坐标 |
| `nc` | `6` | 6 个类别 |
| `names` | `B1 B2 B3 R1 R2 R3` | 蓝方 / 红方各 3 类装甲板 |

**关键点来源**：脚本读取模型 pose 分支的真实预测输出 `results.keypoints.xy`，**不是用检测框的四个角代替**。

## 环境要求

- Python >= 3.9（已在 3.13 上验证）
- 无需 GPU —— 使用 CPU 即可运行，作业不要求实时推理

## 安装

```bash
# 建议使用虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 国内网络建议加清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 默认从 PyPI 安装的是 **CPU 版 torch**（约 200 MB）。若需 GPU 加速，请先参考 [pytorch.org](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch，再执行上面的命令。

## 快速开始

```bash
# 最简用法：默认权重 D:\桌面\best.pt，输出到 runs/infer/video_result.mp4
python infer.py --video D:\test.mp4

# 实时预览画面（按 q 退出）
python infer.py --video D:\test.mp4 --show

# 完整指定路径与参数
python infer.py --weights D:\桌面\best.pt --video D:\test.mp4 --out D:\result.mp4 --conf 0.4

# 打开关键点连线，让 4 个点的归属更直观
python infer.py --video D:\test.mp4 --link-kpts
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--weights` | `D:\桌面\best.pt` | 模型权重路径 |
| `--video` | **必填** | 输入视频路径 |
| `--out` | `runs/infer/video_result.mp4` | 输出视频路径 |
| `--conf` | `0.25` | 置信度阈值 |
| `--iou` | `0.45` | NMS IoU 阈值 |
| `--imgsz` | `640` | 推理分辨率 |
| `--device` | 自动 | `cpu` 或 `0`（GPU 序号） |
| `--show` | 关闭 | 实时显示推理画面 |
| `--kpt-radius` | `5` | 关键点圆点半径（像素） |
| `--link-kpts` | 关闭 | 用连线串起 4 个关键点 |
| `--link-order` | `0-1-2-3-0` | 连线顺序，`none` 表示不连 |
| `--box-thickness` | 自动 | 检测框线宽，默认按分辨率计算 |
| `--no-labels` | 关闭 | 不绘制类别与置信度标签 |

**模型路径、视频路径、输出位置三者都通过命令行参数控制，也可直接修改脚本顶部的 `DEFAULT_*` 常量。**

## 绘制内容说明

检出目标后，脚本在同一帧上绘制：

1. **检测框** —— 颜色按类别 id 生成，同类别颜色固定
2. **类别 + 置信度** —— 例如 `R1 0.87`，带底色块保证任何背景下都清晰
3. **4 个关键点** —— 来自模型预测：
   - 白色外圈 + 彩色内点 + 黑色描边，三层叠加，明暗背景下都醒目
   - 每个点旁标注序号 `0` `1` `2` `3`，便于区分
   - 加 `--link-kpts` 会按顺序连线，四个点的归属一目了然
   - 坐标为 `(0,0)` 或明显偏离检测框的点会被判为无效预测并跳过

## 输出说明

```
runs/infer/
└── video_result.mp4      # 与输入视频内容对应、逐帧对应的结果视频
```

- 输出视频**保持原视频的分辨率与帧率**，画面内容与源视频逐帧对应
- 编码器自动选择：依次尝试 `mp4v` → `avc1` → `XVID` → `MJPG`，取第一个可用的
- 若目标路径无法写入，会自动换用可用的扩展名（`.mp4` / `.avi`）

## 常见问题

**装 ultralytics 报错或很慢**
使用国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**提示 `task='detect'` 而不是 `pose`**
说明传入的不是 pose 模型。本脚本只支持 pose 权重，请确认使用的是参考材料里的 `best.pt`。

**结果视频打不开或花屏**
换一个输出扩展名再试，例如 `--out result.avi`。脚本会自动尝试其他编码器。

**CPU 推理较慢**
属正常现象，作业不要求实时。可适当降低 `--imgsz`（如 480）加快速度，代价是小目标精度略降。

## 目录结构

```
yolo-armor-inference/
├── infer.py             # 视频推理主程序
├── requirements.txt     # 依赖清单
├── README.md            # 说明文档
├── docs/
│   └── model_info.md    # best.pt 模型元数据分析
└── .gitignore
```

## 备注

所用开源项目：[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
