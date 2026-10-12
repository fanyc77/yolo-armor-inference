# YOLO Armor Inference

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的装甲板目标检测推理实现，用于 RoboMaster 视觉学习与验证。

## 功能

- 支持单张图片 / 整个目录批量推理
- 输出带检测框的可视化图片
- 可选导出结构化 JSON 结果（类别、置信度、坐标）
- 命令行参数化，便于脚本调用

## 环境要求

- Python >= 3.9（推荐 3.10 / 3.11）
- 支持 CUDA 的显卡（可选，无显卡时自动使用 CPU）

## 安装

```bash
# 建议使用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 快速开始

```bash
# 单张图片
python infer.py --source assets/test.jpg

# 整个目录，自定义阈值，并导出 JSON
python infer.py --source assets/ --conf 0.4 --save-json

# 使用自训练权重
python infer.py --weights weights/best.pt --source assets/
```

首次运行会自动下载 `yolo26n.pt` 官方预训练权重；若该权重不存在，可改用 `--weights yolo11n.pt`。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--weights` | `yolo26n.pt` | 权重文件路径 |
| `--source` | 必填 | 图片路径或图片目录 |
| `--conf` | `0.25` | 置信度阈值 |
| `--imgsz` | `640` | 推理分辨率 |
| `--out` | `runs/infer` | 输出目录 |
| `--save-json` | 关闭 | 是否导出 JSON |

## 输出说明

```
runs/infer/
├── <图片名>_pred.jpg     # 画框后的可视化结果
└── results.json          # 开启 --save-json 时生成
```

`results.json` 结构：

```json
[
  {
    "image": "assets/test.jpg",
    "detections": [
      {
        "class_id": 0,
        "class_name": "armor",
        "confidence": 0.9123,
        "xyxy": [312.5, 180.0, 401.2, 245.8]
      }
    ]
  }
]
```

## 目录结构

```
yolo-armor-inference/
├── infer.py             # 推理主程序（本分支新增）
├── requirements.txt     # 依赖清单（本分支新增）
├── README.md            # 说明文档
└── assets/              # 测试图片（可选）
```

## 备注

本次作业的推理代码在开发分支 `feat/yolo-inference` 完成，通过 Pull Request 合并回 `main` 分支。

所用开源项目：[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
