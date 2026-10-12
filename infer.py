"""YOLO 装甲板推理脚本（单张图片 / 整个目录）

用法:
    python infer.py --source assets/test.jpg
    python infer.py --source assets/ --conf 0.4 --save-json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: Path) -> list[Path]:
    """source 可以是单张图片，也可以是目录。"""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_SUFFIX)
    raise FileNotFoundError(f"路径不存在: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 推理")
    ap.add_argument("--weights", default="yolo26n.pt", help="权重文件，不存在会自动下载")
    ap.add_argument("--source", required=True, help="图片路径或图片目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    ap.add_argument("--out", default="runs/infer", help="结果输出目录")
    ap.add_argument("--save-json", action="store_true", help="同时导出检测结果 JSON")
    args = ap.parse_args()

    src = Path(args.source)
    images = collect_images(src)
    if not images:
        raise SystemExit(f"没有找到图片: {src}")

    print(f"[1/3] 加载模型: {args.weights}")
    model = YOLO(args.weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] 开始推理，共 {len(images)} 张")
    all_records = []
    for img_path in images:
        results = model.predict(
            source=str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False
        )
        result = results[0]

        # 画框图
        plotted = result.plot()
        save_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(save_path), plotted)

        # 收集结构化结果
        names = result.names
        records = []
        for box in result.boxes:
            records.append({
                "class_id": int(box.cls[0]),
                "class_name": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            })
        all_records.append({"image": str(img_path), "detections": records})
        print(f"      {img_path.name}: 检出 {len(records)} 个目标 -> {save_path}")

    if args.save_json:
        json_path = out_dir / "results.json"
        json_path.write_text(
            json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[3/3] JSON 结果已保存: {json_path}")
    else:
        print("[3/3] 完成")


if __name__ == "__main__":
    main()
