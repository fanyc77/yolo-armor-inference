"""装甲板视频推理脚本（YOLO26n-pose）

功能：
    1. 加载模型权重，读取视频
    2. 逐帧推理
    3. 检出目标时绘制：检测框、类别、置信度、以及模型预测的 4 个关键点
    4. 保存可正常播放的结果视频

关键点来自模型 pose 分支的真实预测（results.keypoints.xy），不是检测框的四个角。

用法示例：
    python infer.py --video D:\\test.mp4
    python infer.py --video D:\\test.mp4 --show
    python infer.py --weights D:\\桌面\\best.pt --video D:\\test.mp4 --out D:\\result.mp4
    python infer.py --video test.mp4 --conf 0.4 --link-kpts --kpt-radius 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认是 GBK，打印中文路径会直接抛 UnicodeEncodeError。
# 这里把标准输出切到 UTF-8，避免因为一句 print 就中断整个推理。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[错误] 未安装 ultralytics。请先执行：pip install -r requirements.txt")
    sys.exit(1)


# ----------------------------------------------------------------------------
# 默认配置（全部可用命令行参数覆盖）
# ----------------------------------------------------------------------------
DEFAULT_WEIGHTS = r"D:\桌面\best.pt"
DEFAULT_OUTPUT = "runs/infer/video_result.mp4"

# 关键点连线顺序（四边形轮廓）。none 表示不连线。
DEFAULT_LINK_ORDER = "0-1-2-3-0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO26n-pose 装甲板视频推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 路径参数 ---
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型权重路径 (.pt)")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--out", default=DEFAULT_OUTPUT, help="输出视频路径")
    # --- 推理参数 ---
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    p.add_argument("--device", default=None, help="推理设备，如 cpu / 0；不填自动选择")
    # --- 显示参数 ---
    p.add_argument("--show", action="store_true", help="实时显示推理画面（按 q 退出）")
    p.add_argument("--kpt-radius", type=int, default=5, help="关键点圆点半径（像素）")
    p.add_argument("--link-kpts", action="store_true", help="用连线把关键点串起来")
    p.add_argument("--link-order", default=DEFAULT_LINK_ORDER,
                   help="连线顺序，如 0-1-2-3-0；设为 none 表示不连")
    p.add_argument("--box-thickness", type=int, default=None,
                   help="检测框线宽；不填则按分辨率自动计算")
    p.add_argument("--no-labels", action="store_true", help="不绘制类别/置信度标签")
    return p.parse_args()


def make_color(class_id: int) -> tuple[int, int, int]:
    """按类别 id 生成稳定且鲜艳的 BGR 颜色。"""
    hue = (class_id * 47) % 180          # OpenCV 色相范围 0-179
    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def parse_link_order(text: str, n_kpts: int) -> list[tuple[int, int]]:
    """把 '0-1-2-3-0' 解析成 [(0,1),(1,2),(2,3),(3,0)]。"""
    if not text or text.strip().lower() in ("none", "no", ""):
        return []
    try:
        idx = [int(x) for x in text.replace(" ", "").split("-") if x != ""]
    except ValueError:
        print(f"[警告] --link-order 格式无法解析: {text}，已忽略连线")
        return []
    pairs = []
    for a, b in zip(idx, idx[1:]):
        if 0 <= a < n_kpts and 0 <= b < n_kpts:
            pairs.append((a, b))
    return pairs


def suggest_thickness(width: int, height: int, base: int | None = None) -> tuple[int, float]:
    """按视频分辨率给出 (框线宽, 字体缩放)。"""
    thickness = base if base is not None else max(2, round(min(width, height) / 320))
    font_scale = max(0.5, min(width, height) / 720)
    return thickness, font_scale


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str, Path]:
    """尝试多种编码器，返回可用的 VideoWriter、编码名和实际输出路径。"""
    candidates = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    for code, ext in candidates:
        target = path if path.suffix.lower() == ext else path.with_suffix(ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code, target
        writer.release()
    raise RuntimeError("无法创建视频写入器，请尝试换一个输出扩展名（.mp4 / .avi）")


def draw_detection(
    frame: np.ndarray,
    box: np.ndarray,
    kpts: np.ndarray | None,
    class_name: str,
    conf: float,
    color: tuple[int, int, int],
    thickness: int,
    font_scale: float,
    kpt_radius: int,
    link_pairs: list[tuple[int, int]],
    show_label: bool,
) -> None:
    """在帧上绘制一个目标的框、标签和关键点（就地修改 frame）。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # 1) 检测框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # 2) 类别 + 置信度标签（带底色，任何背景下都看得清）
    if show_label:
        text = f"{class_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        ty1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, ty1), (x1 + tw + 6, ty1 + th + baseline + 4), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        txt_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, txt_color, max(1, thickness - 1), cv2.LINE_AA)

    # 3) 关键点：必须是模型预测出来的点
    if kpts is None or len(kpts) == 0:
        return

    valid = []
    for px, py in kpts:
        if px <= 0 and py <= 0:                 # (0,0) 视为无效预测
            valid.append(False)
            continue
        # 允许略微超出框，但远在框外的点判为异常，避免画出飞到画面外的点
        mx = 0.35 * (x2 - x1)
        my = 0.35 * (y2 - y1)
        valid.append(bool((x1 - mx) <= px <= (x2 + mx) and (y1 - my) <= py <= (y2 + my)))

    # 先连线（画在圆点下方，视觉更干净）
    for a, b in link_pairs:
        if a < len(kpts) and b < len(kpts) and valid[a] and valid[b]:
            pa = (int(round(kpts[a][0])), int(round(kpts[a][1])))
            pb = (int(round(kpts[b][0])), int(round(kpts[b][1])))
            cv2.line(frame, pa, pb, color, max(1, thickness - 1), cv2.LINE_AA)

    # 再画圆点 + 编号，保证看得出属于哪个目标
    for i, (px, py) in enumerate(kpts):
        if not valid[i]:
            continue
        center = (int(round(px)), int(round(py)))
        cv2.circle(frame, center, kpt_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, kpt_radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, kpt_radius, (0, 0, 0), 1, cv2.LINE_AA)
        pos = (center[0] + kpt_radius + 3, center[1] - kpt_radius - 1)
        num_scale = max(0.4, font_scale * 0.7)
        cv2.putText(frame, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, num_scale,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, num_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    video = Path(args.video)
    out_path = Path(args.out)

    if not weights.exists():
        print(f"[错误] 找不到模型权重: {weights}")
        return 1
    if not video.exists():
        print(f"[错误] 找不到视频文件: {video}")
        return 1

    # ---------------- 1. 加载模型 ----------------
    print(f"[1/4] 加载模型: {weights}")
    model = YOLO(str(weights))
    task = getattr(model, "task", None)
    names = getattr(model, "names", {}) or {}

    print(f"      任务类型: {task}")
    print(f"      类别数量: {len(names)}  ->  {names}")

    if task != "pose":
        print(f"[错误] 本脚本针对 pose 模型编写，但加载到的模型 task='{task}'。")
        print("       请确认传入的是 yolo26n-pose 训练得到的 best.pt。")
        return 1

    kpt_shape = getattr(model.model, "kpt_shape", None)
    if not kpt_shape:
        print("[错误] 模型没有 kpt_shape 属性，不是 pose 模型。")
        return 1
    n_kpts = int(kpt_shape[0])
    print(f"      关键点形状: {kpt_shape}  ->  每个目标 {n_kpts} 个关键点")
    if n_kpts != 4:
        print(f"[提示] 本模型输出 {n_kpts} 个关键点（作业要求 4 个），请确认模型是否正确。")

    link_pairs = parse_link_order(args.link_order, n_kpts) if args.link_kpts else []
    print(f"      关键点连线: {link_pairs if link_pairs else '关闭'}")

    # ---------------- 2. 打开视频 ----------------
    print(f"\n[2/4] 读取视频: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("[错误] 无法打开视频，请检查文件是否损坏或编码是否受支持。")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:       # 无效值 / NaN 保护
        fps = 25.0
        print(f"[警告] 读不到帧率，按 {fps} fps 处理")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      分辨率: {width}x{height}   帧率: {fps:.2f}   总帧数: {total}")

    thickness, font_scale = suggest_thickness(width, height, args.box_thickness)
    print(f"      绘制参数: 线宽={thickness}  字号={font_scale:.2f}  点半径={args.kpt_radius}")

    # ---------------- 3. 逐帧推理 ----------------
    print(f"\n[3/4] 开始推理，输出到: {out_path}")
    writer, codec, actual_out = open_writer(out_path, fps, (width, height))
    print(f"      编码器: {codec}")

    frame_idx = 0
    frames_with_det = 0
    total_boxes = 0
    total_kpts = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(
                source=frame, conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, device=args.device, verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            kpts_all = result.keypoints

            if boxes is not None and len(boxes) > 0:
                frames_with_det += 1
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                kxy = None
                if kpts_all is not None and kpts_all.xy is not None and len(kpts_all.xy) > 0:
                    kxy = kpts_all.xy.cpu().numpy()      # [N, 4, 2]

                for i in range(len(boxes)):
                    cid = int(cls_ids[i])
                    kp = kxy[i] if kxy is not None and i < len(kxy) else None
                    draw_detection(
                        frame=frame, box=xyxy[i], kpts=kp,
                        class_name=names.get(cid, str(cid)), conf=float(confs[i]),
                        color=make_color(cid), thickness=thickness, font_scale=font_scale,
                        kpt_radius=args.kpt_radius, link_pairs=link_pairs,
                        show_label=not args.no_labels,
                    )
                    total_boxes += 1
                    if kp is not None:
                        total_kpts += int(np.sum((kp[:, 0] > 0) | (kp[:, 1] > 0)))

            writer.write(frame)

            if args.show:
                cv2.imshow("infer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("      用户中断显示")
                    break

            if frame_idx % 50 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"      已处理 {frame_idx}/{total} 帧 ({pct:.1f}%)  检出目标 {total_boxes} 个")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ---------------- 4. 结果 ----------------
    print("\n[4/4] 完成")
    print(f"      处理帧数: {frame_idx}")
    print(f"      含检出帧: {frames_with_det}")
    print(f"      检出目标: {total_boxes} 个")
    print(f"      有效关键点: {total_kpts} 个")
    if actual_out.exists() and actual_out.stat().st_size > 0:
        print(f"      结果视频: {actual_out}  ({actual_out.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("      [警告] 未找到输出文件，请检查输出路径权限")
    return 0


if __name__ == "__main__":
    sys.exit(main())
