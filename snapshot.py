"""交互式挑帧工具（命令行驱动版）

为什么不在图片窗口里按键：
    OpenCV 的窗口需要键盘焦点，一旦被别的程序抢走焦点，
    按键就会跑进别的程序（例如聊天输入框），操作很别扭。
    本版本改为在命令行窗口里输入命令，焦点问题彻底消失。

用法：
    python snapshot.py --video runs\\infer\\video_result.mp4

命令（在命令行里输入后回车）：
    n 或回车     下一帧
    p            上一帧
    +30          前进 30 帧（数字可任意改，如 +100 / +1）
    -30          后退 30 帧
    90           直接跳到第 90 帧
    s            保存当前帧为无损 PNG
    q            退出

每显示一帧都会打印帧号、秒数和检测目标数，方便对照记录。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="交互式挑帧工具（命令行驱动）")
    p.add_argument("--video", required=True, help="视频路径")
    p.add_argument("--out", default="runs/snapshots", help="导出目录")
    p.add_argument("--preview", type=int, default=1,
                   help="是否弹出预览窗口：1 弹出（只看不按键），0 不弹")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    if not video.exists():
        print(f"[错误] 找不到视频: {video}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("[错误] 无法打开视频")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 64)
    print(f"视频    : {video}")
    print(f"信息    : {width}x{height}  {fps:.2f}fps  共 {total} 帧  {total/fps:.2f} 秒")
    print(f"导出目录: {out_dir.resolve()}")
    print("=" * 64)
    print("命令说明：")
    print("    n 或直接回车 ....... 下一帧")
    print("    p .................. 上一帧")
    print("    +30 ................ 前进 30 帧（数字随便改，如 +100）")
    print("    -30 ................ 后退 30 帧")
    print("    90 ................. 直接跳到第 90 帧")
    print("    s .................. 保存当前帧（无损 PNG）")
    print("    q .................. 退出")
    print("=" * 64)
    print()
    print("注意：所有命令都输入在【这个命令行窗口】里，不要在图片窗口上按键。")
    print()

    win = "preview (this window ignores keyboard - type commands in the console)"
    show_preview = bool(args.preview)
    if show_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    pos = 0
    saved = 0
    frame = None

    def load(p: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, f = cap.read()
        return f if ok else None

    def show(p: int):
        nonlocal frame
        f = load(p)
        if f is None:
            print(f"  [警告] 第 {p} 帧读取失败")
            return
        frame = f
        print(f"  -> 第 {p} 帧 / 共 {total}   时间 {p / fps:.2f}s")
        if show_preview:
            disp = f
            long_side = max(width, height)
            if long_side > 1400:
                r = 1400 / long_side
                disp = cv2.resize(f, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
            cv2.putText(disp, f"frame {p} / {total - 1}   {p / fps:.2f}s",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(disp, f"frame {p} / {total - 1}   {p / fps:.2f}s",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            cv2.waitKey(1)          # 只为刷新画面，不等待按键

    show(pos)

    while True:
        try:
            cmd = input("命令> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("", "n"):
            pos = min(pos + 1, total - 1)
            show(pos)
        elif cmd == "p":
            pos = max(pos - 1, 0)
            show(pos)
        elif cmd == "s":
            if frame is None:
                print("  [警告] 当前没有可用画面")
                continue
            save_path = out_dir / f"frame_{pos:05d}_{pos / fps:.2f}s.png"
            cv2.imwrite(str(save_path), frame)
            saved += 1
            print(f"  [已保存] {save_path.resolve()}")
        elif cmd.startswith("+"):
            try:
                step = int(cmd[1:])
            except ValueError:
                print("  格式错误，示例：+30")
                continue
            pos = min(pos + step, total - 1)
            show(pos)
        elif cmd.startswith("-"):
            try:
                step = int(cmd[1:])
            except ValueError:
                print("  格式错误，示例：-30")
                continue
            pos = max(pos - step, 0)
            show(pos)
        else:
            try:
                target = int(cmd)
            except ValueError:
                print("  无法识别，可用命令：n / p / +30 / -30 / 数字 / s / q")
                continue
            pos = max(0, min(target, total - 1))
            show(pos)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    print()
    print("=" * 64)
    print(f"结束。共保存 {saved} 张到 {out_dir.resolve()}")
    if saved:
        print("从里面挑一张最满意的，复制到 docs/images/ 作为代表性结果截图。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
