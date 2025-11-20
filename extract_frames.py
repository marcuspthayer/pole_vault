# tools/extract_frames.py
import argparse, cv2, re
from pathlib import Path
import glob

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def extract(video_path: Path, out_dir: Path, fps: float=None, every:int=None,
            start:float=0.0, end:float=None, quality:int=95, root: Path|None=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = max(0, int(start * native_fps))
    end_f = total if end is None else min(total, int(end * native_fps))

    # sampling stride
    if fps and fps > 0:
        stride = max(1, int(round(native_fps / fps)))
    elif every and every > 0:
        stride = every
    else:
        stride = 10

    stem = sanitize(video_path.stem)

    # output directory: mirror original tree if --root given
    if root:
        rel = video_path.resolve().relative_to(root.resolve())
        out_sub = out_dir / rel.with_suffix("")  # keep nested dirs, drop ext
    else:
        out_sub = out_dir / stem

    out_sub.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    saved = 0
    for f in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start_f) % stride == 0:
            out_path = out_sub / f"{stem}_f{f:06d}.jpg"
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out_path.write_bytes(buf.tobytes())
                saved += 1

    cap.release()
    print(f"[OK] {video_path.name} → {out_sub}  (saved {saved} frames)")

def main():
    ap = argparse.ArgumentParser(description="Extract frames for YOLO labeling")
    ap.add_argument("--videos", required=True, help=r'Glob, e.g. "videos\**\*.mp4" (use ** for recursion)')
    ap.add_argument("--out", default="data/cones/images/raw", help="Output folder")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames/sec to sample (or use --every)")
    ap.add_argument("--every", type=int, default=None, help="Keep every Nth frame")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    ap.add_argument("--end", type=float, default=None, help="End time (s)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1–100")
    ap.add_argument("--root", type=str, default=None, help=r'Optional root to mirror, e.g. "videos"')
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root).resolve() if args.root else None

    # recursive glob
    paths = glob.glob(args.videos, recursive=True)
    if not paths:
        raise SystemExit(f"No videos matched: {args.videos}")

    # keep only known video extensions
    exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
    vids = [Path(p) for p in paths if Path(p).suffix.lower() in exts]
    if not vids:
        raise SystemExit("No video files with known extensions were found.")

    for vp in vids:
        extract(vp.resolve(), out_dir, fps=args.fps, every=args.every,
                start=args.start, end=args.end, quality=args.quality, root=root)

if __name__ == "__main__":
    main()
