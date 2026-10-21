from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_analysis.extract import extract_frames
from pose_analysis.landmarker import MediaPipePoseLandmarker, ensure_model
from pose_analysis.overlay import preview_overlay_video, write_overlay_video
from pose_analysis.pipeline import analyze_frames
from pose_analysis.records import read_jsonl, write_jsonl
from pose_analysis.video import iter_rgb_frames, open_video
from pose_analysis.windows import sliding_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from video and analyze jumpshot form."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract_cmd = sub.add_parser("extract", help="Run Pose Landmarker on a video and save JSONL.")
    extract_cmd.add_argument("video")
    extract_cmd.add_argument("--video-id")
    extract_cmd.add_argument("--out", default="data/landmarks")
    extract_cmd.add_argument("--model", choices=("lite", "full"), default="lite")

    analyze_cmd = sub.add_parser(
        "analyze", help="Analyze a video file or a previously extracted JSONL file."
    )
    analyze_cmd.add_argument("source")
    analyze_cmd.add_argument("--video-id")
    analyze_cmd.add_argument("--model", choices=("lite", "full"), default="lite")
    analyze_cmd.add_argument("--window-size", type=int, default=45)
    analyze_cmd.add_argument("--stride", type=int, default=15)

    overlay_cmd = sub.add_parser(
        "overlay",
        help="Replay a video with skeleton lines from a JSONL landmark file.",
    )
    overlay_cmd.add_argument("video")
    overlay_cmd.add_argument("landmarks")
    overlay_cmd.add_argument("--out", help="Write an annotated video (e.g. overlay.mp4).")
    overlay_cmd.add_argument(
        "--preview",
        action="store_true",
        help="Show the overlay in a window (q to quit).",
    )
    overlay_cmd.add_argument(
        "--analyze",
        action="store_true",
        help="Also stamp movement/form text from the heuristic analyzer.",
    )
    overlay_cmd.add_argument("--window-size", type=int, default=45)
    overlay_cmd.add_argument("--stride", type=int, default=15)

    args = parser.parse_args(argv)
    if args.command == "extract":
        return _extract(args)
    if args.command == "analyze":
        return _analyze(args)
    return _overlay(args)


def _extract(args) -> int:
    video_id = args.video_id or Path(args.video).stem
    records = _records_from_video(args.video, video_id, args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.jsonl"
    write_jsonl(out_path, records)
    print(f"wrote {len(records)} frames to {out_path}")
    return 0


def _analyze(args) -> int:
    source = Path(args.source)
    if source.suffix.lower() == ".jsonl":
        records = read_jsonl(source)
    else:
        video_id = args.video_id or source.stem
        records = _records_from_video(source, video_id, args.model)
    results = analyze_frames(records, window_size=args.window_size, stride=args.stride)
    if not results:
        print(json.dumps({"movement": "uncertain", "confidence": 0.0, "phase": None, "issues": []}))
        return 0
    for item in results:
        print(json.dumps(item.to_dict()))
    return 0


def _overlay(args) -> int:
    if not args.out and not args.preview:
        raise SystemExit("overlay needs --out and/or --preview")
    records = read_jsonl(args.landmarks)
    feedbacks = None
    if args.analyze:
        feedbacks = _feedbacks_per_frame(records, args.window_size, args.stride)
    if args.out:
        written = write_overlay_video(args.video, records, args.out, feedbacks)
        print(f"wrote {written} overlay frames to {args.out}")
    if args.preview:
        preview_overlay_video(args.video, records, feedbacks)
    return 0


def _feedbacks_per_frame(records, window_size: int, stride: int):
    results = analyze_frames(records, window_size=window_size, stride=stride)
    windows = sliding_windows(records, size=window_size, stride=stride)
    stamped = [None] * len(records)
    if not windows:
        if results:
            return list(results[:1]) * len(records)
        return stamped
    for window, feedback in zip(windows, results, strict=False):
        for frame in window:
            if 0 <= frame.frame_index < len(stamped):
                stamped[frame.frame_index] = feedback
    return stamped


def _records_from_video(path, video_id: str, model: str):
    model_path = ensure_model(model)
    capture, fps = open_video(path)
    with MediaPipePoseLandmarker(model_path) as landmarker:
        return extract_frames(iter_rgb_frames(capture), landmarker, video_id, fps)


if __name__ == "__main__":
    raise SystemExit(main())
