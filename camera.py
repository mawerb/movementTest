from __future__ import annotations

import argparse
import time

import cv2

from pose_analysis.landmarker import MediaPipePoseLandmarker, ensure_model
from pose_analysis.overlay import draw_overlay
from pose_analysis.pipeline import RollingAnalyzer
from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS

_MISSING = tuple(Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live webcam jumpshot form coach.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "full"), default="lite")
    parser.add_argument("--window-size", type=int, default=45)
    args = parser.parse_args(argv)

    model_path = ensure_model(args.model)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    analyzer = RollingAnalyzer(window_size=args.window_size)
    started = time.monotonic()
    last_ts = -1
    index = 0
    with MediaPipePoseLandmarker(model_path) as landmarker:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - started) * 1000)
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms
            detected = landmarker.detect(rgb, timestamp_ms)
            record = FrameRecord(
                video_id="webcam",
                frame_index=index,
                timestamp=timestamp_ms / 1000.0,
                landmarks=detected if detected is not None else _MISSING,
            )
            feedback = analyzer.push(record)
            overlay = draw_overlay(bgr, record, feedback)
            cv2.imshow("Jumpshot Coach", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            index += 1

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
