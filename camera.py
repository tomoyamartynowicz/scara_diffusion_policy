from __future__ import annotations

import time
from typing import Any

import numpy as np


class CameraError(RuntimeError):
    pass


class RealSenseSource:
    """The same RealSense source used for ACT recording, kept local for eval."""

    def __init__(
        self,
        color_name: str = "wrist_d405",
        depth_name: str | None = None,
        size: tuple[int, int] = (640, 480),
        fps: int = 30,
        serial: str | None = None,
        warmup_frames: int = 30,
        timeout_ms: int = 1000,
        align_depth: bool = False,
    ) -> None:
        self.color_name = color_name
        self.depth_name = depth_name
        self.size = size
        self.fps = fps
        self.serial = serial
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self.align_depth = align_depth
        self.rs: Any = None
        self.pipeline: Any = None
        self.align: Any = None

    def start(self) -> "RealSenseSource":
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError("pyrealsense2 is required for --live or --execute") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        width, height = self.size
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, self.fps)
        if self.depth_name:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)
        try:
            self.pipeline.start(config)
        except Exception as exc:
            raise CameraError(f"Could not start RealSense camera: {exc}") from exc
        if self.depth_name and self.align_depth:
            self.align = rs.align(rs.stream.color)
        for _ in range(self.warmup_frames):
            self._wait_for_frames()
        return self

    def _wait_for_frames(self):
        if self.pipeline is None:
            raise CameraError("Camera has not been started")
        try:
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
        except Exception as exc:
            raise CameraError("Timed out waiting for RealSense frame") from exc
        return self.align.process(frames) if self.align is not None else frames

    def read(self) -> dict:
        frameset = self._wait_for_frames()
        image_time = time.monotonic()
        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise CameraError("Missing RGB frame")
        frames = {self.color_name: np.asanyarray(color_frame.get_data()).copy()}
        if self.depth_name:
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                raise CameraError("Missing depth frame")
            frames[self.depth_name] = np.asanyarray(depth_frame.get_data()).copy()
        return {
            "frames": frames,
            "image_time": image_time,
            "rs_timestamp_ms": float(color_frame.get_timestamp()),
            "frame_number": int(color_frame.get_frame_number()),
        }

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def __enter__(self) -> "RealSenseSource":
        return self.start()

    def __exit__(self, *_args) -> None:
        self.close()
