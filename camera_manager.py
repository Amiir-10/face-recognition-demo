"""Webcam capture wrapper using OpenCV with background capture thread."""
import threading

import cv2
import numpy as np


class CameraManager:
    """Manages webcam capture on a background thread for non-blocking frame reads."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that your webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._ret: bool = False
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._latest_frame = frame

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Return the most recent frame without blocking.

        Returns (False, None) until the background thread has captured the first frame.
        """
        with self._lock:
            return self._ret, self._latest_frame

    def is_opened(self) -> bool:
        """Check if the camera is available."""
        return self._cap.isOpened()

    def release(self) -> None:
        """Stop the capture thread and release the camera resource."""
        self._running = False
        self._thread.join(timeout=2.0)
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.release()
