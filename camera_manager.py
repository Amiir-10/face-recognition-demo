"""Webcam capture wrapper using OpenCV."""
import cv2
import numpy as np


class CameraManager:
    """Manages webcam capture, resolution, and cleanup."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that your webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """Read a single frame. Returns (success, frame)."""
        return self._cap.read()

    def is_opened(self) -> bool:
        """Check if the camera is available."""
        return self._cap.isOpened()

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.release()
