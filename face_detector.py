"""Face detection using the face_recognition library."""
import cv2
import numpy as np
import face_recognition

from config import DETECTION_MODEL, DETECTION_SCALE, DETECTION_EVERY_N_FRAMES


class FaceDetector:
    """Detects face locations in frames. Supports frame skipping and downscaling for performance."""

    def __init__(
        self,
        model: str = DETECTION_MODEL,
        scale: float = DETECTION_SCALE,
        every_n_frames: int = DETECTION_EVERY_N_FRAMES,
    ) -> None:
        self._model = model
        self._scale = scale
        self._every_n_frames = every_n_frames
        self._cached_locations: list[tuple] = []

    def detect(self, frame: np.ndarray, frame_count: int) -> list[tuple]:
        """Detect faces in frame. Uses cached results for skipped frames.

        Args:
            frame: BGR image (full resolution).
            frame_count: Current frame number (0-indexed).

        Returns:
            List of face locations as (top, right, bottom, left) tuples
            in original frame coordinates.
        """
        if frame_count % self._every_n_frames != 0:
            return self._cached_locations

        small_frame = cv2.resize(frame, (0, 0), fx=self._scale, fy=self._scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model=self._model)

        inv_scale = 1.0 / self._scale
        self._cached_locations = [
            (
                int(top * inv_scale),
                int(right * inv_scale),
                int(bottom * inv_scale),
                int(left * inv_scale),
            )
            for top, right, bottom, left in locations
        ]

        return self._cached_locations

    @property
    def model(self) -> str:
        return self._model
