"""Face detection and embedding using InsightFace."""
import numpy as np

from config import DETECTION_EVERY_N_FRAMES


class FaceDetector:
    """Detects faces and computes ArcFace embeddings via InsightFace. Supports frame skipping."""

    def __init__(self, app, every_n_frames: int = DETECTION_EVERY_N_FRAMES) -> None:
        """
        Args:
            app: Initialized insightface.app.FaceAnalysis instance (shared).
            every_n_frames: Run detection every Nth frame; return cached results otherwise.
        """
        self._app = app
        self._every_n_frames = every_n_frames
        self._cached_faces: list = []

    def detect(self, frame: np.ndarray, frame_count: int) -> list:
        """Detect faces in frame. Returns cached results on skipped frames.

        Args:
            frame: BGR image (OpenCV format).
            frame_count: Current frame number (0-indexed).

        Returns:
            List of InsightFace Face objects. Each face has:
                .bbox: [x1, y1, x2, y2] (left, top, right, bottom)
                .embedding: 512D L2-normalized numpy array (ArcFace)
                .det_score: detection confidence
        """
        if frame_count % self._every_n_frames != 0:
            return self._cached_faces

        self._cached_faces = self._app.get(frame)
        return self._cached_faces
