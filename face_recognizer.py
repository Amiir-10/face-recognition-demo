"""Face recognition -- compares unknown encodings against the database."""
import numpy as np
import face_recognition as fr

from config import RECOGNITION_THRESHOLD, UNKNOWN_LABEL


class FaceRecognizer:
    """Matches face encodings against known faces in the database."""

    def __init__(self, threshold: float = RECOGNITION_THRESHOLD) -> None:
        self._threshold = threshold

    def recognize(self, encoding: np.ndarray, database) -> tuple[str, float]:
        """Compare an unknown face encoding against all known faces.

        Args:
            encoding: 128D numpy array of the unknown face.
            database: FaceDatabase instance.

        Returns:
            Tuple of (name, confidence_percentage).
            Returns ("Unknown", 0.0) if no match or empty database.
        """
        known_encodings, known_names = database.get_all_encodings()

        if not known_encodings:
            return UNKNOWN_LABEL, 0.0

        distances = fr.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance < self._threshold:
            confidence = round((1 - best_distance) * 100, 1)
            return known_names[best_idx], confidence

        return UNKNOWN_LABEL, 0.0
