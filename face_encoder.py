"""Face encoding generation using the face_recognition library."""
import numpy as np
import cv2
import face_recognition


class FaceEncoder:
    """Generates 128D face embedding vectors from detected face regions."""

    def encode(self, frame: np.ndarray, face_locations: list[tuple]) -> list[np.ndarray]:
        """Generate encodings for detected faces.

        Args:
            frame: BGR image (full resolution).
            face_locations: List of (top, right, bottom, left) tuples.

        Returns:
            List of 128D numpy arrays, one per face.
        """
        if not face_locations:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
        return encodings
