"""Face embedding extraction from InsightFace detection results."""
import numpy as np


class FaceEncoder:
    """Extracts 512D ArcFace embeddings from InsightFace Face objects.

    InsightFace computes embeddings during detection, so this is a zero-cost extraction.
    """

    def encode(self, faces: list) -> list[np.ndarray]:
        """Extract embeddings from detected faces.

        Args:
            faces: List of InsightFace Face objects (from FaceDetector.detect).

        Returns:
            List of 512D L2-normalized numpy arrays, one per face with a valid embedding.
        """
        return [face.embedding for face in faces if face.embedding is not None]
