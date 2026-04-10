"""Face recognition using cosine similarity on ArcFace embeddings."""
import numpy as np

from config import RECOGNITION_THRESHOLD, UNKNOWN_LABEL


class FaceRecognizer:
    """Matches 512D ArcFace embeddings against the database using cosine similarity."""

    def __init__(self, threshold: float = RECOGNITION_THRESHOLD) -> None:
        self._threshold = threshold

    def recognize(self, embedding: np.ndarray, database) -> tuple[str, float]:
        """Compare an unknown face embedding against all known faces.

        Args:
            embedding: 512D numpy array (InsightFace ArcFace output, L2-normalized).
            database: FaceDatabase instance.

        Returns:
            Tuple of (name, confidence_percentage).
            Returns ("Unknown", 0.0) if no match or empty database.
        """
        known_encodings, known_names = database.get_all_encodings()

        if not known_encodings:
            return UNKNOWN_LABEL, 0.0

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return UNKNOWN_LABEL, 0.0
        query = embedding / norm

        # Stack stored embeddings into matrix and normalize
        matrix = np.array(known_encodings, dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        matrix = matrix / row_norms

        # Cosine similarity = dot product of normalized vectors
        similarities = matrix @ query
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self._threshold:
            confidence = round(best_sim * 100, 1)
            return known_names[best_idx], confidence

        return UNKNOWN_LABEL, 0.0
