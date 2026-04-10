"""Persistent storage for known face embeddings."""
import os
import pickle

import numpy as np

DB_VERSION = 2  # v1 = dlib 128D, v2 = InsightFace 512D


class FaceDatabase:
    """Stores face embeddings mapped to names. Persists to disk via pickle."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._data: dict[str, list[np.ndarray]] = {}
        self.load()

    def add_face(self, name: str, encoding: np.ndarray, auto_save: bool = True) -> None:
        """Add a face embedding for a person.

        Args:
            name: Person's name (will be lowercased and stripped).
            encoding: 512D numpy array.
            auto_save: If True, persist to disk immediately. Set False for batch imports.
        """
        name = name.strip().lower()
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(encoding)
        if auto_save:
            self.save()

    def get_all_encodings(self) -> tuple[list[np.ndarray], list[str]]:
        """Return (flat list of all embeddings, corresponding names)."""
        encodings = []
        names = []
        for name, encs in self._data.items():
            for enc in encs:
                encodings.append(enc)
                names.append(name)
        return encodings, names

    def delete_face(self, name: str) -> bool:
        """Remove a person and all their embeddings. Auto-saves."""
        name = name.strip().lower()
        if name in self._data:
            del self._data[name]
            self.save()
            return True
        return False

    def list_faces(self) -> list[str]:
        """Return list of all registered names."""
        return list(self._data.keys())

    def face_count(self) -> int:
        """Return number of registered people."""
        return len(self._data)

    def save(self) -> None:
        """Persist database to disk."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        payload = {"version": DB_VERSION, "data": self._data}
        with open(self._db_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self) -> None:
        """Load database from disk if it exists."""
        if not os.path.exists(self._db_path):
            return

        with open(self._db_path, "rb") as f:
            payload = pickle.load(f)

        # Old format: plain dict (dlib 128D encodings, incompatible)
        if isinstance(payload, dict) and "version" not in payload:
            print(
                "WARNING: Existing database uses old dlib encodings (128D) which are "
                "incompatible with InsightFace (512D). Database cleared. "
                "Please re-register all faces."
            )
            self._data = {}
            self.save()
        else:
            self._data = payload.get("data", {})
