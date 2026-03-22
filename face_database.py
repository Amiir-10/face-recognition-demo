"""Persistent storage for known face encodings."""
import os
import pickle

import numpy as np


class FaceDatabase:
    """Stores face encodings mapped to names. Persists to disk via pickle."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._data: dict[str, list[np.ndarray]] = {}
        self.load()

    def add_face(self, name: str, encoding: np.ndarray) -> None:
        """Add a face encoding for a person. Auto-saves to disk."""
        name = name.strip().lower()
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(encoding)
        self.save()

    def get_all_encodings(self) -> tuple[list[np.ndarray], list[str]]:
        """Return (flat list of all encodings, corresponding names)."""
        encodings = []
        names = []
        for name, encs in self._data.items():
            for enc in encs:
                encodings.append(enc)
                names.append(name)
        return encodings, names

    def delete_face(self, name: str) -> bool:
        """Remove a person and all their encodings. Auto-saves."""
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
        with open(self._db_path, "wb") as f:
            pickle.dump(self._data, f)

    def load(self) -> None:
        """Load database from disk if it exists."""
        if os.path.exists(self._db_path):
            with open(self._db_path, "rb") as f:
                self._data = pickle.load(f)
