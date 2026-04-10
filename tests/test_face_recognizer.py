import numpy as np
import pytest
import os
from face_recognizer import FaceRecognizer
from face_database import FaceDatabase

TEST_DB_PATH = "data/test_recognizer_db.pkl"


@pytest.fixture
def db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database = FaceDatabase(TEST_DB_PATH)
    yield database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def recognizer():
    return FaceRecognizer(threshold=0.35)


def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized vector."""
    return v / np.linalg.norm(v)


def test_recognize_empty_database(recognizer, db):
    fake_embedding = _unit(np.random.rand(512).astype(np.float32))
    name, confidence = recognizer.recognize(fake_embedding, db)
    assert name == "Unknown"
    assert confidence == 0.0


def test_recognize_exact_match(recognizer, db):
    embedding = _unit(np.random.rand(512).astype(np.float32))
    db.add_face("amir", embedding)
    name, confidence = recognizer.recognize(embedding, db)
    assert name == "amir"
    assert confidence == 100.0


def test_recognize_no_match_above_threshold(recognizer, db):
    # Store a unit vector in one direction, query with an orthogonal vector
    # Cosine similarity = 0.0, which is below threshold 0.35 -> Unknown
    stored = np.zeros(512, dtype=np.float32)
    stored[0] = 1.0
    db.add_face("amir", stored)

    query = np.zeros(512, dtype=np.float32)
    query[1] = 1.0  # orthogonal to stored
    name, confidence = recognizer.recognize(query, db)
    assert name == "Unknown"
