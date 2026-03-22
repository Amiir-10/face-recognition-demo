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
    return FaceRecognizer(threshold=0.6)


def test_recognize_empty_database(recognizer, db):
    fake_encoding = np.random.rand(128)
    name, confidence = recognizer.recognize(fake_encoding, db)
    assert name == "Unknown"
    assert confidence == 0.0


def test_recognize_exact_match(recognizer, db):
    encoding = np.random.rand(128)
    db.add_face("amir", encoding)
    name, confidence = recognizer.recognize(encoding, db)
    assert name == "amir"
    assert confidence == 100.0


def test_recognize_no_match_above_threshold(recognizer, db):
    db.add_face("amir", np.zeros(128))
    far_encoding = np.ones(128)
    name, confidence = recognizer.recognize(far_encoding, db)
    assert name == "Unknown"
