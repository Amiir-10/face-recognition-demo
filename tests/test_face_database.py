import os
import numpy as np
import pytest
from face_database import FaceDatabase

TEST_DB_PATH = "data/test_face_database.pkl"


@pytest.fixture
def db():
    """Fresh database for each test."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database = FaceDatabase(TEST_DB_PATH)
    yield database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def sample_encoding():
    """Fake 128D face encoding."""
    return np.random.rand(128)


def test_empty_database(db):
    assert db.face_count() == 0
    assert db.list_faces() == []


def test_add_face(db, sample_encoding):
    db.add_face("amir", sample_encoding)
    assert db.face_count() == 1
    assert "amir" in db.list_faces()


def test_add_multiple_encodings_same_person(db, sample_encoding):
    db.add_face("amir", sample_encoding)
    db.add_face("amir", np.random.rand(128))
    assert db.face_count() == 1
    encodings, names = db.get_all_encodings()
    assert names.count("amir") == 2


def test_get_all_encodings(db, sample_encoding):
    db.add_face("amir", sample_encoding)
    db.add_face("omar", np.random.rand(128))
    encodings, names = db.get_all_encodings()
    assert len(encodings) == 2
    assert len(names) == 2
    assert set(names) == {"amir", "omar"}


def test_delete_face(db, sample_encoding):
    db.add_face("amir", sample_encoding)
    result = db.delete_face("amir")
    assert result is True
    assert db.face_count() == 0


def test_delete_nonexistent_face(db):
    result = db.delete_face("nobody")
    assert result is False


def test_persistence(sample_encoding):
    """Database survives being closed and reopened."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    db1 = FaceDatabase(TEST_DB_PATH)
    db1.add_face("amir", sample_encoding)
    del db1

    db2 = FaceDatabase(TEST_DB_PATH)
    assert db2.face_count() == 1
    assert "amir" in db2.list_faces()

    os.remove(TEST_DB_PATH)


def test_empty_get_all_encodings(db):
    encodings, names = db.get_all_encodings()
    assert encodings == []
    assert names == []
