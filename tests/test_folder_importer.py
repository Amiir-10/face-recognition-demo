import os
import pytest
from folder_importer import FolderImporter
from face_database import FaceDatabase
from face_encoder import FaceEncoder

TEST_DB_PATH = "data/test_importer_db.pkl"


@pytest.fixture
def db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database = FaceDatabase(TEST_DB_PATH)
    yield database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_import_nonexistent_folder(db):
    importer = FolderImporter()
    encoder = FaceEncoder()
    result = importer.import_faces("/nonexistent/path", db, encoder)
    assert result == {}


def test_import_empty_folder(db, tmp_path):
    importer = FolderImporter()
    encoder = FaceEncoder()
    result = importer.import_faces(str(tmp_path), db, encoder)
    assert result == {}
