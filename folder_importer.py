"""Bulk import face images from a labeled folder structure."""
import os

import cv2
import face_recognition

from face_database import FaceDatabase
from face_encoder import FaceEncoder

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class FolderImporter:
    """Scans a folder of person_name/image.jpg and imports face encodings."""

    def import_faces(
        self,
        folder_path: str,
        database: FaceDatabase,
        encoder: FaceEncoder,
    ) -> dict[str, int]:
        """Import faces from labeled folder structure.

        Expected structure:
            folder_path/
                person_name/
                    image1.jpg
                    image2.png

        Returns:
            Dict of {name: count_of_encodings_imported}.
        """
        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            return {}

        summary: dict[str, int] = {}

        for person_name in sorted(os.listdir(folder_path)):
            person_dir = os.path.join(folder_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            name = person_name.strip().lower()
            count = 0

            for filename in sorted(os.listdir(person_dir)):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(person_dir, filename)
                image = cv2.imread(filepath)
                if image is None:
                    print(f"  Warning: Could not read {filepath}")
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb_image, model="hog")

                if len(locations) == 0:
                    print(f"  Warning: No face found in {filepath}")
                    continue
                elif len(locations) > 1:
                    print(f"  Warning: Multiple faces in {filepath}, skipping")
                    continue

                encodings = encoder.encode(image, locations)
                if encodings:
                    database.add_face(name, encodings[0])
                    count += 1
                    print(f"  Imported: {name}/{filename}")

            if count > 0:
                summary[name] = count
                print(f"  Total for '{name}': {count} encoding(s)")

        return summary
