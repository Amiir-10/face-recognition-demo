"""FaceID -- Real-Time Face Recognition System. Entry point."""
import time
import sys

import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    KNOWN_FACES_DIR,
    DATABASE_PATH,
)
from camera_manager import CameraManager
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_database import FaceDatabase
from face_recognizer import FaceRecognizer
from display_renderer import DisplayRenderer
from registration import RegistrationManager
from folder_importer import FolderImporter


def print_controls() -> None:
    """Print keyboard controls to terminal."""
    print("\n" + "=" * 45)
    print("  FaceID -- Real-Time Face Recognition")
    print("=" * 45)
    print("  Controls:")
    print("    R  -- Register a new face")
    print("    I  -- Import faces from folder")
    print("    L  -- List registered faces")
    print("    D  -- Delete a registered face")
    print("    Q  -- Quit")
    print("=" * 45 + "\n")


def main() -> None:
    print("Starting FaceID...")

    try:
        camera = CameraManager(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    detector = FaceDetector()
    encoder = FaceEncoder()
    database = FaceDatabase(DATABASE_PATH)
    recognizer = FaceRecognizer()
    renderer = DisplayRenderer()
    registration_mgr = RegistrationManager(camera, detector, encoder, database, renderer)
    importer = FolderImporter()

    print(f"Loaded {database.face_count()} registered face(s).")
    print(f"Using detection model: {detector.model}")
    print_controls()

    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    cached_results: list[tuple[str, float]] = []

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("Failed to read from camera. Exiting.")
                break

            is_detection_frame = frame_count % detector._every_n_frames == 0
            face_locations = detector.detect(frame, frame_count)

            if is_detection_frame:
                results = []
                if face_locations:
                    encodings = encoder.encode(frame, face_locations)
                    for enc in encodings:
                        name, confidence = recognizer.recognize(enc, database)
                        results.append((name, confidence))

                while len(results) < len(face_locations):
                    results.append(("Unknown", 0.0))

                cached_results = results
            else:
                results = cached_results

            frame = renderer.draw_results(frame, face_locations, results)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time

            frame = renderer.draw_hud(frame, fps, database.face_count())

            cv2.imshow("FaceID", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("Quitting FaceID...")
                break

            elif key == ord("r") or key == ord("R"):
                registration_mgr.register_face()

            elif key == ord("i") or key == ord("I"):
                print(f"\nImporting faces from: {KNOWN_FACES_DIR}")
                summary = importer.import_faces(KNOWN_FACES_DIR, database, encoder)
                if summary:
                    print(f"Import complete: {sum(summary.values())} total encoding(s).\n")
                else:
                    print("No faces imported. Check folder structure.\n")

            elif key == ord("l") or key == ord("L"):
                faces = database.list_faces()
                if faces:
                    print(f"\nRegistered faces ({len(faces)}):")
                    for name in sorted(faces):
                        print(f"  - {name.title()}")
                    print()
                else:
                    print("\nNo faces registered yet.\n")

            elif key == ord("d") or key == ord("D"):
                name = input("\nEnter name to delete (or 'cancel'): ").strip()
                if name and name.lower() != "cancel":
                    if database.delete_face(name.lower()):
                        print(f"Deleted '{name}'.\n")
                    else:
                        print(f"'{name}' not found in database.\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("FaceID closed.")


if __name__ == "__main__":
    main()
