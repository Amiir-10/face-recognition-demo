"""FaceID -- Real-Time Face Recognition System. Entry point."""
import time
import sys

import cv2
from insightface.app import FaceAnalysis

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    KNOWN_FACES_DIR,
    DATABASE_PATH,
    INSIGHTFACE_MODEL,
    INSIGHTFACE_CTX_ID,
    INSIGHTFACE_DET_SIZE,
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
    print("Loading InsightFace model (first run downloads ~300MB)...")

    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=INSIGHTFACE_DET_SIZE)

    try:
        camera = CameraManager(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    detector = FaceDetector(app)
    encoder = FaceEncoder()
    database = FaceDatabase(DATABASE_PATH)
    recognizer = FaceRecognizer()
    renderer = DisplayRenderer()
    registration_mgr = RegistrationManager(camera, detector, encoder, database, renderer)
    importer = FolderImporter(app)

    print(f"Loaded {database.face_count()} registered face(s).")
    print_controls()

    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    cached_results: list[tuple[str, float]] = []
    cached_faces: list = []

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                # Background thread hasn't captured first frame yet
                continue

            is_detection_frame = frame_count % detector._every_n_frames == 0
            faces = detector.detect(frame, frame_count)

            if is_detection_frame:
                results = []
                if faces:
                    embeddings = encoder.encode(faces)
                    for emb in embeddings:
                        name, confidence = recognizer.recognize(emb, database)
                        results.append((name, confidence))
                while len(results) < len(faces):
                    results.append(("Unknown", 0.0))
                cached_results = results
                cached_faces = faces
            else:
                results = cached_results
                faces = cached_faces

            # InsightFace bbox: [x1, y1, x2, y2] = (left, top, right, bottom)
            bboxes = [
                (int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3]))
                for f in faces
            ]
            frame = renderer.draw_results(frame, bboxes, results)

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
                summary = importer.import_faces(KNOWN_FACES_DIR, database)
                if summary:
                    print(f"Import complete: {sum(summary.values())} total embedding(s).\n")
                else:
                    print("No faces imported. Check folder structure.\n")

            elif key == ord("l") or key == ord("L"):
                faces_list = database.list_faces()
                if faces_list:
                    print(f"\nRegistered faces ({len(faces_list)}):")
                    for name in sorted(faces_list):
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
