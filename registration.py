"""Interactive live face registration flow."""
import time

import cv2
import numpy as np

from camera_manager import CameraManager
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_database import FaceDatabase
from display_renderer import DisplayRenderer
from config import SNAPSHOT_COUNTDOWN_SECONDS, DEFAULT_SNAPSHOTS


class RegistrationManager:
    """Handles the pause-capture-save registration workflow."""

    def __init__(
        self,
        camera: CameraManager,
        detector: FaceDetector,
        encoder: FaceEncoder,
        database: FaceDatabase,
        renderer: DisplayRenderer,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._encoder = encoder
        self._database = database
        self._renderer = renderer

    def register_face(self) -> bool:
        """Run the full registration flow via terminal + OpenCV window.

        Returns:
            True if registration was successful, False if cancelled or failed.
        """
        print("\n--- FACE REGISTRATION ---")
        name = input("Enter the person's name (or 'cancel' to abort): ").strip()
        if not name or name.lower() == "cancel":
            print("Registration cancelled.")
            return False

        num_snapshots = input(
            f"How many snapshots? (1-20, default {DEFAULT_SNAPSHOTS}): "
        ).strip()
        try:
            num_snapshots = int(num_snapshots) if num_snapshots else DEFAULT_SNAPSHOTS
            num_snapshots = max(1, min(20, num_snapshots))
        except ValueError:
            num_snapshots = DEFAULT_SNAPSHOTS

        print(f"\nCapturing {num_snapshots} snapshot(s) for '{name}'.")
        print("Look at the camera. Slight head angle changes between shots improve accuracy.\n")

        successful = 0

        for i in range(num_snapshots):
            for sec in range(SNAPSHOT_COUNTDOWN_SECONDS, 0, -1):
                ret, frame = self._camera.read_frame()
                if not ret:
                    print("Camera error during registration.")
                    return False
                msg = f"Snapshot {i + 1}/{num_snapshots} in {sec}..."
                frame = self._renderer.draw_registration_overlay(frame, msg)
                cv2.imshow("FaceID", frame)
                cv2.waitKey(1000)

            ret, frame = self._camera.read_frame()
            if not ret:
                print("Camera error during capture.")
                return False

            locations = self._detector.detect(frame, 0)

            if len(locations) == 0:
                print(f"  Snapshot {i + 1}: No face detected. Skipping.")
                continue
            elif len(locations) > 1:
                print(f"  Snapshot {i + 1}: Multiple faces detected. Skipping. (Ensure only you are in frame)")
                continue

            encodings = self._encoder.encode(frame, locations)
            if encodings:
                self._database.add_face(name.lower(), encodings[0])
                successful += 1
                print(f"  Snapshot {i + 1}: Captured successfully.")

                msg = f"Captured! ({successful}/{num_snapshots})"
                frame = self._renderer.draw_registration_overlay(frame, msg)
                cv2.imshow("FaceID", frame)
                cv2.waitKey(500)

        if successful > 0:
            print(f"\nRegistered '{name}' with {successful} encoding(s).\n")
            return True
        else:
            print(f"\nFailed to capture any valid snapshots for '{name}'.\n")
            return False
