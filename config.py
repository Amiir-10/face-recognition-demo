"""FaceID configuration constants."""
import os

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection
DETECTION_MODEL = "hog"             # "cnn" (GPU) or "hog" (CPU fallback)
DETECTION_SCALE = 0.25               # Downscale factor for detection speed
DETECTION_EVERY_N_FRAMES = 3        # Run detection every Nth frame

# Recognition
RECOGNITION_THRESHOLD = 0.6         # Lower = stricter matching
UNKNOWN_LABEL = "Unknown"

# Persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "face_database.pkl")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# Display
BOX_COLOR_KNOWN = (0, 255, 0)       # Green BGR
BOX_COLOR_UNKNOWN = (0, 0, 255)     # Red BGR
FONT_SCALE = 0.7
BOX_THICKNESS = 2
LABEL_BG_COLOR = (0, 0, 0)          # Black background behind text

# Registration
SNAPSHOT_COUNTDOWN_SECONDS = 3
DEFAULT_SNAPSHOTS = 5
