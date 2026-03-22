# FaceID -- Real-Time Face Recognition System

Real-time face recognition system using Python, OpenCV, and dlib. Detects, labels, and registers faces from a live camera feed with confidence scoring and persistent storage.

![Demo](docs/demo.png)

## Features

- Real-time face detection and recognition from webcam
- Simultaneous multi-face detection and labeling
- Name + confidence percentage displayed above each face
- Live face registration via keyboard shortcut (pause, capture snapshots, resume)
- Bulk import from labeled image folders
- Persistent face database across sessions
- FPS counter and on-screen controls HUD
- Delete and list registered faces

## Requirements

- Python 3.10+
- CMake
- C++ compiler (Visual Studio Build Tools on Windows, Xcode CLI Tools on macOS)
- Webcam

Optional:
- CUDA Toolkit + cuDNN (for GPU-accelerated CNN detection)

## Installation

### Windows

```bash
# 1. Clone the repo
git clone https://github.com/Amiir-10/face-recognition-demo.git
cd face-recognition-demo

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### macOS

```bash
# 1. Install system dependencies
xcode-select --install
brew install cmake

# 2. Clone and set up
git clone https://github.com/Amiir-10/face-recognition-demo.git
cd face-recognition-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** CUDA is not available on macOS. The system uses the HOG detection model (CPU-based), which runs at ~10-15 FPS.

## Usage

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `R` | Register a new face |
| `I` | Import faces from `known_faces/` folder |
| `L` | List all registered faces |
| `D` | Delete a registered face |
| `Q` | Quit |

### Registering a Face

1. Press `R` while the app is running
2. Enter the person's name in the terminal
3. Choose number of snapshots (1-20, default 5)
4. Look at the camera -- snapshots are captured with a 3-second countdown
5. The face is now recognized in future sessions

### Bulk Import

Place images in the `known_faces/` directory following this structure:

```
known_faces/
    person_name/
        photo1.jpg
        photo2.png
    another_person/
        photo1.jpg
```

Press `I` in the app to import. Each image must contain exactly one face.

## Configuration

Edit `config.py` to adjust:

- `DETECTION_MODEL` -- `"hog"` (CPU) or `"cnn"` (GPU with CUDA)
- `DETECTION_SCALE` -- Downscale factor for detection speed (default 0.5)
- `DETECTION_EVERY_N_FRAMES` -- Skip frames for performance (default 2)
- `RECOGNITION_THRESHOLD` -- Match strictness, lower = stricter (default 0.6)
- `CAMERA_INDEX` -- Camera device index (default 0)
- `CAMERA_WIDTH` / `CAMERA_HEIGHT` -- Resolution (default 1280x720)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Face Detection & Recognition | face_recognition (dlib-based) |
| Video Capture & Display | OpenCV |
| GPU Acceleration | dlib with CUDA (optional) |
| Data Persistence | pickle + numpy |

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
