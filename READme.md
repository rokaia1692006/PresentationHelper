# Real-Time Gesture Recognition

A real-time hand gesture recognition system that maps gestures to system actions using a trained LSTM model and MediaPipe hand tracking.

---

## What It Does

- Captures webcam feed and detects hand landmarks via MediaPipe
- Feeds landmark sequences into a two-layer LSTM classifier
- Triggers system actions on recognized gestures:
 - **swipe right** -> right arrow

 - **swipe left** -> left arrow

 - **Both hands up** -> clapping sound

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand Tracking | MediaPipe Hand Landmarker |
| Model | PyTorch LSTM (2-layer, 126-input) |
| Computer Vision | OpenCV |
| System Automation | PyAutoGUI |
| Audio Playback | ffplay |

---

## Model Architecture

Two stacked LSTM layers (64 → 32 units) with dropout (0.3) after each, followed by a fully connected head. Input is a sliding window of 15 frames, each frame being 126 floats (21 landmarks x 3 coordinates x 2 hands).

---



## Getting Started

### Install dependencies

```bash
pip install opencv-python mediapipe torch numpy pyautogui
```

ffplay must also be available on your system path (ships with ffmpeg).

### Run

```bash
python main.py
```

Press `q` to quit.

---

## Configuration

All tunable values are at the top of `realtimegestureRec()`:

| Variable | Default | Description |
|---|---|---|
| `sql` | 15 | Sliding window length (frames) |
| `holdtime` | 0.2s | How long the label stays on screen |
| `waittine` | 2s | Cooldown between repeated triggers |
| `threshs` | 0.5 | Per-gesture confidence threshold |

---

## Potential Improvements

- Training pipeline and data collection script
- WebSocket output to control remote interfaces
- GPU inference for lower latency
