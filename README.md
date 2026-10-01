# Vehicle Tracking & Counting System

Detects, tracks, and counts vehicles from a video stream using **YOLOv8** and a centroid-based tracker.

## How it works

1. **Detection** — YOLOv8 runs per-frame inference, filtering for vehicle classes (car, motorcycle, bus, truck).
2. **Tracking** — A centroid tracker assigns persistent IDs across frames using nearest-neighbour matching on bounding-box centroids.
3. **Counting** — A virtual counting line (horizontal) detects when a tracked centroid crosses it in the configured direction, incrementing the count once per unique ID.

## Setup

```bash
# Create a virtual environment (required on macOS)
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

## Usage

> Always use `.venv/bin/python` to run the script (macOS does not have a bare `python` command).

```bash
# Run on a video file
.venv/bin/python main.py --source traffic.mp4

# Run on webcam
.venv/bin/python main.py --source 0

# Save output video
.venv/bin/python main.py --source traffic.mp4 --save output.mp4

# Tune confidence & line position
.venv/bin/python main.py --source traffic.mp4 --conf 0.5 --line-ratio 0.55
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolov8n.pt` | YOLOv8 weights (n/s/m/l/x) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--line-ratio` | `0.6` | Counting line height (0.0–1.0 of frame) |
| `--save` | `` | Output `.mp4` path (omit to display only) |

## Project Structure

```
├── main.py          # Pipeline entry point
├── tracker.py       # Centroid tracker
├── counter.py       # Line-crossing counter
└── requirements.txt
```

## Visual Indicators

- Green/coloured boxes — detected vehicles with class label
- Yellow dot — tracked centroid (not yet counted)
- Cyan dot — tracked centroid (already counted)
- Orange line — virtual counting line
- Top-left counter — running total
