import argparse
import cv2
from ultralytics import YOLO
from tracker import CentroidTracker
from counter import LineCounter

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {2: (0, 255, 0), 3: (255, 165, 0), 5: (0, 0, 255), 7: (255, 0, 255)}


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Tracking & Counting")
    p.add_argument("--source", default="0", help="Video file path or webcam index")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--line-ratio", type=float, default=0.6, help="Counting line position (0-1 of frame height)")
    p.add_argument("--save", default="", help="Output video path (empty = display only)")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    line_y = int(height * args.line_ratio)

    tracker = CentroidTracker(max_disappeared=20)
    counter = LineCounter(line_y, direction="down")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, classes=list(VEHICLE_CLASSES.keys()), verbose=False)[0]

        boxes, class_ids = [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            class_ids.append(cls)

        tracked = tracker.update(boxes)
        counter.update(tracked)

        # draw detections
        for (x1, y1, x2, y2), cls in zip(boxes, class_ids):
            color = COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, VEHICLE_CLASSES[cls], (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # draw tracked centroids & IDs
        for obj_id, (cx, cy) in tracked.items():
            cx, cy = int(cx), int(cy)
            dot_color = (0, 255, 255) if obj_id in counter.counted_ids else (255, 255, 0)
            cv2.circle(frame, (cx, cy), 5, dot_color, -1)
            cv2.putText(frame, f"ID {obj_id}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1)

        # draw counting line & count
        cv2.line(frame, (0, line_y), (width, line_y), (0, 165, 255), 2)
        cv2.putText(frame, f"Vehicles: {counter.count}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)

        if writer:
            writer.write(frame)

        cv2.imshow("Vehicle Tracking & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nTotal vehicles counted: {counter.count}")


if __name__ == "__main__":
    main()
