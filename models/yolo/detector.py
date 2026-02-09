# models/yolo/detector.py

from ultralytics import YOLO


class YoloDetector:
    """
    Pure YOLO detector.
    - Loads the model once
    - Runs inference on a single frame
    - Returns raw detections (NO drawing, NO saving)
    """

    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Run YOLO on a single frame.

        Returns:
        [
            {
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            },
            ...
        ]
        """
        results = self.model(frame, conf=self.conf, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": self.class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections
