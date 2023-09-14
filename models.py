from pathlib import Path
from typing import Tuple

import numpy as np
import torch as torch

from ultralytics import YOLO

model = YOLO('models/best_new.pt')

mapping = {
    0: 1,
    1: 2,
    2: 0
}


class YoloModel:
    def select(self, results: np.array) -> Tuple:
        boxes = results[0].boxes  # Boxes object for bbox outputs
        to_int = lambda x: (int(torch.round(num)) for num in x)

        bboxes = tuple(map(to_int, boxes.xyxy))
        classes = np.fromiter(map(lambda x: mapping[x]+1, boxes.cls.numpy()), dtype=int)

        # bboxes = boxes.xyxyn
        # classes = np.fromiter(map(lambda x: mapping[x], boxes.cls.numpy()), dtype=int)

        return bboxes, classes, boxes.conf

    def predict(self, image: Path) -> Tuple:
        results = model([image], max_det=1000, conf=0.0, iou=0.62)
        selected = self.select(results)
        return selected
