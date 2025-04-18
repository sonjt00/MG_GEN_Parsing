import cv2
import numpy as np
from PIL import Image

class YoloAnnotator:
    def __init__(self, classes=["layer"]):
        self.classes = classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_yolo_annotation(self, image, bbox_list, bbox_class_list=None):
        image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        if bbox_class_list is None:
            bbox_class_list = [0] * len(bbox_list)

        for bbox, bbox_class in zip(bbox_list, bbox_class_list):
            x1, y1, x2, y2 = bbox
            color = self.colors[bbox_class]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, self.classes[bbox_class], (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

 