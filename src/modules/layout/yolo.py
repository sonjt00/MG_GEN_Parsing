import numpy as np

from ultralytics import  YOLO
from PIL import Image
from config.config_layersplit import LAYOUT_CONF

class Yolo_Client:
    def __init__(self, model_path):
        self.yolo_model = YOLO(model_path)

    def execute(self, input_image:Image.Image): 
        img_w, img_h = input_image.size
        result = self.yolo_model(
            [input_image, ],
            iou=LAYOUT_CONF.iou,
            conf=LAYOUT_CONF.conf,
            device=[0, ], 
        )[0]

        bbox_list = [list(bbox.cpu().numpy().astype(np.int32)) for bbox in result.boxes.xyxy]
        res_img = Image.fromarray(result.plot()[..., [2, 1, 0]])  
        bbox_list = [bbox.cpu().numpy().astype(np.int32).tolist() for bbox in result.boxes.xyxy]
        bbox_class_list = [bbox.cpu().numpy().astype(np.int32).tolist() for bbox in result.boxes.cls] if result.boxes.cls is not None else [0] * len(bbox_list)

        filtered_bbox = []
        filtered_bbox_class = []
        for box, cls in zip(bbox_list, bbox_class_list): 
            if (box[2]-box[0]) * (box[3]-box[1]) > LAYOUT_CONF.layout_filter.max_region_ratio**2 * img_w * img_h: 
                continue
            if (box[2]-box[0]) * (box[3]-box[1]) < LAYOUT_CONF.layout_filter.min_region_pix: 
                continue
            filtered_bbox.append(box)
            filtered_bbox_class.append(cls)

        return filtered_bbox, filtered_bbox_class, res_img