import cv2
import numpy as np
import PIL.Image as Image

from paddleocr import PaddleOCR


class PaddleOCRClient:
    def __init__(self, mask_extend_size=10):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',det_lang='ml')
        self.mask_extend_size = mask_extend_size

    def draw_polygon(self, image, vertices):
        vertices = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
        cv2.fillPoly(image, [vertices], color=255)
        return image

    def run_ocr(self, image:Image.Image):
        """
        [Input]
        image: Image.Image

        [output]
        ocr_text_list: List[dict] [{"vertices":, "text": }, ...]
        annotated_image: Image.Imgae

        """

        image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        annotation_image = image[:,:,0].copy() * 0
        ocr_result = self.ocr.ocr(image, cls=True)[0]

        ocr_text_list, extract_count = [], 0
        for ocr_item in ocr_result:
            vertices = [{"x": pos[0]+extend[0]*self.mask_extend_size , "y": pos[1]+extend[1]*self.mask_extend_size} for pos, extend in zip(ocr_item[0], [[-1,-1], [1,-1], [1,1],[-1,1]])]
            text = ocr_item[1][0]
            text_id = extract_count
            polygon = self.draw_polygon(image[:,:,0].copy()*0, vertices)
            ocr_text_list.append({"id": text_id, "text": text, "vertices": vertices, "polygon": polygon})
            annotation_image = self.draw_polygon(annotation_image, vertices)
            extract_count += 1            

        return ocr_text_list, Image.fromarray(annotation_image)