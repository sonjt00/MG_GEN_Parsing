import base64
import os
import numpy as np

from io import BytesIO
from PIL import Image, ImageChops, ImageDraw
from omegaconf import OmegaConf
from utility.image import concatenate_images_horizontally
from config.config_layersplit import HISAM_CONF, OCR_CONF, LAMA_CONF, LAYOUT_CONF

from modules.textremover.lama import LaMa
from modules.hisam.inference import HiSam_Inference
from modules.ocr.main import PaddleOCRClient
from modules.layout.yolo import Yolo_Client
from ultralytics import SAM
from gemini.chat import Gemini

from .extract_text_layer import _extract_text_layer2, crop_transparent, xywh2xyxy


class ConfigManeger(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 

class ADImageHTML_Util:
	@classmethod
	def pilimage_to_base64(cls, img):
		buffer = BytesIO()
		img.save(buffer, format="PNG")
		img_bytes = buffer.getvalue()
		img_base64 = base64.b64encode(img_bytes)
		img_base64_str = img_base64.decode('utf-8')
		return img_base64_str

	@staticmethod
	def set_id(number):
		MAPPING = {
			'0': 'A',
			'1': 'B',
			'2': 'C',
			'3': 'D',
			'4': 'E',
			'5': 'F',
			'6': 'G',
			'7': 'H',
			'8': 'I',
			'9': 'J',
		}
		number = f"{int(number)}"
		id = ''.join([MAPPING[c] for c in number])
		return id

	@classmethod
	def layer_to_html(cls, layer_id, layer_image, bbox, layer_name, class_name, alt=None):
		image_base64 = cls.pilimage_to_base64(layer_image)
		x, y, w, h = bbox
		_id = cls.set_id(layer_id)
		alt = alt if alt is not None else f"an image of {layer_name}"
		html_real = f"""<div style="position: absolute; margin: 0; display: flex; justify-content: center; align-items: center; left: {x}px; top: {y}px; width: {w}px; height: {h}px;"><img id="{_id}" class="{class_name}" src="data:image/png;base64,{image_base64}" alt="{alt}" style="position: relative;"></div>"""
		html_fake = f"""<div style="position: absolute; margin: 0; display: flex; justify-content: center; align-items: center; left: {x}px; top: {y}px; width: {w}px; height: {h}px;"><img id="{_id}" class="{class_name}" src="html_images/{layer_id}.png" alt="{alt}" style="position: relative;"></div>"""
		return html_real, html_fake, {'src': f"html_images/{layer_id}.png", 'img': layer_image}


class ADImageHTML_Client:
	def __init__(self, ad_id, export_config):
		self.ad_id = ad_id
		self.data_dir = export_config.data_dir
		self.title = export_config.title
		self.pil_image = Image.open(os.path.join(self.data_dir, f"{self.ad_id}_original.png")).convert("RGB")


	def get_html_contents(self):
		images_real = []
		images_fake = []
		layer_images = []

		config = ConfigManeger()
		config.data_dir = self.data_dir
		os.makedirs(os.path.join(self.data_dir, "html_images"), exist_ok=True)
		os.makedirs(os.path.join(self.data_dir, "debug_images"), exist_ok=True)


		## テキストのセグメンテーションを生成
		_, seg_text_mask = HiSam_Inference(
		    check_point_dir = "../weights",
		    model_path="sam_tss_h_textseg.pth"
		)(input_img=self.pil_image)
		seg_text_mask = seg_text_mask.point(lambda p: 255 if p >= 128 else 0)
		seg_text_mask.save(os.path.join(self.data_dir, "debug_images", "text_segmentation.png"))
		ImageChops.multiply(self.pil_image, Image.merge("RGB", (seg_text_mask, seg_text_mask, seg_text_mask))).save(os.path.join(self.data_dir, "debug_images", "segmentated_text.png"))


		## テキストのOCR
		ocr_text_list, ocr_text_mask  = PaddleOCRClient().run_ocr(self.pil_image)
		ocr_text_mask.save(os.path.join(self.data_dir, "debug_images", "ocr_mask.png"))


		## OCRとセグメンテーションから、アニメーション可能なテキストレイヤーを取得する
		config = OmegaConf.create({'id': "root", 'name': f"{self.ad_id}.png", 'layers': []})
		config.layers = [
			OmegaConf.create({'id': "background", 'name': f"{self.ad_id}.png", 'layers': [
				OmegaConf.create({'id': 9999, 'type': "BackgroundLayer", 'textitem': None, 'bbox': (0, 0, self.pil_image.width, self.pil_image.height)})
			]}),
			OmegaConf.create({'id': "imagegroup", 'name': f"{self.ad_id}.png", 'layers': []}),
			OmegaConf.create({'id': "textgroup", 'name': f"{self.ad_id}.png", 'layers': []})
		]
		text_inpaint_mask = np.zeros(self.pil_image.size[::-1], dtype=np.uint8)
		textlayer_list = []
		for ocr_textitem in ocr_text_list:
			text_layer, bbox = _extract_text_layer2(
				input_image = self.pil_image,
				text_region_mask = ocr_textitem["polygon"],
				text_segmentation_mask = seg_text_mask,
				IoU_threathhold=OCR_CONF.IoU_threathhold
			)

			if text_layer is not None:
				text_inpaint_mask = np.bitwise_or(text_inpaint_mask, np.array(ocr_textitem["polygon"]))
				textlayer_list.append({
					'id': ocr_textitem["id"],
					'bbox': bbox,
					'text': ocr_textitem["text"],
					'layer_image': text_layer
				})
				config.layers[2].layers.append(OmegaConf.create({
					'id': ocr_textitem["id"],
					'type': "TextLayer",
					'bbox': xywh2xyxy(bbox),
					'textitem': OmegaConf.create({'text': ocr_textitem["text"], 'type': None, 'direction': None, 'font': None, 'size': None, 'color': None}),
				}))


		## 背景レイヤーを生成
		text_inpaint_mask = np.logical_and(text_inpaint_mask, np.array(seg_text_mask))
		text_inpaint_mask = Image.fromarray(text_inpaint_mask.astype(np.uint8) * 255, mode='L')
		if len(textlayer_list) > 0:
			text_layer_removed_image, _, _ = LaMa(model_path="../weights/big-lama.pt").remove_text_by_mask(
				self.pil_image,
				text_inpaint_mask
			)
		else:
			text_layer_removed_image = self.pil_image.copy()

		text_all_removed_image, _, _ = LaMa(model_path="../weights/big-lama.pt").remove_text_by_mask(
			self.pil_image,
			seg_text_mask
		)
		
		text_layer_removed_image = text_layer_removed_image.convert("RGB")
		text_all_removed_image = text_all_removed_image.convert("RGB")
		text_layer_removed_image.save(os.path.join(self.data_dir, "debug_images", "text_layer_removed_image.png"))
		text_all_removed_image.save(os.path.join(self.data_dir, "debug_images", "text_all_removed_image.png"))
		ImageChops.multiply(self.pil_image, Image.merge("RGB", (seg_text_mask, seg_text_mask, seg_text_mask))).save(os.path.join(self.data_dir, "debug_images", "text_all_only.png"))
		ImageChops.multiply(self.pil_image, Image.merge("RGB", (text_inpaint_mask, text_inpaint_mask, text_inpaint_mask))).save(os.path.join(self.data_dir, "debug_images", "text_layer_only.png"))


		# YoLoの実行とクラスごとにbboxのリストを分ける
		filtered_bbox, filtered_bbox_class, yolo_annoted_image = Yolo_Client(model_path="../weights/yolov11.pt").execute(text_all_removed_image)  
		yolo_annoted_image.save(os.path.join(self.data_dir, "debug_images", "yolo_result.png"))

		layer_list, mask_list, bbox_list = [], [], []
		inpaint_mask = np.zeros(self.pil_image.size[::-1], dtype=np.uint8)
		objlayer_bbox, piclayer_bbox = [], []
		for box, cls in zip(filtered_bbox, filtered_bbox_class):
			if cls == 1:
				piclayer_bbox.append(box)
			else:
				objlayer_bbox.append(box)


		# YoLoで取得したObjectlayerのbboxを条件にSAM2でマスクを推定
		if len(objlayer_bbox) > 0:
			sam_results = SAM("../weights/sam2_b.pt")(os.path.join(self.data_dir, "debug_images", "text_layer_removed_image.png"), bboxes=objlayer_bbox)
			for bbox, mask in zip(sam_results[0].boxes.xyxy, sam_results[0].masks):
				bbox = list(bbox.cpu().numpy().astype(np.int32))
				mask_image = Image.fromarray((mask.data.squeeze()*255).cpu().numpy().astype(np.uint8))
				
				inpaint_mask = np.bitwise_or(inpaint_mask, np.array(mask_image))
				layer_image = ImageChops.multiply(text_layer_removed_image.convert("RGBA"), Image.merge("RGBA", (mask_image, mask_image, mask_image, mask_image)))

				layer_list = layer_list + [layer_image]
				bbox_list = bbox_list + [bbox]
				mask_list = mask_list + [mask_list]


		# YoLoで取得したPictureLayerのbboxを条件にSAM2でマスクを推定
		if len(piclayer_bbox) > 0:
			for bbox in piclayer_bbox:
				mask_image = Image.new("L", size=text_layer_removed_image.size, color="black")
				ImageDraw.Draw(mask_image).rectangle(xy=bbox, fill="white")
				
				inpaint_mask = np.bitwise_or(inpaint_mask, np.array(mask_image))
				layer_image = ImageChops.multiply(text_layer_removed_image.convert("RGBA"), Image.merge("RGBA", (mask_image, mask_image, mask_image, mask_image)))

				layer_list = layer_list + [layer_image]
				bbox_list = bbox_list + [bbox]
				mask_list = mask_list + [mask_list]

		if len(layer_list) > 0:
			concatenate_images_horizontally(layer_list).save(os.path.join(self.data_dir, "debug_images", "layer_image_only.png"))


		## 背景レイヤーの生成
		if len(layer_list) > 0:
			inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8), mode='L')
			bg_only_image, masked_image, mask = LaMa(model_path="../weights/big-lama.pt").remove_text_by_mask(
				text_layer_removed_image,
				inpaint_mask
			)
		else:
			bg_only_image = text_layer_removed_image.copy()
		
		bg_only_image = bg_only_image.convert("RGB")
		bg_only_image.save(os.path.join(self.data_dir, "debug_images", "layer_bg_only.png"))


		## HTMLの背景レイヤー
		html_real, html_fake, layer_image = ADImageHTML_Util.layer_to_html(
			layer_id=9999,
			layer_image=bg_only_image,
			bbox=(0,0,self.pil_image.width,self.pil_image.height),
			layer_name="background_layer",
			class_name="background_layer",
			alt=f"a simple background image"
		)
		images_real.append(html_real)
		images_fake.append(html_fake)
		layer_images.append(layer_image)
		bg_only_image.save(os.path.join(self.data_dir, "html_images", "9999.png"))


		## HTMLの画像レイヤーの<img>タグとその中身を生成
		for i_id, layer_image in enumerate(layer_list):
			layer_image, bbox = crop_transparent(layer_image)
			layer_image.save(os.path.join(self.data_dir, "html_images", f"{i_id+1000}.png"))

			try:
				gemini_client = Gemini()
				alt = gemini_client.client.models.generate_content(
					model = gemini_client.model_name,
					contents = ["Describe the image's contents and details in about 30 words in English.", gemini_client.upload_image(text_layer_removed_image.crop(xywh2xyxy(bbox)))],
				).text		
			except:
				alt = f"a simple image layer"


			html_real, html_fake, layer_image = ADImageHTML_Util.layer_to_html(
				layer_id=i_id+1000,
				layer_image=layer_image,
				bbox=bbox,
				layer_name="image_layer",
				class_name="image_layer",
				alt=alt
			)
			images_real.append(html_real)
			images_fake.append(html_fake)
			layer_images.append(layer_image)

			config.layers[1].layers.append(OmegaConf.create({
				'id': i_id+1000,
				'type': "ImageLayer",
				'bbox': bbox,
				'alt': alt,
				'textitem': None, 
			}))
			

		## HTMLのテキストレイヤーの<img>タグとその中身を生成
		for textitem in textlayer_list:
			html_real, html_fake, layer_image = ADImageHTML_Util.layer_to_html(
				layer_id=textitem["id"],
				layer_image=textitem["layer_image"],
				bbox=textitem["bbox"],
				layer_name=textitem["text"],
				class_name="text_layer",
				alt=f"an image of text `{textitem['text']}`"
			)
			images_real.append(html_real)
			images_fake.append(html_fake)
			layer_images.append(layer_image)
			textitem["layer_image"].save(os.path.join(self.data_dir, "html_images", f"{textitem['id']}.png"))


		## テキストと画像レイヤーのConfig
		OmegaConf.save(config, os.path.join(self.data_dir, f"{self.ad_id}_config.yaml"))
		

		return images_real, images_fake, layer_images
	
	def create_html(self, images_html, concept=[], script=[]):
		html = [
			"""<!DOCTYPE html>""",
			"""<html lang="ja">""",
			"""<head>""",
			"""<meta charset="UTF-8">""",
			"""<meta name="viewport" content="width=device-width, initial-scale=1.0">""",
			f"""<title>{self.title}</title>""",
			"""<script src="anime.min.js"></script>""",
			"""</head>""",
			"""<body style="margin: 0;">""",
			f"""<div id="LOADING" style="position: absolute; width: {f"{int(self.pil_image.width)}"}px; height: {f"{int(self.pil_image.height)}"}px; background-color: #ffffff; z-index: 10"></div>""",
			f"""<div style="position: relative; width: {f"{int(self.pil_image.width)}"}px; height: {f"{int(self.pil_image.height)}"}px; background-color: #f0f0f0;">"""
		] + images_html +[
			"""</div>""",
			"""<script>document.getElementById('LOADING').style.display = "none";</script>""",
			"""<!--""",
		] + concept + [
			"""-->""",
			"""<script>""",
		] + script + [
			"""</script>""",
			"""</body>""",
			"""</html>"""
		]

		return html
	
	def preview_html(self, html, console_out=False):
		html_out = ""
		for item in html:
			html_out += item + '\n'
			if console_out:
				print(item)
		return html_out

	def save_html(self, html, filename="psd_html"):
		save_path = os.path.join(self.data_dir, f"{filename}.html")
		with open(save_path, "w", encoding='utf-8') as f:
			for item in html:
				f.write(item + '\n')