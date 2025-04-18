import numpy as np
import cv2
import re

from PIL import Image, ImageChops, ImageOps

def reuturn_error():
    return None, None

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def check_character_size(ocr_text, text_region_mask:Image.Image, text_segmentation_mask:Image.Image, charactersize_threathhold:str):
    ## サイズの小さい記号は排除
    KIGOU_ASCII = r'["\'*+,\-.<=>^_`~ ]'
    KIGOU_ZENKAKU = r'[．，。、「」『』〜ー・　]'
    ocr_text = re.sub(KIGOU_ASCII, '', re.sub(KIGOU_ZENKAKU, '', ocr_text))
    num_text = len(ocr_text)

    ## 文字の平均サイズを取得（面積ベース）
    mask = np.logical_and(np.array(text_region_mask), np.array(text_segmentation_mask))
    pixels = list(mask.reshape(-1))
    mask_area_size = pixels.count(True)
    avg_char_size = 0 if num_text == 0 else (mask_area_size/num_text)

    if charactersize_threathhold.endswith("px"):
        return avg_char_size >= int(charactersize_threathhold.replace("px", ""))**2

    elif charactersize_threathhold.endswith("%"):
        return avg_char_size/(np.min(text_segmentation_mask.size)**2) >= (float(charactersize_threathhold.replace("%", ""))*0.01)**2

    else:
        raise ValueError(f"Wrong charactersize format is given: {charactersize_threathhold}")

def make_chromakey_mask(input_image:Image.Image, seg:Image.Image, color_range="5%", lower=7, upper=12):
    def crate_mask(mask, extend_mask_size):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extended_mask = np.zeros_like(mask)
        for contour in contours:
            cv2.drawContours(extended_mask, [contour], -1, 255, extend_mask_size) # マスクを拡張
            cv2.drawContours(extended_mask, [contour], -1, 255, -1) # マスクの中をすべて塗りつぶし
        return extended_mask

    seg = np.array(seg)
    lower_mask = crate_mask(seg, lower)
    upper_mask = crate_mask(seg, upper)

    check_mask = Image.fromarray(upper_mask - lower_mask)
    bg_colorlist = [c for c in set(list(ImageChops.multiply(input_image.convert("L"), Image.merge("L", (check_mask,))).getdata())) if c != 0]
    color_range = 255 / int(color_range.replace('%',''))

    if np.max(bg_colorlist) - np.min(bg_colorlist) <= color_range:
        bg_color_center = np.mean([c for c in list(ImageChops.multiply(input_image.convert("RGB"), Image.merge("RGB", (check_mask,check_mask,check_mask))).getdata()) if c != (0,0,0)], axis=0)
        threshold = ((color_range/2)**2 + (color_range/2)**2 + (color_range/2)**2) ** 0.5
        
        input_image_array = np.array(input_image.convert("RGB"))
        chromakey_mask = np.zeros_like(seg)
        height, width = seg.shape
        for y in range(height):
            for x in range(width):
                if np.sqrt(np.sum((input_image_array[y, x] - bg_color_center) ** 2)) <= threshold:
                    chromakey_mask[y, x] = 255

        return ImageOps.invert(Image.fromarray(chromakey_mask))
    else:
        return None

def crop_transparent(im):
    if im.mode == "RGB":
        return im, (0, 0, im.width, im.height)

    pix = im.load()
    left, top, right, bottom = im.width, im.height, 0, 0

    for y in range(im.height):
        for x in range(im.width):
            if pix[x, y][3] != 0:  # 透明じゃないピクセルを探す（alpha channelが0でない）
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    if right > left and bottom > top:
        return im.crop((left, top, right + 1, bottom + 1)), (left, top, right + 1 - left, bottom + 1 -top)
    else:
        return im, (0, 0, im.width, im.height)

def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

def xywh2xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def _extract_text_layer(
        input_image: Image.Image, 
        text_region_mask:Image.Image, 
        text_segmentation_mask:Image.Image, 
        ocr_text:str,
        IoU_threathhold=0.001,
        do_crop_trasparent=True,
        option_skipsmallchar={'charactersize_threathhold': "16px"},
        option_chromakey={'color_range':'5%', 'lower':7, 'upper':12}
    ):
    input_image = input_image.convert("RGBA")
    mask1, mask2 = np.array(text_region_mask) // 255, np.array(text_segmentation_mask) // 255
    
    ## IoUフィルタリング
    iou = calculate_iou(mask1, mask2)
    if iou < IoU_threathhold:
        return reuturn_error()

    ## 文字サイズフィルタリング
    if option_skipsmallchar is not None:
        charactersize_threathhold = option_skipsmallchar.get("charactersize_threathhold", "16px")
        if check_character_size(ocr_text, text_region_mask, text_segmentation_mask, charactersize_threathhold) == False:
            return reuturn_error()

    ## クロマキー
    mask = Image.fromarray(np.logical_and(mask1, mask2)).convert("L")

    if option_chromakey is not None:
        color_range = option_chromakey.get("color_range", '10%')
        lower = option_chromakey.get("lower", 7)
        upper = option_chromakey.get("upper", 12)
        chromakey_mask = make_chromakey_mask(input_image, mask, color_range, lower, upper)

        if chromakey_mask is not None:
            mask = chromakey_mask

    ## テキストレイヤーの抽出
    text_layer_image = ImageChops.multiply(input_image, Image.merge("RGBA", (mask, mask, mask, mask)))
    if do_crop_trasparent:
        text_layer_image, bbox = crop_transparent(text_layer_image)
    else:
        text_layer_image, bbox = text_layer_image, (0,0,text_layer_image.width,text_layer_image.height)
    
    return text_layer_image, bbox


def _extract_text_layer2(
        input_image: Image.Image, 
        text_region_mask:Image.Image, 
        text_segmentation_mask:Image.Image, 
        IoU_threathhold:float,
    ):
    input_image = input_image.convert("RGBA")
    region_mask, segmentaion_mask = np.array(text_region_mask)//255, np.array(text_segmentation_mask)//255
    text_mask = np.logical_and(region_mask, segmentaion_mask)*1
    
    ## 文字フィルタリング
    if np.sum(text_mask) / np.sum(region_mask) < IoU_threathhold:
        return reuturn_error()

    ## テキストレイヤーの抽出
    mask = Image.fromarray(text_mask.astype(np.uint8)*255).convert("L")
    text_layer_image = ImageChops.multiply(input_image, Image.merge("RGBA", (mask, mask, mask, mask)))
    text_layer_image, bbox = crop_transparent(text_layer_image)
    return text_layer_image, bbox