import os
import torch
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageChops
from .util import crate_mask, ResizeKeepAspectRatio

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

class LaMa:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_jit_model(model_path).eval()

    def load_jit_model(self, model_path):
        model = torch.jit.load(model_path, map_location="cpu").to(self.device)
        model.eval()
        return model

    def forward(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return Image.fromarray(cur_res)
    
    def remove_text_by_mask(self, base_image:Image.Image, mask:Image.Image):
        ## 1024x1024または512x512しか対応しない？ 
        # square_size =  1024 if np.max(base_image.size) > 512 else 512
        square_size = 512 #fix

        resize_func = ResizeKeepAspectRatio(base_image)
        base_image = resize_func.forward(target_size=(square_size, square_size))
        base_image = base_image.convert("RGB")
        mask = resize_func.forward(mask, target_size=(square_size, square_size), bg_color=(0,0,0)).convert("L")
        mask = crate_mask(mask)

        _mask = ImageOps.invert(mask)
        masked_base_image = ImageChops.multiply(base_image, Image.merge("RGB", (_mask, _mask, _mask)))
        masked_base_image = base_image.copy()
        masked_base_image.putalpha(_mask)

        image = self.forward(
            image=np.array(base_image)[:, :, ::-1],
            mask=np.array(mask),
        )
        image = resize_func.reverse(image)
        masked_base_image = resize_func.reverse(masked_base_image)
        mask = resize_func.reverse(mask)

        return image, masked_base_image, mask