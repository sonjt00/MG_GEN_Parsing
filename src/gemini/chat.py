import io
import os

from google import genai
from google.genai.types import GenerateContentConfig, Part
from PIL import Image as PILImage


class Gemini:
    def __init__(self, model_name:str=None):
        self.model_name = model_name if model_name is not None else "gemini-2.0-pro-exp-02-05"  
        self.client = genai.Client(api_key=os.getenv("GEMINI_APIKEY"))
        # self.client = genai.Client(vertexai=True, location="us-central1", project="your-project-id") # for Vertex AI

    def upload_image(self, pil_image:PILImage.Image):
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        return Part.from_bytes(data=image_bytes.read(), mime_type='image/png')