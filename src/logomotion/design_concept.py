import re
import concurrent.futures

from gemini.chat import Gemini, GenerateContentConfig
from config.config_logomotion import Animation_CONF, Gemini_CONF
from .design_concept_util import LayerManeger
from .design_concept_prompt import PopupAnimation

class ConceptGenerator:
    @classmethod
    def create_newline(cls, text): ## 改行文字の削除
        lines = text.split('\\n') 
        lines = [line.replace('\\', '') for line in lines]
        return lines

    @classmethod
    def _generate_concept(cls, args):
        html = args["html"]
        image = args["image"]
        psd_config = args["psd_config"]

        prompt_lang = args["prompt_lang"]
        animation_concept = args["animation_concept"]
        animation_script = args["animation_script"]
        animation_concept_edit = args["animation_concept_edit"]
        
        ## 前処理
        layer_info = LayerManeger.get_layer_info(psd_config, return_as_dict=True)
        grouped_config = LayerManeger.group_and_sort(layer_info, image)
        prompt = PopupAnimation.get_prompt(prompt_lang=prompt_lang)
        script_postprocess = PopupAnimation.script_postprocess
        
        prompt = prompt.replace("[IMAGE_HTML]", html)
        prompt = prompt.replace("[LAYER_INFO]", grouped_config)
        prompt = prompt.replace("[ANIMATION_IDEA]", animation_concept)
        prompt = prompt.replace("[ANIMATION_SCRIPT]", animation_script)
        prompt = prompt.replace("[ANIMATION_IDEA_EDIT]", animation_concept_edit)

        ## Gemini呼び出し
        gemini_client = Gemini()
        res = gemini_client.client.models.generate_content(
            model = gemini_client.model_name,
            contents = [prompt, gemini_client.upload_image(image)],
            config = GenerateContentConfig(
                temperature=0.0,
            )
        ).text


        ## 後処理
        match = re.search(r'<idea>(.*?)</idea>', res, re.DOTALL)
        if match:
            idea = match.group(1).strip()
            idea = cls.create_newline(idea)
            idea = idea + ["", "layer_info"] + cls.create_newline(grouped_config)
        else:
            idea = None

        match = re.search(r'<script>(.*?)</script>', res, re.DOTALL)
        if match:
            script = match.group(1).strip()
            script = script_postprocess(script)
            script = cls.create_newline(script)
        else:
            script = None
            
        return {"has_idea": (idea is not None), "has_script": (script is not None),  "idea":idea, "script":script}

    @classmethod
    def generate_concept(cls, html, image, psd_config, animation_concept="", animation_script="", animation_concept_edit=""):
        args_dict = {
            "html": html, 
            "psd_config": psd_config, 
            "image": image, 
            "prompt_lang": "jp",
            "animation_concept": animation_concept,
            "animation_script": animation_script,
            "animation_concept_edit": animation_concept_edit
        }

        num_generation = Gemini_CONF.num_generation
        num_threads = Gemini_CONF.num_threads

        if num_generation == 1:
            results = [cls._generate_concept(args_dict)]
            return results
        
        elif num_generation > 1 and num_threads == 1:
            results = [cls._generate_concept(args_dict) for _ in range(num_generation)]
            return results

        elif num_generation > 1 and num_threads > 1:
            args_list = [args_dict for _ in range(num_generation)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                responses = [executor.submit(cls._generate_concept, args) for args in args_list]
                results = [response.result() for response in responses]
                return results
            
        else:
            raise ValueError("num_generation must be greater than 0")