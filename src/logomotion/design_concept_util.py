import re
import json

from copy import deepcopy
from omegaconf import OmegaConf
from gemini.chat import Gemini, GenerateContentConfig
from config.config_logomotion import Gemini_CONF


def set_id(layer_id):
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
    layer_id = f"{int(layer_id)}"
    _id = ''.join([MAPPING[c] for c in layer_id])
    return _id


class LayerManeger:
    @classmethod
    def get_layer_info(cls, ad_config, return_as_dict=False):
        def filter_layers(layers, filtered_layers):
            for layer in layers:
                if 'layers' in layer:
                    filter_layers(layer['layers'], filtered_layers)
                
                if layer.get('type') == 'TextLayer':
                    filtered_layers.append({'id': set_id(layer.id), 'type': 'TextLayer', 'content': layer.textitem.text, 'bbox': layer.bbox})
                elif layer.get('type') == 'ImageLayer':
                    filtered_layers.append({'id': set_id(layer.id), 'type': 'ImageLayer', 'content': layer.alt, 'bbox': layer.bbox})

        filtered_layers = []
        filter_layers(ad_config.layers, filtered_layers)

        if return_as_dict:
            return filtered_layers
        else:
            return json.dumps(filtered_layers, ensure_ascii=False, indent=4)

    @classmethod
    def group_and_sort(cls, layer_info, image):
        system_prompt = "Please output in JSON format."
        output_format = """
{
    groups: [
        {
            "group_name": "Assign names to the group",
            "sentence": "Create sentences by rearranging text in the group"
            "items": [
                {
                    "id": "Layer ID",
                    "type": "Layer Type"
                    "content": "The content of the layer",
                    "bbox": "BBox of the layer"
                },
                // List all items within groups in their rearranged order.
            ]
        },
        // List all groups.
    ]
}
        """

        prompt = f"""
Given image 1 is an advertisement image.
We aim to create an animated advertisement from the advertisement in image 1.
Please divide all layers into several animation groups considering the layout and content of each layer in the advertisement based on the JSON data below: 

## JSON data 
```
{layer_info}
```

Output conditions are as follows:
- The JSON data contains information about each layer after breaking down image 1 into layers, including content and layer IDs.
- Please assign appropriate names to groups and store them under the key `group_name`.
- Please rearrange the texts within each group into a natural sentence order and store under the key `sentence`.
- Please store the grouped layers and IDs in list format under the key `items`.

## An example of the output format
```
{output_format}
```
        """

        gemini_client = Gemini()
        result_json = gemini_client.client.models.generate_content(
            model = gemini_client.model_name,
            contents = [prompt, gemini_client.upload_image(image)],
            config = GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        ).text

        try:
            parsed_json = json.loads(result_json)
            key = list(parsed_json.keys())[0]
            return json.dumps(parsed_json[key], indent=4, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return None

class ScriptChecker:
    @staticmethod
    def delete_animation_of_target_id(script, id_list):
        def func(_script, _id_list):
            added_animations = [m.group() for m in re.finditer(r'\.add\(\{(.*?)\}\)', _script, re.DOTALL)]
            delete_count, num_animations = 0, len(added_animations)
            for ani in added_animations:
                for _id in _id_list:
                    if _id in ani:
                        delete_count += 1
                        _script = _script.replace(ani, "")
            
            if delete_count == num_animations:
                return ""
            else:
                return _script

        timeline_animations = [m.group() for m in re.finditer(r'tl\.add\(\{(.*?)\}\);', script, re.DOTALL)] + [m.group() for m in re.finditer(r'loop_block_tl\.add\(\{(.*?)\}\);', script, re.DOTALL)]
        for ani in timeline_animations:
                script = script.replace(ani, func(ani, id_list))

        return script

    @staticmethod
    def check_include_target_id(script, id_list):
        for _id in id_list:
            if _id not in script:
                return True
        return False