import shutil
import uuid
import time
import os
import gradio as gr

from utility.const import SUCCESS, ERROR
from config.config_logomotion import Gemini_CONF
from inference_batch import main
import tempfile

EXP_DIR = "../results/gradio"
NUM_GENERATION = Gemini_CONF.num_generation

def show_videos(image_path: str):
    ad_id = str(time.time()).replace(".", "") + uuid.uuid4().hex[:6]
    save_dir = f"{EXP_DIR}/{ad_id}"; os.makedirs(save_dir, exist_ok=True)
    shutil.move(image_path, f"{save_dir}/{ad_id}_original.png")
    shutil.copyfile("libs/anime.min.js", f"{save_dir}/anime.min.js")

    ret = main(ad_id, save_dir)
    if ret == SUCCESS:
        video_path_list = [] 
        tmp_dir = tempfile.gettempdir()
        for i in range(NUM_GENERATION):
            tmp_path = os.path.join(tmp_dir, f"{str(i).zfill(3)}.mp4")
            shutil.move(f"{save_dir}/{ad_id}_{str(i).zfill(3)}.mp4", tmp_path)
            video_path_list.append(tmp_path)
        return video_path_list

    else:
        raise Exception("Failed to generate videos.")


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML('<h1 align="center">MG-Gen Demo</h1><div height="50px"></div>')

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<h3 align="center">Input Image</h3>')
                image_input = gr.Image(label="Upload Image", type="filepath")
                submit_btn = gr.Button("Generate")

            with gr.Column(scale=NUM_GENERATION):
                gr.HTML('<h3 align="center">Generated Motion Graphics</h3>')
                with gr.Row():
                    videos = []
                    for idx in range(NUM_GENERATION):
                        videos.append(gr.Video(label=f"Video {idx}"))

        submit_btn.click(fn=show_videos, inputs=image_input, outputs=videos)

    demo.launch()