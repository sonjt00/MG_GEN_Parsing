import glob
import os
import shutil
import time
import asyncio
import argparse
import warnings; warnings.filterwarnings("ignore")

from PIL import Image
from omegaconf import OmegaConf
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv; load_dotenv()

from adimage_html.adimage_html import ADImageHTML_Client
from utility.mylogger import Logger
from utility.const import SUCCESS, ERROR
from logomotion.design_concept import ConceptGenerator
from logomotion.html_util import HTMLUtil
from logomotion.browser_capture import BrowserCapture, record_video

from config.config_logomotion import Render_CONF


def main(ad_id, result_dir="./tmp"):
    logger = Logger(result_dir)

    # Image -> HTML
    logger.logging(f"Converting image into layerd-splitted HTML")
    try:
        client = ADImageHTML_Client(
            ad_id = ad_id,
            export_config = OmegaConf.create({
                'data_dir': result_dir,
                'title': "adimage_html",
            })
        )

        images_real, images_fake, layer_images = client.get_html_contents()
        html_real = client.create_html(images_real)
        client.save_html(html_real, filename=f"{ad_id}_noanimation")
        html_fake = client.create_html(images_fake)
        client.save_html(html_fake, filename=f"{ad_id}_exsample")

        ad_config = OmegaConf.load(f"{result_dir}/{ad_id}_config.yaml")
        pil_image = Image.open(f"{result_dir}/{ad_id}_original.png")
        width, height = pil_image.size
        with open(f"{result_dir}/{ad_id}_exsample.html", "r", encoding="utf-8") as f:
            html_noanimation = f.read()
        logger.logging(f"Success converting image to HTML")

    except Exception as e:
        logger.logging(f"Failed Converting image to html ({e})", tag="ERROR")
        return ERROR
    
    logger.logging(f"Success rendering all animated HTML")
    
    
    ## Image -> HTML
    logger.logging(f"Loading HTML, Image and Config")
    try:
        ad_config = OmegaConf.load(f"{result_dir}/{ad_id}_config.yaml")
        pil_image = Image.open(f"{result_dir}/{ad_id}_original.png")
        width, height = pil_image.size
        with open(f"{result_dir}/{ad_id}_exsample.html", "r", encoding="utf-8") as f:
            html_noanimation = f.read()
        logger.logging(f"Success loading")

    except Exception as e:
        logger.logging(f"Failed loading ({e})", tag="ERROR")
        return ERROR


    ## Render HTML and Save screenshot
    logger.logging(f"Rendering HTML without animation")
    try:
        browser_capture = BrowserCapture(
            htmldirs=os.path.abspath(result_dir),
            local_html_file=f"{ad_id}_noanimation",
            viewport_size={"width": width, "height": height},
        )
        asyncio.run(browser_capture.capture_screenshot())
        logger.logging(f"Success rendering and saving HTML as image")
    
    except Exception as e:
        logger.logging(f"Failed rendering HTML ({e})", tag="ERROR")
        return ERROR


    ## Create animation
    logger.logging(f"Creating animation")
    try: 
        created_animations = ConceptGenerator.generate_concept(html_noanimation, pil_image, ad_config)
        success_count = 0

        for i, animation in enumerate(created_animations):
            has_script = animation["has_script"]
            if has_script == False:
                logger.logging(f"Failed creating animation script: {i+1}/{len(created_animations)}", tag="ERROR")
            else:
                logger.logging(f"Success creating animation script: {i+1}/{len(created_animations)}")
                success_count += 1

        if success_count > 0:
            logger.logging(f"Success creating {success_count}/{len(created_animations)} animation scripts")
        else:
            logger.logging(f"Failed creating animation (Wrong format animations are created.)", tag="ERROR")
            return ERROR
    
    except Exception as e:
        logger.logging(f"Failed creating animation ({e})", tag="ERROR")
        return ERROR


    ## Add animation to HTML
    logger.logging("Adding animation to HTML")
    error_count, error_msg_list = 0, []
    for i, animation in enumerate(created_animations):
        if animation["has_script"]:
            try:
                concept, script = animation["idea"], animation["script"]
                html_animation = HTMLUtil.add_concept_to_html(concept, html_noanimation)
                html_animation = HTMLUtil.add_script_to_html(script, html_animation)

                save_html_path = f"{result_dir}/{ad_id}_{i:03d}.html"
                HTMLUtil.save_html(html_animation, save_html_path)
                logger.logging(f"Success adding animation to HTML: {i+1}/{len(created_animations)}")
            except Exception as e:
                logger.logging(f"Failed adding animation to HTML: {i+1}/{len(created_animations)} ({e})", tag="ERROR")
                error_count += 1

        else:
            try:
                concept, script = ["No animation created."], []
                html_animation = HTMLUtil.add_concept_to_html(concept, html_noanimation)
                html_animation = HTMLUtil.add_script_to_html(script, html_animation)

                save_html_path = f"{result_dir}/{ad_id}_{i:03d}.html"
                HTMLUtil.save_html(html_animation, save_html_path)
                logger.logging(f"No animation added to HTML: {i+1}/{len(created_animations)}")
            except Exception as e:
                logger.logging(f"Failed creating HTML: {i+1}/{len(created_animations)} ({e})", tag="ERROR")
                error_count += 1

    if error_count == len(created_animations):
        logger.logging(f"Failed adding animation to all HTML", tag="ERROR")
        return ERROR


    ## Render HTML and Save as .mp4
    logger.logging(f"Rendering HTML with animation")
    error_count = 0

    try:
        args_list = [{
            "ad_id": ad_id,
            "result_dir": result_dir,
            "video_idx": video_idx,
            "width": width,
            "height": height,
        } for video_idx in range(len(created_animations))]
    
        with ProcessPoolExecutor(max_workers=Render_CONF.num_threads) as executor:
            futures = [executor.submit(record_video, args) for args in args_list]
        
        results = []
        for video_idx, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.logging(f"[Error] Failed rendering animated HTML: {video_idx+1}/{len(created_animations)}")

    except Exception as e:
        logger.logging(f"Failed rendering all animated HTML ({e})", tag="ERROR")
        return ERROR

    if len(results) == 0:
        logger.logging(f"Failed rendering all animated HTML", tag="ERROR")
        return ERROR
    

    ## Job Successed.
    logger.logging(f"Success motion graphics generation.")
    
    return SUCCESS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='../results', help='Path to results directory')
    parser.add_argument('--exp_id', default='exp001', help='Experiment ID')
    parser.add_argument('--testset_dir', default='../example_inputs', help='Path to testset directory')
    parser.add_argument('--target_sets', default='*', help='Glob pattern for target sets')
    parser.add_argument('--max_sets', type=int, default=9999, help='Max sets to process')
    
    args = parser.parse_args()
    setattr(args, "exp_dir", f"{args.result_dir}/{args.exp_id}")
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"{args.exp_dir}", exist_ok=True)
    base_logger = Logger(f"{args.result_dir}/{args.exp_id}")

    testset_list = sorted(glob.glob(f"{args.testset_dir}/{args.target_sets}.png"))[:args.max_sets]
    total = len(testset_list)

    for count, testset_path in enumerate(testset_list):
        ad_id = os.path.split(testset_path)[1]
        save_dir = f"{args.exp_dir}/{ad_id}"; os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(testset_path, f"{save_dir}/{ad_id}_original.png")
        shutil.copyfile("libs/anime.min.js", f"{save_dir}/anime.min.js")

        start = time.time()
        try:
            ret = main(ad_id, save_dir)
            error_msg = ""
        
        except Exception as e:
            ret = ERROR
            error_msg = "Error: " + str(e)
        
        end = time.time()
        flag = f"Success" if ret == SUCCESS else f"Failed"
        base_logger.logging(f"Done {count+1:03d}/{total:03d} {flag}: {ad_id} Process Time: {end - start} {error_msg}")
