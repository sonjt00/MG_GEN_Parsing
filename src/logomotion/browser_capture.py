import asyncio
import os
import subprocess
import glob
import re 

import imageio
import concurrent.futures
from PIL import Image
from playwright.async_api import async_playwright
from config.config_logomotion import Render_CONF

class VideoUtil:
    @staticmethod
    def concatnate_videos_by_ffmpeg(video_folder, output_video_name):
        videos = sorted(glob.glob(f"{video_folder}/*_[0-9][0-9][0-9].mp4"))

        args1 = " ".join([f"-i {video}" for video in videos])
        args2 = "".join(f"[{i}:v]scale=320:-2,fps=30[v{i}];" for i in range(len(videos))) + "".join(f"[v{i}] " for i in range(len(videos)))
        args3 = f"{len(videos)}"
        args4 = f"{video_folder}/{output_video_name}"

        command = f"""ffmpeg {args1} -filter_complex "{args2}hstack=inputs={args3}[v]" -map "[v]" {args4}.mp4 -y"""
        os.system(command)
        # subprocess.run([command], capture_output=True, text=True, shell=True)
        # subprocess.run(command.split(" "), capture_output=True, text=True, shell=False)

    @staticmethod
    def preview_concept_and_animation(video_folder, output_file_name):
        pattern = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]"
        videos = sorted(glob.glob(f"{video_folder}/{pattern}.mp4"))
        htmls = sorted(glob.glob(f"{video_folder}/{pattern}.html"))
        
        if len(videos) != len(htmls):
            raise FileNotFoundError("Number of videos and htmls do not match.")

        description_and_video = ""
        for video, html in zip(videos, htmls):
            html = open(html, "r").read()
            match = re.search(r'<!--(.*?)-->', html, re.DOTALL)
            concept = (match.group(1).strip() if match else "No Concept found.").replace("\n", "<br>")
            video = video.split('/')[-1]

            description_and_video += f"""
<div class="container">
    <div class="description">{concept}</div>
    <div class="video"><video controls><source src="{video}" type="video/mp4">あなたのブラウザは video タグをサポートしていません。</video></div>
</div>
            """

        output_html = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>説明文と動画</title>
<style>
.container {
    display: flex;
    justify-content: space-between; /* スペースを均等に分配 */
    padding: 20px;
}
.description {
    margin-right: 20px;
    width: 760px;
}
.video {
    flex: 1; /* 動画の領域を広げる */
}
video {
    width: 320px;
}
</style>
</head>
<body>
        """ + description_and_video + """
</body>
</html>
        """

        save_path = f"{video_folder}/{output_file_name}.html"
        with open(save_path, "w") as f:
            f.write(output_html)



class BrowserCapture:
    def __init__(self, htmldirs, local_html_file, viewport_size={"width":1280, "height":720}):
        self.htmldirs = htmldirs
        self.local_html_file = local_html_file
        self.viewport_size = viewport_size

    async def create_error_html(self, error_message, save_path):
        error_message = error_message.replace("\n", "<br>")
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>psd_html</title>
    <style>
        body {{
            margin: 0;
        }}
        .canvas {{
            display: flex;
            position: relative;
            justify-content: center;
            align-items: center;
            width: {self.viewport_size["width"]}px;
            height: {self.viewport_size["height"]}px;
            border: 10px solid red;
        }}
        .text {{
            font-size: 64px;
            color: red;
        }}
    </style>
</head>
<body>
    <div class="canvas"><p class="text">{error_message}</p></div>
</body>
</html>
        """

        with open(save_path, "w") as f:
            f.write(html)

    async def capture_screenshot(self, screenshot_path=None):
        async with async_playwright() as p:          
            html_file_path = f"{self.htmldirs}/{self.local_html_file}.html"
            if screenshot_path is None:
                screenshot_path = f"{self.htmldirs}/{self.local_html_file}_html.png"
            if os.path.exists(html_file_path) == False:
                await self.create_error_html("HTML file\nNot found", html_file_path)

            browser = await p.chromium.launch()
            page = await browser.new_page(java_script_enabled=True, viewport=self.viewport_size)
            await page.goto(f'file:///{html_file_path}')
            await page.screenshot(path=screenshot_path)
            await browser.close()

    async def record_animation(self, video_path, duration=15, keep_original=False):
        async with async_playwright() as p:
            html_file_path = f"{self.htmldirs}/{self.local_html_file}.html"
            if os.path.exists(html_file_path) == False:
                await self.create_error_html("HTML file\nNot found", html_file_path)
            
            browser = await p.chromium.launch()
            context = await browser.new_context(
                java_script_enabled=True, 
                viewport=self.viewport_size,
                record_video_dir=self.htmldirs,
                record_video_size=self.viewport_size
            )

            page = await context.new_page()
            await page.goto(f'file:///{html_file_path}')
            await asyncio.sleep(duration)
            
            await context.close()
            await browser.close()

            recoded_video_path = await page.video.path()
            converted_video_path = f"{self.htmldirs}/{video_path}"

            if keep_original:
                os.rename(recoded_video_path, converted_video_path)
                return None

            else:
                reader = imageio.get_reader(recoded_video_path, 'ffmpeg')
                fps = reader.get_meta_data()['fps']
                writer = imageio.get_writer(converted_video_path, fps=fps, macro_block_size=1)

                for frame in reader:
                    pixel_values = list(Image.fromarray(frame).convert('L').resize((64,64), Image.NEAREST).getdata()) 
                    if all(pixel > 224 for pixel in pixel_values):
                        pass
                    else:
                        writer.append_data(frame)

                reader.close()
                writer.close()
                os.remove(recoded_video_path)


def record_video(args_dict):
    ad_id = args_dict["ad_id"]
    video_idx = args_dict["video_idx"]
    result_dir = args_dict["result_dir"]
    width = args_dict["width"]
    height = args_dict["height"]

    browser_capture = BrowserCapture(
        htmldirs=os.path.abspath(result_dir),
        local_html_file=f"{ad_id}_{video_idx:03d}",
        viewport_size={"width": width, "height": height},
    )
    flag = asyncio.run(browser_capture.record_animation(
        video_path=f"{ad_id}_{video_idx:03d}.mp4",
        duration=Render_CONF.max_video_duration
    ))

    return flag