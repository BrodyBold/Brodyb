import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import nodes_images, nodes_lt, nodes_custom_sampler
from folder_paths import add_model_folder_path

# Load custom node
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Fluxpromptenhancer")

# Register model folder as ComfyUI checkpoints folder
ltx_model_dir = "/runpod-volume/models/ltxv"
ltx_model_filename = "ltxv-13b-0.9.7-distilled.safetensors"
ltx_model_path = os.path.join(ltx_model_dir, ltx_model_filename)
add_model_folder_path("checkpoints", ltx_model_dir)

# Ensure model exists in volume, or download
if not os.path.exists(ltx_model_path):
    os.makedirs(ltx_model_dir, exist_ok=True)
    print("Downloading LTX model...")
    url = f"https://huggingface.co/Lightricks/LTX-Video/resolve/main/{ltx_model_filename}"
    r = requests.get(url, stream=True)
    with open(ltx_model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Init nodes
FluxPromptEnhance = NODE_CLASS_MAPPINGS["FluxPromptEnhance"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LTXVImgToVideo = nodes_lt.NODE_CLASS_MAPPINGS["LTXVImgToVideo"]()
LTXVConditioning = nodes_lt.NODE_CLASS_MAPPINGS["LTXVConditioning"]()
SamplerCustom = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustom"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
LTXVScheduler = nodes_lt.NODE_CLASS_MAPPINGS["LTXVScheduler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
SaveAnimatedWEBP = nodes_images.NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()

with torch.inference_mode():
    clip = CLIPLoader.load_clip("t5xxl_fp16.safetensors", type="ltxv")[0]
    model, _, vae = CheckpointLoaderSimple.load_checkpoint(ltx_model_filename)

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_path = os.path.join(save_dir, file_name + file_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def webp_to_mp4(input_webp, output_mp4, fps=10):
    with Image.open(input_webp) as img:
        frames = []
        try:
            while True:
                frame = img.copy()
                frames.append(frame)
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    temp_paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame.save(path)
        temp_paths.append(path)
    clip = ImageSequenceClip(temp_paths, fps=fps)
    clip.write_videofile(output_mp4, codec="libx264", fps=fps)
    for path in temp_paths:
        os.remove(path)
    os.rmdir(temp_dir)

@torch.inference_mode()
def generate(input):
    values = input["input"]
    job_id = values.get('job_id', 'unknown')
    notify_uri = values.get('notify_uri')
    notify_token = values.get('notify_token')

    input_image = download_file(values['input_image'], '/content/ComfyUI/input', 'input_image')
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    if values['noise_seed'] == 0:
        values['noise_seed'] = random.randint(0, 18446744073709551615)

    if values['prompt_enhance']:
        positive_prompt = FluxPromptEnhance.enhance_prompt(positive_prompt, values['noise_seed'])[0]

    conditioning_positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    conditioning_negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    image = LoadImage.load_image(input_image)[0]

    positive, negative, latent_image = LTXVImgToVideo.generate(
        conditioning_positive, conditioning_negative, image, vae,
        values['width'], values['height'], values['length'], batch_size=1
    )
    positive, negative = LTXVConditioning.append(positive, negative, values['frame_rate'])
    sampler = KSamplerSelect.get_sampler(values['sampler_name'])[0]
    sigmas = LTXVScheduler.get_sigmas(
        values['steps'], values['max_shift'], values['base_shift'],
        values['stretch'], values['terminal'], latent=None
    )[0]
    samples = SamplerCustom.sample(
        model, values['add_noise'], values['noise_seed'], values['cfg'],
        positive, negative, sampler, sigmas, latent_image
    )[0]
    images = VAEDecode.decode(vae, samples)[0].detach()
    video = SaveAnimatedWEBP.save_images(
        images, values['fps'], filename_prefix=f"ltx-video-{values['noise_seed']}-tost",
        lossless=False, quality=90, method="default"
    )

    source = f"/content/ComfyUI/output/{video['ui']['images'][0]['filename']}"
    destination = f"/content/ltx-video-{values['noise_seed']}-tost.webp"
    shutil.move(source, destination)
    webp_to_mp4(destination, f"/content/ltx-video-{values['noise_seed']}-tost.mp4", fps=values['fps'])

    result = f"/content/ltx-video-{values['noise_seed']}-tost.mp4"
    result_url = result

    try:
        payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if notify_uri == "notify_uri":
            requests.post(web_notify_uri, json=payload, headers={"Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, json=payload, headers={"Authorization": web_notify_token})
            requests.post(notify_uri, json=payload, headers={"Authorization": notify_token})
        return {"result": result_url, "status": "DONE"}
    except Exception as e:
        payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if notify_uri == "notify_uri":
                requests.post(web_notify_uri, json=payload, headers={"Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, json=payload, headers={"Authorization": web_notify_token})
                requests.post(notify_uri, json=payload, headers={"Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        for f in [result, destination]:
            if os.path.exists(f):
                os.remove(f)

runpod.serverless.start({"handler": generate})
