import os
import shutil
import json
import requests
import random
import time
import runpod
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
ltx_model_dir      = "/runpod-volume/models/ltxv"
ltx_model_filename = "ltxv-13b-0.9.7-distilled.safetensors"
ltx_model_path     = os.path.join(ltx_model_dir, ltx_model_filename)
add_model_folder_path("checkpoints", ltx_model_dir)

# Ensure model exists in volume, or download once
if not os.path.exists(ltx_model_path):
    os.makedirs(ltx_model_dir, exist_ok=True)
    print("Downloading LTX model...")
    url = f"https://huggingface.co/Lightricks/LTX-Video/resolve/main/{ltx_model_filename}"
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(ltx_model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Init ComfyUI nodes
FluxPromptEnhance  = NODE_CLASS_MAPPINGS["FluxPromptEnhance"]()
CLIPLoader         = NODE_CLASS_MAPPINGS["CLIPLoader"]()
CLIPTextEncode     = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage          = NODE_CLASS_MAPPINGS["LoadImage"]()
CheckpointLoader   = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LTXVImgToVideo     = nodes_lt.NODE_CLASS_MAPPINGS["LTXVImgToVideo"]()
LTXVConditioning   = nodes_lt.NODE_CLASS_MAPPINGS["LTXVConditioning"]()
SamplerCustom      = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustom"]()
KSamplerSelect     = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
LTXVScheduler      = nodes_lt.NODE_CLASS_MAPPINGS["LTXVScheduler"]()
VAEDecode          = NODE_CLASS_MAPPINGS["VAEDecode"]()
SaveAnimatedWEBP   = nodes_images.NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()

with torch.inference_mode():
    clip  = CLIPLoader.load_clip("t5xxl_fp16.safetensors", type="ltxv")[0]
    model, _, vae = CheckpointLoader.load_checkpoint(ltx_model_filename)

def download_file(url, save_dir, file_name, max_retries=3):
    os.makedirs(save_dir, exist_ok=True)
    suffix   = os.path.splitext(urlsplit(url).path)[1]
    out_path = os.path.join(save_dir, file_name + suffix)
    headers  = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":     "image/*,*/*;q=0.8"
    }
    for attempt in range(1, max_retries+1):
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"[download_file] 429, retrying in {wait}s… (attempt {attempt})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return out_path
    raise requests.exceptions.HTTPError(
        f"Failed to download {url} after {max_retries} retries (last status {resp.status_code})"
    )

def webp_to_mp4(input_webp, output_mp4, fps=10):
    with Image.open(input_webp) as img:
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell()+1)
        except EOFError:
            pass
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame.save(p)
        paths.append(p)
    clip = ImageSequenceClip(paths, fps=fps)
    clip.write_videofile(output_mp4, codec="libx264", fps=fps)
    for p in paths: os.remove(p)
    os.rmdir(temp_dir)

@torch.inference_mode()
def generate(job):
    values      = job["input"]
    job_id      = values.get("job_id", "unknown")
    notify_uri  = values.get("notify_uri")
    notify_tok  = values.get("notify_token")

    # Download input image
    input_image = download_file(
        values["input_image"],
        "/content/ComfyUI/input",
        "input_image"
    )

    # Prepare prompts and seed
    if values["noise_seed"] == 0:
        values["noise_seed"] = random.randint(0, 2**63-1)
    positive = values["positive_prompt"]
    if values["prompt_enhance"]:
        positive = FluxPromptEnhance.enhance_prompt(positive, values["noise_seed"])[0]
    negative = values["negative_prompt"]

    # Encode conditioning
    c_pos = CLIPTextEncode.encode(clip, positive)[0]
    c_neg = CLIPTextEncode.encode(clip, negative)[0]
    img   = LoadImage.load_image(input_image)[0]

    # Generate latent video
    pos, neg, latent = LTXVImgToVideo.generate(
        c_pos, c_neg, img, vae,
        values["width"], values["height"], values["length"],
        batch_size=1
    )
    pos, neg = LTXVConditioning.append(pos, neg, values["frame_rate"])
    sampler = KSamplerSelect.get_sampler(values["sampler_name"])[0]
    sigmas  = LTXVScheduler.get_sigmas(
        values["steps"], values["max_shift"], values["base_shift"],
        values["stretch"], values["terminal"], latent=None
    )[0]
    samples = SamplerCustom.sample(
        model, values["add_noise"], values["noise_seed"],
        values["cfg"], pos, neg, sampler, sigmas, latent
    )[0]
    images  = VAEDecode.decode(vae, samples)[0].detach()

    # Save as WEBP then MP4
    ui = SaveAnimatedWEBP.save_images(
        images, values["fps"],
        filename_prefix=f"ltx-video-{values['noise_seed']}-tost",
        lossless=False, quality=90, method="default"
    )["ui"]["images"][0]["filename"]
    src = f"/content/ComfyUI/output/{ui}"
    dst = f"/content/ltx-video-{values['noise_seed']}-tost.webp"
    shutil.move(src, dst)
    webp_to_mp4(dst, f"/content/ltx-video-{values['noise_seed']}-tost.mp4", fps=values["fps"])

    result = f"/content/ltx-video-{values['noise_seed']}-tost.mp4"
    payload = {"jobId": job_id, "result": result, "status": "DONE"}
    try:
        web_uri = os.getenv("com_camenduru_web_notify_uri")
        web_tok = os.getenv("com_camenduru_web_notify_token")
        if notify_uri == "notify_uri":
            requests.post(web_uri, json=payload, headers={"Authorization": web_tok})
        else:
            requests.post(web_uri, json=payload, headers={"Authorization": web_tok})
            requests.post(notify_uri, json=payload, headers={"Authorization": notify_tok})
        return payload
    except Exception as e:
        err = {"jobId": job_id, "result": f"FAILED: {e}", "status": "FAILED"}
        try:
            if notify_uri == "notify_uri":
                requests.post(web_uri, json=err, headers={"Authorization": web_tok})
            else:
                requests.post(web_uri, json=err, headers={"Authorization": web_tok})
                requests.post(notify_uri, json=err, headers={"Authorization": notify_tok})
        except: pass
        return err
    finally:
        for f in [result, dst]:
            if os.path.exists(f):
                os.remove(f)

runpod.serverless.start({"handler": generate})
