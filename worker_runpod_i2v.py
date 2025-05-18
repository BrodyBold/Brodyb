import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import nodes_images, nodes_lt, nodes_custom_sampler
from folder_paths import add_model_folder_path

# ————————————————
# 1) Setup & Model Download
# ————————————————

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Fluxpromptenhancer")

ltx_dir  = "/runpod-volume/models/ltxv"
ltx_file = "ltxv-13b-0.9.7-distilled.safetensors"
ltx_path = os.path.join(ltx_dir, ltx_file)

add_model_folder_path("checkpoints", ltx_dir)

if not os.path.exists(ltx_path):
    os.makedirs(ltx_dir, exist_ok=True)
    print("Downloading LTX model…")
    dl_url = f"https://huggingface.co/Lightricks/LTX-Video/resolve/main/{ltx_file}"
    resp = requests.get(dl_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(ltx_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk: f.write(chunk)
    print("Download complete.")

# ————————————————
# 2) Init Nodes & Models
# ————————————————

FluxEnhance = NODE_CLASS_MAPPINGS["FluxPromptEnhance"]()
CLIPLoader  = NODE_CLASS_MAPPINGS["CLIPLoader"]()
CLIPTextEnc = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage   = NODE_CLASS_MAPPINGS["LoadImage"]()
ChkptLoader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
Img2Vid     = nodes_lt.NODE_CLASS_MAPPINGS["LTXVImgToVideo"]()
Condition   = nodes_lt.NODE_CLASS_MAPPINGS["LTXVConditioning"]()
SamplerC    = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustom"]()
SamplerSel  = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
Scheduler   = nodes_lt.NODE_CLASS_MAPPINGS["LTXVScheduler"]()
VAEDec      = NODE_CLASS_MAPPINGS["VAEDecode"]()
SaveWEBP    = nodes_images.NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()

with torch.inference_mode():
    clip = CLIPLoader.load_clip("t5xxl_fp16.safetensors", type="ltxv")[0]
    model, _, vae = ChkptLoader.load_checkpoint(ltx_file)

# ————————————————
# 3) Helpers
# ————————————————

def download_file(url, out_dir, name, retries=3):
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(urlsplit(url).path)[1]
    out = os.path.join(out_dir, name + ext)
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept":"image/*,*/*;q=0.8"
    }
    for i in range(1, retries+1):
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        if r.status_code == 429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        with open(out, "wb") as f:
            for c in r.iter_content(8192):
                if c: f.write(c)
        return out
    raise requests.HTTPError(f"429 too many requests for {url}")

def webp_to_mp4(wp, mp4, fps=10):
    with Image.open(wp) as img:
        frames=[]
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell()+1)
        except EOFError: pass
    tmp="temp_frames"; os.makedirs(tmp,exist_ok=True)
    paths=[]
    for idx,fr in enumerate(frames):
        p=f"{tmp}/f{idx:04d}.png"; fr.save(p); paths.append(p)
    clipSeq = ImageSequenceClip(paths, fps=fps)
    clipSeq.write_videofile(mp4, codec="libx264", fps=fps)
    for p in paths: os.remove(p)
    os.rmdir(tmp)

# ————————————————
# 4) Handler
# ————————————————

@torch.inference_mode()
def generate(job):
    vals = job["input"]
    jid  = vals.get("job_id","-")

    try:
        # 1) download input image
        img_path = download_file(vals["input_image"], "/content/ComfyUI/input", "inp")

        # 2) seed & prompts
        seed = vals["noise_seed"] or random.randrange(2**63)
        pos  = vals["positive_prompt"]
        if vals["prompt_enhance"]:
            pos = FluxEnhance.enhance_prompt(pos, seed)[0]
        neg = vals["negative_prompt"]

        # 3) encode & generate latent
        cpos=CLIPTextEnc.encode(clip, pos)[0]
        cneg=CLIPTextEnc.encode(clip, neg)[0]
        img = LoadImage.load_image(img_path)[0]

        # include strength param (default 1.0)
        strength = vals.get("strength", 1.0)
        p,n,lat = Img2Vid.generate(
            cpos, cneg, img, vae,
            vals["width"], vals["height"], vals["length"],
            strength,
            batch_size=1
        )
        p,n    = Condition.append(p, n, vals["frame_rate"])
        sam    = SamplerSel.get_sampler(vals["sampler_name"])[0]
        sigmas = Scheduler.get_sigmas(
            vals["steps"], vals["max_shift"], vals["base_shift"],
            vals["stretch"], vals["terminal"], latent=None
        )[0]
        samp   = SamplerC.sample(model, vals["add_noise"], seed,
                 vals["cfg"], p, n, sam, sigmas, lat)[0]
        imgs   = VAEDec.decode(vae, samp)[0].detach()

        # 4) save WEBP→MP4
        ui = SaveWEBP.save_images(
            imgs, vals["fps"],
            filename_prefix=f"ltx-{seed}",
            lossless=False, quality=90, method="default"
        )["ui"]["images"][0]["filename"]
        src = f"/content/ComfyUI/output/{ui}"
        wp  = f"/content/ltx-{seed}.webp"
        shutil.move(src, wp)
        mp4 = f"/content/ltx-{seed}.mp4"
        webp_to_mp4(wp, mp4, fps=vals["fps"])

        return {"jobId":jid, "result":mp4, "status":"DONE"}

    except Exception as e:
        return {"jobId":jid, "result":f"FAILED: {e}", "status":"FAILED"}

    finally:
        # cleanup
        for f in ["mp4", "wp"]:
            if os.path.exists(f): os.remove(f)

runpod.serverless.start({"handler": generate})
