FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano curl wget git git-lfs unzip unrar ffmpeg && \
    wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -O cuda_12.6.2_560.35.03_linux.run && \
    sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && chmod -R 777 /home

USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torchsde einops diffusers transformers accelerate peft timm kornia scikit-image moviepy==1.0.3 && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone -b dev https://github.com/camenduru/ComfyUI-Fluxpromptenhancer /content/ComfyUI/custom_nodes/ComfyUI-Fluxpromptenhancer && \
    mkdir -p /content/ComfyUI/models/clip && \
    mkdir -p /workspace/models/ltxv && \
    mkdir -p /content/ComfyUI/models/LLM/Flux-Prompt-Enhance && \
    wget -q --show-progress -O /content/ComfyUI/models/clip/t5xxl_fp16.safetensors https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp16.safetensors && \
    wget -q --show-progress -O /workspace/models/ltxv/ltxv-13b-0.9.7-distilled.safetensors https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled.safetensors && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/config.json https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/raw/main/config.json && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/generation_config.json https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/raw/main/generation_config.json && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/model.safetensors https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/resolve/main/model.safetensors && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/special_tokens_map.json https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/raw/main/special_tokens_map.json && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/spiece.model https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/resolve/main/spiece.model && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/tokenizer.json https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/raw/main/tokenizer.json && \
    wget -q --show-progress -O /content/ComfyUI/models/LLM/Flux-Prompt-Enhance/tokenizer_config.json https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/raw/main/tokenizer_config.json

COPY ./worker_runpod_i2v.py /content/ComfyUI/worker_runpod_i2v.py

WORKDIR /content/ComfyUI
CMD python worker_runpod_i2v.py
