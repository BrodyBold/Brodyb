FROM nvcr.io/nvidia/pytorch:25.04-py3
WORKDIR /app

# 1) Install OS deps
RUN apt-get update && apt-get install -y git wget

# 2) Clone FramePack
RUN git clone https://github.com/lllyasviel/FramePack.git .

# 3) Install PyTorch (already in base, but ensure the right CUDA wheel)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# 4) Install FramePack’s Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5) Install the CLI entrypoint
RUN pip install --no-cache-dir framepack

# 6) Copy in your server wrapper
COPY server.py .

EXPOSE 5000
ENTRYPOINT ["python", "server.py"]
