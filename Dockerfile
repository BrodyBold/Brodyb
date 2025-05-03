FROM nvcr.io/nvidia/pytorch:25.04-py3
WORKDIR /app

# Install git, wget and clone FramePack
RUN apt-get update && apt-get install -y git wget \
  && git clone https://github.com/lllyasviel/FramePack.git . \
  && pip install -r requirements.txt \
  && pip install torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu126 \
  && pip install framepack

COPY server.py .

EXPOSE 5000
ENTRYPOINT ["python", "server.py"]
