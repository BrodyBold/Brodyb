FROM nvcr.io/nvidia/pytorch:25.04-py3
WORKDIR /app


RUN apt-get update && apt-get install -y git wget


RUN git clone https://github.com/lllyasviel/FramePack.git .

RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir framepack

COPY server.py .

EXPOSE 5000
ENTRYPOINT ["python", "server.py"]
