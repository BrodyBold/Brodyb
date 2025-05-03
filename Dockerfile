FROM nvcr.io/nvidia/pytorch:24.12-py3
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git wget curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/lllyasviel/FramePack.git .

COPY requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py

EXPOSE 5000
ENTRYPOINT ["python3", "server.py"]
