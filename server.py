import subprocess, tempfile, os, json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    src  = data["image_url"]
    dur  = data.get("duration", 5)
    fps  = data.get("fps", 30)

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = os.path.join(tmp, "in.png")
        out_path = os.path.join(tmp, "out.mp4")

        os.system(f"wget -q '{src}' -O {in_path}")
        subprocess.check_call([
          "framepack", "generate",
          "--source",   in_path,
          "--target",   out_path,
          "--duration", str(dur),
          "--fps",      str(fps)
        ])

        # upload logic (e.g. tmpfiles or S3)
        os.system(f"curl -s -F file=@{out_path} https://tmpfiles.org/api/v1/upload > {tmp}/res.json")
        url = json.load(open(f"{tmp}/res.json"))["data"]["url"]

    return jsonify({"video_url": url})
