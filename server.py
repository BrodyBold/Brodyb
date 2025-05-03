from flask import Flask, request, jsonify
import subprocess, os, tempfile, json

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    img_url  = data["image_url"]
    duration = data.get("duration", 5)
    fps      = data.get("fps", 30)

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = os.path.join(tmp, "in.jpg")
        out_path = os.path.join(tmp, "out.mp4")
        os.system(f"wget -q '{img_url}' -O {in_path}")

        cmd = [
            "framepack", "generate",
            "--source", in_path,
            "--target", out_path,
            "--duration", str(duration),
            "--fps", str(fps)
        ]
        subprocess.check_call(cmd)

        os.system(f"curl -s -F file=@{out_path} https://tmpfiles.org/api/v1/upload > {tmp}/res.json")
        url = json.load(open(f"{tmp}/res.json"))["data"]["url"]

    return jsonify({"video_url": url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
