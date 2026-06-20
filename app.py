from flask import Flask, request, jsonify, send_file, render_template
import os
import uuid
import threading
import json
from pathlib import Path
from groq import Groq
import anthropic
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
JOBS = {}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")


def set_status(job_id, step, progress, message):
    JOBS[job_id].update({"step": step, "progress": progress, "message": message})


def parse_segments(transcription):
    """Handle both dict and object formats from Groq API."""
    raw = transcription.segments or []
    segments = []
    for s in raw:
        if isinstance(s, dict):
            segments.append({
                "start": s.get("start", 0),
                "end":   s.get("end", 0),
                "text":  s.get("text", "").strip(),
            })
        else:
            segments.append({
                "start": s.start,
                "end":   s.end,
                "text":  s.text.strip(),
            })
    return segments


def run_pipeline(job_id, video_path, groq_key, claude_key):
    try:
        job_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_dir, exist_ok=True)
        srt_path    = os.path.join(job_dir, "subtitles.srt")
        output_path = os.path.join(job_dir, "output.mp4")

        # Step 1: Transcribe with Groq Whisper
        set_status(job_id, "transcribe", 10, "Đang nhận diện giọng nói...")
        groq_client = Groq(api_key=groq_key)
        with open(video_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(video_path), f),
                model="whisper-large-v3",
                language="zh",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = parse_segments(transcription)
        set_status(job_id, "transcribe", 30, f"Nhận diện xong {len(segments)} đoạn")

        # Step 2: Translate with Claude
        set_status(job_id, "translate", 35, "Đang dịch sang tiếng Việt...")
        claude_client = anthropic.Anthropic(api_key=claude_key)
        translated = []
        batch_size = 30
        batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]

        for idx, batch in enumerate(batches):
            texts = [s["text"] for s in batch]
            prompt = (
                "Dịch các phụ đề sau từ tiếng Trung sang tiếng Việt. "
                "Trả về ONLY JSON array, không giải thích:\n"
                + json.dumps(texts, ensure_ascii=False)
            )
            resp = claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                vi_texts = json.loads(raw)
            except Exception:
                vi_texts = texts

            for seg, vi in zip(batch, vi_texts):
                translated.append({"start": seg["start"], "end": seg["end"], "text": vi})

            progress = 35 + int((idx + 1) / len(batches) * 35)
            set_status(job_id, "translate", progress, f"Đã dịch {len(translated)}/{len(segments)} đoạn")

        # Step 3: Write SRT
        set_status(job_id, "srt", 72, "Đang tạo file phụ đề...")
        write_srt(translated, srt_path)

        # Step 4: Burn subtitles
        set_status(job_id, "burn", 78, "Đang nhúng phụ đề vào video...")
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Bold=1,Outline=2,MarginV=20"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
            "-c:a", "copy", "-c:v", "libx264",
            "-crf", "22", "-preset", "fast", "-y",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg lỗi: {result.stderr[-500:]}")

        JOBS[job_id].update({
            "step": "done", "progress": 100,
            "message": "Hoàn thành!", "output": output_path
        })

    except Exception as e:
        JOBS[job_id].update({"step": "error", "progress": 0, "message": str(e)})
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "Không có file video"}), 400
    file      = request.files["video"]
    groq_key  = request.form.get("groq_key", "").strip()
    claude_key = request.form.get("claude_key", "").strip()
    if not groq_key or not claude_key:
        return jsonify({"error": "Thiếu API key"}), 400

    job_id     = str(uuid.uuid4())
    ext        = Path(file.filename).suffix or ".mp4"
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    file.save(video_path)

    JOBS[job_id] = {"step": "queued", "progress": 0, "message": "Đang chờ xử lý..."}
    threading.Thread(target=run_pipeline, args=(job_id, video_path, groq_key, claude_key), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job không tồn tại"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("step") != "done":
        return jsonify({"error": "Chưa xong"}), 404
    return send_file(job["output"], as_attachment=True, download_name="video_vietsub.mp4")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
