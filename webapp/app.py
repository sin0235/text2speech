from __future__ import annotations

import mimetypes
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

try:
    from webapp.tts_service import TTSError, TTSStudioService
except ImportError:  # pragma: no cover
    from tts_service import TTSError, TTSStudioService


ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
MAX_UPLOAD_MB = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)

studio = TTSStudioService(ROOT)

TEXT_EXAMPLES = [
    "Xin chào, đây là bản demo chuyển văn bản thành giọng nói tiếng Việt bằng Gwen-TTS.",
    "Thông báo: lớp học xử lý ngôn ngữ tự nhiên sẽ bắt đầu lúc tám giờ sáng tại phòng A3.",
    "Hôm nay chúng ta sẽ thu voice-over cho landing page với tông giọng rõ, sáng và chuyên nghiệp.",
    "Chúc bạn có một ngày làm việc hiệu quả, nhẹ nhàng và nhiều năng lượng tích cực.",
]

REFERENCE_TIPS = [
    "Audio mẫu nên dài 3 đến 8 giây, một người nói, ít nhiễu.",
    "Gwen-TTS và VieNeu Standard cần transcript đúng với câu trong audio mẫu.",
    "Nếu lỗi hoặc thiếu GPU, đổi tạm sang VieNeu hoặc F5.",
]

SETUP_STEPS = [
    {
        "title": "Cài webapp",
        "body": "Cài dependencies tối thiểu để chạy giao diện và API.",
    },
    {
        "title": "Bật Gwen",
        "body": "Cài `qwen-tts`; model sẽ load khi bạn generate.",
    },
    {
        "title": "Thêm fallback",
        "body": "Cài VieNeu hoặc F5 nếu cần engine phụ.",
    },
]

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _base_context(active: str) -> dict:
    engine_cards = studio.get_engine_cards()
    return {
        "active": active,
        "engine_cards": engine_cards,
        "status_summary": studio.summary(),
    }


def _pick_default_engine(engine_cards: list) -> str:
    for card in engine_cards:
        if card.id == studio.default_engine and card.ready:
            return card.id
    for card in engine_cards:
        if card.ready:
            return card.id
    return engine_cards[0].id if engine_cards else "gwen"


def _is_api_request() -> bool:
    return request.path.startswith("/api/")


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(exc: RequestEntityTooLarge):
    if not _is_api_request():
        return exc
    return (
        jsonify(
            {
                "ok": False,
                "error": f"File upload quá lớn. Giới hạn hiện tại là {MAX_UPLOAD_MB} MB.",
            }
        ),
        413,
    )


@app.errorhandler(HTTPException)
def handle_api_http_exception(exc: HTTPException):
    if not _is_api_request():
        return exc
    return jsonify({"ok": False, "error": exc.description or exc.name}), exc.code or 500


@app.route("/")
def home():
    context = _base_context("home")
    context.update(
        examples=TEXT_EXAMPLES,
        reference_tips=REFERENCE_TIPS,
        max_upload_mb=MAX_UPLOAD_MB,
        default_engine_id=_pick_default_engine(context["engine_cards"]),
        engine_cards_payload=[asdict(card) for card in context["engine_cards"]],
    )
    return render_template("home.html", **context)


@app.route("/voices")
def voices_page():
    context = _base_context("voices")
    context.update(reference_tips=REFERENCE_TIPS)
    return render_template("voices.html", **context)


@app.route("/about")
def about():
    context = _base_context("about")
    context.update(setup_steps=SETUP_STEPS)
    return render_template("about.html", **context)


@app.route("/api/tts/status")
def api_tts_status():
    cards = studio.get_engine_cards()
    return jsonify(
        {
            "ok": True,
            "summary": studio.summary(),
            "engines": [
                {
                    "id": card.id,
                    "label": card.label,
                    "ready": card.ready,
                    "summary": card.summary,
                    "warning": card.warning,
                    "metadata": card.metadata,
                }
                for card in cards
            ],
        }
    )


@app.route("/api/tts/generate", methods=["POST"])
def api_tts_generate():
    upload = request.files.get("reference_audio")
    text = (request.form.get("text") or "").strip()
    engine_id = (request.form.get("engine") or _pick_default_engine(studio.get_engine_cards())).strip().lower()
    model_key = (request.form.get("model_key") or "default").strip() or "default"
    custom_model = (request.form.get("custom_model") or "").strip()
    reference_text = (request.form.get("reference_text") or "").strip()
    remove_silence = (request.form.get("remove_silence") or "").strip().lower() in {"1", "true", "yes", "on"}

    speed_raw = (request.form.get("speed") or "1.0").strip()
    seed_raw = (request.form.get("seed") or "").strip()

    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "Cần upload audio tham chiếu trước khi sinh giọng."}), 400

    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify(
            {
                "ok": False,
                "error": "Định dạng audio chưa hỗ trợ. Chỉ nhận .wav, .mp3, .m4a, .flac, .ogg.",
            }
        ), 400

    try:
        speed = min(max(float(speed_raw), 0.7), 1.3)
    except ValueError:
        return jsonify({"ok": False, "error": "Speed phải là số hợp lệ trong khoảng 0.7 - 1.3."}), 400

    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            return jsonify({"ok": False, "error": "Seed phải là số nguyên nếu được nhập."}), 400

    reference_path = studio.save_reference_file(upload.filename, upload.read())

    try:
        result = studio.synthesize(
            engine_id=engine_id,
            text=text,
            reference_audio=reference_path,
            reference_text=reference_text,
            speed=speed,
            remove_silence=remove_silence,
            seed=seed,
            model_key=model_key,
            custom_model=custom_model,
        )
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "error": f"Lỗi không mong muốn: {exc}"}), 500

    audio_url = url_for("serve_generated_audio", filename=result.output_path.name)
    return jsonify(
        {
            "ok": True,
            "engine": result.engine_id,
            "engine_label": result.engine_label,
            "model_key": result.model_key,
            "model_label": result.model_label,
            "audio_url": audio_url,
            "download_name": result.output_path.name,
            "sample_rate": result.sample_rate,
            "duration_seconds": round(result.duration_seconds, 2),
            "inference_seconds": round(result.inference_seconds, 2),
            "chunk_count": result.chunk_count,
            "reference_text_used": result.reference_text_used,
            "seed": result.seed,
            "notes": result.notes,
            "storage_path": str(result.output_path.relative_to(ROOT)),
        }
    )


@app.route("/outputs/<path:filename>")
def serve_generated_audio(filename: str):
    guessed = mimetypes.guess_type(filename)[0] or "audio/wav"
    return send_from_directory(studio.output_dir, filename, mimetype=guessed, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8386)
