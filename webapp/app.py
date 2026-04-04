from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import mimetypes
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

try:
    from webapp.tts_service import (
        GWEN_GENERATION_DEFAULTS,
        GWEN_GENERATION_LIMITS,
        TTSError,
        TEXT_INPUT_LIMIT,
        TTSStudioService,
        _normalize_gwen_generation_config,
        _parse_pronunciation_overrides,
    )
except ImportError:  # pragma: no cover
    from tts_service import (
        GWEN_GENERATION_DEFAULTS,
        GWEN_GENERATION_LIMITS,
        TTSError,
        TEXT_INPUT_LIMIT,
        TTSStudioService,
        _normalize_gwen_generation_config,
        _parse_pronunciation_overrides,
    )


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
JOB_RETENTION_SECONDS = 60 * 60


@dataclass(slots=True)
class SynthesisSubmission:
    engine_id: str
    text: str
    reference_path: Path
    reference_text: str
    speed: float
    remove_silence: bool
    seed: int | None
    model_key: str
    custom_model: str
    gwen_generation_config: dict[str, Any] | None
    pronunciation_overrides: list[tuple[str, str]]
    preset_voice: Any | None = None


class TTSJobManager:
    def __init__(self, *, max_workers: int = 1, retention_seconds: int = JOB_RETENTION_SECONDS) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts-job")
        self._retention_seconds = retention_seconds
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def enqueue(self, submission: SynthesisSubmission) -> str:
        self._purge_expired()
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "message": "Đã nhận yêu cầu. Job đang chờ worker synthesize.",
                "result": None,
                "error": None,
            }
        self._executor.submit(self._run_job, job_id, submission)
        return job_id

    def get_snapshot(self, job_id: str) -> dict[str, Any] | None:
        self._purge_expired()
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None:
                return None
            return dict(snapshot)

    def _purge_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired_ids = [
                job_id
                for job_id, snapshot in self._jobs.items()
                if now - float(snapshot.get("updated_at", snapshot.get("created_at", now))) > self._retention_seconds
            ]
            for job_id in expired_ids:
                self._jobs.pop(job_id, None)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None:
                return
            snapshot.update(updates)
            snapshot["updated_at"] = time.time()

    def _run_job(self, job_id: str, submission: SynthesisSubmission) -> None:
        self._update_job(
            job_id,
            status="running",
            message="Model đang synthesize trong nền. Request đã được tách khỏi kết nối Cloudflare.",
        )
        try:
            payload = _execute_synthesis_submission(submission)
        except TTSError as exc:
            self._update_job(job_id, status="failed", error=str(exc), message=str(exc))
            return
        except Exception as exc:  # pragma: no cover
            error_message = f"Lỗi không mong muốn: {exc}"
            self._update_job(job_id, status="failed", error=error_message, message=error_message)
            return

        self._update_job(
            job_id,
            status="completed",
            result=payload,
            message="Đã synthesize xong audio.",
            error=None,
        )


tts_job_manager = TTSJobManager()


def _parse_gwen_generation_form() -> dict[str, Any]:
    subtalker_do_sample_values = request.form.getlist("gwen_subtalker_do_sample")
    normalized = _normalize_gwen_generation_config(
        {
            "speed": request.form.get("speed"),
            "temperature": request.form.get("gwen_temperature"),
            "top_p": request.form.get("gwen_top_p"),
            "top_k": request.form.get("gwen_top_k"),
            "repetition_penalty": request.form.get("gwen_repetition_penalty"),
            "max_new_tokens": request.form.get("gwen_max_new_tokens"),
            "subtalker_do_sample": any(
                str(value).strip().lower() in {"1", "true", "yes", "on"}
                for value in subtalker_do_sample_values
            ),
            "subtalker_temperature": request.form.get("gwen_subtalker_temperature"),
            "subtalker_top_k": request.form.get("gwen_subtalker_top_k"),
            "subtalker_top_p": request.form.get("gwen_subtalker_top_p"),
            "subtalker_sampling_method": request.form.get("gwen_subtalker_sampling_method"),
        }
    )
    return normalized


def _parse_pronunciation_form() -> list[tuple[str, str]]:
    sources = request.form.getlist("pronunciation_from")
    targets = request.form.getlist("pronunciation_to")
    row_count = max(len(sources), len(targets))
    for index in range(row_count):
        source = (sources[index] if index < len(sources) else "").strip()
        target = (targets[index] if index < len(targets) else "").strip()
        if bool(source) != bool(target):
            raise ValueError(f"Dòng phát âm {index + 1} phải có đủ chữ gốc và cách đọc.")
    return _parse_pronunciation_overrides(sources, targets)


def _gwen_settings_payload() -> dict[str, Any]:
    return {
        "defaults": {
            "speed": GWEN_GENERATION_DEFAULTS["speed"],
            "temperature": GWEN_GENERATION_DEFAULTS["temperature"],
            "topP": GWEN_GENERATION_DEFAULTS["top_p"],
            "topK": GWEN_GENERATION_DEFAULTS["top_k"],
            "repetitionPenalty": GWEN_GENERATION_DEFAULTS["repetition_penalty"],
            "maxNewTokens": GWEN_GENERATION_DEFAULTS["max_new_tokens"],
            "codePredictorDoSample": GWEN_GENERATION_DEFAULTS["subtalker_do_sample"],
            "codePredictorTemperature": GWEN_GENERATION_DEFAULTS["subtalker_temperature"],
            "codePredictorTopK": GWEN_GENERATION_DEFAULTS["subtalker_top_k"],
            "codePredictorTopP": GWEN_GENERATION_DEFAULTS["subtalker_top_p"],
            "codePredictorSamplingMethod": GWEN_GENERATION_DEFAULTS["subtalker_sampling_method"],
        },
        "ranges": {
            "speed": GWEN_GENERATION_LIMITS["speed"],
            "temperature": GWEN_GENERATION_LIMITS["temperature"],
            "topP": GWEN_GENERATION_LIMITS["top_p"],
            "topK": GWEN_GENERATION_LIMITS["top_k"],
            "repetitionPenalty": GWEN_GENERATION_LIMITS["repetition_penalty"],
            "maxNewTokens": GWEN_GENERATION_LIMITS["max_new_tokens"],
            "codePredictorTemperature": GWEN_GENERATION_LIMITS["subtalker_temperature"],
            "codePredictorTopK": GWEN_GENERATION_LIMITS["subtalker_top_k"],
            "codePredictorTopP": GWEN_GENERATION_LIMITS["subtalker_top_p"],
        },
    }


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


def _engine_page_context(engine_id: str) -> dict:
    context = _base_context(engine_id)
    try:
        engine_card = studio.get_engine_card(engine_id)
    except TTSError as exc:
        abort(404, description=str(exc))

    preset_voices = studio.get_preset_voices(engine_card.id)
    context.update(
        examples=TEXT_EXAMPLES,
        reference_tips=REFERENCE_TIPS,
        max_upload_mb=MAX_UPLOAD_MB,
        text_input_limit=TEXT_INPUT_LIMIT,
        engine_card=engine_card,
        engine_card_payload=asdict(engine_card),
        model_selection=engine_card.metadata.get("model_selection", studio.get_model_selection(engine_card.id)),
        gwen_settings_payload=_gwen_settings_payload() if engine_card.id == "gwen" else None,
        preset_voices_payload=[
            {
                "id": voice.id,
                "name": voice.name,
                "avatar": voice.avatar,
                "style": voice.style,
                "reference_text": voice.reference_text,
                "audio_url": url_for("static", filename=f"voice_presets/{voice.audio_filename}"),
            }
            for voice in preset_voices
        ],
    )
    return context


def _is_api_request() -> bool:
    return request.path.startswith("/api/")


def _audio_url_for_filename(filename: str) -> str:
    return f"/outputs/{quote(filename)}"


def _build_synthesis_payload(result, *, preset_voice: Any | None = None) -> dict[str, Any]:
    notes = list(result.notes)
    if preset_voice is not None:
        notes.insert(0, f"Đã dùng preset voice Gwen: {preset_voice.name}.")

    return {
        "ok": True,
        "engine": result.engine_id,
        "engine_label": result.engine_label,
        "model_key": result.model_key,
        "model_label": result.model_label,
        "preset_voice_id": preset_voice.id if preset_voice is not None else "",
        "preset_voice_label": preset_voice.name if preset_voice is not None else "",
        "audio_url": _audio_url_for_filename(result.output_path.name),
        "download_name": result.output_path.name,
        "sample_rate": result.sample_rate,
        "duration_seconds": round(result.duration_seconds, 2),
        "inference_seconds": round(result.inference_seconds, 2),
        "chunk_count": result.chunk_count,
        "reference_text_used": result.reference_text_used,
        "seed": result.seed,
        "notes": notes,
        "storage_path": str(result.output_path.relative_to(ROOT)),
    }


def _build_synthesis_submission() -> SynthesisSubmission:
    upload = request.files.get("reference_audio")
    text = (request.form.get("text") or "").strip()
    engine_id = (request.form.get("engine") or _pick_default_engine(studio.get_engine_cards())).strip().lower()
    preset_voice_id = (request.form.get("preset_voice_id") or "").strip().lower()
    model_key = (request.form.get("model_key") or "default").strip() or "default"
    custom_model = (request.form.get("custom_model") or "").strip()
    reference_text = (request.form.get("reference_text") or "").strip()
    remove_silence = (request.form.get("remove_silence") or "").strip().lower() in {"1", "true", "yes", "on"}

    speed_raw = (request.form.get("speed") or "1.0").strip()
    seed_raw = (request.form.get("seed") or "").strip()
    gwen_generation_config = None
    pronunciation_overrides: list[tuple[str, str]] = []

    preset_voice = None
    if preset_voice_id:
        preset_voice, reference_path = studio.get_preset_voice_reference(engine_id, preset_voice_id)
        reference_text = preset_voice.reference_text
        app.logger.info(
            "Resolved preset voice for synth: engine=%s preset_voice_id=%s preset_name=%s audio=%s reference_chars=%d",
            engine_id,
            preset_voice.id,
            preset_voice.name,
            reference_path.name,
            len(reference_text),
        )
    else:
        if not upload or not upload.filename:
            raise ValueError("Cần upload audio tham chiếu trước khi sinh giọng.")

        extension = Path(upload.filename).suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError("Định dạng audio chưa hỗ trợ. Chỉ nhận .wav, .mp3, .m4a, .flac, .ogg.")

        reference_path = studio.save_reference_file(upload.filename, upload.read())
        app.logger.info(
            "Resolved custom reference for synth: engine=%s filename=%s reference_chars=%d",
            engine_id,
            reference_path.name,
            len(reference_text),
        )

    if engine_id == "gwen":
        gwen_generation_config = _parse_gwen_generation_form()
        pronunciation_overrides = _parse_pronunciation_form()
        speed = float(gwen_generation_config["speed"])
    else:
        try:
            speed = min(max(float(speed_raw), 0.7), 1.3)
        except ValueError as exc:
            raise ValueError("Speed phải là số hợp lệ trong khoảng 0.7 - 1.3.") from exc

    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError("Seed phải là số nguyên nếu được nhập.") from exc

    return SynthesisSubmission(
        engine_id=engine_id,
        text=text,
        reference_path=reference_path,
        reference_text=reference_text,
        speed=speed,
        remove_silence=remove_silence,
        seed=seed,
        model_key=model_key,
        custom_model=custom_model,
        gwen_generation_config=gwen_generation_config,
        pronunciation_overrides=pronunciation_overrides,
        preset_voice=preset_voice,
    )


def _execute_synthesis_submission(submission: SynthesisSubmission) -> dict[str, Any]:
    result = studio.synthesize(
        engine_id=submission.engine_id,
        text=submission.text,
        reference_audio=submission.reference_path,
        reference_text=submission.reference_text,
        speed=submission.speed,
        remove_silence=submission.remove_silence,
        seed=submission.seed,
        model_key=submission.model_key,
        custom_model=submission.custom_model,
        gwen_generation_config=submission.gwen_generation_config,
        pronunciation_overrides=submission.pronunciation_overrides,
    )
    return _build_synthesis_payload(result, preset_voice=submission.preset_voice)


def _should_use_async_generation() -> bool:
    async_raw = (request.form.get("async") or "").strip().lower()
    if async_raw in {"1", "true", "yes", "on"}:
        return True
    if async_raw in {"0", "false", "no", "off"}:
        return False
    return bool(request.headers.get("CF-Connecting-IP") or request.headers.get("CF-Ray"))


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
    return redirect(url_for("engine_page", engine_id=_pick_default_engine(studio.get_engine_cards())))


@app.route("/studio/<engine_id>")
def engine_page(engine_id: str):
    return render_template("studio_engine.html", **_engine_page_context(engine_id))


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
    try:
        submission = _build_synthesis_submission()
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "error": f"Lỗi không mong muốn: {exc}"}), 500

    if _should_use_async_generation():
        job_id = tts_job_manager.enqueue(submission)
        return (
            jsonify(
                {
                    "ok": True,
                    "queued": True,
                    "job_id": job_id,
                    "status": "queued",
                    "message": "Job đã được tách nền để tránh timeout Cloudflare. Hãy poll trạng thái cho tới khi hoàn tất.",
                }
            ),
            202,
        )

    try:
        payload = _execute_synthesis_submission(submission)
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "error": f"Lỗi không mong muốn: {exc}"}), 500
    return jsonify(payload)


@app.route("/api/tts/jobs/<job_id>")
def api_tts_job_status(job_id: str):
    snapshot = tts_job_manager.get_snapshot(job_id.strip())
    if snapshot is None:
        return jsonify({"ok": False, "error": "Không tìm thấy job generate hoặc job đã hết hạn."}), 404

    status = snapshot["status"]
    if status in {"queued", "running"}:
        return jsonify(
            {
                "ok": True,
                "done": False,
                "job_id": snapshot["id"],
                "status": status,
                "message": snapshot["message"],
            }
        )

    if status == "completed":
        return jsonify(
            {
                "ok": True,
                "done": True,
                "job_id": snapshot["id"],
                "status": status,
                "result": snapshot["result"],
            }
        )

    return jsonify(
        {
            "ok": False,
            "done": True,
            "job_id": snapshot["id"],
            "status": status,
            "error": snapshot["error"] or snapshot["message"],
        }
    )


@app.route("/outputs/<path:filename>")
def serve_generated_audio(filename: str):
    guessed = mimetypes.guess_type(filename)[0] or "audio/wav"
    return send_from_directory(studio.output_dir, filename, mimetype=guessed, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8386)
