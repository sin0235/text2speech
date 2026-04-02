from __future__ import annotations

import importlib.util
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf


class TTSError(RuntimeError):
    """Raised when the selected TTS engine cannot complete synthesis."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_text_for_tts(text: str, max_chars: int = 220) -> list[str]:
    """Split long Vietnamese text into short sentences/chunks for stable inference."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        words = sentence.split()
        part = ""
        for word in words:
            attempt = word if not part else f"{part} {word}"
            if len(attempt) <= max_chars:
                part = attempt
            else:
                if part:
                    chunks.append(part)
                part = word
        if part:
            current = part

    if current:
        chunks.append(current)
    return chunks


def _crossfade_join(waves: list[np.ndarray], sample_rate: int, duration_ms: int = 80) -> np.ndarray:
    if not waves:
        return np.zeros(1, dtype=np.float32)
    if len(waves) == 1:
        return waves[0]

    crossfade = max(0, int(sample_rate * duration_ms / 1000))
    output = waves[0].astype(np.float32)

    for wave in waves[1:]:
        current = wave.astype(np.float32)
        if crossfade <= 0 or len(output) <= crossfade or len(current) <= crossfade:
            output = np.concatenate([output, current])
            continue

        fade_out = np.linspace(1.0, 0.0, crossfade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        mixed_tail = output[-crossfade:] * fade_out + current[:crossfade] * fade_in
        output = np.concatenate([output[:-crossfade], mixed_tail, current[crossfade:]])

    return output


def _to_numpy_audio(audio: Any) -> np.ndarray:
    tensor = audio
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor, dtype=np.float32).reshape(-1)


def _numel(obj: Any) -> int | None:
    if hasattr(obj, "numel"):
        try:
            return int(obj.numel())
        except Exception:
            return None
    if hasattr(obj, "size"):
        try:
            size = obj.size
            if isinstance(size, int):
                return int(size)
        except Exception:
            return None
    return None


def _load_audio_mono_float(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
    except Exception:
        try:
            import librosa
        except Exception as exc:
            raise TTSError(
                "Không đọc được audio tham chiếu. Hãy upload WAV/FLAC hoặc cài `librosa` để hỗ trợ MP3/M4A."
            ) from exc
        audio_np, sr = librosa.load(str(path), sr=None, mono=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)

    if audio_np.ndim == 0:
        raise TTSError("Audio tham chiếu rỗng hoặc không hợp lệ.")

    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=-1 if audio_np.shape[-1] <= 8 else 0)

    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0:
        raise TTSError("Audio tham chiếu không chứa sample hợp lệ.")

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak <= 1e-6:
        raise TTSError("Audio tham chiếu gần như im lặng. Hãy dùng file có tiếng nói rõ hơn.")

    if sr != target_sr:
        try:
            import librosa
        except Exception as exc:
            raise TTSError(
                f"Audio tham chiếu đang ở {sr} Hz. Cần cài `librosa` để resample về {target_sr} Hz cho ViRa."
            ) from exc
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio_np = np.clip(audio_np / max(peak, 1.0), -1.0, 1.0).astype(np.float32)
    return audio_np, sr


@dataclass(slots=True)
class EngineCard:
    id: str
    label: str
    headline: str
    description: str
    recommended_for: str
    output_quality: str
    reference_hint: str
    supports_reference_text: bool
    ready: bool
    summary: str
    warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SynthesisResult:
    engine_id: str
    engine_label: str
    output_path: Path
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    chunk_count: int
    reference_text_used: bool
    seed: int | None
    notes: list[str] = field(default_factory=list)


class TTSStudioService:
    """Flask-facing service that orchestrates F5-TTS and ViRa engines."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime_dir = self.root / "webapp" / "runtime"
        self.reference_dir = self.runtime_dir / "references"
        self.output_dir = self.runtime_dir / "generated"
        self.model_dir = self.root / "models"
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.default_engine = os.getenv("TTS_DEFAULT_ENGINE", "vira").strip().lower() or "vira"
        self.vira_model_id = os.getenv("VIRA_MODEL_ID", "dolly-vn/Vira-TTS")
        self.vira_model_path = Path(os.getenv("VIRA_MODEL_PATH", self.model_dir / "vira"))
        self.vira_auto_download = _env_flag("VIRA_AUTO_DOWNLOAD", False)

        self.f5_model_name = os.getenv("F5_MODEL_NAME", "F5TTS_v1_Base")
        self.f5_ckpt_file = os.getenv("F5_CKPT_FILE", "").strip()
        self.f5_vocab_file = os.getenv("F5_VOCAB_FILE", "").strip()
        self.f5_vocoder_local_path = os.getenv("F5_VOCODER_LOCAL_PATH", "").strip() or None

        self._locks = {
            "f5": Lock(),
            "vira": Lock(),
        }
        self._loaded_models: dict[str, Any] = {}

    def summary(self) -> dict[str, Any]:
        cards = self.get_engine_cards()
        ready_count = sum(1 for card in cards if card.ready)
        device_label = "GPU" if self._torch_cuda_available() else "CPU"
        default_card = next((card for card in cards if card.id == self.default_engine), cards[0])
        return {
            "engine_count": len(cards),
            "ready_count": ready_count,
            "device_label": device_label,
            "default_engine": default_card.label,
        }

    def get_engine_cards(self) -> list[EngineCard]:
        return [
            self._f5_card(),
            self._vira_card(),
        ]

    def get_engine_card(self, engine_id: str) -> EngineCard:
        engine = (engine_id or "").strip().lower()
        for card in self.get_engine_cards():
            if card.id == engine:
                return card
        raise TTSError(f"Engine '{engine_id}' không tồn tại.")

    def synthesize(
        self,
        *,
        engine_id: str,
        text: str,
        reference_audio: Path,
        reference_text: str = "",
        speed: float = 1.0,
        remove_silence: bool = False,
        seed: int | None = None,
    ) -> SynthesisResult:
        text = re.sub(r"\s+", " ", (text or "").strip())
        reference_text = re.sub(r"\s+", " ", (reference_text or "").strip())

        if not text:
            raise TTSError("Thiếu nội dung văn bản cần chuyển giọng nói.")
        if len(text) > 2500:
            raise TTSError("Văn bản quá dài. Giới hạn hiện tại là 2500 ký tự mỗi lượt.")
        if not reference_audio.exists():
            raise TTSError("Không tìm thấy file audio tham chiếu.")

        engine = self.get_engine_card(engine_id)
        if not engine.ready:
            raise TTSError(engine.warning or engine.summary)

        output_path = self.output_dir / f"{int(time.time())}-{uuid.uuid4().hex[:10]}-{engine.id}.wav"
        if engine.id == "f5":
            return self._synthesize_with_f5(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speed=speed,
                remove_silence=remove_silence,
                seed=seed,
                output_path=output_path,
            )
        if engine.id == "vira":
            return self._synthesize_with_vira(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                output_path=output_path,
            )
        raise TTSError(f"Engine '{engine_id}' không được hỗ trợ.")

    def save_reference_file(self, filename: str, payload: bytes) -> Path:
        extension = Path(filename).suffix.lower() or ".wav"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", Path(filename).stem).strip("-") or "reference"
        saved = self.reference_dir / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{safe_name}{extension}"
        saved.write_bytes(payload)
        return saved

    def _prepare_reference_audio_for_vira(self, reference_audio: Path) -> tuple[Path, float]:
        target_sr = 48000
        audio_np, sr = _load_audio_mono_float(reference_audio, target_sr=target_sr)
        duration_seconds = float(len(audio_np) / sr)
        if duration_seconds < 1.0:
            raise TTSError("Audio tham chiếu quá ngắn cho ViRa. Hãy dùng đoạn nói rõ dài tối thiểu khoảng 1 giây, tốt nhất 3-10 giây.")
        if duration_seconds > 20.0:
            raise TTSError("Audio tham chiếu quá dài cho ViRa. Hãy cắt còn khoảng 3-10 giây để ổn định hơn.")

        prepared_path = self.reference_dir / f"{reference_audio.stem}-vira-48k.wav"
        sf.write(prepared_path, audio_np, target_sr)
        return prepared_path, duration_seconds

    def _f5_card(self) -> EngineCard:
        module_ok = importlib.util.find_spec("f5_tts") is not None
        summary = "Sẵn sàng dùng zero-shot voice cloning với F5-TTS." if module_ok else "Chưa cài gói `f5_tts`."
        warning = None if module_ok else (
            "Engine F5-TTS chưa khả dụng trong môi trường này. "
            "Cài upstream F5-TTS rồi khởi động lại ứng dụng."
        )
        return EngineCard(
            id="f5",
            label="F5-TTS",
            headline="Voice cloning linh hoạt, hợp cho voice-over và thử nghiệm đa phong cách.",
            description="Adapter này gọi API `F5TTS().infer(...)` của F5-TTS và tự chia câu dài thành nhiều chunk nhỏ để ổn định hơn.",
            recommended_for="Dùng khi cần clone giọng tự nhiên, giữ nhịp đọc mềm và muốn tinh chỉnh speed / seed.",
            output_quality="Output WAV theo sample rate của model F5 đã nạp.",
            reference_hint="Audio tham chiếu 3-12 giây, nói rõ, ít nhạc nền. Có thể nhập transcript để giữ nội dung tham chiếu chính xác hơn.",
            supports_reference_text=True,
            ready=module_ok,
            summary=summary,
            warning=warning,
            metadata={
                "model": self.f5_model_name,
                "checkpoint": self.f5_ckpt_file or "auto-download theo cấu hình upstream",
            },
        )

    def _vira_card(self) -> EngineCard:
        module_ok = importlib.util.find_spec("mira") is not None
        model_ok = self.vira_model_path.exists() and any(self.vira_model_path.iterdir())
        ready = module_ok and (model_ok or self.vira_auto_download)

        if ready:
            summary = "Sẵn sàng dùng ViRa cho tiếng Việt tự nhiên 48kHz."
            warning = None
        elif not module_ok:
            summary = "Chưa cài module `mira` từ ViRa."
            warning = (
                "Engine ViRa chưa khả dụng vì module `mira` chưa được cài. "
                "Cài upstream ViRa/Mira rồi khởi động lại ứng dụng."
            )
        else:
            summary = "Thiếu thư mục model ViRa."
            warning = (
                f"Chưa tìm thấy model tại '{self.vira_model_path}'. "
                "Hãy tải model ViRa hoặc bật `VIRA_AUTO_DOWNLOAD=1`."
            )

        return EngineCard(
            id="vira",
            label="ViRa",
            headline="Tối ưu riêng cho tiếng Việt với zero-shot voice cloning.",
            description="Adapter này dùng `MiraTTS.encode_audio()` và `generate()` theo flow của space ViRa, nhưng sinh tuần tự từng chunk để giảm lỗi codec khi câu dài.",
            recommended_for="Dùng mặc định khi cần phát âm tiếng Việt, dấu thanh và ngữ điệu ổn định hơn.",
            output_quality="48kHz audio sau bước decode của codec ViRa.",
            reference_hint="Audio tham chiếu 3-10 giây, một người nói, không vọng phòng. Không cần transcript nhưng vẫn nên dùng câu mẫu rõ ràng.",
            supports_reference_text=False,
            ready=ready,
            summary=summary,
            warning=warning,
            metadata={
                "model_path": str(self.vira_model_path),
                "auto_download": self.vira_auto_download,
                "model_id": self.vira_model_id,
            },
        )

    def _load_f5(self) -> Any:
        with self._locks["f5"]:
            if "f5" in self._loaded_models:
                return self._loaded_models["f5"]

            try:
                from f5_tts.api import F5TTS
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Không thể import F5-TTS: {exc}") from exc

            kwargs: dict[str, Any] = {
                "model": self.f5_model_name,
            }
            if self.f5_ckpt_file:
                kwargs["ckpt_file"] = self.f5_ckpt_file
            if self.f5_vocab_file:
                kwargs["vocab_file"] = self.f5_vocab_file
            if self.f5_vocoder_local_path:
                kwargs["vocoder_local_path"] = self.f5_vocoder_local_path

            try:
                instance = F5TTS(**kwargs)
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Khởi tạo F5-TTS thất bại: {exc}") from exc

            self._loaded_models["f5"] = instance
            return instance

    def _load_vira(self) -> Any:
        with self._locks["vira"]:
            if "vira" in self._loaded_models:
                return self._loaded_models["vira"]

            model_path = self._ensure_vira_model_path()
            try:
                from mira.model import MiraTTS
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Không thể import ViRa/Mira: {exc}") from exc

            try:
                instance = MiraTTS(str(model_path))
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Khởi tạo ViRa thất bại: {exc}") from exc

            self._loaded_models["vira"] = instance
            return instance

    def _ensure_vira_model_path(self) -> Path:
        if self.vira_model_path.exists() and any(self.vira_model_path.iterdir()):
            return self.vira_model_path

        if not self.vira_auto_download:
            raise TTSError(
                f"Thiếu model ViRa tại '{self.vira_model_path}'. "
                "Bật `VIRA_AUTO_DOWNLOAD=1` hoặc tải model thủ công."
            )

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"Không thể auto-download model ViRa: {exc}") from exc

        snapshot_download(
            repo_id=self.vira_model_id,
            local_dir=str(self.vira_model_path),
            local_dir_use_symlinks=False,
        )
        return self.vira_model_path

    def _synthesize_with_f5(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        speed: float,
        remove_silence: bool,
        seed: int | None,
        output_path: Path,
    ) -> SynthesisResult:
        f5 = self._load_f5()
        notes: list[str] = []
        chunks = _split_text_for_tts(text, max_chars=220)
        chunk_waves: list[np.ndarray] = []
        sample_rate = None
        started = time.perf_counter()

        for index, chunk in enumerate(chunks):
            chunk_seed = None if seed is None else seed + index
            try:
                wav, sr, _ = f5.infer(
                    ref_file=str(reference_audio),
                    ref_text=reference_text,
                    gen_text=chunk,
                    speed=float(speed),
                    remove_silence=False,
                    seed=chunk_seed,
                    show_info=lambda *_args, **_kwargs: None,
                )
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"F5-TTS sinh audio thất bại ở chunk {index + 1}: {exc}") from exc

            wave = _to_numpy_audio(wav)
            chunk_waves.append(wave)
            sample_rate = int(sr)

        combined = _crossfade_join(chunk_waves, sample_rate=sample_rate or 24000)
        sf.write(output_path, combined, sample_rate or 24000)

        if remove_silence:
            try:
                from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

                remove_silence_for_generated_wav(str(output_path))
                notes.append("Đã loại bớt khoảng lặng bằng tiện ích upstream của F5-TTS.")
            except Exception:
                notes.append("Không thể remove silence tự động, nên giữ nguyên audio gốc.")

        elapsed = time.perf_counter() - started
        duration = len(combined) / float(sample_rate or 24000)
        if len(chunks) > 1:
            notes.append(f"Văn bản được tách thành {len(chunks)} chunk để giữ inference ổn định hơn.")
        if reference_text:
            notes.append("Đã dùng transcript tham chiếu để neo chất giọng của F5-TTS.")

        return SynthesisResult(
            engine_id="f5",
            engine_label="F5-TTS",
            output_path=output_path,
            sample_rate=sample_rate or 24000,
            duration_seconds=duration,
            inference_seconds=elapsed,
            chunk_count=len(chunks),
            reference_text_used=bool(reference_text),
            seed=seed,
            notes=notes,
        )

    @staticmethod
    def _count_speech_tokens(llm_response: str) -> int:
        """Count valid speech_token_XXX patterns in the LLM output."""
        return len(re.findall(r"speech_token_\d+", llm_response or ""))

    @staticmethod
    def _mutate_chunk_text(chunk: str, attempt: int) -> str:
        """Slightly modify chunk text on retries to nudge the LLM.

        Sometimes the LLM gets stuck on certain text patterns and produces
        zero or too few speech tokens.  Small formatting changes can break
        the degenerate generation pattern without changing meaning.
        """
        chunk = chunk.strip()
        if attempt <= 0:
            return chunk
        if attempt == 1:
            # Ensure trailing punctuation
            if chunk and chunk[-1] not in ".!?;:":
                return chunk + "."
            return chunk
        if attempt == 2:
            # Add a soft leading prompt word
            if not chunk.startswith("Đọc:"):
                return chunk
            return chunk
        if attempt == 3:
            # Remove trailing punctuation to try a different prompt shape
            if chunk and chunk[-1] in ".!?;:":
                return chunk[:-1]
            return chunk + "."
        # attempt >= 4: add explicit comma break
        words = chunk.split()
        mid = len(words) // 2
        if mid > 0 and "," not in chunk:
            words.insert(mid, ",")
            return " ".join(words)
        return chunk

    def _vira_generate_safe(
        self,
        model: Any,
        chunk: str,
        context_tokens: str,
        *,
        max_retries: int = 5,
        min_speech_tokens: int = 10,
    ) -> tuple[Any, list[str]]:
        """Generate audio for a single chunk with validation and retry logic.

        The upstream AudioDecoder.detokenize() extracts speech tokens via regex.
        If the LLM produces fewer valid speech_token_XXX patterns than the ONNX
        Conv kernel's minimum receptive field, the resulting tensor crashes the
        ONNX Conv node at '/out_project/Conv' with "Invalid input shape: {0}".

        Defence layers:
          1. Token count validation (min_speech_tokens=10)
          2. Separate try/except around codec.decode() so ONNX errors trigger retry
          3. Temperature + text mutation schedule across retries
          4. Escalating max_new_tokens budget
        """
        codec = model.codec
        pipe = model.pipe
        base_gen_config = model.gen_config
        notes: list[str] = []

        # Retry schedule: temperature, top_k, repetition_penalty, max_new_tokens
        # Gradually increase randomness and output budget.
        retry_schedule = [
            {},                                                                          # 0: original
            {"temperature": 0.9,  "top_k": 80},                                         # 1: warmer
            {"temperature": 1.0,  "top_k": 100, "repetition_penalty": 1.05},             # 2: more diverse
            {"temperature": 0.85, "top_k": 60,  "max_new_tokens": 2048},                 # 3: longer budget, lower temp
            {"temperature": 1.1,  "top_k": 120, "repetition_penalty": 1.0, "max_new_tokens": 2048},  # 4: max explore
        ]

        last_error: Exception | None = None
        last_token_count = 0

        for attempt in range(max_retries):
            # Mutate text on retries to break degenerate LLM patterns
            attempt_chunk = self._mutate_chunk_text(chunk, attempt)

            # Build generation config for this attempt
            overrides = retry_schedule[attempt] if attempt < len(retry_schedule) else retry_schedule[-1]
            if overrides:
                try:
                    from lmdeploy import GenerationConfig
                    gen_config = GenerationConfig(
                        top_p=overrides.get("top_p", base_gen_config.top_p),
                        top_k=overrides.get("top_k", base_gen_config.top_k),
                        temperature=overrides.get("temperature", base_gen_config.temperature),
                        max_new_tokens=overrides.get("max_new_tokens", base_gen_config.max_new_tokens),
                        repetition_penalty=overrides.get("repetition_penalty", base_gen_config.repetition_penalty),
                        do_sample=True,
                        min_p=overrides.get("min_p", base_gen_config.min_p),
                    )
                except Exception:
                    gen_config = base_gen_config
            else:
                gen_config = base_gen_config

            # --- Step 1: LLM inference ---
            try:
                formatted_prompt = codec.format_prompt(attempt_chunk, context_tokens, None)
                response = pipe([formatted_prompt], gen_config=gen_config, do_preprocess=False)
                llm_output = response[0].text if response else ""
            except Exception as exc:
                last_error = exc
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries} LLM lỗi: {type(exc).__name__}: {exc}"
                )
                continue

            # --- Step 2: Validate token count ---
            token_count = self._count_speech_tokens(llm_output)
            last_token_count = token_count

            if token_count < min_speech_tokens:
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries}: LLM trả về {token_count} speech token "
                    f"(cần ≥{min_speech_tokens})."
                )
                continue  # retry with next config

            # --- Step 3: Decode tokens to audio (separate try/except for ONNX) ---
            try:
                audio = codec.decode(llm_output, context_tokens)
            except Exception as exc:
                last_error = exc
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries} decode lỗi ({token_count} tokens): "
                    f"{type(exc).__name__}: {exc}"
                )
                # Raise min threshold for next attempt since tokens were insufficient
                min_speech_tokens = max(min_speech_tokens, token_count + 5)
                continue

            # --- Step 4: Apply fade in/out ---
            try:
                from mira.model import apply_fade_in_out
                sample_rate = 48000
                fade_in_samples = int(10 * sample_rate / 1000)
                fade_out_samples = int(50 * sample_rate / 1000)
                audio = apply_fade_in_out(audio, fade_in_samples, fade_out_samples)
            except Exception:
                pass  # fade is cosmetic, don't fail on it

            if attempt > 0:
                notes.append(f"Chunk thành công sau {attempt + 1} lần thử.")
            return audio, notes

        # All retries exhausted
        raise TTSError(
            f"ViRa sinh audio thất bại sau {max_retries} lần thử. "
            f"LLM chỉ trả về {last_token_count} speech token (cần ≥{min_speech_tokens}). "
            f"Lỗi cuối: {last_error}. "
            "Nguyên nhân thường là audio tham chiếu quá ngắn, có khoảng lặng lớn, "
            "hoặc chunk văn bản khiến model không sinh được token hợp lệ. "
            "Hãy thử: (1) đổi audio tham chiếu dài 3-10s, nói rõ; "
            "(2) rút ngắn câu văn; (3) tách thành nhiều câu ngắn hơn."
        )

    def _synthesize_with_vira(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        output_path: Path,
    ) -> SynthesisResult:
        model = self._load_vira()
        try:
            from mira.utils import split_text as vira_split_text
        except Exception:
            vira_split_text = None

        context_started = time.perf_counter()
        prepared_reference_audio, ref_duration = self._prepare_reference_audio_for_vira(reference_audio)
        try:
            context_tokens = model.encode_audio(str(prepared_reference_audio))
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"ViRa không encode được audio tham chiếu: {exc}") from exc

        context_numel = _numel(context_tokens)
        if context_numel is not None and context_numel <= 0:
            raise TTSError(
                "ViRa encode ra context token rỗng từ audio tham chiếu. Hãy đổi sang WAV mono, 48kHz hoặc dùng đoạn nói 3-10 giây rõ hơn."
            )

        # Also validate context_tokens is a non-empty string with actual token patterns
        if isinstance(context_tokens, str):
            ctx_token_count = len(re.findall(r"context_token_\d+", context_tokens))
            if ctx_token_count == 0:
                raise TTSError(
                    "ViRa encode ra context token string rỗng (0 context_token pattern). "
                    "Audio tham chiếu có thể quá ngắn hoặc gần im lặng. "
                    "Hãy dùng WAV mono, 3-10 giây nói rõ ràng."
                )

        chunks = vira_split_text(text) if vira_split_text else _split_text_for_tts(text, max_chars=180)
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            raise TTSError("Không có câu hợp lệ để ViRa sinh audio.")

        chunk_waves: list[np.ndarray] = []
        all_notes: list[str] = []
        skipped_chunks: list[int] = []

        for index, chunk in enumerate(chunks):
            try:
                audio, chunk_notes = self._vira_generate_safe(
                    model, chunk, context_tokens,
                    max_retries=5,
                    min_speech_tokens=10,
                )
                all_notes.extend(chunk_notes)
            except TTSError as chunk_exc:
                # Fallback: try splitting this failed chunk into smaller sub-chunks
                sub_chunks = _split_text_for_tts(chunk, max_chars=80)
                sub_chunks = [sc for sc in sub_chunks if sc.strip()]
                sub_success = False

                if len(sub_chunks) > 1:
                    all_notes.append(
                        f"Chunk {index + 1}/{len(chunks)} thất bại → tách thành {len(sub_chunks)} sub-chunk."
                    )
                    for si, sub_chunk in enumerate(sub_chunks):
                        try:
                            sub_audio, sub_notes = self._vira_generate_safe(
                                model, sub_chunk, context_tokens,
                                max_retries=3,
                                min_speech_tokens=10,
                            )
                            all_notes.extend(sub_notes)
                            sub_np = _to_numpy_audio(sub_audio)
                            if sub_np.size > 0:
                                chunk_waves.append(sub_np)
                                sub_success = True
                            else:
                                all_notes.append(
                                    f"  Sub-chunk {si + 1}/{len(sub_chunks)} trả về audio rỗng."
                                )
                        except TTSError:
                            all_notes.append(
                                f"  Sub-chunk {si + 1}/{len(sub_chunks)} thất bại, bị bỏ qua."
                            )

                if sub_success:
                    continue

                # If this is the only chunk and sub-chunking also failed, re-raise
                if len(chunks) == 1:
                    raise
                # For multi-chunk: skip this chunk and continue
                skipped_chunks.append(index + 1)
                all_notes.append(
                    f"Chunk {index + 1}/{len(chunks)} bị bỏ qua do LLM không sinh được speech token hợp lệ."
                )
                continue

            audio_np = _to_numpy_audio(audio)
            if audio_np.size == 0:
                if len(chunks) == 1:
                    raise TTSError(
                        f"ViRa trả về audio rỗng ở chunk {index + 1}. "
                        "Hãy thử rút ngắn câu, đổi audio tham chiếu sang WAV mono rõ tiếng, "
                        "hoặc tách đoạn văn thành các câu ngắn hơn."
                    )
                skipped_chunks.append(index + 1)
                all_notes.append(f"Chunk {index + 1}/{len(chunks)} trả về audio rỗng, bị bỏ qua.")
                continue

            chunk_waves.append(audio_np)

        if not chunk_waves:
            raise TTSError(
                "ViRa không sinh được audio cho bất kỳ chunk nào. "
                "Hãy thử: (1) đổi audio tham chiếu dài 3-10s, nói rõ; "
                "(2) rút ngắn câu văn; (3) dùng audio WAV mono 48kHz."
            )

        audio_np = _crossfade_join(chunk_waves, sample_rate=48000, duration_ms=65)
        sample_rate = 48000
        sf.write(output_path, audio_np, sample_rate)

        elapsed = time.perf_counter() - context_started
        notes = [
            "ViRa dùng reference audio đã được chuẩn hóa về WAV mono 48kHz trước khi encode context token.",
            f"Audio tham chiếu sau chuẩn hóa dài khoảng {ref_duration:.2f}s.",
        ]
        if len(chunks) > 1:
            notes.append(f"ViRa đã sinh tuần tự {len(chunks)} chunk rồi crossfade để giảm lỗi codec và rớt shape.")
        if skipped_chunks:
            notes.append(f"⚠️ Đã bỏ qua {len(skipped_chunks)} chunk lỗi (chunk {', '.join(map(str, skipped_chunks))}).")
        if reference_text:
            notes.append("ViRa hiện không dùng transcript tham chiếu; trường này chỉ để người dùng đối chiếu.")
        notes.extend(all_notes)

        return SynthesisResult(
            engine_id="vira",
            engine_label="ViRa",
            output_path=output_path,
            sample_rate=sample_rate,
            duration_seconds=len(audio_np) / float(sample_rate),
            inference_seconds=elapsed,
            chunk_count=len(chunks),
            reference_text_used=False,
            seed=None,
            notes=notes,
        )

    @staticmethod
    def _torch_cuda_available() -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
