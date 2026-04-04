from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from webapp.tts_service import (
    EngineCard,
    TTSError,
    _fallback_cleanup_vira_text,
    _format_f5_import_error,
    _format_f5_runtime_error,
    _format_gwen_import_error,
    _format_gwen_runtime_error,
    _format_vieneu_import_error,
    _format_vieneu_runtime_error,
    _map_import_name_to_package,
    _normalize_engine_id,
    _normalize_reference_wave,
    _vieneu_mode_requires_reference_text,
    TTSStudioService,
)


class TTSServiceHelperTest(unittest.TestCase):
    def test_normalize_reference_wave_boosts_quiet_audio(self) -> None:
        audio = np.array([0.0, 0.02, -0.04, 0.01], dtype=np.float32)

        normalized = _normalize_reference_wave(audio)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.92, places=2)
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0, places=2)

    def test_fallback_cleanup_vira_text_removes_uncommon_symbols(self) -> None:
        cleaned = _fallback_cleanup_vira_text(
            '  TP.HCM — AI/ML (2025)…  ',
            aggressive=True,
            ensure_terminal=True,
        )

        self.assertNotIn("—", cleaned)
        self.assertNotIn("…", cleaned)
        self.assertNotIn("(", cleaned)
        self.assertNotIn(")", cleaned)
        self.assertTrue(cleaned.endswith((".", "!", "?", ",")))
        self.assertIn("AI, ML", cleaned)

    def test_map_import_name_to_package_handles_hydra(self) -> None:
        self.assertEqual(_map_import_name_to_package("hydra"), "hydra-core")
        self.assertEqual(_map_import_name_to_package("qwen_tts"), "qwen-tts")

    def test_format_f5_import_error_mentions_reinstall_hint(self) -> None:
        exc = ModuleNotFoundError("No module named 'cached_path'", name="cached_path")

        message = _format_f5_import_error(exc)

        self.assertIn("cached_path", message)
        self.assertIn("f5-tts", message)
        self.assertIn("--no-deps", message)

    def test_format_vieneu_import_error_mentions_package(self) -> None:
        exc = ModuleNotFoundError("No module named 'llama_cpp'", name="llama_cpp")

        message = _format_vieneu_import_error(exc)

        self.assertIn("vieneu", message)
        self.assertIn("llama-cpp-python", message)

    def test_format_gwen_import_error_mentions_package(self) -> None:
        exc = ModuleNotFoundError("No module named 'flash_attn'", name="flash_attn")

        message = _format_gwen_import_error(exc)

        self.assertIn("qwen-tts", message)
        self.assertIn("flash-attn", message)

    def test_format_vieneu_runtime_error_mentions_torch_reinstall(self) -> None:
        exc = RuntimeError(
            "Codec 'neuphonic/distill-neucodec' requires PyTorch. Or install torch via: pip install vieneu[gpu]"
        )

        message = _format_vieneu_runtime_error(exc)

        self.assertIn("PyTorch hiện tại", message)
        self.assertIn("CUDA 12.8", message)
        self.assertIn("VIENEU_MODE=turbo", message)

    def test_format_f5_runtime_error_shortens_torchaudio_ffmpeg_failure(self) -> None:
        exc = RuntimeError(
            "Could not load libtorchaudio codec. Likely causes: FFmpeg is not properly installed in your environment."
        )

        message = _format_f5_runtime_error(exc)

        self.assertIn("torchaudio/FFmpeg", message)
        self.assertIn("Gwen-TTS", message)

    def test_format_gwen_runtime_error_mentions_sdpa_fallback(self) -> None:
        exc = RuntimeError("flash_attn kernel failed to initialize")

        message = _format_gwen_runtime_error(exc)

        self.assertIn("sdpa", message)
        self.assertIn("GWEN_ATTN_IMPLEMENTATION", message)

    def test_normalize_engine_id_maps_legacy_vira(self) -> None:
        self.assertEqual(_normalize_engine_id("vira"), "vieneu")
        self.assertEqual(_normalize_engine_id("vieneu"), "vieneu")

    def test_vieneu_mode_requires_reference_text_for_standard_and_fast(self) -> None:
        self.assertTrue(_vieneu_mode_requires_reference_text("standard"))
        self.assertTrue(_vieneu_mode_requires_reference_text("fast"))
        self.assertFalse(_vieneu_mode_requires_reference_text("turbo"))
        self.assertFalse(_vieneu_mode_requires_reference_text("turbo_gpu"))

    def test_default_engine_is_gwen(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))

        self.assertEqual(service.default_engine, "gwen")
        self.assertEqual(service.gwen_model_id, "g-group-ai-lab/gwen-tts-0.6B")
        self.assertEqual(service.vieneu_mode, "standard")

    def test_offline_f5_is_still_shown_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                service._probe_f5_import = lambda: (False, "Chưa cài gói `f5_tts`.")

                cards = service.get_engine_cards()

        self.assertEqual([card.id for card in cards], ["gwen", "vieneu", "f5"])

    def test_gwen_card_is_not_ready_without_cuda(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                service._probe_gwen_import = lambda: (True, None)
                service._torch_version_label = lambda: "2.11.0+cu128"
                service._torch_cuda_available = lambda: False

                card = service.get_engine_card("gwen")

        self.assertFalse(card.ready)
        self.assertIn("GPU CUDA", card.warning or "")

    def test_vieneu_standard_card_is_not_ready_on_old_torch(self) -> None:
        with patch.dict(os.environ, {"VIENEU_MODE": "standard"}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                service._probe_vieneu_import = lambda: (True, None)
                service._torch_version_label = lambda: "2.6.0+cu124"

                card = service.get_engine_card("vieneu")

        self.assertFalse(card.ready)
        self.assertIn("torch >= 2.11", card.warning or "")

    def test_vieneu_standard_requires_reference_text_before_inference(self) -> None:
        with patch.dict(os.environ, {"VIENEU_MODE": "standard"}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                reference_audio = Path(tmpdir) / "reference.wav"
                reference_audio.write_bytes(b"fake")

                service.get_engine_card = lambda _engine_id: EngineCard(
                    id="vieneu",
                    label="VieNeu-TTS",
                    headline="",
                    description="",
                    recommended_for="",
                    output_quality="",
                    reference_hint="",
                    supports_reference_text=True,
                    ready=True,
                    summary="",
                )

                with self.assertRaises(TTSError) as exc_info:
                    service.synthesize(
                        engine_id="vieneu",
                        text="xin chao",
                        reference_audio=reference_audio,
                        reference_text="",
                    )

        self.assertIn("transcript tham chiếu", str(exc_info.exception))

    def test_prepare_reference_audio_for_f5_normalizes_wav(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                reference_audio = Path(tmpdir) / "reference.wav"
                wave = np.linspace(-0.2, 0.2, 24000, dtype=np.float32)
                sf.write(reference_audio, wave, 24000)

                prepared_path, duration_seconds, notes = service._prepare_reference_audio_for_f5(reference_audio)
                prepared_exists = prepared_path.exists()
                prepared_name = prepared_path.name

        self.assertTrue(prepared_exists)
        self.assertTrue(prepared_name.endswith("-f5-24k.wav"))
        self.assertGreater(duration_seconds, 0.9)
        self.assertTrue(any("WAV mono 24kHz" in note for note in notes))

    def test_prepare_reference_audio_for_f5_requires_ffmpeg_for_compressed_input(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                service = TTSStudioService(Path(tmpdir))
                reference_audio = Path(tmpdir) / "reference.m4a"
                reference_audio.write_bytes(b"fake")

                with patch("webapp.tts_service.shutil.which", return_value=None):
                    with self.assertRaises(TTSError) as exc_info:
                        service._prepare_reference_audio_for_f5(reference_audio)

        self.assertIn("cần `ffmpeg`", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
