from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

from webapp.app import MAX_UPLOAD_MB, app, studio
from webapp.tts_service import EngineCard, PresetVoice, SynthesisResult, TranscriptionResult, _normalize_tts_prompt_text


class ApiErrorHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_home_redirects_to_default_engine_page(self) -> None:
        response = self.client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/studio/gwen"))

    def test_gwen_engine_page_renders(self) -> None:
        response = self.client.get("/studio/gwen")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Gwen-TTS", response.get_data(as_text=True))
        self.assertNotIn('href="/asr"', response.get_data(as_text=True))

    def test_asr_page_redirects_to_studio_when_disabled(self) -> None:
        response = self.client.get("/asr", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/studio/gwen"))

    def test_gwen_page_lists_official_preset_voices(self) -> None:
        response = self.client.get("/studio/gwen")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Yến Nhi", html)
        self.assertIn("NSND Kim Cúc", html)
        self.assertIn('id="advancedGwenSettings"', html)
        self.assertIn('id="advancedGwenSettingsToggle"', html)
        self.assertIn('id="pronunciationSettings"', html)
        self.assertIn('id="pronunciationSettingsToggle"', html)
        self.assertIn('maxlength="5000"', html)
        self.assertIn("0 / 5000", html)
        self.assertIn('id="textProgressBar"', html)

    def test_unknown_engine_page_returns_404(self) -> None:
        response = self.client.get("/studio/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_generate_without_reference_audio_returns_json(self) -> None:
        response = self.client.post("/api/tts/generate", data={"text": "xin chao"})

        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.is_json)
        self.assertEqual(
            response.get_json(),
            {
                "ok": False,
                "error": "Cần upload audio tham chiếu trước khi sinh giọng.",
            },
        )

    def test_transcribe_reference_returns_disabled_when_asr_off(self) -> None:
        response = self.client.post("/api/tts/transcribe-reference", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.is_json)
        self.assertIn("ASR đang tắt", response.get_json()["error"])

    def test_transcribe_reference_without_audio_returns_json(self) -> None:
        with patch("webapp.app.studio.is_asr_enabled", return_value=True):
            response = self.client.post("/api/tts/transcribe-reference", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.is_json)
        self.assertEqual(
            response.get_json(),
            {
                "ok": False,
                "error": "Cần upload audio tham chiếu để nhận diện transcript.",
            },
        )

    def test_transcribe_reference_returns_payload(self) -> None:
        result = TranscriptionResult(
            text="xin chao tat ca moi nguoi",
            language="vi",
            model_id="openai/whisper-small",
            duration_seconds=2.35,
            inference_seconds=0.61,
            notes=["ASR model: openai/whisper-small."],
        )

        with patch("webapp.app.studio.is_asr_enabled", return_value=True):
            with patch("webapp.app.studio.transcribe_reference_audio", return_value=result):
                response = self.client.post(
                    "/api/tts/transcribe-reference",
                    data={
                        "reference_audio": (BytesIO(b"fake wav bytes"), "clone.wav"),
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["text"], "xin chao tat ca moi nguoi")
        self.assertEqual(payload["language"], "vi")
        self.assertEqual(payload["model_id"], "openai/whisper-small")

    def test_api_asr_transcribe_returns_disabled_when_asr_off(self) -> None:
        response = self.client.post("/api/asr/transcribe", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.is_json)
        self.assertIn("ASR đang tắt", response.get_json()["error"])

    def test_safe_tts_prompt_normalization_merges_lines_and_spacing(self) -> None:
        normalized, notes = _normalize_tts_prompt_text("Xin chào\n- hôm nay   ưu đãi lớn\n- miễn phí giao hàng")

        self.assertEqual(normalized, "Xin chào. hôm nay ưu đãi lớn. miễn phí giao hàng.")
        self.assertTrue(notes)

    def test_safe_tts_prompt_normalization_cleans_punctuation(self) -> None:
        normalized, notes = _normalize_tts_prompt_text("  Xin  chào  ,bạn!!!  ")

        self.assertEqual(normalized, "Xin chào, bạn!")
        self.assertTrue(notes)

    def test_synthesize_applies_safe_prompt_normalization_before_engine_call(self) -> None:
        reference_path = studio.reference_dir / "normalization-test.wav"
        reference_path.write_bytes(b"RIFF")
        engine_card = EngineCard(
            id="gwen",
            label="Gwen-TTS",
            headline="demo",
            description="demo",
            recommended_for="demo",
            output_quality="demo",
            reference_hint="demo",
            supports_reference_text=True,
            ready=True,
            summary="ready",
        )
        result = SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key="default",
            model_label="Default",
            output_path=studio.output_dir / "norm-pass.wav",
            sample_rate=24000,
            duration_seconds=1.0,
            inference_seconds=0.2,
            chunk_count=1,
            reference_text_used=True,
            seed=None,
            notes=[],
        )

        try:
            with patch.object(studio, "get_engine_card", return_value=engine_card):
                with patch.object(studio, "resolve_model_spec", return_value={"key": "default", "label": "Default"}):
                    with patch.object(studio, "_synthesize_with_gwen", return_value=result) as synth_mock:
                        output = studio.synthesize(
                            text="Xin chào\n- hôm nay sale lớn",
                            reference_audio=reference_path,
                            reference_text="xin chao",
                            speed=1.0,
                            model_key="default",
                            custom_model="",
                        )
        finally:
            reference_path.unlink(missing_ok=True)

        self.assertEqual(synth_mock.call_args.kwargs["text"], "Xin chào. hôm nay sale lớn.")
        self.assertTrue(output.notes)
        self.assertIn("Đã chuẩn hóa", output.notes[0])

    def test_generate_with_preset_voice_skips_upload_requirement(self) -> None:
        preset_voice = PresetVoice(
            id="yen_nhi",
            name="Yến Nhi",
            avatar="YN",
            style="Tự nhiên",
            audio_filename="yen_nhi.wav",
            reference_text="xin chao",
        )
        result = SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key="default",
            model_label="Default",
            output_path=studio.output_dir / "preset-test.wav",
            sample_rate=24000,
            duration_seconds=1.25,
            inference_seconds=0.42,
            chunk_count=1,
            reference_text_used=True,
            seed=None,
            notes=["synth ok"],
        )

        with patch("webapp.app.studio.get_preset_voice_reference", return_value=(preset_voice, studio.output_dir / "yen_nhi.wav")):
            with patch("webapp.app.studio.synthesize", return_value=result):
                response = self.client.post(
                    "/api/tts/generate",
                    data={
                        "engine": "gwen",
                        "text": "xin chao",
                        "preset_voice_id": "yen_nhi",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["preset_voice_id"], "yen_nhi")
        self.assertEqual(payload["preset_voice_label"], "Yến Nhi")
        self.assertIn("Đã dùng preset voice Gwen: Yến Nhi.", payload["notes"])

    def test_generate_async_returns_job_id(self) -> None:
        preset_voice = PresetVoice(
            id="yen_nhi",
            name="Yến Nhi",
            avatar="YN",
            style="Tự nhiên",
            audio_filename="yen_nhi.wav",
            reference_text="xin chao",
        )

        with patch("webapp.app.studio.get_preset_voice_reference", return_value=(preset_voice, studio.output_dir / "yen_nhi.wav")):
            with patch("webapp.app.tts_job_manager.enqueue", return_value="job-123") as enqueue_mock:
                response = self.client.post(
                    "/api/tts/generate",
                    data={
                        "engine": "gwen",
                        "text": "xin chao",
                        "preset_voice_id": "yen_nhi",
                        "async": "1",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["queued"], True)
        self.assertEqual(payload["job_id"], "job-123")
        enqueue_mock.assert_called_once()

    def test_job_status_returns_progress_payload(self) -> None:
        with patch(
            "webapp.app.tts_job_manager.get_snapshot",
            return_value={
                "id": "job-123",
                "status": "running",
                "message": "dang synthesize",
            },
        ):
            response = self.client.get("/api/tts/jobs/job-123")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["done"], False)
        self.assertEqual(payload["status"], "running")

    def test_job_status_returns_completed_result(self) -> None:
        with patch(
            "webapp.app.tts_job_manager.get_snapshot",
            return_value={
                "id": "job-123",
                "status": "completed",
                "result": {"ok": True, "download_name": "done.wav"},
            },
        ):
            response = self.client.get("/api/tts/jobs/job-123")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["done"], True)
        self.assertEqual(payload["result"]["download_name"], "done.wav")

    def test_job_status_returns_404_when_missing(self) -> None:
        with patch("webapp.app.tts_job_manager.get_snapshot", return_value=None):
            response = self.client.get("/api/tts/jobs/missing-job")

        self.assertEqual(response.status_code, 404)
        payload = response.get_json()
        self.assertEqual(payload["ok"], False)
        self.assertIn("job", payload["error"].lower())

    def test_generate_with_gwen_settings_and_pronunciation_rules_passes_through(self) -> None:
        preset_voice = PresetVoice(
            id="yen_nhi",
            name="Yến Nhi",
            avatar="YN",
            style="Tự nhiên",
            audio_filename="yen_nhi.wav",
            reference_text="xin chao",
        )
        result = SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key="default",
            model_label="Default",
            output_path=studio.output_dir / "preset-advanced-test.wav",
            sample_rate=24000,
            duration_seconds=1.25,
            inference_seconds=0.42,
            chunk_count=1,
            reference_text_used=True,
            seed=None,
            notes=["advanced ok"],
        )

        with patch("webapp.app.studio.get_preset_voice_reference", return_value=(preset_voice, studio.output_dir / "yen_nhi.wav")):
            with patch("webapp.app.studio.synthesize", return_value=result) as synthesize_mock:
                response = self.client.post(
                    "/api/tts/generate",
                    data={
                        "engine": "gwen",
                        "text": "AI giup KPI ro hon",
                        "preset_voice_id": "yen_nhi",
                        "speed": "1.2",
                        "gwen_temperature": "0.4",
                        "gwen_top_p": "0.85",
                        "gwen_top_k": "42",
                        "gwen_repetition_penalty": "2.3",
                        "gwen_max_new_tokens": "5000",
                        "gwen_subtalker_do_sample": "on",
                        "gwen_subtalker_temperature": "0.2",
                        "gwen_subtalker_top_k": "11",
                        "gwen_subtalker_top_p": "0.75",
                        "gwen_subtalker_sampling_method": "gumbel",
                        "pronunciation_from": ["AI", "KPI"],
                        "pronunciation_to": ["ây ai", "cây pi ai"],
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        kwargs = synthesize_mock.call_args.kwargs
        self.assertEqual(kwargs["gwen_generation_config"]["speed"], 1.2)
        self.assertEqual(kwargs["gwen_generation_config"]["temperature"], 0.4)
        self.assertEqual(kwargs["gwen_generation_config"]["top_p"], 0.85)
        self.assertEqual(kwargs["gwen_generation_config"]["top_k"], 42)
        self.assertEqual(kwargs["gwen_generation_config"]["max_new_tokens"], 5000)
        self.assertTrue(kwargs["gwen_generation_config"]["subtalker_do_sample"])
        self.assertEqual(kwargs["pronunciation_overrides"], [("AI", "ây ai"), ("KPI", "cây pi ai")])

    def test_generate_oversize_upload_returns_json(self) -> None:
        response = self.client.post(
            "/api/tts/generate",
            data={
                "text": "xin chao",
                "reference_audio": (BytesIO(b"0" * (33 * 1024 * 1024)), "oversize.wav"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], False)
        self.assertIn(f"{MAX_UPLOAD_MB} MB", payload["error"])

    def test_generate_gwen_without_subtalker_checkbox_keeps_default(self) -> None:
        preset_voice = PresetVoice(
            id="yen_nhi",
            name="Yến Nhi",
            avatar="YN",
            style="Tự nhiên",
            audio_filename="yen_nhi.wav",
            reference_text="xin chao",
        )
        result = SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key="default",
            model_label="Default",
            output_path=studio.output_dir / "preset-default-cp.wav",
            sample_rate=24000,
            duration_seconds=1.25,
            inference_seconds=0.42,
            chunk_count=1,
            reference_text_used=True,
            seed=None,
            notes=["default cp ok"],
        )

        with patch("webapp.app.studio.get_preset_voice_reference", return_value=(preset_voice, studio.output_dir / "yen_nhi.wav")):
            with patch("webapp.app.studio.synthesize", return_value=result) as synthesize_mock:
                response = self.client.post(
                    "/api/tts/generate",
                    data={
                        "engine": "gwen",
                        "text": "AI giup KPI ro hon",
                        "preset_voice_id": "yen_nhi",
                        "speed": "1.0",
                        "gwen_temperature": "0.3",
                        "gwen_top_p": "0.9",
                        "gwen_top_k": "20",
                        "gwen_repetition_penalty": "2.0",
                        "gwen_max_new_tokens": "4096",
                        "gwen_subtalker_temperature": "0.1",
                        "gwen_subtalker_top_k": "20",
                        "gwen_subtalker_top_p": "1.0",
                        "gwen_subtalker_sampling_method": "gumbel",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        kwargs = synthesize_mock.call_args.kwargs
        self.assertTrue(kwargs["gwen_generation_config"]["subtalker_do_sample"])

    def test_unknown_api_route_returns_json(self) -> None:
        response = self.client.get("/api/does-not-exist")

        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], False)
        self.assertIn("URL", payload["error"])


if __name__ == "__main__":
    unittest.main()
