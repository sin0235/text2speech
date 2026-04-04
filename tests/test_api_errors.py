from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

from webapp.app import MAX_UPLOAD_MB, app, studio
from webapp.tts_service import PresetVoice, SynthesisResult


class ApiErrorHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_home_redirects_to_default_engine_page(self) -> None:
        response = self.client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/studio/gwen"))

    def test_engine_pages_render(self) -> None:
        for engine_id, label in (
            ("gwen", "Gwen-TTS"),
            ("vieneu", "VieNeu-TTS"),
            ("f5", "F5-TTS"),
        ):
            with self.subTest(engine_id=engine_id):
                response = self.client.get(f"/studio/{engine_id}")

                self.assertEqual(response.status_code, 200)
                self.assertIn(label, response.get_data(as_text=True))

    def test_gwen_page_lists_official_preset_voices(self) -> None:
        response = self.client.get("/studio/gwen")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("10 giọng có sẵn", html)
        self.assertIn("Yến Nhi", html)
        self.assertIn("NSND Kim Cúc", html)
        self.assertIn("Cài đặt nâng cao", html)
        self.assertIn("Cài Đặt Phát Âm", html)

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

    def test_unknown_api_route_returns_json(self) -> None:
        response = self.client.get("/api/does-not-exist")

        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertEqual(payload["ok"], False)
        self.assertIn("URL", payload["error"])


if __name__ == "__main__":
    unittest.main()
