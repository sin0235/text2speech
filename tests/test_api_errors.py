from __future__ import annotations

from io import BytesIO
import unittest

from webapp.app import MAX_UPLOAD_MB, app


class ApiErrorHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

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
