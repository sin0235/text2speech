from __future__ import annotations

import unittest

import numpy as np

from webapp.tts_service import _fallback_cleanup_vira_text, _normalize_reference_wave


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


if __name__ == "__main__":
    unittest.main()
