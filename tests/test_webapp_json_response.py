import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import _json_safe


class WebappJsonResponseTests(unittest.TestCase):
    def test_non_finite_values_become_json_null(self):
        payload = {
            "missing_word_reason": float("nan"),
            "debug": [pd.NA, float("inf"), float("-inf")],
        }

        cleaned = _json_safe(payload)
        encoded = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)

        self.assertEqual(cleaned["missing_word_reason"], None)
        self.assertEqual(cleaned["debug"], [None, None, None])
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_frontend_does_not_show_version_banner(self):
        index_html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("当前版本：", index_html)
        self.assertNotIn('class="version-banner"', index_html)


if __name__ == "__main__":
    unittest.main()
