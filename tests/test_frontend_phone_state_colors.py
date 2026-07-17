from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendPhoneStateColorTests(unittest.TestCase):
    def test_phone_states_have_distinct_component_colors(self):
        css = (ROOT / "webapp" / "static" / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".phone-pill.correct", css)
        self.assertIn(".phone-pill.error", css)
        self.assertIn(".phone-pill.review", css)
        self.assertIn("--correct-bg: #dcfce7", css)
        self.assertIn("--error-bg: #fee2e2", css)
        self.assertIn("--review-bg: #fef3c7", css)

    def test_table_rows_receive_the_same_state_class(self):
        javascript = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("tr.className = `phone-row ${cls}`", javascript)
        self.assertIn('if (displayedDecision === "\\u6b63\\u786e") return "correct"', javascript)
        self.assertIn('return "error"', javascript)
        self.assertIn('return "review"', javascript)
        self.assertIn('return "correct"', javascript)

    def test_correct_decision_badge_is_explicitly_green(self):
        css = (ROOT / "webapp" / "static" / "styles.css").read_text(encoding="utf-8")
        start = css.rindex(".badge.correct {")
        rule = css[start:css.index("}", start)]
        self.assertIn("color: #15803d", rule)
        self.assertIn("background: #dcfce7", rule)

    def test_static_assets_are_cache_busted(self):
        html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("styles.css?v=PHONE_THREE_STATE_V5_IPA", html)
        self.assertIn("app.js?v=PHONE_THREE_STATE_V5_IPA", html)

    def test_phone_display_uses_common_ipa_notation(self):
        javascript = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('M: "m"', javascript)
        self.assertIn('EY: "eɪ"', javascript)
        self.assertIn('AH: "ʌ"', javascript)
        self.assertIn("function ipaPhone(phone)", javascript)
        self.assertIn("pill.textContent = displayedPhone", javascript)
        self.assertIn('class="ipa-phone"', javascript)
        self.assertIn("<th>IPA</th>", html)


if __name__ == "__main__":
    unittest.main()
