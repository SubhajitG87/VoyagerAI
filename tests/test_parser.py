import unittest

from voyager_ai.parser import extract_json


class ParserTests(unittest.TestCase):
    def test_extract_json_from_fenced_response(self):
        payload = extract_json('```json\n{"destinations":[{"name":"Tokyo"}]}\n```')
        self.assertEqual(payload["destinations"][0]["name"], "Tokyo")

    def test_extract_json_ignores_preface(self):
        payload = extract_json('Here you go:\n{"ok": true, "items": [1, 2]}')
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["items"], [1, 2])


def test_strip_reasoning():
    from voyager_ai.parser import strip_reasoning

    assert strip_reasoning('<think>a\nb</think>\n{"x": 1}') == '{"x": 1}'
    assert strip_reasoning("<think>never closed") == ""
    assert strip_reasoning("plain") == "plain"


if __name__ == "__main__":
    unittest.main()
