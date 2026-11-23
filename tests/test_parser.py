import unittest

from voyager_ai.parser import extract_json, parse_itinerary


class ParserTests(unittest.TestCase):
    def test_extract_json_from_fenced_response(self):
        payload = extract_json('```json\n{"destinations":[{"name":"Tokyo"}]}\n```')
        self.assertEqual(payload["destinations"][0]["name"], "Tokyo")

    def test_extract_json_ignores_preface(self):
        payload = extract_json('Here you go:\n{"ok": true, "items": [1, 2]}')
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["items"], [1, 2])

    def test_parse_itinerary_days_and_extras(self):
        parsed = parse_itinerary(
            "## Day 1 - Arrival\nMorning: Land.\n\n"
            "## Day 2 - Markets\nMorning: Explore.\n\n"
            "## Practical Tips\nPack light."
        )
        self.assertEqual(len(parsed.days), 2)
        self.assertEqual(parsed.days[0].title, "Arrival")
        self.assertEqual(parsed.extras[0].title, "Practical Tips")


def test_strip_reasoning():
    from voyager_ai.parser import strip_reasoning

    assert strip_reasoning('<think>a\nb</think>\n{"x": 1}') == '{"x": 1}'
    assert strip_reasoning("<think>never closed") == ""
    assert strip_reasoning("plain") == "plain"


if __name__ == "__main__":
    unittest.main()
