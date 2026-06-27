import unittest

from voyager_ai.models import Destination, TravelPreferences, ValidationError, dedupe_destinations


class ModelTests(unittest.TestCase):
    def test_preferences_validate_required_departure(self):
        with self.assertRaises(ValidationError):
            TravelPreferences(departure_city="", months=6, nights=7, vibe="city")

    def test_preferences_validate_nights_range(self):
        with self.assertRaises(ValidationError):
            TravelPreferences(departure_city="Mumbai", months=6, nights=31, vibe="city")

    def test_destination_from_mapping_trims_tags(self):
        destination = Destination.from_mapping(
            {
                "name": " Tokyo ",
                "country": " Japan ",
                "why": " Food and design. ",
                "tags": [" food ", " design ", " culture ", " extra "],
            }
        )
        self.assertEqual(destination.name, "Tokyo")
        self.assertEqual(destination.tags, ["food", "design", "culture"])

    def test_dedupe_destinations(self):
        destinations = [
            Destination("Tokyo", "Japan", "A"),
            Destination("tokyo", "japan", "B"),
            Destination("Seoul", "South Korea", "C"),
        ]
        self.assertEqual(len(dedupe_destinations(destinations)), 2)


if __name__ == "__main__":
    unittest.main()
