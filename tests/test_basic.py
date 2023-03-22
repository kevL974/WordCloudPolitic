import unittest
from wordcloudpolitics.cleaning import to_lower_case, remove_multispaces, remove_url, transform_accented_character, remove_special_character


class CleaningTestCase(unittest.TestCase):

    def test_to_lower_case(self):
        self.assertEqual("texte en majuscule", to_lower_case("TEXTE EN MAJUSCULE"))
        self.assertEqual("texte avec 1 majuscule", to_lower_case("texte avec 1 Majuscule"))
        self.assertEqual("texte sans majuscule", to_lower_case("texte sans majuscule"))

    def test_remove_url(self):
        self.assertEqual("texte avec URL ", remove_url("texte avec URL https://t.co/D22enVH3h7"))
        self.assertEqual("texte sans URL ", remove_url("texte sans URL "))

    def test_remove_multispace(self):
        self.assertEqual("| | | |", remove_multispaces("|   | |       |"))
        self.assertEqual("texte avec un grand espace .", remove_multispaces("texte avec un grand espace            ."))
        self.assertEqual(" texte des tabulations et des espaces.",
                         remove_multispaces(" texte des tabulations       et des     espaces."))

    def test_transform_accented_character(self):
        self.assertEqual("eeee", transform_accented_character("éèêë"))
        self.assertEqual("aa", transform_accented_character("aà"))
        self.assertEqual("iii", transform_accented_character("iïî"))
        self.assertEqual("oo", transform_accented_character("oô"))
        self.assertEqual("u", transform_accented_character("ù"))

    def test_remove_special_character(self):
        self.assertEqual("cestàdire  10", remove_special_character("c'est-à-dire : 10#"))


if __name__ == "__main__":
    unittest.main()
