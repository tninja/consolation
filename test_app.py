
import unittest

import app


class TestComfort(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_comfort_from_openai_real_call(self):
        """Real call to OpenAI API without mocks.
        Skips if OPENAI_API_KEY is not set in environment.
        """
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set; skipping real API call test")

        q = app.ComfortQuery(
            language="en",
            situation="I feel anxious about uncertainty in my job search.",
            philosophy_background="stoicism",
            max_passages=1,
            guidance="focus on dichotomy of control; concise",
        )

        data = app.get_comfort_from_openai(q)
        print(data)

        # Basic shape checks
        self.assertIsInstance(data, dict)
        for key in ("passages", "reflection", "exercise", "disclaimer"):
            self.assertIn(key, data)

        self.assertIsInstance(data["passages"], list)
        self.assertLessEqual(len(data["passages"]), 1)
        if data["passages"]:
            p = data["passages"][0]
            for k in ("ref", "short_quote", "reason", "full_passage_text"):
                self.assertIn(k, p)
            self.assertIsInstance(p["ref"], str)

        self.assertIsInstance(data["reflection"], str)
        self.assertIsInstance(data["exercise"], str)
        self.assertIsInstance(data["disclaimer"], str)
