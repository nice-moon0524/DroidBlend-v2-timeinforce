import unittest

from droidblend.hotpotqa import _answers, hotpotqa_prompt


class HotpotQATests(unittest.TestCase):
    def test_prompt_and_answer_normalization(self):
        prompt = hotpotqa_prompt("A context", "Who?")
        self.assertIn("Context:\nA context", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertEqual(_answers({"text": ["Paris"]}), ["Paris"])
        self.assertEqual(_answers("Paris"), ["Paris"])


if __name__ == "__main__":
    unittest.main()
