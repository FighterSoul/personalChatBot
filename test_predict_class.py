import unittest
import json
from chatbot import predict_class


class TestPredictClass(unittest.TestCase):
    def setUp(self):
        # Load test data from test_data.json
        with open('test_data.json', 'r') as file:
            self.test_data = json.load(file)

    def test_predict_class(self):
        for test_case in self.test_data:
            input_text = test_case['input']
            expected_output = test_case['expected_output']

            with self.subTest(input_text=input_text):
                result = predict_class(input_text)
                intents = [intent['intent'] for intent in result]
                
                # Check if any expected output is in the list of predicted intents
                self.assertTrue(any(intent in expected_output for intent in intents),
                                f"Failed for input: {input_text}. Expected: {expected_output}, Got: {intents}")

if __name__ == '__main__':
    unittest.main()