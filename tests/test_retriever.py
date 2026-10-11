import unittest
from retriever import build_index, retrieve

class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"source": "sales.txt", "text": "Monthly sales revenue increased in the second quarter."},
            {"source": "delivery.txt", "text": "Late deliveries happened because of warehouse delays."},
        ]
        self.index = build_index(self.chunks)

    def test_relevant_chunk_ranked_first(self):
        results = retrieve("Why were deliveries late?", self.chunks, self.index, top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "delivery.txt")

    def test_blank_query(self):
        self.assertEqual(retrieve(" ", self.chunks, self.index), [])

    def test_no_match_returns_empty(self):
        self.assertEqual(retrieve("quantum astronomy", self.chunks, self.index), [])

if __name__ == "__main__":
    unittest.main()
