import unittest
from document_processor import chunk_text

class ChunkTextTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_one_chunk(self):
        self.assertEqual(
    chunk_text("hello world", chunk_size=100, overlap=20),
    ["hello world"]
)

    def test_long_text_multiple_chunks(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)

if __name__ == "__main__":
    unittest.main()
