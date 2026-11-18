"""Long tables stay atomic while embedding batches preserve order and memory."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from embeddings import embedder


class EmbeddingBatchTests(unittest.TestCase):
    def test_padded_token_budget_and_original_order(self):
        texts = ['short','long table','paragraph','small table']
        lengths = [5,7000,600,100]
        calls = []

        def encode(batch, **kwargs):
            calls.append(batch)
            self.assertEqual(kwargs['batch_size'],len(batch))
            self.assertTrue(kwargs['normalize_embeddings'])
            return np.array([[texts.index(text),1.0] for text in batch])

        model = SimpleNamespace(tokenizer=lambda *args,**kwargs:{'length':lengths},encode=encode)
        with patch.object(embedder,'get_embedder',return_value=model), \
                patch.dict('os.environ',{'EMBED_BATCH_SIZE':'3','EMBED_BATCH_TOKEN_LIMIT':'1400'}):
            vectors = embedder.embed_texts(texts)
        self.assertEqual(vectors,[[0.0,1.0],[1.0,1.0],[2.0,1.0],[3.0,1.0]])
        self.assertEqual(calls,[['long table'],['paragraph','small table'],['short']])

    def test_empty_input_does_not_load_model(self):
        with patch.object(embedder,'get_embedder') as load:
            self.assertEqual(embedder.embed_texts([]),[])
            load.assert_not_called()
