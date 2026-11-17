"""Chunk browsing uses saved index content or the shared PDF splitter."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from streamlit.testing.v1 import AppTest

from app.api import app
from app import documents
from app.chunk_browser import _load_chunks
from ingestion.ingest import load_pdf_chunks


def create_pdf(path):
    writer = PdfWriter()
    for page_no in range(2):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
            NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        content = DecodedStreamObject()
        text = (f'Page {page_no + 1} describes document chunk browsing and filtering. ' * 30).encode()
        content.set_data(b'BT /F1 12 Tf 72 720 Td (' + text + b') Tj ET')
        page[NameObject('/Contents')] = writer._add_object(content)
    with path.open('wb') as output:
        writer.write(output)


class ChunkApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / 'data'
        self.data.mkdir()
        self.meta = self.base / 'metadata.pkl'
        for target, value in [('DATA_DIR', self.data), ('META_PATH', self.meta)]:
            ctx = patch.object(documents, target, value)
            ctx.start()
            self.addCleanup(ctx.stop)
        self.client = TestClient(app)
        create_pdf(self.data / 'sample.pdf')

    def test_saved_chunks_match_actual_index_without_parsing_pdf(self):
        texts = ['Exact indexed block one', 'Exact indexed block two']
        metas = [{'source': 'sample.pdf', 'page': 1}, {'source': 'sample.pdf', 'page': 2}]
        self.meta.write_bytes(pickle.dumps({'chunks': texts, 'metadatas': metas}))
        listing = self.client.get('/documents').json()['documents']
        self.assertEqual(listing[0]['indexed_chunks'], 2)
        with patch.object(documents, '_parse_pdf', side_effect=AssertionError('Must use stored chunks')):
            response = self.client.get('/documents/chunks', params={'filename': 'sample.pdf'})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['indexed'])
        self.assertEqual([chunk['text'] for chunk in result['chunks']], texts)
        self.assertEqual([chunk['id'] for chunk in result['chunks']], [1, 2])

    def test_unindexed_preview_uses_ingestion_splitter_and_page_numbers(self):
        expected, metas = load_pdf_chunks(self.data / 'sample.pdf')
        response = self.client.get('/documents/chunks', params={'filename': 'sample.pdf'})
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(result['indexed'])
        self.assertEqual([chunk['text'] for chunk in result['chunks']], expected)
        self.assertEqual([chunk['page'] for chunk in result['chunks']], [meta['page'] for meta in metas])
        self.assertTrue(all(chunk['char_count'] <= 800 for chunk in result['chunks']))
        self.assertFalse(self.meta.exists())

    def test_arbitrary_upload_preview_does_not_save_to_library(self):
        before = set(self.data.iterdir())
        response = self.client.post('/documents/preview', files={
            'file': ('another.pdf', (self.data / 'sample.pdf').read_bytes(), 'application/pdf'),
        })
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()['total_chunks'], 0)
        self.assertFalse(response.json()['indexed'])
        self.assertEqual(set(self.data.iterdir()), before)

    def test_invalid_paths_missing_file_and_bad_pdf(self):
        for filename in ['../secret.pdf', '..\\secret.pdf', 'C:secret.pdf']:
            self.assertEqual(self.client.get('/documents/chunks', params={'filename': filename}).status_code, 400)
        self.assertEqual(self.client.get('/documents/chunks', params={'filename': 'missing.pdf'}).status_code, 404)
        with patch.object(documents.logger, 'exception'):
            response = self.client.post('/documents/preview', files={'file': ('bad.pdf', b'not a PDF', 'application/pdf')})
        self.assertEqual(response.status_code, 422)


class ChunkUiTests(unittest.TestCase):
    def setUp(self):
        _load_chunks.clear()
        self.documents = {'documents': [
            {'filename': 'first.pdf', 'revision': '1'}, {'filename': 'second.pdf', 'revision': '2'},
        ]}
        self.result = {'filename': 'first.pdf', 'indexed': True, 'total_chunks': 12,
            'chunk_size': 800, 'chunk_overlap': 150,
            'chunks': [{'id': i, 'page': 1 if i <= 10 else 2, 'char_count': 12,
                'text': f'Full block {i} text'} for i in range(1, 13)]}

    def fake_get(self, url, **kwargs):
        self.assertFalse(kwargs['trust_env'])
        response = Mock()
        if url.endswith('/documents'):
            response.json.return_value = self.documents
        else:
            result = {**self.result, 'filename': kwargs['params']['filename']}
            response.json.return_value = result
        return response

    def test_choose_document_filter_and_paginate_without_losing_chat(self):
        with patch('httpx.get', side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py', default_timeout=20).run()
            ui.session_state['messages'] = [{'role': 'assistant', 'content': 'Keep this conversation'}]
            ui.radio(key='view').set_value('PDF 分块').run()
            self.assertFalse(ui.exception, str(ui.exception))
            self.assertEqual(len(ui.expander), 10)
            ui.number_input(key='chunk_page').set_value(2).run()
            self.assertEqual(len(ui.expander), 2)
            ui.selectbox[0].set_value('second.pdf').run()
            self.assertEqual(ui.number_input(key='chunk_page').value, 1)
            ui.text_input[0].set_value('block 12 ').run()
            self.assertEqual(len(ui.expander), 1)
            self.assertIn('Full block 12 text', [text.value for text in ui.text])
            ui.radio(key='view').set_value('对话问答').run()
            self.assertEqual(ui.session_state['messages'][0]['content'], 'Keep this conversation')
            self.assertTrue(ui.chat_input)

    def test_empty_library_shows_guidance(self):
        self.documents = {'documents': []}
        with patch('httpx.get', side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py', default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
        self.assertFalse(ui.exception, str(ui.exception))
        self.assertTrue(any('暂无 PDF' in item.value for item in ui.info))


if __name__ == '__main__':
    unittest.main()
