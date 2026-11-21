"""Chunk browsing uses saved index content or the shared PDF splitter."""
import pickle
import base64
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from streamlit.testing.v1 import AppTest
from PIL import Image

from app.api import app
from app import documents
from app.chunk_browser import _load_chunks
from ingestion.ingest import load_pdf_chunks
from ingestion.mineru_parser import normalize_content, MinerUError, MinerUUnavailable


def png_bytes():
    output = BytesIO()
    Image.new('RGB',(8,8),'blue').save(output,format='PNG')
    return output.getvalue()


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
        parser = patch('ingestion.ingest.parse_pdf', return_value=normalize_content([
            {'type': 'text', 'text': 'Chapter one', 'text_level': 1, 'page_idx': 0},
            {'type': 'text', 'text': 'Page one describes chunk browsing. ' * 45, 'page_idx': 0},
            {'type': 'text', 'text': 'Page two explains document filtering. ' * 45, 'page_idx': 1},
        ]))
        self.parser = parser.start()
        self.addCleanup(parser.stop)

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
        with patch.object(documents.logger, 'exception'), patch('ingestion.ingest.parse_pdf', side_effect=MinerUError('Bad PDF')):
            response = self.client.post('/documents/preview', files={'file': ('bad.pdf', b'not a PDF', 'application/pdf')})
        self.assertEqual(response.status_code, 422)

    def test_missing_mineru_reports_installation_error(self):
        self.parser.side_effect = MinerUUnavailable('Install MinerU')
        response = self.client.get('/documents/chunks', params={'filename': 'sample.pdf'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail'], 'Install MinerU')

    def test_table_metadata_survives_preview_and_saved_index(self):
        parsed = normalize_content([
            {'type': 'text', 'text': 'Results', 'text_level': 1, 'page_idx': 1},
            {'type': 'table', 'table_body': '<table>' + 'cell' * 300 + '</table>',
             'table_caption': ['Table 1'], 'table_footnote': ['Footnote'], 'page_idx': 1},
        ])
        self.parser.return_value = parsed
        texts, metas = load_pdf_chunks(self.data / 'sample.pdf')
        for stored in (False, True):
            if stored:
                self.meta.write_bytes(pickle.dumps({'chunks': texts, 'metadatas': metas}))
            result = self.client.get('/documents/chunks', params={'filename': 'sample.pdf'}).json()
            table = next(c for c in result['chunks'] if c['type'] == 'table')
            self.assertGreater(table['char_count'], 800)
            self.assertEqual(table['caption'], ['Table 1'])
            self.assertEqual(table['section'], 'Results')
            self.assertEqual(table['page'], 2)

    def test_indexed_images_are_previewed_without_becoming_chunks_or_running_ocr(self):
        root = self.base/'assets'
        root.mkdir()
        (root/'figure.png').write_bytes(png_bytes())
        parsed = normalize_content([{'type':'image','page_idx':1,'img_path':'figure.png',
            'image_caption':['Figure 1: Results'],'image_footnote':['Note'],
            'bbox':[100,100,800,500]}])
        parsed['artifact_dir'] = str(root)
        self.meta.write_bytes(pickle.dumps({'chunks':['Indexed text'],
            'metadatas':[{'source':'sample.pdf','page':1}]}))
        with patch.object(documents,'read_cached_pdf',return_value=parsed), \
                patch.object(documents,'_parse_pdf',side_effect=AssertionError('No OCR for saved index')):
            response = self.client.get('/documents/chunks',params={'filename':'sample.pdf'})
        self.assertEqual(response.status_code,200)
        data = response.json()
        self.assertEqual(data['total_chunks'],1)
        self.assertEqual(data['total_images'],1)
        self.assertFalse(data['images'][0]['indexed'])
        self.assertEqual(data['images'][0]['caption'],['Figure 1: Results'])
        encoded = data['images'][0]['asset_images'][0]
        self.assertEqual(base64.b64decode(encoded.split(',',1)[1]),png_bytes())
        self.assertNotIn(str(root),response.text)

    def test_temporary_image_preview_survives_upload_cleanup(self):
        folders = []

        def cache(path):
            root = path.parent/'assets'
            root.mkdir()
            (root/'figure.png').write_bytes(png_bytes())
            folders.append(root)
            return {'artifact_dir':str(root),'images':[{'type':'image','page':1,
                'img_path':'figure.png','caption':['Figure 1'],'footnote':[]}]}

        with patch.object(documents,'read_cached_pdf',side_effect=cache):
            response = self.client.post('/documents/preview',files={
                'file':('another.pdf',(self.data/'sample.pdf').read_bytes(),'application/pdf')})
        self.assertEqual(response.status_code,200)
        self.assertFalse(folders[0].exists())
        self.assertTrue(response.json()['images'][0]['asset_images'][0].startswith('data:image/png;base64,'))
        self.assertFalse(self.meta.exists())

    def test_original_table_parts_and_asset_path_boundaries(self):
        root = self.base/'assets'
        root.mkdir()
        for name in ['left.png','right.png']:
            (root/name).write_bytes(png_bytes())
        (self.base/'secret.png').write_bytes(b'private')
        meta = {'source':'sample.pdf','page':1,'type':'table','artifact_dir':str(root),
                'img_paths':['left.png','right.png','../secret.png']}
        result = documents._chunk_response('sample.pdf',['<table>data</table>'],[meta],indexed=True)
        self.assertEqual(len(result['chunks'][0]['asset_images']),2)
        self.assertNotIn('artifact_dir',result['chunks'][0])

    def _patch_store(self):
        from vectorstore import store as vs
        idx_dir = self.base / 'idx'
        idx_dir.mkdir(exist_ok=True)
        meta_file = idx_dir / 'metadata.pkl'
        patches = [
            patch.object(vs, 'STORE_DIR', idx_dir),
            patch.object(vs, 'FAISS_PATH', idx_dir / 'faiss.index'),
            patch.object(vs, 'FAISS_TEXT_PATH', idx_dir / 'faiss_text.index'),
            patch.object(vs, 'BM25_PATH', idx_dir / 'bm25.pkl'),
            patch.object(vs, 'META_PATH', meta_file),
            patch.object(documents, 'META_PATH', meta_file),
            patch.object(vs, 'embed_multimodal_chunks', side_effect=lambda chunks, metas, multimodal=True: [
                [float(i + 1)] * vs.EMBED_DIM for i, _ in enumerate(chunks)
            ]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return idx_dir

    def test_index_single_unindexed_pdf_and_delete_index_preserving_or_removing_file(self):
        idx_dir = self._patch_store()
        create_pdf(self.data / 'other.pdf')

        listing = {d['filename']: d for d in self.client.get('/documents').json()['documents']}
        self.assertEqual(listing['sample.pdf']['indexed_chunks'], 0)
        self.assertFalse(listing['sample.pdf']['indexed'])
        self.assertEqual(listing['other.pdf']['indexed_chunks'], 0)

        res1 = self.client.post('/documents/sample.pdf/index')
        self.assertEqual(res1.status_code, 200)
        self.assertGreater(res1.json()['chunks'], 0)

        listing = {d['filename']: d for d in self.client.get('/documents').json()['documents']}
        self.assertGreater(listing['sample.pdf']['indexed_chunks'], 0)
        self.assertTrue(listing['sample.pdf']['indexed'])
        self.assertEqual(listing['other.pdf']['indexed_chunks'], 0)

        res2 = self.client.post('/documents/other.pdf/index')
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()['total_chunks'], res1.json()['chunks'] + res2.json()['chunks'])

        del_idx = self.client.delete('/documents/sample.pdf/index')
        self.assertEqual(del_idx.status_code, 200)
        self.assertFalse(del_idx.json()['file_deleted'])
        self.assertEqual(del_idx.json()['removed_chunks'], res1.json()['chunks'])
        self.assertTrue((self.data / 'sample.pdf').is_file())

        listing = {d['filename']: d for d in self.client.get('/documents').json()['documents']}
        self.assertEqual(listing['sample.pdf']['indexed_chunks'], 0)
        self.assertEqual(listing['other.pdf']['indexed_chunks'], res2.json()['chunks'])

        del_file = self.client.delete('/documents/sample.pdf/index', params={'delete_file': 'true'})
        self.assertEqual(del_file.status_code, 200)
        self.assertTrue(del_file.json()['file_deleted'])
        self.assertFalse((self.data / 'sample.pdf').exists())

        del_last = self.client.delete('/documents/other.pdf/index')
        self.assertEqual(del_last.status_code, 200)
        self.assertEqual(del_last.json()['total_chunks'], 0)
        self.assertFalse((idx_dir / 'faiss.index').exists())
        self.assertFalse((idx_dir / 'metadata.pkl').exists())

    def test_upload_without_indexing_and_batch_index_unindexed(self):
        self._patch_store()
        up = self.client.post(
            '/documents/upload',
            params={'build_index_now': 'false'},
            files={'file': ('uploaded.pdf', (self.data / 'sample.pdf').read_bytes(), 'application/pdf')},
        )
        self.assertEqual(up.status_code, 200)
        self.assertEqual(up.json()['status'], 'uploaded')
        self.assertEqual(up.json()['chunks'], 0)
        self.assertTrue((self.data / 'uploaded.pdf').is_file())

        listing = {d['filename']: d for d in self.client.get('/documents').json()['documents']}
        self.assertEqual(listing['uploaded.pdf']['indexed_chunks'], 0)
        self.assertEqual(listing['sample.pdf']['indexed_chunks'], 0)

        batch = self.client.post('/documents/index-unindexed')
        self.assertEqual(batch.status_code, 200)
        names = [item['filename'] for item in batch.json()['indexed_files']]
        self.assertEqual(sorted(names), ['sample.pdf', 'uploaded.pdf'])

        listing = {d['filename']: d for d in self.client.get('/documents').json()['documents']}
        self.assertGreater(listing['sample.pdf']['indexed_chunks'], 0)
        self.assertGreater(listing['uploaded.pdf']['indexed_chunks'], 0)


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

    def test_cross_page_chunk_matches_every_covered_page(self):
        self.result['chunks'][9]['page_end'] = 2
        with patch('httpx.get',side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
            ui.selectbox[1].set_value(2).run()
        self.assertFalse(ui.exception,str(ui.exception))
        self.assertEqual(len(ui.expander),3)
        self.assertIn('Full block 10 text',[text.value for text in ui.text])

    def test_parent_section_and_image_only_filter(self):
        for chunk in self.result['chunks']:
            chunk['section_path'] = ['Results','Details']
            chunk['section'] = 'Results / Details'
        self.result['images'] = [{'id':1,'type':'image','page':2,'char_count':15,
            'text':'Figure 1: Results','caption':['Figure 1: Results'],'footnote':[],
            'section':'Results / Details','section_path':['Results','Details'],
            'asset_images':['data:image/png;base64,'+base64.b64encode(png_bytes()).decode()]}]
        with patch('httpx.get',side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
            ui.number_input(key='chunk_page').set_value(2).run()
            ui.selectbox(key='chunk_section').set_value('Results').run()
            ui.multiselect(key='chunk_types').set_value(['图片']).run()
            self.assertFalse(ui.exception,str(ui.exception))
            self.assertEqual(len(ui.expander),1)
            self.assertEqual(len(ui.get('imgs')),1)
            self.assertIn('Caption：Figure 1: Results',[t.value for t in ui.text])
            self.assertEqual(ui.number_input(key='chunk_page').value,1)

    def test_table_only_view_and_empty_type_selection(self):
        chunk = self.result['chunks'][3]
        chunk.update(type='table',section='Methods / Data',section_path=['Methods','Data'],
            text='Table 1: Scores\n<table><tr><td>Score</td><td>42</td></tr></table>',
            caption=['Table 1: Scores'])
        with patch('httpx.get',side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
            ui.selectbox(key='chunk_section').set_value('Methods').run()
            ui.multiselect(key='chunk_types').set_value(['表格']).run()
            self.assertFalse(ui.exception,str(ui.exception))
            self.assertEqual(len(ui.expander),1)
            self.assertEqual(len(ui.get('html')),1)
            self.assertIn('Caption：Table 1: Scores',[t.value for t in ui.text])
            ui.multiselect(key='chunk_types').set_value([]).run()
            self.assertEqual(len(ui.expander),0)
            self.assertTrue(any('没有匹配的内容' in t.value for t in ui.info))

    def test_body_chapter_click_clears_hidden_filters_and_locates_first_chunk(self):
        for chunk in self.result['chunks']:
            path = ['Introduction'] if chunk['id'] <= 10 else ['Results','Details']
            chunk.update(section_path=path,section=' / '.join(path))
        with patch('httpx.get',side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
            ui.multiselect(key='chunk_types').set_value(['正文']).run()
            ui.number_input(key='chunk_page').set_value(2).run()
            ui.selectbox(key='chunk_source_page').set_value(1).run()
            ui.text_input(key='chunk_keyword').set_value('no matching text').run()
            self.assertEqual(len(ui.expander),0)
            button = next(b for b in ui.button if b.label.startswith('Results ·'))
            button.click().run()
            self.assertFalse(ui.exception,str(ui.exception))
            self.assertEqual(ui.selectbox(key='chunk_section').value,'Results')
            self.assertEqual(ui.selectbox(key='chunk_source_page').value,'全部页码')
            self.assertEqual(ui.text_input(key='chunk_keyword').value,'')
            self.assertEqual(ui.number_input(key='chunk_page').value,1)
            self.assertEqual(len(ui.expander),2)
            self.assertTrue(ui.expander[0].label.startswith('📍 块 11'))
            self.assertIn('Full block 11 text',[t.value for t in ui.text])
            self.assertTrue(any(b.label.lstrip('　').startswith('Details ·') for b in ui.button))
            ui.selectbox[0].set_value('second.pdf').run()
            self.assertEqual(ui.selectbox(key='chunk_section').value,'全部章节')
            self.assertFalse(any(e.label.startswith('📍') for e in ui.expander))

    def test_body_outline_is_shown_whenever_body_type_is_selected(self):
        with patch('httpx.get',side_effect=self.fake_get):
            ui = AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('PDF 分块').run()
            self.assertIn('正文章节导航',[h.value for h in ui.subheader])
            ui.multiselect(key='chunk_types').set_value(['表格']).run()
            self.assertNotIn('正文章节导航',[h.value for h in ui.subheader])
            ui.multiselect(key='chunk_types').set_value(['正文','图片']).run()
            self.assertIn('正文章节导航',[h.value for h in ui.subheader])
            self.assertFalse(ui.exception,str(ui.exception))


if __name__ == '__main__':
    unittest.main()
