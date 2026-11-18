"""MinerU schema, CLI contract, cache and semantic chunking regressions."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.chunking import chunk_document
from ingestion.mineru_parser import parse_pdf, normalize_content, MinerUError, MinerUUnavailable
from ingestion.native_symbols import recover_table_symbols


def text(value, page=0, level=0):
    return {'type': 'text', 'text': value, 'page_idx': page, 'text_level': level}


class ChunkingTests(unittest.TestCase):
    def split(self, content, size=800):
        return chunk_document(normalize_content(content), 'test.pdf', size)

    def test_sections_paragraphs_and_cross_page_provenance(self):
        chunks, metas = self.split([text('A', level=1), text('短段', page=0),
            text('后续段落', page=1), text('B', page=1, level=1), text('结束', page=2)], 20)
        self.assertEqual(chunks, ['A\n\n短段\n\n后续段落', 'B\n\n结束'])
        self.assertEqual([(m['page'], m['page_end']) for m in metas], [(1, 2), (2, 3)])
        self.assertEqual([m['section'] for m in metas], ['A', 'B'])

    def test_whole_paragraphs_precede_sentence_splitting(self):
        chunks, _ = self.split([text('aaaa.\n\nbbbb.\n\ncccc.')], 12)
        self.assertEqual(chunks, ['aaaa.\n\nbbbb.', 'cccc.'])
        chunks, _ = self.split([text('甲乙丙丁。戊己庚辛。壬癸子丑。')], 12)
        self.assertEqual(chunks, ['甲乙丙丁。戊己庚辛。', '壬癸子丑。'])

    def test_long_sentences_and_english_whitespace_are_not_lost(self):
        for value in ['长' * 1701, 'First sentence. Second sentence! Third sentence? Last.', 'x' * 800]:
            chunks, _ = self.split([text(value)], 19)
            self.assertEqual(''.join(chunks), value)
            self.assertTrue(all(0 < len(c) <= 19 for c in chunks))
        with self.assertRaises(ValueError):
            self.split([text('x')], 0)

    def test_table_and_image_are_atomic_chunks(self):
        table = '<table><tr><td>' + '数据' * 700 + '</td></tr></table>'
        content = [text('Results', level=1), text('正文'),
            {'type': 'image', 'img_path': 'images/figure.jpg', 'image_caption': ['图一'],
             'image_footnote': ['图注脚注'], 'page_idx': 0},
            {'type': 'table', 'table_body': table, 'table_caption': ['表一'],
             'table_footnote': ['注：单位为元'], 'page_idx': 1}, text('结尾', page=1)]
        parsed = normalize_content(content)
        chunks, metas = chunk_document(parsed, 'test.pdf')
        tables = [(c, m) for c, m in zip(chunks, metas) if m['type'] == 'table']
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0][0], '表一\n\n' + table + '\n\n注：单位为元')
        self.assertGreater(len(tables[0][0]), 800)
        images = [(c, m) for c, m in zip(chunks, metas) if m['type'] == 'image']
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][0], '图一\n\n图注脚注')
        self.assertEqual(images[0][1]['img_path'], 'images/figure.jpg')
        self.assertEqual(parsed['images'][0]['caption'], ['图一'])
        self.assertEqual(parsed['tables'][0]['footnote'], ['注：单位为元'])
        self.assertEqual(len(parsed['body']), 3)

    def test_nested_headings_and_empty_text(self):
        chunks, metas = self.split([text('Chapter', level=1), text('Sub', level=2),
            text('内容'), text('Next', level=2), text(''), text('End', level=1)])
        self.assertEqual([m['section'] for m in metas], ['Chapter', 'Chapter / Sub', 'Chapter / Next', 'End'])
        self.assertEqual(metas[1]['section_path'], ['Chapter', 'Sub'])
        self.assertTrue(all(chunks))
        self.assertEqual(self.split([]), ([], []))

    def test_flat_mineru_titles_recover_numbered_hierarchy(self):
        chunks, metas = self.split([text('4 Experiments', level=1),
            text('4.1 Setup', level=1), text('Details'),
            text('4.1.1 Models', level=1), text('4.2 Results', level=1),
            text('5 Conclusion', level=1)])
        self.assertEqual(metas[1]['section_path'], ['4 Experiments', '4.1 Setup'])
        self.assertEqual(metas[2]['section_path'], ['4 Experiments', '4.1 Setup', '4.1.1 Models'])
        self.assertEqual(metas[3]['section_path'], ['4 Experiments', '4.2 Results'])
        self.assertEqual(metas[4]['section_path'], ['5 Conclusion'])

    def test_detached_and_swapped_captions_follow_page_geometry(self):
        parsed = normalize_content([
            {'type':'text','text':'Table 4: Filtering.','page_idx':0,'bbox':[500,80,890,125]},
            {'type':'table','table_body':'<table>A</table>','page_idx':0,
             'bbox':[500,140,890,215],'table_caption':['Table 5: Fusion.']},
            {'type':'table','table_body':'<table>B</table>','page_idx':0,
             'bbox':[500,257,880,315],'table_caption':[]},
        ], [{'text':'Table 5: Fusion.','page':1,'bbox':[500,225,890,245]}])
        self.assertEqual(parsed['tables'][0]['caption'], ['Table 4: Filtering.'])
        self.assertEqual(parsed['tables'][1]['caption'], ['Table 5: Fusion.'])
        self.assertEqual(parsed['body'], [])
        chunks, _ = chunk_document(parsed, 'test.pdf')
        self.assertEqual(len(chunks), 2)
        self.assertIn('Table 4: Filtering.', chunks[0])
        self.assertNotIn('Table 5: Fusion.', chunks[0])

    def test_caption_misclassified_as_heading_does_not_change_section(self):
        parsed = normalize_content([text('Methods',level=1),
            {'type':'text','text':'Table 1: Results.','text_level':1,'page_idx':0,'bbox':[100,100,900,120]},
            {'type':'table','table_body':'<table>data</table>','page_idx':0,'bbox':[100,130,900,200]},
            text('Following paragraph')])
        self.assertEqual(parsed['body'][-1]['section'], 'Methods')
        self.assertEqual(parsed['tables'][0]['caption'], ['Table 1: Results.'])

    def test_side_by_side_parts_with_one_caption_form_one_table_chunk(self):
        parsed = normalize_content([
            {'type':'table','table_body':'<table>left</table>','page_idx':0,
             'bbox':[100,150,450,250],'img_path':'left.jpg','table_caption':['Table 6: Ablations.']},
            {'type':'table','table_body':'<table>right</table>','page_idx':0,
             'bbox':[520,150,880,250],'img_path':'right.jpg','table_caption':[]},
        ],[{'text':'Table 6: Ablations.','page':1,'bbox':[100,80,880,130]}])
        self.assertEqual(len(parsed['tables']),1)
        self.assertEqual(parsed['tables'][0]['img_paths'],['left.jpg','right.jpg'])
        chunks, metas = chunk_document(parsed,'test.pdf')
        self.assertEqual(len(chunks),1)
        self.assertIn('<table>left</table>',chunks[0])
        self.assertIn('<table>right</table>',chunks[0])

    def test_native_symbols_are_recovered_only_inside_table_region(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, NumberObject, ArrayObject
        with tempfile.TemporaryDirectory() as folder:
            pdf = Path(folder)/'symbols.pdf'
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            fonts = DictionaryObject()
            for name, base in [('F1','Helvetica'),('F2','bbding')]:
                font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),
                    NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/'+base)})
                if base == 'bbding':
                    descriptor = DictionaryObject({NameObject('/Type'):NameObject('/FontDescriptor'),
                        NameObject('/FontName'):NameObject('/bbding'), NameObject('/Flags'):NumberObject(4),
                        NameObject('/FontBBox'):ArrayObject([NumberObject(n) for n in [0,-200,1000,1000]]),
                        NameObject('/Ascent'):NumberObject(800),NameObject('/Descent'):NumberObject(-200)})
                    font[NameObject('/FontDescriptor')] = writer._add_object(descriptor)
                    font[NameObject('/FirstChar')] = NumberObject(33)
                    font[NameObject('/LastChar')] = NumberObject(37)
                    font[NameObject('/Widths')] = ArrayObject([NumberObject(600)]*5)
                fonts[NameObject('/'+name)] = writer._add_object(font)
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):fonts})
            stream = DecodedStreamObject()
            stream.set_data(b'BT /F1 12 Tf 100 700 Td (Positive) Tj ET '
                b'BT /F2 12 Tf 250 700 Td (!) Tj ET '
                b'BT /F1 12 Tf 300 700 Td (42) Tj ET '
                b'BT /F2 12 Tf 250 600 Td (%) Tj ET')
            page[NameObject('/Contents')] = writer._add_object(stream)
            with pdf.open('wb') as output: writer.write(output)
            parsed = normalize_content([{'type':'table','table_body':'<table>Positive</table>',
                'page_idx':0,'bbox':[0,90,900,130]}])
            recover_table_symbols(pdf, parsed['tables'])
            table = parsed['tables'][0]
            self.assertEqual([s['symbol'] for s in table['recovered_symbols']], ['✓'])
            self.assertIn('Positive',table['native_symbol_rows'][0])
            self.assertNotIn('42',table['native_symbol_rows'][0])
            chunks, _ = chunk_document(parsed,'symbols.pdf')
            self.assertEqual(len(chunks),1)
            self.assertIn('✓',chunks[0])


class MinerUAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pdf = Path(self.temp.name) / 'sample.pdf'
        self.pdf.write_bytes(b'%PDF fixture')
        env = patch.dict('os.environ', {'MINERU_EXECUTABLE': 'mineru-test', 'MINERU_DEVICE': 'cpu'})
        env.start()
        self.addCleanup(env.stop)

    def fake_cli(self, command, **kwargs):
        self.assertEqual(command[command.index('-b') + 1], 'pipeline')
        self.assertEqual(command[command.index('-m') + 1], 'auto')
        self.assertEqual(command[command.index('-t') + 1], 'true')
        self.assertFalse(kwargs.get('shell', False))
        folder = Path(command[command.index('-o') + 1]) / 'sample' / 'auto'
        (folder / 'images').mkdir(parents=True)
        (folder / 'images' / 'figure.jpg').write_bytes(b'asset')
        (folder / 'sample_content_list.json').write_text(json.dumps([
            text('正文', level=1),
            {'type': 'image', 'img_path': 'images/figure.jpg', 'image_caption': ['caption']},
            {'type': 'table', 'table_body': '<table>cell</table>', 'table_caption': ['Table 1']},
        ]), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def test_artifacts_and_assets_survive_and_cache_invalidates(self):
        with patch('ingestion.mineru_parser.subprocess.run', side_effect=self.fake_cli) as run:
            result = parse_pdf(self.pdf)
            artifact = Path(result['artifact_dir'])
            for name in ['body', 'tables', 'images', 'parsed']:
                self.assertTrue((artifact / (name + '.json')).is_file())
            image = json.loads((artifact / 'images.json').read_text(encoding='utf-8'))[0]
            self.assertEqual((artifact / image['img_path']).read_bytes(), b'asset')
            self.assertEqual(result, parse_pdf(self.pdf))
            self.assertEqual(run.call_count, 1)
            self.pdf.write_bytes(b'%PDF changed')
            self.assertNotEqual(parse_pdf(self.pdf)['artifact_dir'], str(artifact))
            self.assertEqual(run.call_count, 2)

    def test_failures_do_not_create_a_success_cache(self):
        outcomes = [SimpleNamespace(returncode=1, stderr='model unavailable', stdout=''),
                    SimpleNamespace(returncode=0, stderr='', stdout=''),
                    subprocess.TimeoutExpired('mineru', 1), FileNotFoundError('missing')]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                kwargs = {'side_effect': outcome} if isinstance(outcome, Exception) else {'return_value': outcome}
                with patch('ingestion.mineru_parser.subprocess.run', **kwargs), self.assertRaises(MinerUError):
                    parse_pdf(self.pdf)
                self.assertEqual(list(self.pdf.parent.rglob('parsed.json')), [])

    def test_normalization_upgrade_reuses_raw_output_without_ocr(self):
        with patch('ingestion.mineru_parser.subprocess.run', side_effect=self.fake_cli) as run:
            parsed = parse_pdf(self.pdf)
            cache = Path(parsed['artifact_dir']) / 'parsed.json'
            stale = json.loads(cache.read_text(encoding='utf-8'))
            stale['normalization_version'] = 0
            stale['body'][0]['section'] = 'stale hierarchy'
            cache.write_text(json.dumps(stale), encoding='utf-8')
            updated = parse_pdf(self.pdf)
            self.assertEqual(updated['body'][0]['section'], '正文')
            self.assertEqual(run.call_count, 1)

    def test_missing_executable_is_actionable(self):
        with patch.dict('os.environ', {'MINERU_EXECUTABLE': ''}), \
                patch('ingestion.mineru_parser.shutil.which', return_value=None), \
                patch('ingestion.mineru_parser.Path.is_file', return_value=False):
            with self.assertRaisesRegex(MinerUUnavailable, 'MINERU_EXECUTABLE'):
                parse_pdf(self.pdf)


if __name__ == '__main__':
    unittest.main()
