import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

from app.pipeline_config import PipelineConfig
from app.retrieval import retrieve_documents, TranslationCache
from app import retrieval
from app import agent


def snapshot():
    return SimpleNamespace(chunks=('Same text', 'Same text', 'Third text'),
        metadatas=({'source':'p.pdf','page':1},{'source':'p.pdf','page':2},{'source':'p.pdf','page':3}),
        chunk_ids=('p.pdf::1','p.pdf::2','p.pdf::3'), fingerprint='frozen-index')


class RetrievalTests(unittest.TestCase):
    def test_english_multi_query_merges_translation_and_preserves_duplicate_text_sources(self):
        model=FakeListChatModel(responses=[
            '### What architecture does AffectGPT use?\n### Which dataset trains AffectGPT?\n### How is AffectGPT evaluated?'
        ])
        with patch.object(retrieval,'search_candidates',return_value=[0,1,2]) as search, patch.object(retrieval,'rerank_indices',return_value=[2,0,1]) as rerank:
            result=retrieve_documents('介绍 AffectGPT',PipelineConfig(top_k=3),llm=model,snapshot=snapshot())
        self.assertEqual(search.call_count,3)
        self.assertEqual(rerank.call_count,3)
        self.assertEqual(len(result['sources']),3)
        self.assertEqual(len({c['chunk_id'] for c in result['sources']}),3)
        self.assertTrue(all(not any('\u4e00'<=ch<='\u9fff' for ch in q) for q in result['queries']))

    def test_model_can_choose_no_decomposition_or_any_number_of_subqueries(self):
        original='What is AffectGPT?'
        for output,expected_count in [
            ('End | What is AffectGPT?',1),
            ('### Architecture of AffectGPT?\n### Training data of AffectGPT?',2),
            ('\n'.join(f'### AffectGPT subquestion {i}?' for i in range(7)),7),
        ]:
            with self.subTest(expected_count=expected_count):
                model=FakeListChatModel(responses=[output])
                with patch.object(retrieval,'search_candidates',return_value=[0]) as search:
                    result=retrieve_documents(original,PipelineConfig(rerank=False),llm=model,snapshot=snapshot())
                self.assertEqual(search.call_count,expected_count)
                self.assertEqual(len(result['queries']),expected_count)

    def test_query_cleanup_removes_duplicates_without_padding(self):
        original='What is AffectGPT?'
        model=FakeListChatModel(responses=['### Training data for AffectGPT?\n### Training data for AffectGPT?\n### Evaluation of AffectGPT?'])
        with patch.object(retrieval,'search_candidates',return_value=[0]):
            result=retrieve_documents(original,PipelineConfig(rerank=False),llm=model,snapshot=snapshot())
        self.assertEqual(result['queries'],['Training data for AffectGPT?','Evaluation of AffectGPT?'])

    def test_legacy_fixed_count_is_removed_from_saved_configuration(self):
        config=PipelineConfig.model_validate({'num_queries':2,'multi_query':True})
        self.assertNotIn('num_queries',config.model_dump())
        self.assertNotIn('num_queries',PipelineConfig.model_fields)

    def test_no_multi_query_still_translates_and_can_disable_reranking(self):
        model=FakeListChatModel(responses=['Explain AffectGPT.'])
        with patch.object(retrieval,'search_candidates',return_value=[1,0]) as search, patch.object(retrieval,'rerank_indices') as rerank:
            result=retrieve_documents('解释 AffectGPT',PipelineConfig(multi_query=False,rerank=False),llm=model,snapshot=snapshot())
        self.assertEqual(search.call_count,1)
        self.assertEqual(result['queries'],['Explain AffectGPT.'])
        rerank.assert_not_called()

    def test_translation_ablation_passes_original_query(self):
        model=Mock()
        with patch.object(retrieval,'search_candidates',return_value=[0]) as search, patch.object(retrieval,'rerank_indices',return_value=[0]):
            result=retrieve_documents('中文问题',PipelineConfig(translate_query=False,multi_query=False),llm=model,snapshot=snapshot())
        model.invoke.assert_not_called()
        self.assertEqual(result['queries'],['中文问题'])
        self.assertEqual(search.call_args.args[0],'中文问题')

    def test_non_english_expansion_is_skipped_when_translation_enabled(self):
        model=FakeListChatModel(responses=['### English question?\n### 中文扩展'])
        with patch.object(retrieval,'search_candidates',return_value=[0]) as search, patch.object(retrieval,'rerank_indices',return_value=[0]):
            result=retrieve_documents('中文问题',llm=model,snapshot=snapshot())
        self.assertEqual(result['queries'],['English question?'])
        self.assertEqual(search.call_count,1)

    def test_shared_translation_is_computed_once_under_concurrency(self):
        cache=TranslationCache()
        calls=[]
        def translate():
            calls.append(1)
            time.sleep(0.02)
            return 'Shared English question'
        with ThreadPoolExecutor(max_workers=5) as pool:
            results=list(pool.map(lambda _:cache.get('question',translate),range(5)))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(set(results)),1)

    def test_disabling_modules_skips_router_grader_and_uses_chinese_answer_instruction(self):
        model=Mock()
        model.stream.return_value=iter([AIMessageChunk(content='中文回答')])
        sources=[{'text':'Context','source':'p.pdf','page':1,'chunk_id':'p.pdf::1'}]
        with patch.object(agent,'llm',model), patch.object(agent,'retrieve_documents',return_value={
            'sources':sources,'queries':['English question'],'retrieval_query':'English question','snapshot_id':'s'}):
            result=agent.ask('English user question',config=PipelineConfig(route_query=False,grade_documents=False))
        model.invoke.assert_not_called()
        self.assertIn('Simplified Chinese',model.stream.call_args.args[0][0].content)
        self.assertEqual(result['answer'],'中文回答')

    def test_dense_search_discards_negative_and_out_of_range_faiss_ids(self):
        import numpy as np
        from vectorstore import store
        snap=snapshot()
        snap.index=Mock()
        snap.index.search.return_value=(None,np.array([[0,-1,20]]))
        with patch.object(store,'embed_query',return_value=[0.1,0.2]):
            ids=store.search_candidates('English',snap,PipelineConfig(bm25=False))
        self.assertEqual(ids,[0])

    def test_retrieval_switch_validation(self):
        with self.assertRaises(ValidationError):
            PipelineConfig(dense=False,bm25=False)
        with self.assertRaises(ValidationError):
            PipelineConfig(top_k=10,candidate_k=5)

    def test_merged_grade_documents_filters_irrelevant_and_plans_gap_for_multi_query(self):
        model=Mock()
        model.invoke.return_value=SimpleNamespace(content=(
            'IRRELEVANT_CHUNK_IDS: p.pdf::2\n'
            'DECISION: INSUFFICIENT\n'
            'GAP: missing evaluation table details in Table 11\n'
            'TARGETED_SUB_QUERIES:\n'
            '### Table 11 evaluation details'
        ))
        state=agent._initial_state('What is AffectGPT performance?')
        state['context']=[
            {'text':'Relevant overview','source':'p.pdf','page':1,'section':'1. Intro','chunk_id':'p.pdf::1','type':'text'},
            {'text':'Unrelated bibliography','source':'p.pdf','page':9,'section':'References','chunk_id':'p.pdf::2','type':'text'},
        ]
        state['query_doc_history']=[{
            'cycle':1,'cycle_name':'首轮多模态初检','input_query':'What is AffectGPT performance?',
            'sub_groups':[],'new_docs':state['context'],
        }]
        with patch.object(agent,'llm',model):
            graded=agent.grade_documents(state)
            rewritten=agent.rewrite_query(graded)
        self.assertEqual(model.invoke.call_count,1)
        self.assertEqual(graded['_grade'],'rewrite')
        self.assertEqual([c['chunk_id'] for c in graded['context']],['p.pdf::1'])
        self.assertIn('p.pdf::2',graded['rejected_chunk_ids'])
        self.assertIn('p.pdf::1',graded['accepted_chunk_ids'])
        self.assertEqual(rewritten['query'],'Table 11 evaluation details')
        self.assertEqual(rewritten['targeted_sub_queries'],['Table 11 evaluation details'])
        self.assertEqual(rewritten['rewrite_count'],1)

    def test_generate_prompt_includes_section_and_table_image_verification(self):
        model=Mock()
        model.stream.return_value=iter([AIMessageChunk(content='答案 [p.pdf, #p.pdf::5]')])
        state=agent._initial_state('Compare table results')
        state['context']=[
            {'text':'| Model | Acc |\n| A | 90 |','source':'p.pdf','page':3,'section':'4.2 Results','chunk_id':'p.pdf::5','type':'table'}
        ]
        with patch.object(agent,'llm',model), patch.object(agent,'get_stream_writer',return_value=lambda *_:None), patch.object(agent,'_extract_context_images',return_value=[{
            'data_uri':'data:image/jpeg;base64,AAAA','mime':'image/jpeg','caption':'Table 1',
            'source':'p.pdf','page':3,'section':'4.2 Results','chunk_id':'p.pdf::5','type':'table'
        }]):
            out=agent.generate(state)
        sent_parts=model.stream.call_args.args[0][0].content
        text_payload='\n'.join(p['text'] for p in sent_parts if p.get('type')=='text')
        self.assertIn('[p.pdf, #p.pdf::5]',text_payload)
        self.assertNotIn('Section: 4.2 Results',text_payload)
        self.assertIn('cross-check the parsed table text against the attached table image',text_payload)
        self.assertIn('Attached Table Image for [p.pdf, #p.pdf::5]',text_payload)
        self.assertIn('[文件名.pdf, #chunk_id]',text_payload)
        self.assertEqual(out['answer'],'答案 [p.pdf, #p.pdf::5]')


if __name__=='__main__':
    unittest.main()
