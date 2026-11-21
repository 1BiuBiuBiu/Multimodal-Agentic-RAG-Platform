import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from app.pipeline_config import PipelineConfig, default_variants
from evaluation import experiments as runner


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        ctx=patch.object(runner,'RUN_DIR',Path(self.temp.name))
        ctx.start(); self.addCleanup(ctx.stop)
        runner.JOBS.clear()
        runner.CANCEL_EVENTS.clear()

    def test_labelled_metrics_are_rank_sensitive_and_unlabelled_is_missing(self):
        scores=runner.label_metrics(['a','b','c'],['b','d'],3)
        self.assertAlmostEqual(scores['precision_at_k'],1/3)
        self.assertEqual(scores['recall_at_k'],0.5)
        self.assertEqual(scores['mrr'],0.5)
        self.assertEqual(runner.label_metrics([],[],5),{})
        self.assertEqual(runner.label_metrics([],['b'],5)['recall_at_k'],0)

    def test_batch_rejects_incomparable_budgets_and_unlabelled_exact_evaluation(self):
        dataset=[{'question':'Q','ground_truth':'A'}]
        with self.assertRaises(ValidationError):
            runner.BatchRequest(variants=[{'name':'a','config':{'top_k':3}},{'name':'b','config':{'top_k':5}}],dataset=dataset)
        with self.assertRaises(ValidationError):
            runner.BatchRequest(variants=[{'name':'a'}],dataset=dataset,use_ragas=False)

    def test_five_variants_really_overlap_share_snapshot_and_preserve_per_question_results(self):
        barrier=threading.Barrier(5)
        mutex=threading.Lock()
        active=0
        peak=0
        snapshots=[]
        def retrieve(question,options,**kwargs):
            nonlocal active,peak
            with mutex:
                active+=1; peak=max(peak,active); snapshots.append(id(kwargs['snapshot']))
            try:
                if question=='Q1': barrier.wait(timeout=10)
                return {'sources':[{'text':'T','source':'p.pdf','page':1,'chunk_id':'p.pdf::1'}],
                    'queries':['English query'],'retrieval_query':'English query','snapshot_id':'same'}
            finally:
                with mutex: active-=1
        request=runner.BatchRequest(variants=default_variants(),dataset=[
            {'question':'Q1','ground_truth':'A','relevant_chunk_ids':['p.pdf::1']},
            {'question':'Q2','ground_truth':'B','relevant_chunk_ids':['p.pdf::1']}],concurrency=5,use_ragas=False)
        snap=SimpleNamespace(fingerprint='same',chunks=('T',),chunk_ids=('p.pdf::1',))
        with patch.object(runner,'load_snapshot',return_value=snap), patch.object(runner,'retrieve_documents',side_effect=retrieve), patch.object(runner,'create_llm',return_value=Mock()), patch('embeddings.embedder.get_embedder'), patch('embeddings.embedder.get_reranker'):
            job=runner.start_batch(request)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                done=runner.get_job(job['id'])
                if done['status'] not in ('queued','running'):break
                time.sleep(0.02)
            self.assertEqual(done['status'],'completed',done)
        self.assertEqual(peak,5)
        self.assertEqual(len(set(snapshots)),1)
        for result in done['results'].values():
            self.assertEqual([q['question'] for q in result['samples']],['Q1','Q2'])
            self.assertEqual(result['metrics']['recall_at_k'],1)
            self.assertEqual(result['valid_counts']['recall_at_k'],2)
        saved=json.loads((Path(self.temp.name)/(job['id']+'.json')).read_text(encoding='utf-8'))
        self.assertEqual(saved['dataset_hash'],done['dataset_hash'])
        self.assertEqual(saved['request']['concurrency'],5)

    def test_incomplete_judge_metric_is_not_silently_averaged(self):
        request=runner.BatchRequest(variants=[{'name':'one'}],dataset=[{'question':'Q','ground_truth':'A'},{'question':'Q2','ground_truth':'B'}])
        job_id='a'*32
        runner.JOBS[job_id]={'id':job_id,'results':{'one':{}}}
        result={'sources':[{'text':'T','source':'p','page':1,'chunk_id':'p::1'}], 'queries':['Q'],'retrieval_query':'Q','snapshot_id':'s'}
        with patch.object(runner,'create_llm',return_value=Mock()),patch.object(runner,'retrieve_documents',return_value=result),patch.object(runner,'_judge',return_value=[{'context_recall':1.0},{'context_recall':None}]):
            runner._run_variant(job_id,request.variants[0],request,Mock(),runner.TranslationCache())
        values=runner.JOBS[job_id]['results']['one']
        self.assertIsNone(values['metrics']['context_recall'])
        self.assertEqual(values['valid_counts']['context_recall'],1)

    def test_failed_variant_has_no_misleading_aggregate(self):
        request=runner.BatchRequest(variants=[{'name':'broken'}],dataset=[{'question':'Q','ground_truth':'A','relevant_chunk_ids':['p::1']}],use_ragas=False)
        job_id='d'*32
        runner.JOBS[job_id]={'id':job_id,'results':{'broken':{}}}
        with patch.object(runner,'create_llm',return_value=Mock()),patch.object(runner,'retrieve_documents',side_effect=ValueError('Simulated failure')),patch('loguru.logger.exception'):
            runner._run_variant(job_id,request.variants[0],request,Mock(),runner.TranslationCache())
        result=runner.JOBS[job_id]['results']['broken']
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['metrics'],{})

    def test_judging_has_fresh_metric_objects_and_scores_web_evidence_when_used(self):
        import pandas as pd
        captures=[]
        def evaluate(dataset,**kwargs):
            captures.append((dataset,kwargs['metrics'],kwargs.get('run_config')))
            return SimpleNamespace(to_pandas=lambda:pd.DataFrame([{
                metric.name:0.5 for metric in kwargs['metrics']}]))
        sample={'question':'Q','ground_truth':'A','sources':[],'web_contexts':['Web evidence'],'answer':'Chinese answer'}
        with patch('ragas.evaluate',side_effect=evaluate),patch.object(runner,'create_llm',return_value=Mock()):
            runner._judge([sample],'full',max_workers=4)
            runner._judge([sample],'full')
        self.assertEqual(captures[0][0][0]['contexts'],['Web evidence'])
        self.assertEqual(captures[0][2].max_workers,4)
        self.assertEqual(captures[1][2].max_workers,1)
        self.assertTrue(all(a is not b for a,b in zip(captures[0][1],captures[1][1])))

    def test_old_running_job_becomes_interrupted_on_restart(self):
        job_id='b'*32
        path=Path(self.temp.name)/(job_id+'.json')
        path.write_text(json.dumps({'id':job_id,'status':'running'}),encoding='utf-8')
        self.assertEqual(runner.get_job(job_id)['status'],'interrupted')

    def test_simple_and_complex_questions_record_separate_ragas_and_efficiency_metrics(self):
        from evaluation.eval_data import EVAL_DATASET, EVAL_DATASET_SIMPLE, EVAL_DATASET_COMPLEX
        self.assertEqual(len(EVAL_DATASET), len(EVAL_DATASET_SIMPLE) + len(EVAL_DATASET_COMPLEX))
        self.assertTrue(all(item.get('question_type') == 'simple' for item in EVAL_DATASET_SIMPLE))
        self.assertTrue(all(item.get('question_type') == 'complex' for item in EVAL_DATASET_COMPLEX))

        dataset = [
            {'question': 'Simple Q', 'ground_truth': 'Simple A', 'question_type': 'simple'},
            {'question': 'Complex Q', 'ground_truth': 'Complex A', 'question_type': 'complex'},
        ]
        request = runner.BatchRequest(variants=[{'name': 'v1'}], dataset=dataset, scope='full', concurrency=4, use_ragas=True)
        job_id = 'f' * 32
        runner.JOBS[job_id] = {'id': job_id, 'results': {'v1': {}}}

        def fake_ask(q, **kwargs):
            if q == 'Simple Q':
                return {
                    'answer': 'Ans S',
                    'sources': [{'text': 'Ctx S', 'source': 'a.pdf', 'page': 1, 'chunk_id': 'a.pdf::1'}],
                    'retrieval_trace': [{'queries': ['Simple Q']}],
                    'rewrite_count': 0,
                    'used_web': False,
                    'total_tokens': 200,
                }
            return {
                'answer': 'Ans C',
                'sources': [{'text': 'Ctx C', 'source': 'a.pdf', 'page': 2, 'chunk_id': 'a.pdf::2'}],
                'retrieval_trace': [{'queries': ['C1']}, {'queries': ['C2']}, {'queries': ['C3']}],
                'rewrite_count': 2,
                'used_web': False,
                'total_tokens': 1000,
            }

        judged_scores = [
            {
                'llm_context_precision_with_reference': 0.9,
                'llm_context_recall': 1.0,
                'faithfulness': 0.95,
                'response_relevancy': 0.88,
            },
            {
                'llm_context_precision_with_reference': 0.7,
                'llm_context_recall': 0.8,
                'faithfulness': 0.85,
                'response_relevancy': 0.76,
            },
        ]
        with patch.object(runner, 'create_llm', return_value=Mock()), \
                patch('app.agent.ask', side_effect=fake_ask), \
                patch.object(runner, '_judge', return_value=judged_scores) as judge_mock:
            runner._run_variant(job_id, request.variants[0], request, Mock(), runner.TranslationCache())

        self.assertEqual(judge_mock.call_args.kwargs['max_workers'], 4)

        res = runner.JOBS[job_id]['results']['v1']
        self.assertEqual(res['status'], 'completed')
        by_type = res['metrics_by_type']
        self.assertEqual(by_type['simple']['question_count'], 1)
        self.assertEqual(by_type['simple']['avg_retrieval_count'], 1.0)
        self.assertEqual(by_type['simple']['avg_tokens'], 200.0)
        self.assertEqual(by_type['simple']['faithfulness'], 0.95)
        self.assertEqual(by_type['simple']['response_relevancy'], 0.88)
        self.assertEqual(by_type['simple']['llm_context_precision_with_reference'], 0.9)
        self.assertEqual(by_type['simple']['llm_context_recall'], 1.0)

        self.assertEqual(by_type['complex']['question_count'], 1)
        self.assertEqual(by_type['complex']['avg_retrieval_count'], 3.0)
        self.assertEqual(by_type['complex']['avg_tokens'], 1000.0)
        self.assertEqual(by_type['complex']['faithfulness'], 0.85)
        self.assertEqual(by_type['complex']['response_relevancy'], 0.76)
        self.assertEqual(by_type['complex']['llm_context_precision_with_reference'], 0.7)
        self.assertEqual(by_type['complex']['llm_context_recall'], 0.8)

        self.assertEqual(by_type['overall']['question_count'], 2)
        self.assertEqual(by_type['overall']['avg_retrieval_count'], 2.0)
        self.assertEqual(by_type['overall']['avg_tokens'], 600.0)
        self.assertEqual(by_type['overall']['faithfulness'], 0.9)
        self.assertGreaterEqual(by_type['overall']['latency_ms'], 0.0)

    def test_cancel_running_job_and_restore_last_request_preferences(self):
        started_event = threading.Event()
        release_event = threading.Event()

        def slow_retrieve(question, **kwargs):
            started_event.set()
            release_event.wait(timeout=5)
            return {
                'answer': 'A',
                'sources': [{'text': 'T', 'source': 'p.pdf', 'page': 1, 'chunk_id': 'p.pdf::1'}],
                'retrieval_trace': [{'queries': [question]}],
                'rewrite_count': 0,
                'used_web': False,
                'total_tokens': 10,
            }

        dataset = [
            {'question': 'S1', 'ground_truth': 'A1', 'question_type': 'simple', 'relevant_chunk_ids': ['p.pdf::1']},
            {'question': 'C1', 'ground_truth': 'A2', 'question_type': 'complex', 'relevant_chunk_ids': ['p.pdf::1']},
        ]
        request = runner.BatchRequest(
            variants=[{'name': 'v_custom'}],
            dataset=dataset,
            scope='full',
            concurrency=3,
            use_ragas=False,
        )
        snap = SimpleNamespace(fingerprint='snap', chunks=('T',), chunk_ids=('p.pdf::1',))
        with patch.object(runner, 'load_snapshot', return_value=snap), \
                patch('app.agent.ask', side_effect=slow_retrieve), \
                patch.object(runner, 'create_llm', return_value=Mock()), \
                patch('embeddings.embedder.get_embedder'), \
                patch('embeddings.embedder.get_reranker'), \
                patch('embeddings.visual_embedder.get_visual_embedder'):
            job = runner.start_batch(request)
            self.assertTrue(started_event.wait(timeout=5))
            cancelled = runner.cancel_job(job['id'])
            release_event.set()
            self.assertEqual(cancelled['status'], 'cancelled')
            self.assertEqual(cancelled['results']['v_custom']['status'], 'cancelled')
            self.assertEqual(cancelled['results']['v_custom']['phase'], '已手动停止')

            prefs = runner.get_last_request_prefs()
            self.assertIsNotNone(prefs)
            self.assertEqual(prefs['variant_names'], ['v_custom'])
            self.assertEqual(prefs['scope'], 'full')
            self.assertEqual(prefs['simple_count'], 1)
            self.assertEqual(prefs['complex_count'], 1)
            self.assertEqual(prefs['concurrency'], 3)

            # Should allow starting a new batch immediately after cancellation
            release_event.clear()
            job2 = runner.start_batch(request)
            release_event.set()
            runner.cancel_job(job2['id'])

    def test_llm_only_questions_skip_retrieval_metrics_and_score_answer_correctness(self):
        import pandas as pd
        captures = []

        def fake_evaluate(dataset, **kwargs):
            metric_names = [m.name for m in kwargs['metrics']]
            captures.append((len(dataset), metric_names))
            rows = [{name: (0.9 if name in ('answer_correctness', 'answer_relevancy') else 0.8) for name in metric_names}
                    for _ in range(len(dataset))]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

        samples_for_judge = [
            {'question': 'Simple LLM only', 'ground_truth': 'GT1', 'route': 'llm_only',
             'sources': [], 'trace': [], 'web_contexts': [], 'answer': 'A1'},
            {'question': 'Complex docs', 'ground_truth': 'GT2', 'route': 'documents',
             'sources': [{'text': 'Chunk text'}], 'trace': [{'queries': ['q']}], 'web_contexts': [], 'answer': 'A2'},
        ]
        with patch('ragas.evaluate', side_effect=fake_evaluate), patch.object(runner, 'create_llm', return_value=Mock()):
            judged = runner._judge(samples_for_judge, 'full', max_workers=2)

        self.assertEqual(len(captures), 2)
        # Subset 0 in _judge is retrieved_indices (1 sample, all 5 metrics); subset 1 is llm_only_indices (1 sample, 2 metrics)
        self.assertEqual(captures[0][0], 1)
        self.assertIn('faithfulness', captures[0][1])
        self.assertIn('answer_correctness', captures[0][1])
        self.assertEqual(captures[1][0], 1)
        self.assertEqual(set(captures[1][1]), {'answer_correctness', 'answer_relevancy'})

        self.assertEqual(set(judged[0].keys()), {'answer_correctness', 'answer_relevancy'})
        self.assertIn('faithfulness', judged[1])
        self.assertIn('llm_context_precision_with_reference', judged[1])
        self.assertIn('context_recall', judged[1])

        # Also verify _run_variant sets retrieval_count=0, skips label_metrics for llm_only,
        # and does not let llm_only samples drag down retrieval metric averages.
        dataset = [
            {'question': 'Q_llm', 'ground_truth': 'GT1', 'question_type': 'simple', 'relevant_chunk_ids': ['p.pdf::1']},
            {'question': 'Q_doc', 'ground_truth': 'GT2', 'question_type': 'simple', 'relevant_chunk_ids': ['p.pdf::1']},
        ]
        request = runner.BatchRequest(variants=[{'name': 'v1'}], dataset=dataset, scope='full', concurrency=1, use_ragas=True)
        job_id = 'e' * 32
        runner.JOBS[job_id] = {'id': job_id, 'results': {'v1': {}}}

        def fake_ask(q, **kwargs):
            if q == 'Q_llm':
                return {
                    'answer': 'Ans LLM',
                    'route': 'llm_only',
                    'sources': [],
                    'retrieval_trace': [],
                    'rewrite_count': 0,
                    'used_web': False,
                    'total_tokens': 50,
                }
            return {
                'answer': 'Ans Doc',
                'route': 'documents',
                'sources': [{'text': 'Chunk', 'source': 'p.pdf', 'page': 1, 'chunk_id': 'p.pdf::1'}],
                'retrieval_trace': [{'queries': ['Q_doc']}],
                'rewrite_count': 0,
                'used_web': False,
                'total_tokens': 150,
            }

        with patch.object(runner, 'create_llm', return_value=Mock()), \
                patch('app.agent.ask', side_effect=fake_ask), \
                patch('ragas.evaluate', side_effect=fake_evaluate):
            runner._run_variant(job_id, request.variants[0], request, Mock(), runner.TranslationCache())

        res = runner.JOBS[job_id]['results']['v1']
        self.assertEqual(res['status'], 'completed')
        self.assertEqual(res['samples'][0]['retrieval_count'], 0)
        self.assertEqual(res['samples'][0]['route'], 'llm_only')
        self.assertNotIn('recall_at_k', res['samples'][0]['metrics'])
        self.assertNotIn('faithfulness', res['samples'][0]['metrics'])
        self.assertEqual(res['samples'][1]['retrieval_count'], 1)
        self.assertEqual(res['metrics']['faithfulness'], 0.8)
        self.assertEqual(res['metrics']['recall_at_k'], 1.0)
        self.assertEqual(res['valid_counts']['faithfulness'], 1)
        self.assertEqual(res['valid_counts']['answer_correctness'], 2)
        self.assertEqual(res['metrics']['avg_retrieval_count'], 0.5)

    def test_generate_first_then_batch_evaluate_ragas_with_high_concurrency(self):
        import pandas as pd
        # 1. Start two jobs with use_ragas=False, defer_ragas=True (no relevant_chunk_ids required).
        req1 = runner.BatchRequest(
            variants=[{'name': 'v1'}],
            dataset=[{'question': 'Q1', 'ground_truth': 'GT1', 'question_type': 'simple'}],
            scope='full', concurrency=2, ragas_concurrency=24, use_ragas=False, defer_ragas=True,
        )
        req2 = runner.BatchRequest(
            variants=[{'name': 'v1'}],
            dataset=[{'question': 'Q2', 'ground_truth': 'GT2', 'question_type': 'complex'}],
            scope='full', concurrency=2, ragas_concurrency=24, use_ragas=False, defer_ragas=True,
        )
        snap = SimpleNamespace(fingerprint='snap', chunks=('T',), chunk_ids=('p.pdf::1',))

        def fake_ask(q, **kwargs):
            return {
                'answer': f'Ans for {q}',
                'route': 'documents',
                'sources': [{'text': f'Ctx for {q}', 'source': 'p.pdf', 'page': 1, 'chunk_id': 'p.pdf::1'}],
                'retrieval_trace': [{'queries': [q]}],
                'rewrite_count': 0,
                'used_web': False,
                'total_tokens': 120,
            }

        with patch.object(runner, 'load_snapshot', return_value=snap), \
                patch('app.agent.ask', side_effect=fake_ask), \
                patch.object(runner, 'create_llm', return_value=Mock()), \
                patch('embeddings.embedder.get_embedder'), \
                patch('embeddings.embedder.get_reranker'), \
                patch('embeddings.visual_embedder.get_visual_embedder'):
            job1 = runner.start_batch(req1)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                done1 = runner.get_job(job1['id'])
                if done1['status'] not in ('queued', 'running'):
                    break
                time.sleep(0.02)
            self.assertEqual(done1['status'], 'completed')

            job2 = runner.start_batch(req2)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                done2 = runner.get_job(job2['id'])
                if done2['status'] not in ('queued', 'running'):
                    break
                time.sleep(0.02)
            self.assertEqual(done2['status'], 'completed')

        self.assertFalse(runner._is_job_ragas_scored(done1))
        self.assertEqual(done1['results']['v1']['phase'], '已生成结果(待RAGAs评分)')
        self.assertFalse(done1['results']['v1']['ragas_scored'])
        self.assertFalse(runner._is_job_ragas_scored(done2))

        # 2. Batch evaluate both jobs together with ragas_concurrency=24.
        judge_calls = []

        def fake_evaluate(dataset, **kwargs):
            metric_names = [m.name for m in kwargs['metrics']]
            judge_calls.append((len(dataset), kwargs['run_config'].max_workers, metric_names))
            rows = [{name: 0.91 for name in metric_names} for _ in range(len(dataset))]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

        with patch('ragas.evaluate', side_effect=fake_evaluate), patch.object(runner, 'create_llm', return_value=Mock()):
            runner.start_ragas_evaluation([job1['id'], job2['id']], ragas_concurrency=24)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                j1 = runner.get_job(job1['id'])
                j2 = runner.get_job(job2['id'])
                if j1['status'] not in ('queued', 'running') and j2['status'] not in ('queued', 'running'):
                    break
                time.sleep(0.02)

        self.assertEqual(j1['status'], 'completed')
        self.assertEqual(j2['status'], 'completed')
        # Pooled into a single ragas.evaluate call with 2 samples and max_workers=24!
        self.assertEqual(len(judge_calls), 1)
        self.assertEqual(judge_calls[0][0], 2)
        self.assertEqual(judge_calls[0][1], 24)
        self.assertTrue(runner._is_job_ragas_scored(j1))
        self.assertTrue(runner._is_job_ragas_scored(j2))
        self.assertEqual(j1['results']['v1']['metrics']['faithfulness'], 0.91)
        self.assertEqual(j1['results']['v1']['metrics']['answer_correctness'], 0.91)
        self.assertEqual(j2['results']['v1']['metrics_by_type']['complex']['answer_correctness'], 0.91)
        self.assertEqual(j1['results']['v1']['phase'], '完成')

    def test_multi_round_subquery_context_precision_timeout_600_and_strict_api_error_reporting(self):
        import pandas as pd
        from app import agent

        # Verify _deduplicate_mention_chunks keeps text chunks cited by tables/images.
        chunks_with_mention = [
            {'chunk_id': 'p.pdf::10', 'type': 'text', 'text': 'Full paragraph findings.'},
            {'chunk_id': 'p.pdf::11', 'type': 'table', 'text': '| A | B |', 'mention_chunk_id': 'p.pdf::10', 'mention_context': 'Table 1 summarizes.'},
            {'chunk_id': 'p.pdf::10', 'type': 'text', 'text': 'Full paragraph findings.'},
        ]
        deduped, removed = agent._deduplicate_mention_chunks(chunks_with_mention)
        self.assertEqual([c['chunk_id'] for c in deduped], ['p.pdf::10', 'p.pdf::11'])
        self.assertEqual(removed, [])

        # Verify _format_chunk_for_judge includes mention_context and mentioned_visuals.
        formatted = runner._format_chunk_for_judge({
            'text': 'Body sentence mentioning Figure 1.',
            'mention_context': 'Figure 1 caption detail.',
            'mentioned_visuals': [{'caption': 'Figure 1: Overview', 'text': 'Visual body.'}],
        })
        self.assertIn('Body sentence mentioning Figure 1.', formatted)
        self.assertIn('Figure 1 caption detail.', formatted)
        self.assertIn('Figure 1: Overview', formatted)

        # Verify 2 rounds x 2 sub-queries (4 groups) averages Context Precision across all 4 groups,
        # and uses timeout=600, max_retries=1, raise_exceptions=True.
        multi_sample = {
            'question': 'Complex multi-round Q',
            'ground_truth': 'GT',
            'route': 'documents',
            'sources': [{'chunk_id': 'p.pdf::1', 'text': 'R1Q1'}],
            'trace': [
                {'sub_groups': [
                    {'sub_query': 'q1', 'docs': [{'text': 'R1Q1-a'}, {'text': 'R1Q1-b'}]},
                    {'sub_query': 'q2', 'docs': [{'text': 'R1Q2-a'}]},
                ]},
                {'sub_groups': [
                    {'sub_query': 'q3', 'docs': [{'text': 'R2Q3-a'}]},
                    {'sub_query': 'q4', 'docs': [{'text': 'R2Q4-a'}]},
                ]},
            ],
            'answer': 'Ans',
        }
        prec_calls = []
        seen_eval_configs = []

        def fake_evaluate_multi(dataset, **kwargs):
            seen_eval_configs.append((kwargs.get('raise_exceptions'), kwargs['run_config'].timeout, kwargs['run_config'].max_retries))
            metric_names = [m.name for m in kwargs['metrics']]
            if metric_names == ['llm_context_precision_with_reference']:
                prec_calls.append([row['contexts'] for row in dataset])
                return SimpleNamespace(to_pandas=lambda: pd.DataFrame([
                    {'llm_context_precision_with_reference': s} for s in (0.5, 1.0, 0.75, 0.75)
                ]))
            rows = [{name: 0.85 for name in metric_names} for _ in range(len(dataset))]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

        with patch('ragas.evaluate', side_effect=fake_evaluate_multi), patch.object(runner, 'create_llm', return_value=Mock()) as llm_mock:
            judged = runner._judge([multi_sample], 'full', max_workers=8)

        self.assertEqual(llm_mock.call_args.kwargs.get('timeout'), 600)
        self.assertEqual(llm_mock.call_args.kwargs.get('max_retries'), 1)
        self.assertEqual(seen_eval_configs, [(True, 600, 1), (True, 600, 1)])
        self.assertEqual(len(prec_calls), 1)
        self.assertEqual(len(prec_calls[0]), 4)
        self.assertAlmostEqual(judged[0]['llm_context_precision_with_reference'], 0.75)
        self.assertEqual(judged[0]['faithfulness'], 0.85)

        # Verify _judge fails strictly instead of silently returning None when RAGAs returns NaN.
        def fake_evaluate_nan(dataset, **kwargs):
            metric_names = [m.name for m in kwargs['metrics']]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame([
                {name: float('nan') for name in metric_names} for _ in range(len(dataset))
            ]))

        with patch('ragas.evaluate', side_effect=fake_evaluate_nan), patch.object(runner, 'create_llm', return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, '未能从大模型获得有效数值结果'):
                runner._judge([multi_sample], 'full', max_workers=8, selected_metrics=['faithfulness'])

        # Verify 402 Insufficient Balance aborts scoring immediately, surfaces clear Chinese error, and preserves existing scores.
        job_id = '9' * 32
        job_data = {
            'id': job_id,
            'status': 'completed',
            'created_at': '2026-11-19T00:00:00+00:00',
            'request': {
                'variants': [{'name': 'v1'}],
                'dataset': [
                    {'question': 'Q_ok', 'ground_truth': 'GT1', 'question_type': 'simple'},
                    {'question': 'Q_two', 'ground_truth': 'GT2', 'question_type': 'complex'},
                ],
                'scope': 'full',
                'concurrency': 2,
                'ragas_concurrency': 16,
                'use_ragas': True,
            },
            'results': {
                'v1': {
                    'status': 'completed',
                    'phase': '完成',
                    'completed_questions': 2,
                    'ragas_scored': True,
                    'samples': [
                        {
                            'question': 'Q_ok', 'ground_truth': 'GT1', 'question_type': 'simple',
                            'route': 'documents', 'sources': [{'text': 'Ctx1'}], 'trace': [{'queries': ['q1']}],
                            'answer': 'Ans1', 'retrieval_count': 1, 'total_tokens': 100, 'latency_ms': 100.0,
                            'metrics': {
                                'llm_context_precision_with_reference': 0.8,
                                'context_recall': 0.9,
                                'faithfulness': 0.95,
                                'answer_relevancy': 0.88,
                                'answer_correctness': 0.77,
                            },
                        },
                        {
                            'question': 'Q_two', 'ground_truth': 'GT2', 'question_type': 'complex',
                            'route': 'documents', 'sources': [{'text': 'Ctx2'}], 'trace': [{'queries': ['q2']}],
                            'answer': 'Ans2', 'retrieval_count': 2, 'total_tokens': 300, 'latency_ms': 200.0,
                            'metrics': {
                                'llm_context_precision_with_reference': 0.6,
                                'context_recall': 0.7,
                                'faithfulness': 0.81,
                                'answer_relevancy': 0.85,
                                'answer_correctness': 0.73,
                            },
                        },
                    ],
                    'metrics': {'faithfulness': 0.88, 'answer_correctness': 0.75},
                }
            },
        }
        runner._write(runner.RUN_DIR / f'{job_id}.json', job_data)

        def fake_evaluate_402(dataset, **kwargs):
            raise RuntimeError("Error code: 402 - {'error': {'message': 'Insufficient Balance'}}")

        with patch('ragas.evaluate', side_effect=fake_evaluate_402), patch.object(runner, 'create_llm', return_value=Mock()):
            runner.start_ragas_evaluation([job_id], ragas_concurrency=16, selected_metrics=['answer_correctness'])
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                failed_job = runner.get_job(job_id)
                if failed_job['status'] not in ('queued', 'running'):
                    break
                time.sleep(0.02)

        self.assertEqual(failed_job['status'], 'failed')
        self.assertIn('账户余额不足', failed_job['error'])
        self.assertIn('HTTP 402', failed_job['error'])
        self.assertIn('Insufficient Balance', failed_job['error'])
        self.assertEqual(failed_job['results']['v1']['error'], failed_job['error'])
        s0_m = failed_job['results']['v1']['samples'][0]['metrics']
        s1_m = failed_job['results']['v1']['samples'][1]['metrics']
        self.assertEqual(s0_m['faithfulness'], 0.95)
        self.assertEqual(s0_m['answer_correctness'], 0.77)
        self.assertEqual(s1_m['context_recall'], 0.7)
        self.assertEqual(s1_m['answer_correctness'], 0.73)

    def test_semantic_atomic_context_recall_prompt_and_ragas_progress_tracker(self):
        import pandas as pd
        from ragas.metrics._context_recall import ContextRecallClassificationPrompt

        original_default_instr = ContextRecallClassificationPrompt.instruction
        seen_recall_instr = []
        pbar_events = []

        def fake_evaluate(dataset, **kwargs):
            pbar = kwargs.get('_pbar')
            for m in kwargs['metrics']:
                if m.name == 'context_recall':
                    seen_recall_instr.append(m.context_recall_prompt.instruction)
            rows = []
            for _ in range(len(dataset)):
                row_scores = {}
                for m in kwargs['metrics']:
                    row_scores[m.name] = 0.88
                    if pbar is not None:
                        pbar.update(1)
                rows.append(row_scores)
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

        dataset = [
            {'question': 'Q1', 'ground_truth': 'Fact A; Fact B; Fact C.', 'question_type': 'simple'},
            {'question': 'Q2', 'ground_truth': 'Fact D; Fact E.', 'question_type': 'complex'},
        ]
        request = runner.BatchRequest(variants=[{'name': 'v1'}], dataset=dataset, scope='full', concurrency=1, ragas_concurrency=8, use_ragas=True)
        job_id = '7' * 32
        runner.JOBS[job_id] = {'id': job_id, 'status': 'running', 'results': {'v1': {'status': 'running', 'completed_questions': 2}}}

        def fake_ask(q, **kwargs):
            return {
                'answer': f'Ans {q}',
                'route': 'documents',
                'sources': [{'text': f'Ctx {q}', 'source': 'p.pdf', 'page': 1, 'chunk_id': 'p.pdf::1'}],
                'retrieval_trace': [{'queries': [q]}],
                'rewrite_count': 0,
                'used_web': False,
                'total_tokens': 100,
            }

        orig_progress = runner._RagasProgressTracker.update

        def spy_update(self, n=1):
            orig_progress(self, n)
            pbar_events.append((self.n, self.total))

        with patch.object(runner, 'create_llm', return_value=Mock()), \
                patch('app.agent.ask', side_effect=fake_ask), \
                patch('ragas.evaluate', side_effect=fake_evaluate), \
                patch.object(runner._RagasProgressTracker, 'update', spy_update):
            runner._run_variant(job_id, request.variants[0], request, Mock(), runner.TranslationCache())

        # Default ContextRecallClassificationPrompt must stay untouched; per-instance prompt uses atomic decomposition.
        self.assertEqual(ContextRecallClassificationPrompt.instruction, original_default_instr)
        self.assertEqual(len(seen_recall_instr), 1)
        self.assertIn('decompose the answer by its semantic content', seen_recall_instr[0])
        self.assertIn('atomic', seen_recall_instr[0])

        # 2 samples * 5 metrics = 10 total RAGAs steps progressed via _pbar.update(1).
        v_res = runner.JOBS[job_id]['results']['v1']
        self.assertEqual(v_res['ragas_total_steps'], 10)
        self.assertEqual(v_res['ragas_completed_steps'], 10)
        self.assertEqual(pbar_events, [(i, 10) for i in range(1, 11)])

    def test_completeness_answer_correctness_prompt_selective_metrics_and_mention_chunk_restore(self):
        import pandas as pd
        from ragas.metrics._answer_correctness import CorrectnessClassifier

        original_correctness_instr = CorrectnessClassifier.instruction
        seen_correctness_instr = []

        # 1. Verify _restore_missing_mention_chunks recovers dropped body text chunk referenced by mention_chunk_id.
        fake_snap = SimpleNamespace(
            chunk_ids=('p.pdf::10', 'p.pdf::11'),
            chunks=('Restored full paragraph explaining S3 and S4 strategies.', 'Figure 7: Role of external LLMs.'),
            metadatas=({'source': 'p.pdf', 'type': 'text', 'page': 5}, {'source': 'p.pdf', 'type': 'image', 'page': 6, 'mention_chunk_id': 'p.pdf::10'}),
        )
        legacy_samples = [{
            'question': 'Q1', 'ground_truth': 'GT1', 'question_type': 'simple',
            'route': 'documents',
            'sources': [{'chunk_id': 'p.pdf::11', 'type': 'image', 'page': 6, 'mention_chunk_id': 'p.pdf::10', 'sub_query': 'S3 S4 comparison', 'text': 'Figure 7: Role of external LLMs.'}],
            'trace': [{'queries': ['S3 S4 comparison']}],
            'answer': 'Ans1', 'retrieval_count': 1, 'total_tokens': 100, 'latency_ms': 100.0,
            'metrics': {
                'llm_context_precision_with_reference': 0.5,
                'context_recall': 0.0,
                'faithfulness': None,
                'answer_relevancy': 0.88,
                'answer_correctness': 0.72,
            },
        }]
        restored_copy = [dict(legacy_samples[0], sources=list(legacy_samples[0]['sources']))]
        runner._restore_missing_mention_chunks(restored_copy, snapshot=fake_snap)
        self.assertEqual([c['chunk_id'] for c in restored_copy[0]['sources']], ['p.pdf::10', 'p.pdf::11'])
        self.assertEqual(restored_copy[0]['sources'][0]['sub_query'], 'S3 S4 comparison')

        # 2. Verify AnswerCorrectness uses completeness prompt (without mutating global default) &
        # selected_metrics recomputes only selected metrics + any None metric, keeping valid unselected scores.
        job_id = '8' * 32
        job_data = {
            'id': job_id,
            'status': 'completed',
            'created_at': '2026-11-19T00:00:00+00:00',
            'request': {
                'variants': [{'name': 'v1'}],
                'dataset': [{'question': 'Q1', 'ground_truth': 'GT1', 'question_type': 'simple'}],
                'scope': 'full',
                'concurrency': 2,
                'ragas_concurrency': 16,
                'use_ragas': True,
            },
            'results': {
                'v1': {
                    'status': 'completed',
                    'phase': '完成',
                    'completed_questions': 1,
                    'ragas_scored': True,
                    'samples': legacy_samples,
                    'metrics': {},
                }
            },
        }
        runner._write(runner.RUN_DIR / f'{job_id}.json', job_data)

        eval_calls = []

        def fake_evaluate(dataset, **kwargs):
            metric_names = [m.name for m in kwargs['metrics']]
            eval_calls.append((metric_names, dataset[0]['contexts']))
            for m in kwargs['metrics']:
                if m.name == 'answer_correctness':
                    seen_correctness_instr.append(m.correctness_prompt.instruction)
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame([
                {name: 0.93 for name in metric_names} for _ in range(len(dataset))
            ]))

        with patch.object(runner, 'load_snapshot', return_value=fake_snap), \
                patch('ragas.evaluate', side_effect=fake_evaluate), \
                patch.object(runner, 'create_llm', return_value=Mock()):
            # Select context_recall, llm_context_precision_with_reference, answer_correctness.
            # Unselected metrics (answer_relevancy=0.88, faithfulness=None) must NOT be recomputed.
            runner.start_ragas_evaluation(
                [job_id],
                ragas_concurrency=16,
                selected_metrics=['context_recall', 'llm_context_precision_with_reference', 'answer_correctness'],
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                done = runner.get_job(job_id)
                if done['status'] not in ('queued', 'running'):
                    break
                time.sleep(0.02)

        self.assertEqual(done['status'], 'completed')
        self.assertEqual(CorrectnessClassifier.instruction, original_correctness_instr)
        self.assertEqual(len(seen_correctness_instr), 1)
        self.assertIn('do NOT penalize or classify as FP valid background explanations', seen_correctness_instr[0])
        self.assertEqual(len(eval_calls), 1)
        self.assertEqual(
            set(eval_calls[0][0]),
            {'llm_context_precision_with_reference', 'context_recall', 'answer_correctness'},
        )
        # Restored paragraph text is included in contexts passed to RAGAs.
        self.assertTrue(any('Restored full paragraph explaining S3 and S4 strategies.' in c for c in eval_calls[0][1]))
        updated_m = done['results']['v1']['samples'][0]['metrics']
        self.assertEqual(updated_m['answer_relevancy'], 0.88)
        self.assertIsNone(updated_m['faithfulness'])
        self.assertEqual(updated_m['context_recall'], 0.93)
        self.assertEqual(updated_m['llm_context_precision_with_reference'], 0.93)
        self.assertEqual(updated_m['answer_correctness'], 0.93)

    def test_context_precision_uses_raw_subgroups_partial_support_prompt_and_no_dedup_splitting(self):
        import pandas as pd
        from ragas.metrics._context_precision import ContextPrecisionPrompt

        original_prec_instr = ContextPrecisionPrompt.instruction

        # 1. Legacy sample without raw trace sub_groups must NOT split deduplicated sources by sub_query.
        legacy_sample = {
            'question': 'Legacy Q',
            'ground_truth': 'Legacy GT',
            'route': 'documents',
            'sources': [
                {'chunk_id': 'p.pdf::1', 'text': 'Core chunk', 'sub_query': 'q1'},
                {'chunk_id': 'p.pdf::2', 'text': 'Leftover chunk', 'sub_query': 'q2'},
            ],
            'trace': [{'queries': ['q1', 'q2'], 'chunk_ids': ['p.pdf::1', 'p.pdf::2']}],
            'answer': 'Legacy Ans',
        }
        # 2. Single raw sub_group sample must evaluate using raw trace sub_groups (not filtered sources).
        single_raw_sample = {
            'question': 'Single Raw Q',
            'ground_truth': 'Single Raw GT',
            'route': 'documents',
            'sources': [{'chunk_id': 'p.pdf::9', 'text': 'Filtered source only'}],
            'trace': [{'sub_groups': [
                {'sub_query': 'q1', 'docs': [{'text': 'Raw Rank1'}, {'text': 'Raw Rank2'}]},
            ]}],
            'answer': 'Single Raw Ans',
        }

        self.assertEqual(runner._extract_raw_subquery_context_groups(legacy_sample), [])
        self.assertEqual(
            runner._extract_subquery_context_groups(legacy_sample),
            [['Core chunk', 'Leftover chunk']],
        )
        self.assertEqual(
            runner._extract_raw_subquery_context_groups(single_raw_sample),
            [['Raw Rank1', 'Raw Rank2']],
        )

        seen_calls = []
        seen_instructions = []

        def fake_evaluate(dataset, **kwargs):
            metric_names = [m.name for m in kwargs['metrics']]
            for m in kwargs['metrics']:
                if m.name == 'llm_context_precision_with_reference':
                    seen_instructions.append(m.context_precision_prompt.instruction)
            seen_calls.append((metric_names, [row['contexts'] for row in dataset]))
            rows = [{name: 0.88 for name in metric_names} for _ in range(len(dataset))]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

        with patch('ragas.evaluate', side_effect=fake_evaluate), patch.object(runner, 'create_llm', return_value=Mock()):
            judged = runner._judge(
                [legacy_sample, single_raw_sample],
                'full',
                max_workers=4,
                selected_metrics=['llm_context_precision_with_reference'],
            )

        self.assertEqual(ContextPrecisionPrompt.instruction, original_prec_instr)
        self.assertEqual(len(seen_instructions), 2)
        for instr in seen_instructions:
            self.assertIn('Do NOT require a single chunk to cover the entire reference answer', instr)

        # Call 0: legacy_sample evaluated as a single full-sources group (not split into q1/q2).
        # Call 1: single_raw_sample evaluated via raw trace sub_groups.
        self.assertEqual(len(seen_calls), 2)
        self.assertEqual(seen_calls[0][1], [['Core chunk', 'Leftover chunk']])
        self.assertEqual(seen_calls[1][1], [['Raw Rank1', 'Raw Rank2']])
        self.assertEqual(judged[0]['llm_context_precision_with_reference'], 0.88)
        self.assertEqual(judged[1]['llm_context_precision_with_reference'], 0.88)


class ExperimentUiTests(unittest.TestCase):
    def test_ui_queues_all_five_saved_variants_with_parallelism_five(self):
        variants=default_variants()
        variants[0]["config"]["num_queries"]=3
        calls=[]
        job={}
        def request(base,method,path,**kwargs):
            if path=='/evaluations/variants':return {'variants':variants}
            if path=='/evaluations/dataset':return {'dataset':[{'question':'Q','ground_truth':'A'}]}
            if path=='/evaluations/jobs' and method=='POST':
                payload=kwargs['json']; calls.append(payload)
                job.update(id='c'*32,status='queued',request=payload,dataset_hash='hash',
                    results={v['name']:{'status':'queued','phase':'等待执行','completed_questions':0,'metrics':{},'samples':[]} for v in payload['variants']})
                return job
            if path.startswith('/evaluations/jobs/'):return job
            raise AssertionError((method,path))
        with patch('app.experiments_ui._request',side_effect=request):
            ui=AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('消融评估').run()
            self.assertFalse(ui.exception,str(ui.exception))
            next(w for w in ui.slider if w.label=='并行变体数').set_value(5).run()
            next(w for w in ui.button if w.label=='开始批量并行评估').click().run()
            self.assertFalse(ui.exception,str(ui.exception))
            self.assertFalse(ui.error,str(ui.error))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0]['variants']),5)
        self.assertEqual(calls[0]['concurrency'],5)
        self.assertEqual(calls[0]['scope'],'retrieval')
        self.assertTrue(all('num_queries' not in variant['config'] for variant in calls[0]['variants']))

    def test_ui_surfaces_job_and_variant_errors_without_only_missing_button(self):
        self.assertNotIn('only_missing', runner.RagasEvalRequest.model_fields)
        variants=default_variants()[:1]
        job_id='f'*32
        failed_job={
            'id':job_id,
            'status':'failed',
            'error':'💰 DeepSeek API 账户余额不足 (HTTP 402: Insufficient Balance)',
            'dataset_hash':'hash'*16,
            'snapshot_id':'snap'*16,
            'request':{
                'dataset':[{'question':'Q','ground_truth':'A','question_type':'simple'}],
                'top_k':5,
                'ragas_concurrency':16,
            },
            'results':{
                variants[0]['name']:{
                    'status':'failed',
                    'phase':'RAGAs 评分失败',
                    'error':'🚦 变体限流 (HTTP 429)',
                    'completed_questions':1,
                    'ragas_scored':True,
                    'metrics':{'answer_relevancy':0.88,'context_recall':None,'question_count':1,'latency_ms':100.0,'avg_retrieval_count':1.0,'avg_tokens':100.0},
                    'samples':[{
                        'question':'Q','ground_truth':'A','question_type':'simple','route':'documents',
                        'sources':[],'trace':[],'answer':'Ans','retrieval_count':1,'total_tokens':100,'latency_ms':100.0,
                        'metrics':{'answer_relevancy':0.88,'context_recall':None},
                    }],
                }
            },
        }
        def request(base,method,path,**kwargs):
            if path=='/evaluations/variants':return {'variants':variants}
            if path=='/evaluations/dataset':return {'dataset':[{'question':'Q','ground_truth':'A'}]}
            if path==f'/evaluations/jobs/{job_id}':return failed_job
            raise AssertionError((method,path))
        with patch('app.experiments_ui._request',side_effect=request):
            ui=AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.session_state['ablation_job']=job_id
            ui.session_state['ablation_active']=False
            ui.radio(key='view').set_value('消融评估').run()
            self.assertFalse(ui.exception,str(ui.exception))
        err_texts=[e.value for e in ui.error]
        self.assertTrue(any('账户余额不足' in msg and 'HTTP 402' in msg for msg in err_texts),err_texts)
        self.assertTrue(any(variants[0]['name'] in msg and 'HTTP 429' in msg for msg in err_texts),err_texts)
        btn_labels=[b.label for b in ui.button]
        self.assertFalse(any('仅补评' in lbl for lbl in btn_labels),btn_labels)
        self.assertTrue(any('重算所选 RAGAs 指标' in lbl for lbl in btn_labels),btn_labels)


class DatasetEditorRegressionTests(unittest.TestCase):
    def fake_request(self, base, method, path, **kwargs):
        if path=='/evaluations/variants':return {'variants':default_variants()}
        if path=='/evaluations/dataset':return {'dataset':[{'question':'Project question','ground_truth':'Project answer'}]}
        if path.startswith('/evaluations/jobs/'):
            return {'id':'e'*32,'status':'running','dataset_hash':'hash',
                'request':{'dataset':[{'question':'Q','ground_truth':'A'}]},
                'results':{'existing':{'status':'running','phase':'RAGAs 评分中','completed_questions':1,'metrics':{},'samples':[]}}}
        raise AssertionError((method,path))

    def test_custom_dataset_survives_switching_to_pdf_view_and_back(self):
        custom=json.dumps([{'question':'My custom question','ground_truth':'My custom answer'}])
        with patch('app.experiments_ui._request',side_effect=self.fake_request),patch('app.chunk_browser.render_chunk_browser'):
            ui=AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('消融评估').run()
            ui.text_area(key='_ablation_dataset_editor').set_value(custom).run()
            self.assertEqual(ui.session_state['ablation_dataset_text'],custom)
            ui.radio(key='view').set_value('PDF 分块').run()
            ui.radio(key='view').set_value('消融评估').run()
        self.assertFalse(ui.exception,str(ui.exception))
        self.assertFalse(ui.error,str(ui.error))
        self.assertEqual(ui.text_area(key='_ablation_dataset_editor').value,custom)

    def test_empty_draft_does_not_hide_existing_job_and_can_restore_dataset(self):
        with patch('app.experiments_ui._request',side_effect=self.fake_request):
            ui=AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.radio(key='view').set_value('消融评估').run()
            ui.session_state['ablation_job']='e'*32
            ui.session_state['ablation_active']=True
            ui.text_area(key='_ablation_dataset_editor').set_value('').run()
            self.assertFalse(ui.exception,str(ui.exception))
            self.assertTrue(any('题集为空' in error.value for error in ui.error))
            self.assertTrue(any('批次进度与结果' in header.value for header in ui.subheader))
            self.assertTrue(next(w for w in ui.button if w.label=='开始批量并行评估').disabled)
            next(w for w in ui.button if w.label=='恢复项目题集').click().run()
            self.assertFalse(ui.error,str(ui.error))
            self.assertIn('Project question',ui.text_area(key='_ablation_dataset_editor').value)

    def test_legacy_empty_widget_state_is_recovered_from_project_dataset(self):
        with patch('app.experiments_ui._request',side_effect=self.fake_request):
            ui=AppTest.from_file('app/main.py',default_timeout=20).run()
            ui.session_state['ablation_dataset']=''
            ui.radio(key='view').set_value('消融评估').run()
        self.assertFalse(ui.exception,str(ui.exception))
        self.assertFalse(ui.error,str(ui.error))
        self.assertIn('Project question',ui.text_area(key='_ablation_dataset_editor').value)

    def test_empty_and_invalid_json_have_specific_messages(self):
        from app.experiments_ui import _parse_dataset
        self.assertIn('题集为空',_parse_dataset('')[1])
        self.assertIn('第 1 行',_parse_dataset('not json')[1])
        self.assertIn('非空 JSON 数组',_parse_dataset('{}')[1])

    def test_non_json_api_response_names_the_endpoint_and_status(self):
        import httpx
        from app.experiments_ui import _request
        with patch('httpx.request',return_value=httpx.Response(200,text='')):
            with self.assertRaisesRegex(ValueError,'/evaluations/jobs.*HTTP 200'):
                _request('http://127.0.0.1:8000','GET','/evaluations/jobs')


class DeepSeekJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_judge_completions_use_separate_single_completion_prompts(self):
        from evaluation.judge import DeepSeekJudge
        from langchain_core.outputs import LLMResult, Generation
        from langchain_core.prompt_values import StringPromptValue
        from unittest.mock import AsyncMock
        model=Mock()
        model.temperature=0
        model.n=1
        model.agenerate_prompt=AsyncMock(return_value=LLMResult(generations=[
            [Generation(text='a')],[Generation(text='b')],[Generation(text='c')]]))
        judge=DeepSeekJudge(model)
        result=await judge.agenerate_text(StringPromptValue(text='prompt'),n=3)
        self.assertEqual([g.text for g in result.generations[0]],['a','b','c'])
        self.assertEqual(model.agenerate_prompt.call_args.kwargs['n'],1)
        self.assertEqual(len(model.agenerate_prompt.call_args.kwargs['prompts']),3)
        self.assertEqual(model.temperature,0)
        self.assertEqual(model.n,1)


if __name__=='__main__':
    unittest.main()
