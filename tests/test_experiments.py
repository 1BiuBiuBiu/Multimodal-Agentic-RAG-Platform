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
            captures.append((dataset,kwargs['metrics']))
            return SimpleNamespace(to_pandas=lambda:pd.DataFrame([{
                metric.name:0.5 for metric in kwargs['metrics']}]))
        sample={'question':'Q','ground_truth':'A','sources':[],'web_contexts':['Web evidence'],'answer':'Chinese answer'}
        with patch('ragas.evaluate',side_effect=evaluate),patch.object(runner,'create_llm',return_value=Mock()):
            runner._judge([sample],'full')
            runner._judge([sample],'full')
        self.assertEqual(captures[0][0][0]['contexts'],['Web evidence'])
        self.assertTrue(all(a is not b for a,b in zip(captures[0][1],captures[1][1])))

    def test_old_running_job_becomes_interrupted_on_restart(self):
        job_id='b'*32
        path=Path(self.temp.name)/(job_id+'.json')
        path.write_text(json.dumps({'id':job_id,'status':'running'}),encoding='utf-8')
        self.assertEqual(runner.get_job(job_id)['status'],'interrupted')


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
