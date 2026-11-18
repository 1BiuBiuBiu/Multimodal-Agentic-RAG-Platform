"""Streaming protocol, incremental delivery and Streamlit rendering checks."""
import asyncio
import json
import threading
import unittest
import httpx
from contextlib import nullcontext
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from streamlit.testing.v1 import AppTest

from app import api
import app.agent as agent


def payload(raw):
    return json.loads(raw.removeprefix("data: ").strip())


class StreamApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_arrive_before_generation_finishes_and_history_is_used(self):
        finish = threading.Event()
        model = Mock()
        model.invoke.side_effect = [AIMessage(content="documents"), AIMessage(content="yes")]
        def chunks(*args, **kwargs):
            yield AIMessageChunk(content="你好")
            if not finish.wait(5):
                raise RuntimeError("Stream did not deliver the first token in time")
            yield AIMessageChunk(content="世界")
        model.stream.side_effect = chunks
        source = {"text": "Test document", "source": "test.pdf", "page": 2}
        request = api.QueryRequest(question="Explain the document", history=[
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ])
        with patch.object(agent, "llm", model), patch.object(agent, "retrieve_documents", return_value={"sources": [source], "queries": ["test"], "retrieval_query": "test", "snapshot_id": "test"}):
            response = await api.query_stream(request)
            self.assertEqual(response.media_type, "text/event-stream")
            events = []
            try:
                while not any(e["type"] == "token" for e in events):
                    events.append(payload(await asyncio.wait_for(anext(response.body_iterator), 5)))
                self.assertFalse(finish.is_set())
                self.assertEqual(events[-1]["content"], "你好")
                finish.set()
                events.extend([payload(raw) async for raw in response.body_iterator])
            finally:
                finish.set()
            tokens = [e["content"] for e in events if e["type"] == "token"]
            self.assertEqual(tokens, ["你好", "世界"])
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["answer"], "你好世界")
            self.assertEqual(events[-1]["sources"], [source])
            self.assertIn("latency_ms", events[-1])
            prompt = model.stream.call_args.args[0][0].content
            self.assertIn("Previous answer", prompt)

    async def test_missing_index_becomes_error_event(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="documents")
        with patch.object(agent, "llm", model), patch.object(agent, "retrieve_documents", side_effect=FileNotFoundError):
            response = await api.query_stream(api.QueryRequest(question="Look up document"))
            events = [payload(raw) async for raw in response.body_iterator]
        self.assertEqual(events[-1]["type"], "error")
        self.assertFalse(any(e["type"] == "done" for e in events))

    async def test_generation_failure_keeps_partial_tokens_without_done(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="llm_only")
        def chunks(*args, **kwargs):
            yield AIMessageChunk(content="Partial answer")
            raise RuntimeError("Simulated disconnect")
        model.stream.side_effect = chunks
        with patch.object(agent, "llm", model), patch.object(api.logger, "exception"):
            response = await api.query_stream(api.QueryRequest(question="Hello"))
            events = [payload(raw) async for raw in response.body_iterator]
        self.assertIn({"type": "token", "content": "Partial answer"}, events)
        self.assertEqual(events[-1]["type"], "error")
        self.assertFalse(any(e["type"] == "done" for e in events))


class StreamUiTests(unittest.TestCase):
    def run_ui(self, events, status_code=200):
        response = Mock()
        if status_code != 200:
            request = httpx.Request("POST", "http://127.0.0.1:8000/query/stream")
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Backend unavailable", request=request,
                response=httpx.Response(status_code, request=request),
            )
        response.iter_lines.return_value = iter(
            [": keep-alive", ""] + ["data: " + json.dumps(e, ensure_ascii=False) for e in events]
        )
        ui = AppTest.from_file("app/main.py", default_timeout=15).run()
        with patch("httpx.stream", return_value=nullcontext(response)) as stream:
            ui.chat_input[0].set_value("Hello").run()
        self.assertFalse(ui.exception, str(ui.exception))
        self.assertTrue(stream.call_args.args[1].endswith("/query/stream"))
        self.assertFalse(stream.call_args.kwargs["trust_env"])
        self.assertNotIn("num_queries",stream.call_args.kwargs["json"]["config"])
        return ui

    def test_completed_stream_saves_answer_and_sources(self):
        source = {"text": "Test", "source": "test.pdf", "page": 2}
        ui = self.run_ui([
            {"type": "status", "content": "Retrieving"},
            {"type": "token", "content": "你好"},
            {"type": "token", "content": "世界"},
            {"type": "done", "answer": "你好世界", "sources": [source],
                "rewrite_count": 0, "used_web": False, "latency_ms": 10},
        ])
        self.assertFalse(ui.error, str(ui.error))
        message = ui.session_state["messages"][-1]
        self.assertEqual(message["content"], "你好世界")
        self.assertEqual(message["sources"], [source])

    def test_truncated_stream_is_marked_incomplete(self):
        ui = self.run_ui([{"type": "token", "content": "Partial answer"}])
        self.assertTrue(ui.error)
        self.assertTrue(ui.session_state["messages"][-1]["incomplete"])
        self.assertEqual(ui.session_state["messages"][-1]["content"], "Partial answer")

    def test_http_failure_shows_status_code_without_saving_answer(self):
        ui = self.run_ui([], status_code=502)
        self.assertIn("502", ui.error[0].value)
        self.assertEqual(ui.session_state["messages"][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
