"""Offline checks: python -m unittest discover -s tests -v."""

import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

from app.llm import create_llm


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_requires_deepseek_key_even_with_openai_key(self):
        os.environ.pop("DEEPSEEK_API_KEY")
        os.environ["OPENAI_API_KEY"] = "unrelated-key"
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            create_llm()

    def test_rejects_example_key(self):
        os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            create_llm()

    def test_defaults_and_overrides(self):
        client = create_llm()
        self.assertEqual(client.openai_api_base, "https://api.deepseek.com")
        self.assertEqual(client.model_name, "deepseek-flash")
        self.assertEqual(client.request_timeout, 60)
        self.assertEqual(client.max_retries, 2)
        os.environ.update(DEEPSEEK_MODEL="custom-model", DEEPSEEK_BASE_URL="https://example.test/v1")
        client = create_llm(temperature=0)
        self.assertEqual(client.openai_api_base, "https://example.test/v1")
        self.assertEqual(client.model_name, "custom-model")
        self.assertEqual(client.temperature, 0)

    def test_empty_options_use_defaults(self):
        os.environ.update(DEEPSEEK_MODEL=" ", DEEPSEEK_BASE_URL=" ")
        client = create_llm()
        self.assertEqual(client.model_name, "deepseek-flash")
        self.assertEqual(client.openai_api_base, "https://api.deepseek.com")

    def test_chat_and_document_rag_use_deepseek_http(self):
        requests = []
        answers = iter(["Hello", "documents", "yes", "Use DeepSeek [guide.pdf, Page 1]"])

        def respond(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "deepseek-flash")
            requests.append(body)
            if body.get("stream"):
                answer = next(answers)
                events = []
                for text in (answer[:8], answer[8:]):
                    events.append("data: " + json.dumps({
                        "id": "test-completion", "object": "chat.completion.chunk", "created": 1,
                        "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": None,
                            "delta": {"role": "assistant", "content": text}}],
                    }) + "\n\n")
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                    text="".join(events) + "data: [DONE]\n\n")
            return httpx.Response(200, json={
                "id": "test-completion", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": next(answers),
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        chunk = {"text": "Use the DeepSeek API.", "source": "guide.pdf", "page": 1}
        store = types.ModuleType("vectorstore.store")
        store.hybrid_search = lambda query, top_k: [chunk]
        search = types.ModuleType("langchain_community.tools.tavily_search")
        search.TavilySearchResults = lambda **kwargs: None
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            def build_client(**kwargs):
                return ChatOpenAI(**kwargs, http_client=http)

            with patch("app.llm.ChatOpenAI", side_effect=build_client):
                self.assertEqual(create_llm().invoke("Hi").content, "Hello")
                with patch.dict(sys.modules, {
                    "vectorstore.store": store,
                    "langchain_community.tools.tavily_search": search,
                }):
                    sys.modules.pop("app.agent", None)
                    try:
                        agent = importlib.import_module("app.agent")
                        with patch.object(agent, "retrieve_documents", return_value={
                            "sources": [chunk], "queries": ["test"], "retrieval_query": "test", "snapshot_id": "test",
                        }):
                            result = agent.ask("What API does guide.pdf use?")
                    finally:
                        sys.modules.pop("app.agent", None)
        self.assertEqual(result["sources"], [chunk])
        self.assertIn("DeepSeek", result["answer"])
        self.assertFalse(result["used_web"])
        self.assertEqual(len(requests), 4)
        self.assertIn(chunk["text"], requests[-1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
