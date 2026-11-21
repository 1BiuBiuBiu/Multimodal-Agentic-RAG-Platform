import unittest
from pathlib import Path
from app.export_utils import export_to_markdown, export_to_html


class TestExportUtils(unittest.TestCase):
    def test_export_markdown_and_html_with_trace_tokens_and_full_sources(self):
        long_snippet = "A" * 320 + " END_OF_FULL_SOURCE"
        sample_messages = [
            {"role": "user", "content": "AffectGPT测评了哪些数据集？"},
            {
                "role": "assistant",
                "original_query": "AffectGPT测评了哪些数据集？",
                "content": "AffectGPT在MER-UniBench等数据集上进行了测评。",
                "decision_trace": [
                    {"step": "🧭 意图路由", "detail": "判定为多模态论文检索问题"},
                    {"step": "🔍 子问题拆解与检索", "detail": "拆解为 2 个子问题并执行 Cross-Encoder 重排"},
                ],
                "query_doc_history": [
                    {
                        "cycle": 1,
                        "cycle_name": "第 1 轮检索",
                        "input_query": "AffectGPT 评测数据集",
                        "sub_groups": [
                            {
                                "sub_query": "MER-UniBench 数据集构成",
                                "docs": [
                                    {
                                        "source": "2501.16566v2.pdf",
                                        "page": 7,
                                        "chunk_id": "2501.16566v2.pdf::10",
                                        "text": "MER-UniBench includes 9 datasets.",
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "meta": {"latency_ms": 1200, "rewrite_count": 1, "total_tokens": 1500, "used_web": False},
                "token_breakdown": [
                    {"step": "🧭 意图路由", "prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                    {"step": "✍️ 最终回答生成", "prompt_tokens": 900, "completion_tokens": 350, "total_tokens": 1250},
                ],
                "sources": [
                    {
                        "source": "2501.16566v2.pdf",
                        "page": 7,
                        "page_end": 8,
                        "section": "4.2",
                        "chunk_id": "c1",
                        "mention_context": "Table 2 汇总了所有评测基准。",
                        "text": long_snippet,
                    }
                ],
            },
        ]

        md = export_to_markdown(sample_messages, "测试会话")
        self.assertIn("# 测试会话", md)
        self.assertIn("AffectGPT测评了哪些数据集？", md)
        self.assertIn("完整思考决策链路与子问题检索组 (2 步)", md)
        self.assertIn("MER-UniBench 数据集构成", md)
        self.assertIn("详细 Token 分步消耗明细", md)
        self.assertIn("2501.16566v2.pdf", md)
        self.assertIn("Page 7-8", md)
        self.assertIn("Table 2 汇总了所有评测基准。", md)
        self.assertIn("END_OF_FULL_SOURCE", md)

        html_doc = export_to_html(sample_messages, "测试会话")
        self.assertIn("<!DOCTYPE html>", html_doc)
        self.assertIn("测试会话 - 会话导出", html_doc)
        self.assertIn('<details open class="trace-box">', html_doc)
        self.assertIn("MER-UniBench 数据集构成", html_doc)
        self.assertIn("详细 Token 分步消耗明细", html_doc)
        self.assertIn("END_OF_FULL_SOURCE", html_doc)
        self.assertIn("break-inside: auto;", html_doc)

    def test_main_ui_print_stylesheet_and_expand_hooks(self):
        main_src = Path("app/main.py").read_text(encoding="utf-8")
        # Must not render st.info print prompt banner into the printable page
        self.assertNotIn('st.info("💡 正在调起打印窗口', main_src)
        # Must hide tooltips and toasts in print CSS
        self.assertIn('[data-baseweb="tooltip"]', main_src)
        self.assertIn('[data-testid="stToastContainer"]', main_src)
        # Must allow chat messages to flow naturally from page 1
        self.assertIn("break-inside: auto !important;", main_src)
        # Must unclip scrollable trace/snippet containers during print
        self.assertIn(".decision-trace-inner, .source-snippet, .source-table-scroll", main_src)
        # Must auto-expand decision trace, sources, tokens, and handle Ctrl+P via beforeprint
        self.assertIn("force_expand=is_printing", main_src)
        self.assertIn('addEventListener("beforeprint"', main_src)
        self.assertIn('addEventListener("afterprint"', main_src)


if __name__ == "__main__":
    unittest.main()
