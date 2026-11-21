import unittest
import re
from app.agent import _track_tokens

def format_latex_formulas(text: str) -> str:
    r"""
    Detects LaTeX math expressions incorrectly enclosed in parentheses like:
      (Z_{av}\in\mathbb{R}^{(t_a+t_v)\times d})
    and transforms them into standard KaTeX markdown:
      $Z_{av}\in\mathbb{R}^{(t_a+t_v)\times d}$
    """
    if not text:
        return text

    # 1. Standard LaTeX delimiters \( ... \) and \[ ... \]
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    if "(" not in text or "\\" not in text:
        return text

    math_indicators = (
        r"\in", r"\mathbb", r"\times", r"\sum", r"\prod", r"\frac", r"\sqrt",
        r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda", r"\sigma", r"\mu",
        r"\partial", r"\nabla", r"\mathcal", r"\mathbf", r"\mathrm", r"\text",
        r"\hat", r"\bar", r"\tilde", r"\vec", r"\cdot", r"\le", r"\ge", r"\neq",
        r"\approx", r"\infty", r"\cup", r"\cap", r"\subset", r"\subseteq",
        r"\forall", r"\exists", r"^{", r"_{", r"\to", r"\leftarrow", r"\rightarrow",
        r"\pm", r"\mp", r"\equiv", r"\sim", r"\propto", r"\circ", r"\bullet"
    )

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$":
            is_double = (i + 1 < n and text[i + 1] == "$")
            delim = "$$" if is_double else "$"
            k = text.find(delim, i + len(delim))
            if k != -1:
                result.append(text[i : k + len(delim)])
                i = k + len(delim)
                continue
        if text[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = text[i + 1 : j - 1]
                has_math = any(ind in inner for ind in math_indicators)
                has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in inner)
                if has_math and not has_chinese and "\n\n" not in inner and 2 < len(inner) < 400:
                    result.append(f"${inner.strip()}$")
                    i = j
                    continue
        result.append(text[i])
        i += 1
    return "".join(result)


class TestUIFormatting(unittest.TestCase):
    def test_latex_parentheses_conversion(self):
        sample = (
            "Q-Former 方案中，音视频特征沿时间维拼接为 (Z_{av}\\in\\mathbb{R}^{(t_a+t_v)\\times d})，"
            "输出融合特征 (Z_f\\in\\mathbb{R}^{K\\times d})，并使用位置编码 (例如在第 5 页提到)。"
        )
        converted = format_latex_formulas(sample)
        self.assertIn("$Z_{av}\\in\\mathbb{R}^{(t_a+t_v)\\times d}$", converted)
        self.assertIn("$Z_f\\in\\mathbb{R}^{K\\times d}$", converted)
        self.assertIn("(例如在第 5 页提到)", converted)

    def test_citation_regex_punct(self):
        from app.main import CITATION_REGEX
        text1 = "[2605.06126v1 (1).pdf, #2605.06126v1 (1).pdf::13]。"
        m1 = CITATION_REGEX.search(text1)
        self.assertIsNotNone(m1)
        self.assertEqual(m1.group(1), "2605.06126v1 (1).pdf, #2605.06126v1 (1).pdf::13")
        self.assertEqual(m1.group(2), "。")

        text2 = "[Source: 2605.06126v1 (1).pdf, Page 3, Section: 2 AffectGPT-RL, #2605.06126v1 (1).pdf::13] ，接下来"
        m2 = CITATION_REGEX.search(text2)
        self.assertIsNotNone(m2)
        self.assertEqual(m2.group(2), "，")

    def test_track_tokens_breakdown(self):
        state = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "token_breakdown": [],
        }
        usage, breakdown = _track_tokens(state, prompt_text="abcd " * 10, completion_text="xyz", step_name="🧭 意图路由")
        self.assertEqual(len(breakdown), 1)
        self.assertEqual(breakdown[0]["step"], "🧭 意图路由")
        self.assertGreater(breakdown[0]["prompt_tokens"], 0)
        self.assertGreater(breakdown[0]["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 150 + breakdown[0]["total_tokens"])

    def test_linkify_citations_with_types(self):
        from app.main import linkify_citations
        sources = [
            {"source": "2605.06126v1 (1).pdf", "page": 2, "section": "2 AffectGPT-RL", "chunk_id": "2605.06126v1 (1).pdf::11", "type": "text"},
            {"source": "2605.06126v1 (1).pdf", "page": 3, "section": "3 Experiments", "chunk_id": "2605.06126v1 (1).pdf::12", "type": "image"},
            {"source": "2605.06126v1 (1).pdf", "page": 4, "page_end": 5, "section": "4 Ablation", "chunk_id": "2605.06126v1 (1).pdf::13", "type": "table"},
        ]
        raw_text = (
            "冷启动训练[2605.06126v1 (1).pdf, #2605.06126v1 (1).pdf::11]。"
            "架构图[2605.06126v1 (1).pdf, #12]和"
            "表格结果[2605.06126v1 (1).pdf, Page 99, Section: Ignored, #2605.06126v1 (1).pdf::13]"
        )
        rendered = linkify_citations(raw_text, sources, "msg-0")
        self.assertIn("📄 文本", rendered)
        self.assertIn("cite-type-text", rendered)
        self.assertIn("[2605.06126v1 (1).pdf, Page 2, Section: 2 AffectGPT-RL, #11]", rendered)
        self.assertIn("🖼️ 图片", rendered)
        self.assertIn("cite-type-image", rendered)
        self.assertIn("[2605.06126v1 (1).pdf, Page 3, Section: 3 Experiments, #12]", rendered)
        self.assertIn("📊 表格", rendered)
        self.assertIn("cite-type-table", rendered)
        self.assertIn("[2605.06126v1 (1).pdf, Page 4-5, Section: 4 Ablation, #13]", rendered)

    def test_extract_and_anchor_headings_preserves_markdown(self):
        from app.main import extract_and_anchor_headings
        sample = (
            "### 1. 训练方式\n\n"
            "五种奖励的具体定义为：\n"
            "- **Format Reward**：检查输出格式\n"
            "- **Accuracy Reward**：检查准确率"
        )
        anchored, toc = extract_and_anchor_headings(sample, 0)
        lines = anchored.split("\n")
        self.assertTrue(lines[0].startswith("### <span id=\"nav-head-0-1\""))
        self.assertTrue(lines[3].startswith("- <span id=\"nav-head-0-2\""))
        self.assertTrue(lines[4].startswith("- <span id=\"nav-head-0-3\""))
        self.assertEqual([item["title"] for item in toc], ["1. 训练方式", "Format Reward", "Accuracy Reward"])


if __name__ == "__main__":
    unittest.main()
