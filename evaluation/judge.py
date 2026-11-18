"""RAGAs adapter: independent n=1 calls, fixed temperature, no shared mutation."""
from ragas.llms import LangchainLLMWrapper


class DeepSeekJudge(LangchainLLMWrapper):
    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        result = self.langchain_llm.generate_prompt(
            prompts=[prompt] * n, n=1, temperature=0, stop=stop, callbacks=callbacks,
        )
        result.generations = [[row[0] for row in result.generations]]
        return result

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        result = await self.langchain_llm.agenerate_prompt(
            prompts=[prompt] * n, n=1, temperature=0, stop=stop, callbacks=callbacks,
        )
        result.generations = [[row[0] for row in result.generations]]
        return result
