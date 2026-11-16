"""Shared DeepSeek chat client for the agent, UI and evaluation."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def create_llm(*, temperature: float = 0.1) -> ChatOpenAI:
    """Use DeepSeek's OpenAI-compatible endpoint with explicit credentials."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your-deepseek-api-key":
        raise ValueError(
            "DEEPSEEK_API_KEY is required. Copy .env.example to .env "
            "and set your DeepSeek API key."
        )
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com",
        model=os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash",
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
