"""Standalone DeepSeek chat UI: streamlit run app/ui.py."""

import streamlit as st

from app.llm import create_llm

st.set_page_config(page_title="DeepSeek Chat", layout="centered")
st.title("🧠 DeepSeek Chat")
st.write("Chat using the DeepSeek API.")

question = st.text_input("Ask a question:")
if question:
    try:
        st.subheader("🤖 Answer")
        st.write_stream(
            chunk.content for chunk in create_llm().stream(question)
            if isinstance(chunk.content, str) and chunk.content
        )
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("DeepSeek request failed. Check your API key, model, balance and network connection.")
