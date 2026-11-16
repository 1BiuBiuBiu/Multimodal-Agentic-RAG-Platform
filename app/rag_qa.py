"""Command-line RAG using DeepSeek: python -m app.rag_qa."""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import ask


def main() -> None:
    history = []
    print("RAG system ready (DeepSeek API). Type a question or 'exit'.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() == "exit":
            break
        if not query:
            continue
        result = ask(query, history=history)
        print("\nAI:", result["answer"])
        history.extend([HumanMessage(content=query), AIMessage(content=result["answer"])])


if __name__ == "__main__":
    main()
