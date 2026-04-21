import asyncio
from typing import Any, Dict

from dotenv import load_dotenv

from backend.chatbot import VinLexChatbot
from config import CHAT_MODEL

load_dotenv()


class MainAgent:
    """Day 14 benchmark wrapper around the Day 06 VinLex runtime."""

    def __init__(self):
        self.name = "VinLex-Day14"
        self._chatbot = VinLexChatbot()

    async def query(self, question: str) -> Dict[str, Any]:
        """Run the synchronous Day 06 chatbot in a worker thread."""
        return await asyncio.to_thread(self._query_sync, question)

    def _query_sync(self, question: str) -> Dict[str, Any]:
        history: list[dict] = []

        try:
            result = self._chatbot.process(question, history)
            chunks = []

            if result.get("query_type") == "academic_regulation":
                chunks = self._chatbot._rag.retrieve(question)

            return {
                "answer": result["answer"],
                "contexts": [chunk["text"] for chunk in chunks],
                "retrieved_ids": [chunk["id"] for chunk in chunks],
                "metadata": {
                    "model": CHAT_MODEL,
                    "tokens_used": None,
                    "sources": result.get("sources", []),
                    "query_type": result.get("query_type"),
                    "redirect_to_contact": result.get("redirect_to_contact", False),
                    "suggest_counseling": result.get("suggest_counseling", False),
                },
            }
        except Exception as exc:
            return {
                "answer": (
                    "The agent encountered an internal error while processing this question. "
                    "Please inspect the configuration or API credentials."
                ),
                "contexts": [],
                "retrieved_ids": [],
                "metadata": {
                    "model": CHAT_MODEL,
                    "tokens_used": None,
                    "sources": [],
                    "query_type": "error",
                    "error": str(exc),
                },
            }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("How do I apply for a leave of absence?")
        print(resp)

    asyncio.run(test())
