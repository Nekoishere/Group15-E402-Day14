import asyncio
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from backend.chatbot import VinLexChatbot
from config import CHAT_MODEL

load_dotenv()


class MainAgent:
    """Day 14 benchmark wrapper around the Day 06 VinLex runtime."""

    def __init__(
        self,
        generation_temperature: Optional[float] = None,
        prompt_addon: Optional[str] = None,
    ):
        self.name = "VinLex-Day14"
        self._chatbot = VinLexChatbot(
            generation_temperature=generation_temperature,
            prompt_addon=prompt_addon,
        )

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


_V2_PROMPT_ADDON = """\
## V2 Enhancement: Structured & Comprehensive Responses
- Provide well-structured answers using markdown: numbered steps for procedures, \
bullet points for requirements.
- Include EVERY specific detail (deadlines, GPA thresholds, credit counts) from \
the retrieved documents — do not summarise away numbers.
- Explicitly cite each factual claim: "According to [document name], ..."
- If the retrieved documents cover multiple aspects of the question, address each one."""


class MainAgentV2(MainAgent):
    """Optimized agent (V2): temperature=0 generation + structured-response prompt.

    V2 hypothesis: lower temperature and an explicit formatting/citation instruction
    produce more complete, accurate answers that score higher on the Multi-Judge.
    This is what the regression test validates.
    """

    def __init__(self):
        super().__init__(
            generation_temperature=0.0,
            prompt_addon=_V2_PROMPT_ADDON,
        )
        self.name = "VinLex-Day14-V2"


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("How do I apply for a leave of absence?")
        print(resp)

    asyncio.run(test())
