import os
from openai import OpenAI
from dotenv import load_dotenv

from config import CHAT_MODEL, FINANCIAL_KEYWORDS, MENTAL_HEALTH_KEYWORDS, OFFENSIVE_KEYWORDS, CHROMA_DIR
from backend.vector_store import VectorStore
from backend.rag import RAGPipeline

load_dotenv()


_FINANCIAL_RESPONSE = {
    "vi": (
        "Câu hỏi của bạn liên quan đến **tài chính, học phí hoặc học bổng**. "
        "Đây là lĩnh vực nhạy cảm mà tôi không được phép cung cấp thông tin cụ thể để tránh sai sót.\n\n"
        "Vui lòng liên hệ trực tiếp với **Phòng Đào Tạo** để được tư vấn chính xác "
        "(xem tab **Liên Hệ** trên thanh menu)."
    ),
    "en": (
        "Your question relates to **finances, tuition fees, or scholarships**. "
        "This is a sensitive area where I avoid providing specific figures to prevent inaccuracies.\n\n"
        "Please contact the **Registrar's Office** directly for accurate information "
        "(see the **Contact** tab in the menu)."
    ),
}

_COUNSELING_RESPONSE = {
    "vi": (
        "💙 Tôi nhận thấy bạn có vẻ đang trải qua giai đoạn khó khăn. "
        "Sức khỏe tinh thần của bạn rất quan trọng.\n\n"
        "Vui lòng liên hệ ngay với **Bộ phận Tư vấn Tâm lý VinUniversity** hoặc "
        "**Phòng Đào Tạo** (tab **Liên Hệ**) để được hỗ trợ kịp thời. "
        "Bạn không cần phải đối mặt một mình với điều này."
    ),
    "en": (
        "💙 I can sense you might be going through a tough time. Your mental health matters.\n\n"
        "Please reach out to **VinUniversity's Student Counseling** or the "
        "**Registrar's Office** (Contact tab) for support. You don't have to face this alone."
    ),
}

_OFFENSIVE_RESPONSE = {
    "vi": "Vui lòng sử dụng ngôn ngữ lịch sự. Tôi là trợ lý học thuật VinLex, sẵn sàng hỗ trợ bạn về quy chế nếu bạn trao đổi một cách tôn trọng.",
    "en": "Please use polite language. I am VinLex, an academic assistant ready to help you with university regulations if we communicate respectfully.",
}

# System prompt for the general conversational / greeting path
_CONVERSATIONAL_SYSTEM = """You are VinLex AI, a friendly academic assistant for VinUniversity students.

Your personality: warm, helpful, and knowledgeable about university life and academic regulations.

When greeting or having casual conversation:
- Be warm and natural, as if chatting with a student
- If the conversation has history, reference it naturally (e.g., "Based on what we discussed...")
- Briefly remind the student of what you can help with (academic regulations, policies, procedures)
- Keep responses concise — 2-4 sentences for greetings/small talk

You DO NOT answer questions about tuition fees, scholarships, or financial aid — direct those to the Registrar's Office.

IMPORTANT: Respond in the same language the student uses."""

_OUT_OF_SCOPE_VI = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi hỗ trợ của tôi. "
    "Tôi chuyên về **quy chế học vụ** của VinUniversity — đăng ký môn, tốt nghiệp, bảo lưu, thi cử, v.v.\n\n"
    "Bạn có câu hỏi nào về học vụ không? Tôi rất sẵn lòng giúp!"
)
_OUT_OF_SCOPE_EN = (
    "That's outside my area of expertise. I specialize in **VinUniversity academic regulations** — "
    "course registration, graduation, leave of absence, exams, and similar topics.\n\n"
    "Do you have any academic regulation questions I can help with?"
)


class VinLexChatbot:
    """Main chatbot orchestrator implementing the decision loop."""

    def __init__(self, generation_temperature: float = None, prompt_addon: str = None):
        self._vs = VectorStore(str(CHROMA_DIR))
        self._rag = RAGPipeline(
            self._vs,
            generation_temperature=generation_temperature,
            prompt_addon=prompt_addon,
        )
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def process(self, query: str, history: list[dict]) -> dict:
        """
        Main decision loop.
        Returns: {answer, query_type, sources, redirect_to_contact, suggest_counseling}
        """
        query_lower = query.lower()
        language = self._detect_language(query)

        # 0. Offensive language check (keyword-based, fast)
        if self._check_offensive(query_lower):
            return {
                "answer": _OFFENSIVE_RESPONSE[language],
                "query_type": "offensive",
                "sources": [],
                "redirect_to_contact": False,
                "suggest_counseling": False,
            }

        # 1. Mental health check (keyword-based, fast)
        if self._check_mental_health(query_lower):
            return {
                "answer": _COUNSELING_RESPONSE[language],
                "query_type": "mental_health",
                "sources": [],
                "redirect_to_contact": True,
                "suggest_counseling": True,
            }

        # 2. Financial check (keyword-based, fast)
        if self._check_financial(query_lower):
            return {
                "answer": _FINANCIAL_RESPONSE[language],
                "query_type": "financial",
                "sources": [],
                "redirect_to_contact": True,
                "suggest_counseling": False,
            }

        # 3. Classify intent via LLM (with conversation history for context)
        intent = self._classify_intent(query, history)

        # 4. Greeting / casual chat — use LLM with history for natural conversation
        if intent == "greeting":
            answer = self._conversational_reply(query, history, language)
            return {
                "answer": answer,
                "query_type": "greeting",
                "sources": [],
                "redirect_to_contact": False,
                "suggest_counseling": False,
            }

        # 5. Financial (caught by LLM classifier, not keywords)
        if intent == "financial":
            return {
                "answer": _FINANCIAL_RESPONSE[language],
                "query_type": "financial",
                "sources": [],
                "redirect_to_contact": True,
                "suggest_counseling": False,
            }

        # 6. Academic regulation OR out_of_scope → always try RAG first
        # The classifier may be wrong; if we find relevant documents, answer them.
        chunks = self._rag.retrieve(query)

        if intent == "out_of_scope" and not chunks:
            # Truly nothing found and clearly off-topic
            out_msg = _OUT_OF_SCOPE_EN if language == "en" else _OUT_OF_SCOPE_VI
            return {
                "answer": out_msg,
                "query_type": "out_of_scope",
                "sources": [],
                "redirect_to_contact": False,
                "suggest_counseling": False,
            }

        # 7. Generate answer (with or without chunks)
        result = self._rag.generate_answer(query, chunks, history, language)
        return {
            "answer": result["answer"],
            "query_type": "academic_regulation",
            "sources": result["sources"],
            "redirect_to_contact": False,
            "suggest_counseling": False,
        }

    # ── Conversational reply ─────────────────────────────────────

    def _conversational_reply(self, query: str, history: list[dict], language: str) -> str:
        """Generate a natural, context-aware greeting/casual response using LLM."""
        lang_note = (
            "Respond in ENGLISH only."
            if language == "en"
            else "Trả lời bằng TIẾNG VIỆT."
        )
        system = f"{_CONVERSATIONAL_SYSTEM}\n\n{lang_note}"

        messages = [{"role": "system", "content": system}]
        # Include recent history for context-aware replies
        for msg in history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content or ""
        except Exception:
            if language == "en":
                return "Hello! How can I help you with VinUniversity academic regulations today?"
            return "Xin chào! Tôi có thể giúp gì cho bạn về quy chế học vụ VinUniversity hôm nay?"

    # ── Intent classification ────────────────────────────────────

    def _classify_intent(self, query: str, history: list[dict]) -> str:
        """
        Classify query intent using gpt-4o-mini, with conversation history for context.
        Returns: 'academic_regulation' | 'financial' | 'greeting' | 'out_of_scope'
        Default fallback: 'academic_regulation' (safe — RAG will handle it)
        """
        # Build a brief history context (last 3 exchanges)
        recent = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        history_lines = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {m['content'][:150]}")
        history_ctx = "\n".join(history_lines)

        system = (
            "You are a query classifier for VinUniversity's academic chatbot.\n\n"
            "Classify the CURRENT question into EXACTLY ONE category:\n\n"
            "- academic_regulation: ANY question about university academic life, including:\n"
            "  course registration, credits, GPA, graduation, prerequisites, retakes,\n"
            "  academic probation, leave of absence, enrollment, exams, grading,\n"
            "  academic calendar, transfer credits, degree requirements, student status,\n"
            "  academic policies, student handbook, university procedures.\n"
            "  → When in doubt, choose this category.\n\n"
            "- financial: tuition fees, scholarship amounts, payment, financial aid,\n"
            "  fee waivers, refunds. Only if SPECIFICALLY about money.\n\n"
            "- greeting: hello, hi, thanks, goodbye, how are you, simple pleasantries.\n\n"
            "- out_of_scope: truly unrelated topics — weather, sports, cooking, entertainment,\n"
            "  personal advice unrelated to university, political topics.\n\n"
            "IMPORTANT: If the question is a follow-up to a previous academic discussion\n"
            "(e.g., 'what about...', 'and if...', 'so then...'), classify as academic_regulation.\n\n"
            "Respond with ONLY the category name, nothing else."
        )

        user_content = query
        if history_ctx:
            user_content = (
                f"[Recent conversation for context:]\n{history_ctx}\n\n"
                f"[Current question to classify:] {query}"
            )

        try:
            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=20,
            )
            result = response.choices[0].message.content.strip().lower()
            valid = {"academic_regulation", "financial", "greeting", "out_of_scope"}
            return result if result in valid else "academic_regulation"
        except Exception:
            return "academic_regulation"

    # ── Language detection ───────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Vietnamese or English.
        Vietnamese has many diacritical characters (non-ASCII).
        Returns 'vi' or 'en'.
        """
        if not text:
            return "vi"
        non_ascii = sum(1 for c in text if ord(c) > 127)
        # Vietnamese text typically >10% non-ASCII due to diacritics and tone marks.
        # English text is nearly all ASCII (even with some punctuation/symbols).
        return "vi" if non_ascii / len(text) > 0.08 else "en"

    # ── Keyword checks (fast, no LLM) ───────────────────────────

    def _check_offensive(self, query_lower: str) -> bool:
        return any(kw in query_lower for kw in OFFENSIVE_KEYWORDS)

    def _check_financial(self, query_lower: str) -> bool:
        return any(kw in query_lower for kw in FINANCIAL_KEYWORDS)

    def _check_mental_health(self, query_lower: str) -> bool:
        return any(kw in query_lower for kw in MENTAL_HEALTH_KEYWORDS)
