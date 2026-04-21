import os
from openai import OpenAI
from dotenv import load_dotenv

from config import CHAT_MODEL, MEMORY_WINDOW, RETRIEVAL_DISTANCE_THRESHOLD
from backend.vector_store import VectorStore

load_dotenv()


# Base system prompt (language instruction is prepended dynamically)
_BASE_SYSTEM_PROMPT = """You are VinLex AI, an academic regulation assistant for VinUniversity.

## Your Role
Answer student questions based on the retrieved regulation documents provided in [CONTEXT].

## Absolute Rules

1. **Ground answers in [CONTEXT].** For specific VinUniversity policies, numbers, deadlines, and procedures — only use retrieved documents. For well-known general academic concepts (e.g., "what is a GPA?"), you may use general knowledge.

2. **Cite sources for all regulation-specific answers.** Format: `[Source: filename, page X]`

3. **When the documents don't cover the question**, respond:
   - If asked about a VinUni-specific policy not found: "I couldn't find this in the available documents. Please contact the Registrar's Office (Contact tab)."
   - If the question is general academic knowledge: answer from general knowledge without citation.

4. **Never make commitments or guarantees** about outcomes for specific students.

5. **Be conversational and helpful.** Remember the conversation history and refer back to earlier topics naturally when relevant. If the student's follow-up question refers to something discussed earlier, acknowledge it.

6. **Format:** Use markdown (bullets, bold, numbered lists) for clarity. Keep answers concise but complete.

7. **Financial/scholarship questions:** Do not answer. Redirect to Registrar's Office."""

_LANG_RULE_EN = (
    "🔴 LANGUAGE RULE — HIGHEST PRIORITY: You MUST write your ENTIRE response in ENGLISH. "
    "The source documents may be in Vietnamese — that is fine. "
    "Read and understand the Vietnamese content, then write your answer in English. "
    "Do NOT output any Vietnamese text in your response. English only."
)

_LANG_RULE_VI = (
    "🔴 QUY TẮC NGÔN NGỮ — ƯU TIÊN CAO NHẤT: Bạn PHẢI viết toàn bộ câu trả lời bằng TIẾNG VIỆT. "
    "Tài liệu nguồn có thể bằng tiếng Anh — không sao. "
    "Đọc và hiểu nội dung tiếng Anh, sau đó trả lời bằng tiếng Việt. "
    "KHÔNG viết tiếng Anh trong phần trả lời."
)

_NO_CONTEXT_VI = (
    "Tôi không tìm thấy thông tin này trong tài liệu hiện có của VinUniversity. "
    "Vui lòng liên hệ **Phòng Đào Tạo** (xem tab **Liên Hệ**) để được hỗ trợ trực tiếp."
)
_NO_CONTEXT_EN = (
    "I couldn't find this information in the available VinUniversity documents. "
    "Please contact the **Registrar's Office** (see the **Contact** tab) for direct assistance."
)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for academic regulation Q&A."""

    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        Filters out low-relevance results (distance > threshold).
        """
        chunks = self._vs.query(query)
        relevant = [c for c in chunks if c["distance"] <= RETRIEVAL_DISTANCE_THRESHOLD]
        return relevant

    def generate_answer(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict],
        language: str = "vi",
    ) -> dict:
        """
        Generate an answer using GPT-4o-mini.
        language: "vi" or "en" — enforced strictly in the system prompt.
        Returns {"answer": str, "sources": list[dict]}
        """
        # Build language-specific system prompt — language rule comes FIRST
        lang_rule = _LANG_RULE_EN if language == "en" else _LANG_RULE_VI
        system_prompt = f"{lang_rule}\n\n{_BASE_SYSTEM_PROMPT}"

        if not chunks:
            # No RAG context — still make LLM call so it can use history and
            # answer general academic questions from its own knowledge
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-MEMORY_WINDOW:]:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

            no_context_note = (
                "[NOTE: No relevant documents were retrieved for this query. "
                "If this is a general academic concept, answer from general knowledge. "
                "If this is a VinUniversity-specific policy, say you couldn't find it.]"
            )
            messages.append({"role": "user", "content": f"{query}\n\n{no_context_note}"})

            try:
                response = self._client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600,
                )
                answer = response.choices[0].message.content or (
                    _NO_CONTEXT_EN if language == "en" else _NO_CONTEXT_VI
                )
            except Exception:
                answer = _NO_CONTEXT_EN if language == "en" else _NO_CONTEXT_VI

            return {"answer": answer, "sources": []}

        context_block = self._build_context_block(chunks, language)
        sources = self._extract_sources(chunks)

        # Build message array with history
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-MEMORY_WINDOW:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Inject context into user message
        user_content = f"{query}\n\n{context_block}"
        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content or ""
        return {"answer": answer, "sources": sources}

    def _build_context_block(self, chunks: list[dict], language: str = "vi") -> str:
        """Format retrieved chunks into a context injection block."""
        header = "=== REFERENCE DOCUMENTS ===" if language == "en" else "=== TÀI LIỆU THAM KHẢO ==="
        src_label = "Source" if language == "en" else "Nguồn"
        page_label = "page" if language == "en" else "trang"
        section_label = "section" if language == "en" else "mục"

        lines = [header]
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            src = meta.get("source_pdf", "Document")
            page = meta.get("page", "?")
            section = meta.get("section", "")
            entry = f"[{src_label}: {src}, {page_label} {page}"
            if section:
                entry += f", {section_label}: {section}"
            entry += "]"
            lines.append(f"\n{entry}")
            lines.append(chunk["text"])
            lines.append("---")
        return "\n".join(lines)

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate and format source citations from chunks."""
        seen = set()
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = (meta.get("source_pdf", ""), meta.get("page", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "pdf_name": meta.get("source_pdf", "Document"),
                    "page": meta.get("page", ""),
                    "section": meta.get("section", ""),
                })
        return sources
