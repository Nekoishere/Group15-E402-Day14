"""
Synthetic Data Generator (SDG) for AI Evaluation - Day 14
Reads documents from ChromaDB and uses GPT-4o-mini to generate 50+ QA test cases.
Follows the Lab 14 Checklist for few-shot prompting and metadata generation.
"""
import json
import asyncio
import os
import sys
import io
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load .env explicitly from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
from dotenv import load_dotenv
load_dotenv(dotenv_path)

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHROMA_DIR, CHROMA_COLLECTION

from openai import AsyncOpenAI

# ─── SDG CONFIG ──────────────────────────────────────────────────────────────
TARGET_CASES = 50
QUESTIONS_PER_CHUNK = 3    
MAX_CHUNKS = 20            
MAX_CONCURRENT = 5         
# ─────────────────────────────────────────────────────────────────────────────

# Initialize client after loading environment variables
try:
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"❌ Lỗi khởi tạo OpenAI Client. Check lại file .env: {e}")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert QA dataset creator for a Vietnamese university academic chatbot called VinLex.
Given a document chunk about university academic regulations, generate {n} distinct question-answer pair(s).

REQUIREMENTS:
1. Generate specific questions, their exact answers based ONLY on the chunk, and provide the correct chunk ID, difficulty, and category.
2. You must generate diverse difficulties: 
   - 'easy': direct factual question.
   - 'medium': multi-hop reasoning or synthesis from multiple sentences.
   - 'hard' ('adversarial'): tricky question, boundary condition, or slightly misleading wording to test the system's robustness (e.g. asking for something that is an exception).

CATEGORIES: 
Choose from: "registration", "graduation", "grading", "attendance", "discipline", or "other".

FEW-SHOT EXAMPLES:
[Good Example (Easy)]
Document Chunk: "Sinh viên vắng mặt trên quá 20% số tiết học của môn học sẽ không được dự thi cuối kỳ và nhận điểm F."
-> question: "Em nghỉ học 3 buổi trên tổng số 10 buổi của môn Toán thì có được thi cuối kỳ không?"
-> expected_answer: "Theo quy định, bạn đã vắng mặt 30% (quá 20%) số tiết học nên sẽ không được dự thi cuối kỳ và phải nhận điểm F."

[Hard Case Example (Adversarial/Boundary)]
Document Chunk: "Sinh viên có GPA dưới 2.0 trong 2 học kỳ liên tiếp sẽ bị cảnh báo học vụ. Nếu kỳ tiếp theo vẫn dưới 2.0 sẽ bị buộc thôi học, ngoại trừ sinh viên năm nhất đang trong giai đoạn thích nghi (có thể được gia hạn thêm 1 kỳ)."
-> question: "Em là sinh viên năm nhất bị điểm GPA 1.9 trong 3 kỳ liên tiếp. Em có bị đuổi học luôn không và tại sao?"
-> expected_answer: "Vì bạn là sinh viên năm nhất, đang trong giai đoạn thích nghi, bạn sẽ không bị buộc thôi học ngay lập tức mà có thể được xem xét gia hạn thêm 1 kỳ nữa mặc dù đã 3 kỳ GPA dưới 2.0."

Now, output ONLY a JSON array with exactly {n} objects for the provided chunk text. 
Each object must have exactly these keys:
{{
  "question": "<question text>",
  "expected_answer": "<concise, accurate answer from the chunk>",
  "difficulty": "easy" | "medium" | "hard",
  "category": "<one of the categories>"
}}
No extra text outside the JSON."""


async def fetch_chunks_from_chroma(num_chunks: int) -> list[dict]:
    """Fetch diverse document chunks directly from ChromaDB."""
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()
        if count == 0:
            print("❌ ChromaDB collection is empty. Please ingest documents first.")
            return []

        all_ids_result = collection.get(include=[])
        all_ids = all_ids_result.get("ids", [])
        sample_ids = random.sample(all_ids, min(num_chunks, len(all_ids)))

        result = collection.get(
            ids=sample_ids,
            include=["documents", "metadatas"],
        )

        chunks = []
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        for chunk_id, doc, meta in zip(ids, docs, metas):
            if doc and len(doc.strip()) > 100:
                chunks.append({"id": chunk_id, "text": doc, "metadata": meta})

        print(f"✅ Fetched {len(chunks)} chunks from ChromaDB (total: {count})")
        return chunks

    except Exception as e:
        print(f"❌ Error fetching from ChromaDB (Môi trường python có thể bị lỗi thư viện): {e}")
        # Return some fallback chunks to allow testing if Chroma fails
        print("⚠️ Sử dụng Fallback Chunks để generate test cases...")
        return [
            {
                "id": "fallback-chunk-01",
                "text": "Điều 15. Cảnh báo học vụ. Sinh viên bị cảnh báo học vụ nếu điểm trung bình chung học kỳ đạt dưới 1.0 đối với học kỳ đầu tiên khóa học, dưới 1.2 đối với các học kỳ tiếp theo.",
                "metadata": {"source": "Quy-che-dao-tao.pdf", "page": 5}
            },
            {
                "id": "fallback-chunk-02",
                "text": "Điều 17. Nghỉ ốm và vắng mặt. Sinh viên vắng mặt vì lý do ốm đau phải nộp giấy khám bệnh hợp lệ cho Phòng Đào tạo trong vòng 7 ngày kể từ ngày nghỉ đầu tiên. Quá hạn sẽ không được chấp nhận.",
                "metadata": {"source": "Quy-che-dao-tao.pdf", "page": 7}
            },
            {
                "id": "fallback-chunk-03",
                "text": "Điều 20. Học phí. Sinh viên phải nộp học phí đúng hạn quy định. Trễ hạn nộp học phí quá 30 ngày sẽ bị đình chỉ học tập tạm thời. Mọi khiếu nại về học phí nộp về Phòng Tài chính.",
                "metadata": {"source": "So-tay-sinh-vien.pdf", "page": 10}
            }
        ] * 4  # Repeat to have enough chunks to generate dataset


async def generate_qa_from_chunk(
    chunk: dict,
    semaphore: asyncio.Semaphore,
    n: int = QUESTIONS_PER_CHUNK,
) -> list[dict]:
    """Use GPT-4o-mini to generate QA pairs."""
    async with semaphore:
        prompt = SYSTEM_PROMPT.format(n=n)
        user_content = f"Document chunk (ID: {chunk['id']}):\n\"\"\"\n{chunk['text'][:2000]}\n\"\"\""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)

            qa_list = []
            if isinstance(parsed, list):
                qa_list = parsed
            else:
                qa_list = next((v for v in parsed.values() if isinstance(v, list)), [])

            results = []
            for qa in qa_list[:n]:
                if "question" in qa and "expected_answer" in qa:
                    results.append({
                        "question": qa["question"],
                        "expected_answer": qa["expected_answer"],
                        "context": chunk["text"][:500],
                        "expected_retrieval_ids": [chunk["id"]],
                        "metadata": {
                            "difficulty": qa.get("difficulty", "medium"),
                            "category": qa.get("category", "other"),
                            "source_chunk_id": chunk["id"],
                            "source_metadata": chunk.get("metadata", {}),
                        },
                    })
            return results

        except Exception as e:
            print(f"  ⚠️ Failed to generate QA for chunk {chunk['id'][:20]}: {e}")
            return []


async def main():
    print("🚀 Synthetic Data Generator – Day 14 AI Evaluation")
    print(f"Target: {TARGET_CASES} QA pairs from up to {MAX_CHUNKS} chunks\n")

    chunks = await fetch_chunks_from_chroma(MAX_CHUNKS)
    if not chunks:
        print("❌ No chunks available. Exiting.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [generate_qa_from_chunk(chunk, semaphore) for chunk in chunks]

    print(f"⚙️  Generating questions from {len(chunks)} chunks (async, batch={MAX_CONCURRENT})...")
    all_results = await asyncio.gather(*tasks)

    qa_pairs: list[dict] = []
    for batch in all_results:
        qa_pairs.extend(batch)
    random.shuffle(qa_pairs)

    if len(qa_pairs) < TARGET_CASES and chunks:
        print(f"\n🔄 Only got {len(qa_pairs)} cases so far, generating more...")
        extra_needed = TARGET_CASES - len(qa_pairs)
        extra_chunks = random.choices(chunks, k=max(extra_needed // QUESTIONS_PER_CHUNK + 1, 1))
        extra_tasks = [generate_qa_from_chunk(chunk, semaphore, n=QUESTIONS_PER_CHUNK) for chunk in extra_chunks]
        extra_results = await asyncio.gather(*extra_tasks)
        for batch in extra_results:
            qa_pairs.extend(batch)
        random.shuffle(qa_pairs)

    final_pairs = qa_pairs[:max(TARGET_CASES, len(qa_pairs))]

    output_path = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in final_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(final_pairs)} QA pairs to {output_path}")

    difficulties = [p["metadata"]["difficulty"] for p in final_pairs]
    print(f"📊 Difficulty breakdown: Easy={difficulties.count('easy')}, "
          f"Medium={difficulties.count('medium')}, Hard={difficulties.count('hard')}")


if __name__ == "__main__":
    asyncio.run(main())
