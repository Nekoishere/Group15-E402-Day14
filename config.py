import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PDFS_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONVS_DIR = DATA_DIR / "conversations"


# Ensure data directories exist
for _dir in [PDFS_DIR, CHROMA_DIR, CONVS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# OpenAI models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# RAG settings
CHUNK_SIZE_CHARS = 1800      # ~450 tokens at 4 chars/token
CHUNK_OVERLAP_CHARS = 200
TOP_K = 5
RETRIEVAL_DISTANCE_THRESHOLD = 0.75  # higher = less relevant; filter out if distance > this

# Conversation memory window (last N messages sent to API)
MEMORY_WINDOW = 10

# ChromaDB collection name
CHROMA_COLLECTION = "vinlex_regulations"

# Hardcoded suggested questions for the chat home screen
SUGGESTED_QUESTIONS = [
    "Điều kiện để đăng ký học vượt môn là gì?",
    "Quy trình xin bảo lưu kết quả học tập?",
    "Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?",
    "Quy định về điểm F và học lại như thế nào?",
    "What are the prerequisites for advanced courses?",
    "How do I apply for a leave of absence?",
]

# Keywords for pre-classification (bypass LLM call)
FINANCIAL_KEYWORDS = [
    "học phí", "học bổng", "tài chính", "tiền", "phí", "miễn giảm",
    "scholarship", "tuition", "fee", "financial aid", "payment", "cost",
    "chi phí", "nộp tiền", "hoàn trả", "refund",
]

MENTAL_HEALTH_KEYWORDS = [
    "trầm cảm", "tự tử", "không muốn sống", "căng thẳng quá mức",
    "muốn chết", "tuyệt vọng", "depression", "suicide", "self-harm",
    "không còn muốn cố gắng", "bỏ cuộc", "overwhelmed",
]

OFFENSIVE_KEYWORDS = [
    "ngu", "dốt", "đần", "ngu ngốc", "vô học", "clgt", "đm", "vcl",
    "chửi", "fuck", "stupid", "idiot", "bitch", "shit",
]


