"""
LightRAG 설정 및 초기화
"""
from pathlib import Path
from functools import partial
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.base import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed

from models.model_config import EMBEDDING_MODEL, EMBEDDING_DIM, LLM_MODEL

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# LightRAG 데이터 저장 경로
LIGHTRAG_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "lightrag_data"


def get_lightrag_instance() -> LightRAG:
    """
    LightRAG 인스턴스 생성 및 반환
    """
    # OpenAI 임베딩 함수 설정
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        func=partial(openai_embed, model=EMBEDDING_MODEL),
    )

    rag = LightRAG(
        working_dir=str(LIGHTRAG_DATA_DIR),

        # LLM 설정 (엔티티/관계 추출용)
        llm_model_func=openai_complete,
        llm_model_name=LLM_MODEL,

        # 임베딩 설정 (벡터 검색용)
        embedding_func=embedding_func,

        # 한글 엔티티 추출 설정
        addon_params={
            "language": "Korean",
        }
    )

    return rag


if __name__ == "__main__":
    # 초기화 테스트
    print(f"LightRAG Data Dir: {LIGHTRAG_DATA_DIR}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print("Initializing LightRAG...")

    rag = get_lightrag_instance()
    print("[OK] LightRAG initialized successfully!")
