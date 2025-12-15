"""
모델 설정
"""

# 임베딩 모델 설정
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# LLM 모델 설정
LLM_MODEL = "gpt-3.5-turbo"  # 또는 다른 모델

# 모델 경로 설정
MODEL_CACHE_DIR = "models/cache"

