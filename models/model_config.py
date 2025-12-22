"""
모델 설정
"""

# 임베딩 모델 설정
# 로컬 모델 (GPU 환경)
# EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
# EMBEDDING_DIM = 4096

# API 모델 (GPU 없는 환경)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_PROVIDER = "openai"  # "openai" 또는 "local"

# LLM 모델 설정
LLM_MODEL = "gpt-4o-mini"

# 모델 경로 설정
MODEL_CACHE_DIR = "models/cache"

