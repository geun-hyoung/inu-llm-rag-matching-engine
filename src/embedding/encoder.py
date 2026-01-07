"""
텍스트 임베딩 생성기
EmbeddingModel을 감싸는 래퍼 클래스
"""

from typing import List, Union
import numpy as np

from .model import EmbeddingModel


class Embedder:
    """텍스트를 벡터 임베딩으로 변환하는 클래스"""

    def __init__(self, force_api: bool = False):
        """
        Embedder 초기화

        Args:
            force_api: True면 GPU 유무와 관계없이 OpenAI API 사용
        """
        self.model = EmbeddingModel(force_api=force_api)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 단일 텍스트 또는 텍스트 리스트

        Returns:
            임베딩 벡터 (numpy array)
        """
        return self.model.encode(texts)

    @property
    def dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.model.dimension

    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        return self.model.model_name


if __name__ == "__main__":
    # 테스트
    embedder = Embedder()
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # 엔티티/관계/질의 임베딩 테스트
    test_texts = ["딥러닝", "CNN", "딥러닝 uses CNN", "의료영상 전문가 찾아줘"]
    embeddings = embedder.encode(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")