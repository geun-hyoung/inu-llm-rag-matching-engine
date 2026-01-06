"""
Embedding Model
Text embedding model implementation
"""


class EmbeddingModel:
    """임베딩 모델 클래스"""
    
    def __init__(self, model_name: str = None):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: 모델 이름
        """
        self.model_name = model_name
        # TODO: 모델 로드 로직 구현
    
    def encode(self, texts):
        """
        텍스트를 임베딩 벡터로 변환합니다.
        
        Args:
            texts: 텍스트 리스트 또는 단일 텍스트
            
        Returns:
            임베딩 벡터 또는 벡터 리스트
        """
        # TODO: 임베딩 생성 로직 구현
        pass

