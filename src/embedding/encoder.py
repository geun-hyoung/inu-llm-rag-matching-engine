"""
텍스트 임베딩 생성기
"""

import json
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_PROVIDER, PROCESSED_DATA_DIR


class Embedder:
    """텍스트를 벡터 임베딩으로 변환하는 클래스"""
    
    def __init__(self, model_name: str = None, output_dir: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM
        self.provider = EMBEDDING_PROVIDER
        self.output_dir = Path(output_dir or PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_text_representation(self, item: Dict) -> str:
        """
        데이터 항목을 텍스트 표현으로 변환합니다.
        
        Args:
            item: 데이터 항목
            
        Returns:
            텍스트 표현
        """
        parts = []
        
        if "name" in item:
            parts.append(f"이름: {item['name']}")
        if "department" in item:
            parts.append(f"학과: {item['department']}")
        if "expertise" in item:
            expertise = ", ".join(item["expertise"]) if isinstance(item["expertise"], list) else item["expertise"]
            parts.append(f"전문분야: {expertise}")
        if "research_areas" in item:
            research = ", ".join(item["research_areas"]) if isinstance(item["research_areas"], list) else item["research_areas"]
            parts.append(f"연구분야: {research}")
        
        return " | ".join(parts)
    
    def generate_embeddings(self, data: List[Dict]) -> List[Dict]:
        """
        데이터를 임베딩으로 변환합니다.
        
        Args:
            data: 데이터 리스트
            
        Returns:
            임베딩이 포함된 데이터 리스트
        """
        embedded_data = []
        
        for item in data:
            text = self.create_text_representation(item)
            # 실제로는 여기서 임베딩 모델을 사용하여 벡터 생성
            # 예시로 더미 벡터 생성 (실제 구현 시 모델 사용)
            embedding = np.random.rand(self.embedding_dim).tolist()
            
            embedded_item = {
                **item,
                "text_representation": text,
                "embedding": embedding
            }
            embedded_data.append(embedded_item)
        
        return embedded_data
    
    def save_embeddings(self, source: str, data: List[Dict]) -> str:
        """
        임베딩을 저장합니다.
        
        Args:
            source: 데이터 소스 이름
            data: 임베딩이 포함된 데이터
            
        Returns:
            저장된 파일 경로
        """
        output_file = self.output_dir / f"{source}_embeddings.json"
        
        # 임베딩은 큰 배열이므로 별도로 저장하는 것이 좋을 수 있음
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"임베딩이 저장되었습니다: {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # 예시 사용법
    embedder = Embedder()
    
    # 샘플 데이터
    sample_data = [
        {
            "name": "홍길동",
            "department": "컴퓨터공학과",
            "expertise": ["인공지능", "머신러닝"],
            "research_areas": ["딥러닝", "자연어처리"]
        }
    ]
    
    embedded = embedder.generate_embeddings(sample_data)
    embedder.save_embeddings("sample_faculty", embedded)