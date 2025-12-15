"""
벡터 저장소
임베딩을 저장하고 검색하는 기능을 제공합니다.
"""

import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class VectorStore:
    """벡터 임베딩을 저장하고 검색하는 클래스"""
    
    def __init__(self, store_path: str = "data/rag_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.vectors = []
        self.metadata = []
    
    def load_embeddings(self, embeddings_file: str):
        """
        저장된 임베딩을 불러옵니다.
        
        Args:
            embeddings_file: 임베딩 파일 경로
        """
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if "embedding" in item:
                self.vectors.append(np.array(item["embedding"]))
                # 메타데이터 저장 (임베딩 제외)
                metadata = {k: v for k, v in item.items() if k != "embedding"}
                self.metadata.append(metadata)
        
        print(f"{len(self.vectors)}개의 벡터를 로드했습니다.")
    
    def add_vectors(self, embeddings: List[Dict]):
        """
        벡터를 저장소에 추가합니다.
        
        Args:
            embeddings: 임베딩이 포함된 데이터 리스트
        """
        for item in embeddings:
            if "embedding" in item:
                self.vectors.append(np.array(item["embedding"]))
                metadata = {k: v for k, v in item.items() if k != "embedding"}
                self.metadata.append(metadata)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        유사한 벡터를 검색합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 상위 k개 결과
            
        Returns:
            검색 결과 리스트
        """
        if not self.vectors:
            return []
        
        query_vec = np.array(query_embedding)
        
        # 코사인 유사도 계산
        similarities = []
        for vec in self.vectors:
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            similarities.append(similarity)
        
        # 상위 k개 인덱스
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.metadata[idx],
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def save_store(self, filename: str = "vector_store.json"):
        """
        벡터 저장소를 파일로 저장합니다.
        
        Args:
            filename: 저장할 파일명
        """
        output_file = self.store_path / filename
        
        store_data = {
            "vectors": [vec.tolist() for vec in self.vectors],
            "metadata": self.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(store_data, f, ensure_ascii=False, indent=2)
        
        print(f"벡터 저장소가 저장되었습니다: {output_file}")


if __name__ == "__main__":
    # 예시 사용법
    store = VectorStore()
    
    # 샘플 임베딩 데이터 로드
    sample_embeddings = [
        {
            "name": "홍길동",
            "department": "컴퓨터공학과",
            "expertise": ["인공지능", "머신러닝"],
            "embedding": np.random.rand(384).tolist()
        }
    ]
    
    store.add_vectors(sample_embeddings)
    store.save_store()

