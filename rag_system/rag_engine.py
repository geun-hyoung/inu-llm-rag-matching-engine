"""
RAG 엔진
벡터 저장소를 활용한 RAG 시스템 구현
"""

from typing import List, Dict
from .vector_store import VectorStore
from embedding.embedder import Embedder


class RAGEngine:
    """RAG 시스템을 구현하는 클래스"""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        쿼리에 대한 검색 결과를 반환합니다.
        
        Args:
            query_text: 검색 쿼리 텍스트
            top_k: 반환할 상위 k개 결과
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리를 임베딩으로 변환
        query_item = {"text": query_text}
        query_embedding = self.embedder.generate_embeddings([query_item])[0]["embedding"]
        
        # 벡터 저장소에서 검색
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    def add_documents(self, documents: List[Dict]):
        """
        문서를 RAG 시스템에 추가합니다.
        
        Args:
            documents: 추가할 문서 리스트
        """
        embeddings = self.embedder.generate_embeddings(documents)
        self.vector_store.add_vectors(embeddings)


if __name__ == "__main__":
    # 예시 사용법
    store = VectorStore()
    embedder = Embedder()
    rag = RAGEngine(store, embedder)
    
    # 문서 추가
    documents = [
        {
            "name": "홍길동",
            "department": "컴퓨터공학과",
            "expertise": ["인공지능", "머신러닝"]
        }
    ]
    
    rag.add_documents(documents)
    
    # 검색
    results = rag.query("인공지능 전문가", top_k=3)
    print(results)

