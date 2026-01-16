"""
NaiveRetriever
단순 벡터 검색 기반 Retriever (베이스라인 비교용)
- LLM 키워드 추출 없음
- 그래프 확장 없음
- 청크 벡터 검색만 수행
"""

import sys
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL, TOP_K_RESULTS
from src.rag.prompts import RAG_RESPONSE_PROMPT
from src.rag.embedding.embedder import Embedder
from src.rag.store.vector_store import ChromaVectorStore


class NaiveRetriever:
    """단순 벡터 검색 기반 Retriever (베이스라인)"""

    def __init__(
        self,
        doc_types: List[str] = None,
        force_api: bool = False
    ):
        """
        Naive 검색기 초기화

        Args:
            doc_types: 검색할 문서 타입 리스트 (기본: patent, article, project)
            force_api: OpenAI API 강제 사용 여부 (기본: False, Qwen3 사용)
        """
        self.doc_types = doc_types or ["patent", "article", "project"]

        # OpenAI 클라이언트 (응답 생성용)
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        self.llm_model = LLM_MODEL

        # 임베딩 모델 (기본적으로 Qwen3 사용)
        self.embedder = Embedder(force_api=force_api)

        # 벡터 저장소
        self.vector_store = ChromaVectorStore()

        print(f"NaiveRetriever initialized for doc_types: {self.doc_types}")

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict:
        """
        쿼리에 대한 청크 검색 (단순 벡터 검색)

        Args:
            query: 사용자 쿼리
            top_k: 반환할 결과 수 (기본: settings.TOP_K_RESULTS)

        Returns:
            검색 결과 딕셔너리
        """
        top_k = top_k or TOP_K_RESULTS

        # 쿼리 임베딩 (LLM 키워드 추출 없이 직접 임베딩)
        query_embedding = self.embedder.encode(query)

        # 청크 벡터 검색
        chunk_results = self.vector_store.search_chunks(
            query_embedding=query_embedding,
            doc_types=self.doc_types,
            top_k=top_k
        )

        # 결과에 search_type 추가
        for chunk in chunk_results:
            chunk["search_type"] = "naive"

        return {
            "query": query,
            "results": chunk_results,
            "mode": "naive"
        }

    def _format_context(self, results: List[Dict]) -> str:
        """
        검색 결과를 LLM 컨텍스트 형식으로 변환

        Args:
            results: 검색 결과 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        context_parts = []

        for i, r in enumerate(results):
            metadata = r.get("metadata", {})
            doc_id = metadata.get("doc_id", "N/A")
            title = metadata.get("title", "N/A")
            text = r.get("document", "")

            context_parts.append(
                f"[문서 {i+1}] {title}\n"
                f"  ID: {doc_id}\n"
                f"  내용: {text[:500]}..."  # 길이 제한
            )

        return "\n\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        contexts: List[Dict],
        response_type: str = "간결하게 2-3문장으로 답변"
    ) -> str:
        """
        검색 결과를 바탕으로 자연어 응답 생성

        Args:
            query: 사용자 쿼리
            contexts: 검색 결과 리스트
            response_type: 응답 형식 지정

        Returns:
            생성된 자연어 응답
        """
        if not contexts:
            return "검색 결과가 없습니다."

        # 컨텍스트 포맷팅
        context_data = self._format_context(contexts)

        # 프롬프트 생성
        prompt = RAG_RESPONSE_PROMPT.format(
            response_type=response_type,
            context_data=context_data
        )

        # 사용자 질문 추가
        full_prompt = f"{prompt}\n\n---Question---\n{query}"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Response generation error: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {e}"

    def query(
        self,
        query: str,
        top_k: int = None,
        generate: bool = True
    ) -> Dict:
        """
        검색 + 응답 생성 통합 메서드

        Args:
            query: 사용자 쿼리
            top_k: 검색 결과 수
            generate: 응답 생성 여부

        Returns:
            검색 결과 + 생성된 응답
        """
        # 검색 수행
        results = self.retrieve(query, top_k=top_k)

        # 응답 생성
        if generate:
            response = self.generate_response(
                query=query,
                contexts=results["results"]
            )
            results["response"] = response
        else:
            results["response"] = None

        return results


if __name__ == "__main__":
    # 테스트
    print("Testing NaiveRetriever...")

    retriever = NaiveRetriever(doc_types=["patent"])

    # 테스트 쿼리
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    print(f"\nQuery: {test_query}")

    # 검색 + 응답 생성
    results = retriever.query(test_query, top_k=5, generate=True)

    print(f"\nResults: {len(results['results'])}")
    for r in results['results']:
        print(f"  - {r['metadata'].get('title', 'N/A')}: {r['similarity']:.4f}")

    print("\n=== Generated Response ===")
    print(results['response'])
