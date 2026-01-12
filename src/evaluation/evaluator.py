"""
RAG Evaluator
Hybrid vs Naive 비교 평가 + 프롬프트 버전 관리
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RESULTS_DIR

from src.rag.query.retriever import HybridRetriever
from src.rag.query.naive_retriever import NaiveRetriever
from src.evaluation.metrics import evaluate_all


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(
        self,
        doc_types: List[str] = None,
        prompt_version: str = "v1.0"
    ):
        """
        Args:
            doc_types: 검색할 문서 타입
            prompt_version: 프롬프트 버전 태그
        """
        self.doc_types = doc_types or ["patent"]
        self.prompt_version = prompt_version

        # Retriever 초기화
        print("Initializing retrievers...")
        self.hybrid_retriever = HybridRetriever(doc_types=self.doc_types)
        self.naive_retriever = NaiveRetriever(doc_types=self.doc_types)

        # 결과 저장 경로
        self.results_dir = Path(RESULTS_DIR) / "experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _extract_contexts(self, results: Dict, retriever_type: str) -> List[str]:
        """검색 결과에서 컨텍스트 텍스트 추출"""
        contexts = []

        if retriever_type == "hybrid":
            for r in results.get("merged_results", []):
                doc = r.get("document", "")
                if doc:
                    contexts.append(doc)
        else:  # naive
            for r in results.get("results", []):
                doc = r.get("document", "")
                if doc:
                    contexts.append(doc)

        return contexts

    def evaluate_single(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        단일 쿼리 평가

        Args:
            query: 평가할 쿼리
            top_k: 검색 결과 수

        Returns:
            평가 결과 딕셔너리
        """
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "prompt_version": self.prompt_version,
            "top_k": top_k
        }

        # Hybrid RAG 평가
        print(f"\n[Hybrid] Evaluating: {query[:50]}...")
        hybrid_results = self.hybrid_retriever.query(query, top_k=top_k, generate=True)
        hybrid_contexts = self._extract_contexts(hybrid_results, "hybrid")
        hybrid_answer = hybrid_results.get("response", "")

        hybrid_scores = evaluate_all(query, hybrid_contexts, hybrid_answer)
        result["hybrid"] = {
            "scores": hybrid_scores,
            "answer": hybrid_answer,
            "context_count": len(hybrid_contexts),
            "keywords": {
                "high_level": hybrid_results.get("high_level_keywords", []),
                "low_level": hybrid_results.get("low_level_keywords", [])
            }
        }
        print(f"  Scores: {hybrid_scores}")

        # Naive RAG 평가
        print(f"[Naive] Evaluating: {query[:50]}...")
        naive_results = self.naive_retriever.query(query, top_k=top_k, generate=True)
        naive_contexts = self._extract_contexts(naive_results, "naive")
        naive_answer = naive_results.get("response", "")

        naive_scores = evaluate_all(query, naive_contexts, naive_answer)
        result["naive"] = {
            "scores": naive_scores,
            "answer": naive_answer,
            "context_count": len(naive_contexts)
        }
        print(f"  Scores: {naive_scores}")

        # 비교 (Hybrid - Naive)
        result["comparison"] = {
            metric: hybrid_scores[metric] - naive_scores[metric]
            for metric in hybrid_scores
        }

        return result

    def evaluate_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        save: bool = True
    ) -> Dict:
        """
        배치 쿼리 평가

        Args:
            queries: 평가할 쿼리 리스트
            top_k: 검색 결과 수
            save: 결과 저장 여부

        Returns:
            전체 평가 결과
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {len(queries)} queries...")
        print(f"Prompt version: {self.prompt_version}")
        print(f"{'='*60}")

        all_results = []
        hybrid_totals = {"context_relevance": 0, "context_precision": 0, "faithfulness": 0, "answer_relevance": 0}
        naive_totals = {"context_relevance": 0, "context_precision": 0, "faithfulness": 0, "answer_relevance": 0}

        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {query[:50]}...")
            result = self.evaluate_single(query, top_k=top_k)
            all_results.append(result)

            # 합계 누적
            for metric in hybrid_totals:
                hybrid_totals[metric] += result["hybrid"]["scores"][metric]
                naive_totals[metric] += result["naive"]["scores"][metric]

        # 평균 계산
        n = len(queries)
        hybrid_avg = {k: v / n for k, v in hybrid_totals.items()}
        naive_avg = {k: v / n for k, v in naive_totals.items()}

        summary = {
            "experiment_info": {
                "prompt_version": self.prompt_version,
                "doc_types": self.doc_types,
                "top_k": top_k,
                "query_count": len(queries),
                "timestamp": datetime.now().isoformat()
            },
            "summary": {
                "hybrid_avg": hybrid_avg,
                "naive_avg": naive_avg,
                "improvement": {
                    metric: hybrid_avg[metric] - naive_avg[metric]
                    for metric in hybrid_avg
                }
            },
            "results": all_results
        }

        # 결과 출력
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"\nHybrid RAG (avg):")
        for metric, score in hybrid_avg.items():
            print(f"  {metric}: {score:.3f}")

        print(f"\nNaive RAG (avg):")
        for metric, score in naive_avg.items():
            print(f"  {metric}: {score:.3f}")

        print(f"\nImprovement (Hybrid - Naive):")
        for metric, diff in summary["summary"]["improvement"].items():
            sign = "+" if diff > 0 else ""
            print(f"  {metric}: {sign}{diff:.3f}")

        # 결과 저장
        if save:
            filename = f"eval_{self.prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {filepath}")

        return summary

    def compare_versions(
        self,
        version_files: List[str]
    ) -> Dict:
        """
        여러 프롬프트 버전의 결과 비교

        Args:
            version_files: 비교할 결과 파일 경로 리스트

        Returns:
            버전별 비교 결과
        """
        versions = {}

        for filepath in version_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            version = data["experiment_info"]["prompt_version"]
            versions[version] = data["summary"]["hybrid_avg"]

        # 비교 테이블 출력
        print(f"\n{'='*60}")
        print("VERSION COMPARISON")
        print(f"{'='*60}")

        metrics = ["context_relevance", "context_precision", "faithfulness", "answer_relevance"]

        # 헤더
        header = "Metric".ljust(20)
        for version in versions:
            header += version.center(12)
        print(header)
        print("-" * len(header))

        # 각 메트릭별 점수
        for metric in metrics:
            row = metric.ljust(20)
            for version, scores in versions.items():
                row += f"{scores[metric]:.3f}".center(12)
            print(row)

        return versions


if __name__ == "__main__":
    # 테스트
    print("Testing RAGEvaluator...")

    evaluator = RAGEvaluator(
        doc_types=["patent"],
        prompt_version="v1.0"
    )

    # 단일 쿼리 테스트
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    result = evaluator.evaluate_single(test_query, top_k=5)

    print("\n=== Single Query Result ===")
    print(f"Hybrid scores: {result['hybrid']['scores']}")
    print(f"Naive scores: {result['naive']['scores']}")
    print(f"Improvement: {result['comparison']}")
