"""
평가 실행 CLI
Hybrid vs Naive RAG 비교 평가 스크립트
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import RAGEvaluator


def load_test_queries(query_file: str = None) -> list:
    """테스트 쿼리 로드"""
    if query_file is None:
        query_file = Path(__file__).parent.parent / "src" / "evaluation" / "test_queries.json"

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [q["query"] for q in data["queries"]]


def main():
    parser = argparse.ArgumentParser(
        description="RAG 시스템 평가 - Hybrid vs Naive 비교"
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1.0",
        help="프롬프트 버전 태그 (기본: v1.0)"
    )
    parser.add_argument(
        "--doc-types",
        type=str,
        nargs="+",
        default=["patent"],
        choices=["patent", "article", "project"],
        help="평가할 문서 타입 (기본: patent)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색 결과 수 (기본: 5)"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="테스트 쿼리 파일 경로 (기본: test_queries.json)"
    )
    parser.add_argument(
        "--single-query",
        type=str,
        default=None,
        help="단일 쿼리 테스트 (배치 대신)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 저장 안함"
    )

    args = parser.parse_args()

    # Evaluator 초기화
    print(f"\n{'='*60}")
    print("RAG Evaluation System")
    print(f"{'='*60}")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Doc types: {args.doc_types}")
    print(f"Top-K: {args.top_k}")

    evaluator = RAGEvaluator(
        doc_types=args.doc_types,
        prompt_version=args.prompt_version
    )

    if args.single_query:
        # 단일 쿼리 평가
        print(f"\nSingle query evaluation: {args.single_query}")
        result = evaluator.evaluate_single(args.single_query, top_k=args.top_k)

        print(f"\n{'='*60}")
        print("RESULT")
        print(f"{'='*60}")
        print(f"\nQuery: {result['query']}")
        print(f"\nHybrid RAG:")
        print(f"  Answer: {result['hybrid']['answer'][:200]}...")
        print(f"  Scores: {result['hybrid']['scores']}")
        print(f"\nNaive RAG:")
        print(f"  Answer: {result['naive']['answer'][:200]}...")
        print(f"  Scores: {result['naive']['scores']}")
        print(f"\nImprovement: {result['comparison']}")

    else:
        # 배치 평가
        queries = load_test_queries(args.query_file)
        print(f"\nLoaded {len(queries)} test queries")

        summary = evaluator.evaluate_batch(
            queries=queries,
            top_k=args.top_k,
            save=not args.no_save
        )


if __name__ == "__main__":
    main()
