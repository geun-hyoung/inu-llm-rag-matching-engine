"""
Query Time CLI
HybridRetriever를 사용한 검색 실행 스크립트
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.query.retriever import HybridRetriever


def format_result(result: dict, verbose: bool = False) -> str:
    """검색 결과 포맷팅"""
    output = []

    # 키워드 정보
    output.append(f"\n{'='*60}")
    output.append(f"Query: {result['query']}")
    output.append(f"Mode: {result['mode']}")
    output.append(f"{'='*60}")
    output.append(f"\nHigh-level keywords: {result['high_level_keywords']}")
    output.append(f"Low-level keywords: {result['low_level_keywords']}")

    # 결과 요약
    output.append(f"\n--- Results Summary ---")
    output.append(f"{result}")
    output.append(f"Local results: {len(result['local_results'])}")
    output.append(f"Global results: {len(result['global_results'])}")
    output.append(f"Merged results: {len(result['merged_results'])}")

    # 병합된 결과 출력
    output.append(f"\n--- Merged Results ---")
    for i, r in enumerate(result['merged_results']):
        search_type = r.get('search_type', 'unknown')
        metadata = r.get('metadata', {})
        similarity = r.get('similarity', 0)

        if search_type == 'local':
            # 엔티티 결과
            name = metadata.get('name', 'N/A')
            entity_type = metadata.get('entity_type', 'N/A')
            output.append(f"\n{i+1}. [LOCAL] {name} ({entity_type})")
            output.append(f"   Similarity: {similarity:.4f}")
            output.append(f"   Source: {metadata.get('source_doc_id', 'N/A')}")

            if verbose and 'neighbors' in r:
                neighbors = r['neighbors']
                if neighbors:
                    output.append(f"   Neighbors: {[n['name'] for n in neighbors[:3]]}")
        else:
            # 관계 결과
            source = metadata.get('source_entity', 'N/A')
            target = metadata.get('target_entity', 'N/A')
            keywords = metadata.get('keywords', 'N/A')
            output.append(f"\n{i+1}. [GLOBAL] {source} → {target}")
            output.append(f"   Keywords: {keywords}")
            output.append(f"   Similarity: {similarity:.4f}")
            output.append(f"   Source: {metadata.get('source_doc_id', 'N/A')}")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Query Time - HybridRetriever 검색 실행"
    )
    parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리 (예: '딥러닝 의료영상 전문가')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "local", "global"],
        help="검색 모드 (기본: hybrid)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="반환할 결과 수 (기본: 5)"
    )
    parser.add_argument(
        "--doc-types",
        type=str,
        nargs="+",
        default=["patent"],
        choices=["patent", "article", "project"],
        help="검색할 문서 타입 (기본: patent)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 형식으로 출력"
    )

    args = parser.parse_args()

    # Retriever 초기화
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(doc_types=args.doc_types)

    # 검색 실행
    print(f"\nSearching: '{args.query}'")
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode
    )

    # 결과 출력
    if args.json:
        # JSON 출력 (직렬화 가능한 부분만)
        output = {
            "query": results["query"],
            "mode": results["mode"],
            "high_level_keywords": results["high_level_keywords"],
            "low_level_keywords": results["low_level_keywords"],
            "result_count": {
                "local": len(results["local_results"]),
                "global": len(results["global_results"]),
                "merged": len(results["merged_results"])
            },
            "merged_results": [
                {
                    "search_type": r.get("search_type"),
                    "metadata": r.get("metadata"),
                    "similarity": r.get("similarity")
                }
                for r in results["merged_results"]
            ]
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(format_result(results, verbose=args.verbose))


if __name__ == "__main__":
    main()
