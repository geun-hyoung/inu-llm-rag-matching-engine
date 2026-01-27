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
from config.settings import RESULTS_DIR


def save_query_result(result: dict):
    """
    검색 결과를 JSON 파일로 저장

    구조: query + 문서별 매칭 정보 (1-hop 포함)
    """
    # 고정 경로: results/test/rag/test_rag.json
    filepath = Path(RESULTS_DIR) / "test" / "rag" / "test_rag.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 문서별로 매칭 정보 구성 (임시 dict → 최종 list 변환)
    docs_dict = {}

    # local_results 처리
    for r in result['local_results']:
        no = str(r.get('metadata', {}).get('source_doc_id', ''))
        if not no:
            continue

        doc_type = r.get('doc_type', 'unknown')

        if no not in docs_dict:
            docs_dict[no] = {
                "no": no,
                "data_type": doc_type,
                "matches": []
            }

        # 매칭 정보 구성
        match_info = {
            "search_type": "local",
            "similarity": r.get('similarity', 0),
            "matched_entity": {
                "name": r.get('metadata', {}).get('name', ''),
                "entity_type": r.get('metadata', {}).get('entity_type', ''),
                "description": r.get('document', '')
            },
            "neighbors_1hop": [
                {
                    "name": n.get('name', ''),
                    "entity_type": n.get('entity_type', ''),
                    "relation_keywords": n.get('relation_keywords', []),
                    "relation_description": n.get('relation_description', '')
                }
                for n in r.get('neighbors', [])
            ]
        }
        docs_dict[no]["matches"].append(match_info)

    # global_results 처리
    for r in result['global_results']:
        no = str(r.get('metadata', {}).get('source_doc_id', ''))
        if not no:
            continue

        doc_type = r.get('doc_type', 'unknown')

        if no not in docs_dict:
            docs_dict[no] = {
                "no": no,
                "data_type": doc_type,
                "matches": []
            }

        # 매칭 정보 구성
        match_info = {
            "search_type": "global",
            "similarity": r.get('similarity', 0),
            "matched_relation": {
                "source_entity": r.get('metadata', {}).get('source_entity', ''),
                "target_entity": r.get('metadata', {}).get('target_entity', ''),
                "keywords": r.get('metadata', {}).get('keywords', ''),
                "description": r.get('document', '')
            },
            "source_entity_info": r.get('source_entity_info'),
            "target_entity_info": r.get('target_entity_info')
        }
        docs_dict[no]["matches"].append(match_info)

    # matches 내부도 similarity 기준 내림차순 정렬
    for doc in docs_dict.values():
        doc['matches'] = sorted(
            doc['matches'],
            key=lambda m: m.get('similarity', 0),
            reverse=True
        )

    # dict → list 변환 후 similarity 기준 내림차순 정렬
    # 각 문서의 최고 similarity로 정렬
    retrieved_docs = sorted(
        docs_dict.values(),
        key=lambda doc: max((m.get('similarity', 0) for m in doc['matches']), default=0),
        reverse=True
    )

    # JSON 구조 생성
    output = {
        "query": result['query'],
        "keywords": {
            "high_level": result['high_level_keywords'],
            "low_level": result['low_level_keywords']
        },
        "retrieved_docs": retrieved_docs
    }

    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    return filepath


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
                    output.append(f"   1-hop Neighbors: {[n['name'] for n in neighbors[:5]]}")
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
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Local/Global 검색 시 각각 가져올 개수 (기본: settings.RETRIEVAL_TOP_K=10)"
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help="최종 병합 후 반환할 개수 (기본: settings.FINAL_TOP_K=5)"
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
        retrieval_top_k=args.retrieval_top_k,
        mode=args.mode
    )

    # 결과를 JSON 파일로 저장
    save_query_result(results)

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
