"""
Query Time CLI
HybridRetriever를 사용한 검색 실행 스크립트
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.query.retriever import HybridRetriever
from config.settings import RESULTS_DIR


def build_doc_candidates(local_results: list, global_results: list) -> list:
    """
    AHP 입력용 문서 단위 정규화 결과 생성

    - source_doc_id 기준으로 그룹핑
    - 대표 similarity는 max 사용
    - 채널(local/global) 출현 정보 포함

    Args:
        local_results: local search 전체 결과
        global_results: global search 전체 결과

    Returns:
        문서 단위로 집계된 후보 리스트
    """
    doc_map = {}  # source_doc_id -> 집계 정보

    # local + global 전체 결과에서 집계
    all_results = []
    for r in local_results:
        all_results.append({**r, '_channel': 'local'})
    for r in global_results:
        all_results.append({**r, '_channel': 'global'})

    for r in all_results:
        doc_id = r.get('metadata', {}).get('source_doc_id')
        if not doc_id:
            continue

        doc_id = str(doc_id)
        similarity = r.get('similarity', 0)
        channel = r.get('_channel', 'unknown')
        search_type = r.get('search_type', channel)

        if doc_id not in doc_map:
            doc_map[doc_id] = {
                'source_doc_id': doc_id,
                'max_similarity': similarity,
                'similarities': [similarity],
                'hit_count': 0,
                'channels': set(),
                'local_hits': 0,
                'global_hits': 0,
                'best_match': None
            }

        doc_info = doc_map[doc_id]
        doc_info['similarities'].append(similarity)
        doc_info['hit_count'] += 1
        doc_info['channels'].add(channel)

        if channel == 'local':
            doc_info['local_hits'] += 1
        else:
            doc_info['global_hits'] += 1

        # max similarity 갱신 및 best_match 업데이트
        if similarity > doc_info['max_similarity']:
            doc_info['max_similarity'] = similarity

        if doc_info['best_match'] is None or similarity >= doc_info['best_match'].get('similarity', 0):
            if search_type == 'local':
                doc_info['best_match'] = {
                    'search_type': 'local',
                    'similarity': similarity,
                    'entity_name': r.get('metadata', {}).get('name', 'N/A'),
                    'entity_type': r.get('metadata', {}).get('entity_type', 'N/A')
                }
            else:
                doc_info['best_match'] = {
                    'search_type': 'global',
                    'similarity': similarity,
                    'source_entity': r.get('metadata', {}).get('source_entity', 'N/A'),
                    'target_entity': r.get('metadata', {}).get('target_entity', 'N/A'),
                    'keywords': r.get('metadata', {}).get('keywords', 'N/A')
                }

    # 결과 정리 및 정렬
    candidates = []
    for doc_id, info in doc_map.items():
        candidates.append({
            'source_doc_id': info['source_doc_id'],
            'max_similarity': info['max_similarity'],
            'hit_count': info['hit_count'],
            'channels': sorted(list(info['channels'])),
            'local_hits': info['local_hits'],
            'global_hits': info['global_hits'],
            'best_match': info['best_match']
        })

    # max_similarity 기준 내림차순 정렬
    candidates.sort(key=lambda x: x['max_similarity'], reverse=True)

    return candidates


def load_article_data():
    """원본 논문 데이터 로드"""
    try:
        with open('data/article/article_sample.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
        # doc_id를 키로 하는 딕셔너리로 변환
        article_dict = {}
        for article in articles:
            doc_id = article.get('no')
            if doc_id:
                article_dict[str(doc_id)] = article
        return article_dict
    except Exception as e:
        print(f"Warning: Could not load article data: {e}")
        return {}


def save_query_result(result: dict, article_data: dict, output_dir: str = None):
    """검색 결과를 JSON 파일로 저장 (LightRAG 단계별 정보 + 원본 데이터)"""
    if output_dir is None:
        output_dir = Path(RESULTS_DIR) / "experiments"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"query_result_{timestamp}.json"
    filepath = output_dir / filename

    # Top 5 결과에서 doc_id 추출
    doc_ids = set()
    for r in result['merged_results'][:5]:
        doc_id = r.get('metadata', {}).get('source_doc_id')
        if doc_id:
            doc_ids.add(str(doc_id))

    # 원본 논문 데이터 추가
    source_papers = {}
    for doc_id in doc_ids:
        if doc_id in article_data:
            paper = article_data[doc_id]
            source_papers[doc_id] = {
                "title": paper.get('THSS_NM', 'N/A'),
                "year": paper.get('YY', 'N/A'),
                "abstract": paper.get('abstract', 'N/A'),
                "professor": paper.get('professor_info', {}).get('NM', 'N/A'),
                "department": paper.get('professor_info', {}).get('HG_NM', 'N/A'),
                "college": paper.get('professor_info', {}).get('COLG_NM', 'N/A')
            }

    # JSON 구조 생성 (PDF 문서의 Query Time 단계에 맞춤)
    output = {
        "query_info": {
            "query": result['query'],
            "mode": result['mode'],
            "timestamp": datetime.now().isoformat()
        },
        "lightrag_process": {
            # Step 1: Keyword Extraction (LLM)
            "step1_keyword_extraction": {
                "description": "LLM을 통해 쿼리에서 Low-level/High-level 키워드 추출",
                "high_level_keywords": result['high_level_keywords'],
                "low_level_keywords": result['low_level_keywords']
            },
            # Step 2: Dual-level Retrieval (Vector DB)
            "step2_dual_level_retrieval": {
                "description": "Vector DB에서 Entity/Relation 유사도 검색",
                "low_level_retrieval": {
                    "description": "Low-level keywords → Entity name과 유사도 비교",
                    "count": len(result['local_results']),
                    "results": [
                        {
                            "name": r.get('metadata', {}).get('name', 'N/A'),
                            "entity_type": r.get('metadata', {}).get('entity_type', 'N/A'),
                            "similarity": r.get('similarity', 0),
                            "source_doc_id": r.get('metadata', {}).get('source_doc_id', 'N/A')
                        }
                        for r in result['local_results']
                    ]
                },
                "high_level_retrieval": {
                    "description": "High-level keywords → Relation keywords와 유사도 비교",
                    "count": len(result['global_results']),
                    "results": [
                        {
                            "source_entity": r.get('metadata', {}).get('source_entity', 'N/A'),
                            "target_entity": r.get('metadata', {}).get('target_entity', 'N/A'),
                            "keywords": r.get('metadata', {}).get('keywords', 'N/A'),
                            "similarity": r.get('similarity', 0),
                            "source_doc_id": r.get('metadata', {}).get('source_doc_id', 'N/A')
                        }
                        for r in result['global_results']
                    ]
                }
            },
            # Step 3: Graph Traversal (Graph DB) - 1-hop 확장
            "step3_graph_traversal": {
                "description": "Graph DB에서 1-hop 탐색하여 연결된 정보 수집",
                "low_level_1hop": {
                    "description": "Entity top-k → 연결된 Edge들 수집",
                    "results": [
                        {
                            "entity_name": r.get('metadata', {}).get('name', 'N/A'),
                            "connected_edges": [
                                {
                                    "name": n.get('name', 'N/A'),
                                    "entity_type": n.get('entity_type', 'N/A'),
                                    "relation_description": n.get('relation_description', ''),
                                    "relation_keywords": n.get('relation_keywords', [])
                                } for n in r.get('neighbors', [])
                            ]
                        }
                        for r in result['local_results'] if r.get('neighbors')
                    ]
                },
                "high_level_1hop": {
                    "description": "Relation top-k → src/tgt Node 정보 수집",
                    "results": [
                        {
                            "relation": f"{r.get('metadata', {}).get('source_entity', 'N/A')} → {r.get('metadata', {}).get('target_entity', 'N/A')}",
                            "source_entity_info": {
                                "name": r.get('source_entity_info', {}).get('name', 'N/A'),
                                "entity_type": r.get('source_entity_info', {}).get('entity_type', 'N/A'),
                                "description": r.get('source_entity_info', {}).get('description', 'N/A')
                            } if r.get('source_entity_info') else None,
                            "target_entity_info": {
                                "name": r.get('target_entity_info', {}).get('name', 'N/A'),
                                "entity_type": r.get('target_entity_info', {}).get('entity_type', 'N/A'),
                                "description": r.get('target_entity_info', {}).get('description', 'N/A')
                            } if r.get('target_entity_info') else None
                        }
                        for r in result['global_results']
                    ]
                }
            },
            # Step 4: Ranking & Token Truncation
            "step4_ranking_and_truncation": {
                "description": "degree/weight 기반 정렬 후 상위 결과 선정",
                "total_count": len(result['merged_results']),
                "top_results": [
                    {
                        "rank": i + 1,
                        "search_type": r.get('search_type', 'unknown'),
                        "similarity": r.get('similarity', 0),
                        "metadata": r.get('metadata', {}),
                        "neighbors_1hop": [
                            {
                                "name": n.get('name', 'N/A'),
                                "entity_type": n.get('entity_type', 'N/A'),
                                "relation_description": n.get('relation_description', ''),
                                "relation_keywords": n.get('relation_keywords', [])
                            } for n in r.get('neighbors', [])
                        ] if r.get('neighbors') else (
                            [
                                {"source_entity_info": r.get('source_entity_info')},
                                {"target_entity_info": r.get('target_entity_info')}
                            ] if r.get('search_type') == 'global' and (r.get('source_entity_info') or r.get('target_entity_info')) else []
                        )
                    }
                    for i, r in enumerate(result['merged_results'][:5])
                ]
            }
        },
        "source_papers": source_papers,
        # AHP 입력용 문서 단위 정규화 결과
        "doc_candidates": build_doc_candidates(
            local_results=result['local_results'],
            global_results=result['global_results']
        ),
        # 검색 설정값 (재현성/디버깅용)
        "retrieval_config": {
            "local_top_k": len(result['local_results']),
            "global_top_k": len(result['global_results']),
            "final_top_k": 5,
            "merge_strategy": "round_robin",
            "dedup_policy": "max_similarity",
            "graph_hops": 1,
            "score_types": {
                "local_similarity": "entity_name ↔ low_level_keywords cosine",
                "global_similarity": "relation_keywords ↔ high_level_keywords cosine"
            }
        }
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

    # 원본 논문 데이터 로드
    print("Loading article data...")
    article_data = load_article_data()

    # Retriever 초기화
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(doc_types=args.doc_types)

    # 검색 실행
    print(f"\nSearching: '{args.query}'")
    results = retriever.retrieve(
        query=args.query,
        retrieval_top_k=args.retrieval_top_k,
        final_top_k=args.final_top_k,
        mode=args.mode
    )

    # 결과를 JSON 파일로 저장
    save_query_result(results, article_data)

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
