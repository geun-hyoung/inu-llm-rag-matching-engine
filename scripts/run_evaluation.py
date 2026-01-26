"""
Retrieval 평가 실행 CLI
Context Relevance + Noise Rate@K 평가 스크립트
NaiveRetriever (Vector RAG) vs HybridRetriever (GraphRAG) 비교 지원
"""

import asyncio
import sys

# Windows Event Loop Policy 설정
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 종료 시 Event loop closed 경고 무시
_original_unraisablehook = sys.unraisablehook
def _suppress_event_loop_closed(unraisable):
    if "Event loop is closed" in str(unraisable.exc_value):
        return  # 무시
    _original_unraisablehook(unraisable)
sys.unraisablehook = _suppress_event_loop_closed

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Union

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.query.retriever import HybridRetriever
from src.rag.query.naive_retriever import NaiveRetriever
from src.evaluation import evaluate_context_relevance, evaluate_noise_rate
from config.settings import RESULTS_DIR, FINAL_TOP_K


def load_test_queries(query_file: str = None) -> list:
    """테스트 쿼리 로드 (그룹 기반 형식 지원)"""
    if query_file is None:
        query_file = Path(__file__).parent.parent / "src" / "evaluation" / "test_queries.json"

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 쿼리와 source_doc_ids 함께 반환
    queries = []
    for q in data["queries"]:
        queries.append({
            "query": q["query"],
            "source_doc_ids": q.get("source_doc_ids", []),
            "data_type": q.get("data_type", "")
        })
    return queries


def extract_docs_from_hybrid_results(merged_results: list) -> list:
    """
    HybridRetriever의 merged_results에서 평가용 문서 리스트 추출

    Args:
        merged_results: HybridRetriever의 merged_results

    Returns:
        [{"doc_id": str, "content": str, "similarity": float}, ...]
    """
    docs = []
    for r in merged_results:
        doc_id = r.get('metadata', {}).get('source_doc_id', '')
        # 원본 chunk content 우선 사용, 없으면 entity/relation description 사용
        content = r.get('original_content') or r.get('document', '')
        similarity = r.get('similarity', 0)

        if doc_id:
            docs.append({
                'doc_id': str(doc_id),
                'content': content,
                'similarity': similarity
            })

    return docs


def extract_docs_from_naive_results(results: list) -> list:
    """
    NaiveRetriever의 results에서 평가용 문서 리스트 추출

    Args:
        results: NaiveRetriever의 results

    Returns:
        [{"doc_id": str, "content": str, "similarity": float}, ...]
    """
    docs = []
    for r in results:
        metadata = r.get('metadata', {})
        # NaiveRetriever는 chunk를 검색하므로 doc_id 사용
        doc_id = metadata.get('doc_id', '') or metadata.get('source_doc_id', '')
        content = r.get('document', '')
        similarity = r.get('similarity', 0)

        if doc_id:
            docs.append({
                'doc_id': str(doc_id),
                'content': content,
                'similarity': similarity
            })

    return docs


def evaluate_single_query(
    retriever: Union[HybridRetriever, NaiveRetriever],
    query: str,
    k: int = 5,
    retriever_type: str = "hybrid",
    keywords: tuple = None
) -> dict:
    """
    단일 쿼리 평가

    Args:
        retriever: HybridRetriever 또는 NaiveRetriever 인스턴스
        query: 평가할 쿼리
        k: Top-K
        retriever_type: "hybrid" 또는 "naive"
        keywords: 미리 추출된 키워드 튜플 (high_level, low_level). hybrid에서만 사용

    Returns:
        평가 결과 딕셔너리
    """
    if retriever_type == "hybrid":
        # HybridRetriever 검색 (키워드 전달)
        results = retriever.retrieve(query=query, final_top_k=k, keywords=keywords)
        top_k_docs = extract_docs_from_hybrid_results(results['merged_results'])
        keywords_dict = {
            "high_level": results.get('high_level_keywords', []),
            "low_level": results.get('low_level_keywords', [])
        }
    else:
        # NaiveRetriever 검색
        results = retriever.retrieve(query=query, top_k=k)
        top_k_docs = extract_docs_from_naive_results(results['results'])
        keywords_dict = None  # NaiveRetriever는 키워드 추출 없음

    # contexts 추출 (Context Relevance용)
    contexts = [doc['content'] for doc in top_k_docs if doc['content']]

    # Context Relevance 평가
    context_relevance = evaluate_context_relevance(query, contexts) if contexts else 0.0

    # Noise Rate@K 평가
    noise_result = evaluate_noise_rate(query, top_k_docs, k=k)

    result = {
        "query": query,
        "retriever_type": retriever_type,
        "context_relevance": context_relevance,
        "noise_rate": noise_result.noise_rate,
        "noise_judgments": [j.to_dict() for j in noise_result.judgments],
        "retrieved_count": len(top_k_docs),
        "retrieved_docs": [
            {
                "doc_id": doc['doc_id'],
                "similarity": doc['similarity'],
                "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            }
            for doc in top_k_docs
        ]
    }

    if keywords_dict:
        result["keywords"] = keywords_dict

    return result


def calculate_recall_at_k(retrieved_doc_ids: list, ground_truth_ids: list) -> float:
    """
    Recall@K 계산: Ground Truth 문서 중 검색된 비율

    Args:
        retrieved_doc_ids: 검색된 문서 ID 리스트
        ground_truth_ids: Ground Truth 문서 ID 리스트

    Returns:
        Recall 점수 (0~1)
    """
    if not ground_truth_ids:
        return 0.0

    # 문자열로 변환하여 비교
    retrieved_set = set(str(doc_id) for doc_id in retrieved_doc_ids)
    gt_set = set(str(doc_id) for doc_id in ground_truth_ids)

    hits = len(retrieved_set & gt_set)
    return hits / len(gt_set)


def run_batch_evaluation(
    queries: list,
    doc_types: list,
    k: int = 5,
    save: bool = True,
    retriever_type: str = "hybrid",
    compare: bool = False
) -> dict:
    """
    배치 평가 실행 (그룹 기반 쿼리 지원)

    Args:
        queries: 쿼리 리스트 [{"query": str, "source_doc_ids": list, "data_type": str}, ...]
        doc_types: 문서 타입
        k: Top-K
        save: 결과 저장 여부
        retriever_type: "hybrid" 또는 "naive"
        compare: True면 두 retriever 비교 평가

    Returns:
        전체 평가 결과
    """
    if compare:
        return run_comparison_evaluation(queries, doc_types, k, save)

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation ({retriever_type.upper()})")
    print(f"{'='*60}")
    print(f"Retriever: {retriever_type}")
    print(f"Doc types: {doc_types}")
    print(f"Top-K: {k}")
    print(f"Query count: {len(queries)}")

    # Retriever 초기화
    if retriever_type == "hybrid":
        print("\nInitializing HybridRetriever...")
        retriever = HybridRetriever(doc_types=doc_types)
    else:
        print("\nInitializing NaiveRetriever...")
        retriever = NaiveRetriever(doc_types=doc_types)

    # 평가 실행
    all_results = []
    total_relevance = 0.0
    total_noise_rate = 0.0

    for i, query_info in enumerate(queries):
        query_text = query_info["query"] if isinstance(query_info, dict) else query_info

        print(f"\n[{i+1}/{len(queries)}] {query_text[:50]}...")

        result = evaluate_single_query(retriever, query_text, k=k, retriever_type=retriever_type)
        all_results.append(result)

        total_relevance += result['context_relevance']
        total_noise_rate += result['noise_rate']

        print(f"  Context Relevance: {result['context_relevance']:.3f}")
        print(f"  Noise Rate@{k}: {result['noise_rate']:.2%}")

    # 평균 계산
    n = len(queries)
    avg_relevance = total_relevance / n if n > 0 else 0.0
    avg_noise_rate = total_noise_rate / n if n > 0 else 0.0

    # 결과 구조
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "retriever_type": retriever_type,
            "doc_types": doc_types,
            "k": k,
            "query_count": n
        },
        "summary": {
            "avg_context_relevance": avg_relevance,
            "avg_noise_rate": avg_noise_rate
        },
        "results": all_results
    }

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({retriever_type.upper()})")
    print(f"{'='*60}")
    print(f"  Avg Context Relevance: {avg_relevance:.3f}")
    print(f"  Avg Noise Rate@{k}: {avg_noise_rate:.2%}")

    # 결과 저장
    if save:
        results_dir = Path(RESULTS_DIR) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"eval_{retriever_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {filepath}")

    return summary


def run_comparison_evaluation(
    queries: list,
    _doc_types: list,
    k: int = 5,
    save: bool = True
) -> dict:
    """
    NaiveRetriever vs HybridRetriever 비교 평가 실행 (타입별 분리 검색)

    Args:
        queries: 쿼리 리스트
        _doc_types: 문서 타입 (사용되지 않음, 항상 3개 타입 개별 평가)
        k: Top-K
        save: 결과 저장 여부

    Returns:
        비교 평가 결과
    """
    doc_type_list = ["patent", "article", "project"]

    print(f"\n{'='*70}")
    print("COMPARISON: NaiveRetriever (Vector) vs HybridRetriever (Graph)")
    print(f"{'='*70}")
    print(f"Doc types: {doc_type_list} (evaluated separately)")
    print(f"Top-K: {k}")
    print(f"Query count: {len(queries)}")
    print(f"Total evaluations: {len(queries)} queries × 3 types × 2 retrievers = {len(queries) * 6}")

    # 각 타입별 Retriever 초기화
    print("\nInitializing retrievers for each doc type...")
    retrievers = {}
    for doc_type in doc_type_list:
        print(f"  Initializing {doc_type} retrievers...")
        retrievers[doc_type] = {
            "naive": NaiveRetriever(doc_types=[doc_type]),
            "hybrid": HybridRetriever(doc_types=[doc_type])
        }

    # 결과 저장
    results = []

    # 타입별 통계 누적용
    type_stats = {
        doc_type: {
            "naive": {"total_relevance": 0.0, "total_noise": 0.0},
            "hybrid": {"total_relevance": 0.0, "total_noise": 0.0}
        }
        for doc_type in doc_type_list
    }

    for i, query_info in enumerate(queries):
        query_text = query_info["query"] if isinstance(query_info, dict) else query_info

        print(f"\n[{i+1}/{len(queries)}] {query_text[:50]}...")

        query_result = {"query": query_text}

        # 쿼리당 키워드 1번만 추출 (첫 번째 HybridRetriever 사용)
        first_hybrid = retrievers[doc_type_list[0]]["hybrid"]
        extracted_keywords = first_hybrid._extract_keywords(query_text)
        print(f"  Extracted keywords - High: {extracted_keywords[0]}, Low: {extracted_keywords[1]}")

        # 각 타입별로 평가
        for doc_type in doc_type_list:
            # NaiveRetriever 평가
            naive_result = evaluate_single_query(
                retrievers[doc_type]["naive"],
                query_text,
                k=k,
                retriever_type="naive"
            )

            # HybridRetriever 평가 (추출된 키워드 재사용)
            hybrid_result = evaluate_single_query(
                retrievers[doc_type]["hybrid"],
                query_text,
                k=k,
                retriever_type="hybrid",
                keywords=extracted_keywords
            )

            # noise_judgment 매핑 (reason만 추출)
            naive_noise_reasons = {j["doc_id"]: j["reason"] for j in naive_result["noise_judgments"]}
            hybrid_noise_reasons = {j["doc_id"]: j["reason"] for j in hybrid_result["noise_judgments"]}

            # retrieved_docs에 noise_judgment 추가
            naive_docs = [
                {
                    "doc_id": doc["doc_id"],
                    "similarity": doc["similarity"],
                    "content_preview": doc["content_preview"],
                    "noise_judgment": naive_noise_reasons.get(doc["doc_id"], "")
                }
                for doc in naive_result["retrieved_docs"]
            ]

            hybrid_docs = [
                {
                    "doc_id": doc["doc_id"],
                    "similarity": doc["similarity"],
                    "content_preview": doc["content_preview"],
                    "noise_judgment": hybrid_noise_reasons.get(doc["doc_id"], "")
                }
                for doc in hybrid_result["retrieved_docs"]
            ]

            # 결과 저장
            query_result[doc_type] = {
                "naive": {
                    "context_relevance": naive_result["context_relevance"],
                    "noise_rate": naive_result["noise_rate"],
                    "retrieved_docs": naive_docs
                },
                "hybrid": {
                    "context_relevance": hybrid_result["context_relevance"],
                    "noise_rate": hybrid_result["noise_rate"],
                    "retrieved_docs": hybrid_docs
                }
            }

            # 통계 누적
            type_stats[doc_type]["naive"]["total_relevance"] += naive_result["context_relevance"]
            type_stats[doc_type]["naive"]["total_noise"] += naive_result["noise_rate"]
            type_stats[doc_type]["hybrid"]["total_relevance"] += hybrid_result["context_relevance"]
            type_stats[doc_type]["hybrid"]["total_noise"] += hybrid_result["noise_rate"]

            # 개별 결과 출력
            print(f"  [{doc_type}] Naive: rel={naive_result['context_relevance']:.3f}, noise={naive_result['noise_rate']:.2%}")
            print(f"  [{doc_type}] Hybrid: rel={hybrid_result['context_relevance']:.3f}, noise={hybrid_result['noise_rate']:.2%}")

        results.append(query_result)

    # 평균 계산
    n = len(queries)
    comparison_summary = {}

    # 타입별 평균 계산
    for doc_type in doc_type_list:
        naive_avg_rel = type_stats[doc_type]["naive"]["total_relevance"] / n if n > 0 else 0.0
        naive_avg_noise = type_stats[doc_type]["naive"]["total_noise"] / n if n > 0 else 0.0
        hybrid_avg_rel = type_stats[doc_type]["hybrid"]["total_relevance"] / n if n > 0 else 0.0
        hybrid_avg_noise = type_stats[doc_type]["hybrid"]["total_noise"] / n if n > 0 else 0.0

        comparison_summary[doc_type] = {
            "naive_retriever": {
                "avg_context_relevance": naive_avg_rel,
                "avg_noise_rate": naive_avg_noise
            },
            "hybrid_retriever": {
                "avg_context_relevance": hybrid_avg_rel,
                "avg_noise_rate": hybrid_avg_noise
            },
            "winner": {
                "context_relevance": "hybrid" if hybrid_avg_rel > naive_avg_rel else "naive" if naive_avg_rel > hybrid_avg_rel else "tie",
                "noise_rate": "hybrid" if hybrid_avg_noise < naive_avg_noise else "naive" if naive_avg_noise < hybrid_avg_noise else "tie"
            }
        }

    # 전체 평균 계산
    total_naive_rel = sum(comparison_summary[t]["naive_retriever"]["avg_context_relevance"] for t in doc_type_list) / 3
    total_naive_noise = sum(comparison_summary[t]["naive_retriever"]["avg_noise_rate"] for t in doc_type_list) / 3
    total_hybrid_rel = sum(comparison_summary[t]["hybrid_retriever"]["avg_context_relevance"] for t in doc_type_list) / 3
    total_hybrid_noise = sum(comparison_summary[t]["hybrid_retriever"]["avg_noise_rate"] for t in doc_type_list) / 3

    comparison_summary["average"] = {
        "naive_retriever": {
            "avg_context_relevance": total_naive_rel,
            "avg_noise_rate": total_naive_noise
        },
        "hybrid_retriever": {
            "avg_context_relevance": total_hybrid_rel,
            "avg_noise_rate": total_hybrid_noise
        },
        "winner": {
            "context_relevance": "hybrid" if total_hybrid_rel > total_naive_rel else "naive" if total_naive_rel > total_hybrid_rel else "tie",
            "noise_rate": "hybrid" if total_hybrid_noise < total_naive_noise else "naive" if total_naive_noise < total_hybrid_noise else "tie"
        }
    }

    # 결과 구조
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "mode": "comparison_by_type",
            "k": k,
            "query_count": n
        },
        "comparison_summary": comparison_summary,
        "results": results
    }

    # 결과 출력
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY BY DOC TYPE")
    print(f"{'='*70}")

    for doc_type in doc_type_list:
        stats = comparison_summary[doc_type]
        print(f"\n[{doc_type.upper()}]")
        print(f"{'Metric':<25} {'Naive':<15} {'Hybrid':<15} {'Winner':<10}")
        print("-" * 65)
        print(f"{'Avg Context Relevance':<25} {stats['naive_retriever']['avg_context_relevance']:<15.3f} {stats['hybrid_retriever']['avg_context_relevance']:<15.3f} {stats['winner']['context_relevance']}")
        print(f"{'Avg Noise Rate@' + str(k):<25} {stats['naive_retriever']['avg_noise_rate']:<15.2%} {stats['hybrid_retriever']['avg_noise_rate']:<15.2%} {stats['winner']['noise_rate']}")

    print(f"\n{'='*70}")
    print("OVERALL AVERAGE")
    print(f"{'='*70}")
    avg_stats = comparison_summary["average"]
    print(f"{'Metric':<25} {'Naive':<15} {'Hybrid':<15} {'Winner':<10}")
    print("-" * 65)
    print(f"{'Avg Context Relevance':<25} {avg_stats['naive_retriever']['avg_context_relevance']:<15.3f} {avg_stats['hybrid_retriever']['avg_context_relevance']:<15.3f} {avg_stats['winner']['context_relevance']}")
    print(f"{'Avg Noise Rate@' + str(k):<25} {avg_stats['naive_retriever']['avg_noise_rate']:<15.2%} {avg_stats['hybrid_retriever']['avg_noise_rate']:<15.2%} {avg_stats['winner']['noise_rate']}")

    # 결과 저장
    if save:
        results_dir = Path(RESULTS_DIR) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {filepath}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval 평가 - Context Relevance + Noise Rate@K (Naive vs Hybrid 비교 지원)"
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
        default=FINAL_TOP_K,
        help=f"Top-K 문서 수 (기본: {FINAL_TOP_K})"
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
        help="단일 쿼리 테스트"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid",
        choices=["naive", "hybrid"],
        help="사용할 Retriever 타입 (기본: hybrid)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Naive vs Hybrid 비교 평가 실행"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 저장 안함"
    )

    args = parser.parse_args()

    if args.single_query:
        # 단일 쿼리 평가
        if args.compare:
            # 비교 모드: 두 retriever 모두 평가
            print(f"\n{'='*70}")
            print("Single Query Comparison")
            print(f"{'='*70}")
            print(f"Query: {args.single_query}")

            naive_retriever = NaiveRetriever(doc_types=args.doc_types)
            hybrid_retriever = HybridRetriever(doc_types=args.doc_types)

            print("\n--- NaiveRetriever (Vector RAG) ---")
            naive_result = evaluate_single_query(naive_retriever, args.single_query, k=args.top_k, retriever_type="naive")
            print(f"Context Relevance: {naive_result['context_relevance']:.3f}")
            print(f"Noise Rate@{args.top_k}: {naive_result['noise_rate']:.2%}")
            print(f"Retrieved: {naive_result['retrieved_count']} docs")

            print("\n--- HybridRetriever (GraphRAG) ---")
            hybrid_result = evaluate_single_query(hybrid_retriever, args.single_query, k=args.top_k, retriever_type="hybrid")
            print(f"Context Relevance: {hybrid_result['context_relevance']:.3f}")
            print(f"Noise Rate@{args.top_k}: {hybrid_result['noise_rate']:.2%}")
            print(f"Retrieved: {hybrid_result['retrieved_count']} docs")
            if hybrid_result.get('keywords'):
                print(f"Keywords: {hybrid_result['keywords']}")

            print("\n--- Comparison ---")
            rel_diff = hybrid_result['context_relevance'] - naive_result['context_relevance']
            noise_diff = hybrid_result['noise_rate'] - naive_result['noise_rate']
            print(f"Context Relevance: Hybrid {'+' if rel_diff >= 0 else ''}{rel_diff:.3f}")
            print(f"Noise Rate: Hybrid {'+' if noise_diff >= 0 else ''}{noise_diff:.2%}")

        else:
            # 단일 retriever 평가
            print(f"\n{'='*60}")
            print(f"Single Query Evaluation ({args.retriever.upper()})")
            print(f"{'='*60}")
            print(f"Query: {args.single_query}")

            if args.retriever == "hybrid":
                retriever = HybridRetriever(doc_types=args.doc_types)
            else:
                retriever = NaiveRetriever(doc_types=args.doc_types)

            result = evaluate_single_query(retriever, args.single_query, k=args.top_k, retriever_type=args.retriever)

            print(f"\n--- Results ---")
            print(f"Context Relevance: {result['context_relevance']:.3f}")
            print(f"Noise Rate@{args.top_k}: {result['noise_rate']:.2%}")

            print(f"\n--- Retrieved Documents ---")
            for i, doc in enumerate(result['retrieved_docs']):
                print(f"\n{i+1}. [doc_id: {doc['doc_id']}] similarity: {doc['similarity']:.4f}")
                print(f"   {doc['content_preview']}")

            print(f"\n--- Noise Judgments ---")
            for j in result['noise_judgments']:
                status = "Noise" if j['is_noise'] else "Non-noise"
                print(f"[{j['doc_id']}] {status}")
                print(f"  A1(도메인+기술): {j['A1_domain_tech']}")
                print(f"  A2(적용/구현): {j['A2_implementation']}")
                print(f"  사유: {j['reason']}")

    else:
        # 배치 평가
        queries = load_test_queries(args.query_file)
        print(f"Loaded {len(queries)} test queries")

        run_batch_evaluation(
            queries=queries,
            doc_types=args.doc_types,
            k=args.top_k,
            save=not args.no_save,
            retriever_type=args.retriever,
            compare=args.compare
        )


if __name__ == "__main__":
    main()
