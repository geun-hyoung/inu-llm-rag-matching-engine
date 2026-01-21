"""
Retrieval 평가 실행 CLI
Context Relevance + Noise Rate@K 평가 스크립트
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.query.retriever import HybridRetriever
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


def extract_docs_from_results(merged_results: list) -> list:
    """
    merged_results에서 평가용 문서 리스트 추출

    Args:
        merged_results: retriever의 merged_results

    Returns:
        [{"doc_id": str, "content": str, "similarity": float}, ...]
    """
    docs = []
    for r in merged_results:
        doc_id = r.get('metadata', {}).get('source_doc_id', '')
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
    retriever: HybridRetriever,
    query: str,
    k: int = 5
) -> dict:
    """
    단일 쿼리 평가

    Args:
        retriever: HybridRetriever 인스턴스
        query: 평가할 쿼리
        k: Top-K

    Returns:
        평가 결과 딕셔너리
    """
    # 검색 실행 (retriever에서 dedup 처리됨)
    results = retriever.retrieve(query=query, final_top_k=k)

    # merged_results에서 문서 추출
    top_k_docs = extract_docs_from_results(results['merged_results'])

    # contexts 추출 (Context Relevance용)
    contexts = [doc['content'] for doc in top_k_docs if doc['content']]

    # Context Relevance 평가
    context_relevance = evaluate_context_relevance(query, contexts) if contexts else 0.0

    # Noise Rate@K 평가
    noise_result = evaluate_noise_rate(query, top_k_docs, k=k)

    return {
        "query": query,
        "context_relevance": context_relevance,
        "noise_rate": noise_result.noise_rate,
        "noise_judgments": [j.to_dict() for j in noise_result.judgments],
        "retrieved_docs": [
            {
                "doc_id": doc['doc_id'],
                "similarity": doc['similarity'],
                "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            }
            for doc in top_k_docs
        ],
        "keywords": {
            "high_level": results.get('high_level_keywords', []),
            "low_level": results.get('low_level_keywords', [])
        }
    }


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
    save: bool = True
) -> dict:
    """
    배치 평가 실행 (그룹 기반 쿼리 지원)

    Args:
        queries: 쿼리 리스트 [{"query": str, "source_doc_ids": list, "data_type": str}, ...]
        doc_types: 문서 타입
        k: Top-K
        save: 결과 저장 여부

    Returns:
        전체 평가 결과
    """
    print(f"\n{'='*60}")
    print("Retrieval Evaluation")
    print(f"{'='*60}")
    print(f"Doc types: {doc_types}")
    print(f"Top-K: {k}")
    print(f"Query count: {len(queries)}")

    # Retriever 초기화
    print("\nInitializing HybridRetriever...")
    retriever = HybridRetriever(doc_types=doc_types)

    # 평가 실행
    all_results = []
    total_relevance = 0.0
    total_noise_rate = 0.0

    for i, query_info in enumerate(queries):
        query_text = query_info["query"] if isinstance(query_info, dict) else query_info

        print(f"\n[{i+1}/{len(queries)}] {query_text[:50]}...")

        result = evaluate_single_query(retriever, query_text, k=k)
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
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Avg Context Relevance: {avg_relevance:.3f}")
    print(f"  Avg Noise Rate@{k}: {avg_noise_rate:.2%}")

    # 결과 저장
    if save:
        results_dir = Path(RESULTS_DIR) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {filepath}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval 평가 - Context Relevance + Noise Rate@K"
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
        "--no-save",
        action="store_true",
        help="결과 저장 안함"
    )

    args = parser.parse_args()

    if args.single_query:
        # 단일 쿼리 평가
        print(f"\n{'='*60}")
        print("Single Query Evaluation")
        print(f"{'='*60}")
        print(f"Query: {args.single_query}")

        retriever = HybridRetriever(doc_types=args.doc_types)
        result = evaluate_single_query(retriever, args.single_query, k=args.top_k)

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
            save=not args.no_save
        )


if __name__ == "__main__":
    main()
