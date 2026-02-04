"""
산학 매칭 파이프라인 통합 실행 스크립트
RAG 검색 → 교수 집계 → AHP 랭킹 → 보고서 생성
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.query.retriever import HybridRetriever
from src.ranking.professor_aggregator import ProfessorAggregator
from src.ranking.ranker import ProfessorRanker
from src.reporting.report_generator import ReportGenerator
from config.settings import RETRIEVAL_TOP_K, FINAL_TOP_K


def main():
    parser = argparse.ArgumentParser(
        description="산학 매칭 파이프라인: RAG 검색 → AHP 랭킹 → 보고서 생성"
    )
    parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리 (예: '딥러닝 의료영상 전문가')"
    )
    parser.add_argument(
        "--doc-types",
        type=str,
        nargs="+",
        default=["patent", "article", "project"],
        choices=["patent", "article", "project"],
        help="검색할 문서 타입 (기본: patent article project)"
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help=f"Local/Global 검색 시 각각 가져올 개수 (기본: {RETRIEVAL_TOP_K})"
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help=f"최종 병합 후 반환할 문서 개수 (기본: {FINAL_TOP_K})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="보고서에 포함할 상위 교수 수 (기본: 10)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "pdf", "html", "all"],
        default="json",
        help="보고서 출력 형식 (기본: json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="보고서 출력 디렉토리 (기본: results/reports)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("산학 매칭 파이프라인 시작")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Document Types: {args.doc_types}")
    print()
    
    # 1. RAG 검색
    print("[1/4] RAG 검색 수행 중...")
    retriever = HybridRetriever(doc_types=args.doc_types)
    rag_results = retriever.retrieve(
        query=args.query,
        retrieval_top_k=args.retrieval_top_k,
        final_top_k=args.final_top_k,
        mode="hybrid"
    )
    
    if args.verbose:
        print(f"  - Local results: {len(rag_results['local_results'])}")
        print(f"  - Global results: {len(rag_results['global_results'])}")
        print(f"  - Merged results: {len(rag_results['merged_results'])}")
    print("✓ RAG 검색 완료\n")
    
    # 2. 교수별 집계
    print("[2/4] 교수별 문서 집계 중...")
    aggregator = ProfessorAggregator()
    professor_data = aggregator.aggregate_by_professor(
        rag_results=rag_results,
        doc_types=args.doc_types
    )
    
    if args.verbose:
        print(f"  - 총 교수 수: {len(professor_data)}")
    print("✓ 교수별 집계 완료\n")
    
    # 3. AHP 랭킹
    print("[3/4] AHP 기반 교수 순위 평가 중...")
    ranker = ProfessorRanker()
    ranked_professors = ranker.rank_professors(professor_data)
    
    if args.verbose:
        print(f"  - 상위 5명:")
        for i, prof in enumerate(ranked_professors[:5], 1):
            prof_id = prof.get('professor_id', 'N/A')
            score = prof.get('total_score', 0)
            print(f"    {i}. {prof_id}: {score:.4f}")
    print("✓ AHP 랭킹 완료\n")
    
    # 4. AHP 결과 형식으로 변환
    from config.ahp_config import DEFAULT_TYPE_WEIGHTS
    from datetime import datetime
    
    ahp_results = {
        "query": args.query,
        "keywords": rag_results.get("keywords", {}),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_professors": len(ranked_professors),
        "type_weights": DEFAULT_TYPE_WEIGHTS,
        "ranked_professors": ranked_professors
    }
    
    # 5. 보고서 생성
    print("[4/4] 보고서 생성 중...")
    report_gen = ReportGenerator(output_dir=args.output_dir)
    report_data = report_gen.generate_report(
        ahp_results=ahp_results,
        rag_results=rag_results
    )
    
    # 보고서 저장
    saved_files = []
    if args.output_format in ["json", "all"]:
        json_path = report_gen.save_json(report_data)
        saved_files.append(json_path)
        print(f"  ✓ JSON 저장: {json_path}")
    
    if args.output_format in ["pdf", "all"]:
        pdf_path, ok = report_gen.save_pdf(report_data)
        if pdf_path:
            saved_files.append(pdf_path)
            print(f"  ✓ PDF 저장: {pdf_path}")
        else:
            print("  ✗ PDF 저장 실패 (playwright install chromium 필요)")
    
    if args.output_format in ["html", "all"]:
        html_path = report_gen.save_html(report_data)
        saved_files.append(html_path)
        print(f"  ✓ HTML 저장: {html_path}")
    
    print("✓ 보고서 생성 완료\n")
    
    print("=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)
    print(f"상위 {args.top_n}명 교수 추천 완료")
    if saved_files:
        print(f"보고서 저장 위치:")
        for f in saved_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
