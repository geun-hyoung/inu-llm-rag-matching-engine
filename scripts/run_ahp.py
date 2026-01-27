"""
AHP 랭킹 실행 스크립트
RAG 결과를 읽어서 AHP 점수를 계산하고 결과를 저장
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.ranking.professor_aggregator import ProfessorAggregator
from src.ranking.ranker import ProfessorRanker
from config.ahp_config import DEFAULT_TYPE_WEIGHTS


def main():
    """AHP 랭킹 실행"""
    # 입력 파일 경로
    rag_result_file = Path("results/test/rag/test_rag.json")
    
    if not rag_result_file.exists():
        print(f"Error: RAG 결과 파일을 찾을 수 없습니다: {rag_result_file}")
        return
    
    # 출력 디렉토리 생성
    output_dir = Path("results/test/ahp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # RAG 결과 로드
    print("=" * 60)
    print("AHP 랭킹 시작")
    print("=" * 60)
    print(f"RAG 결과 파일: {rag_result_file}")
    
    with open(rag_result_file, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
    
    query = rag_results.get("query", "")
    retrieved_docs = rag_results.get("retrieved_docs", [])
    print(f"Query: {query}")
    print(f"검색된 문서 수: {len(retrieved_docs)}")
    print()
    
    # 1. 교수별 집계
    print("[1/2] 교수별 문서 집계 중...")
    aggregator = ProfessorAggregator()
    professor_data = aggregator.aggregate_by_professor(rag_results)
    print(f"  - 총 교수 수: {len(professor_data)}")
    
    # 교수별 문서 수 출력
    for prof_id, prof_data in list(professor_data.items())[:5]:
        docs = prof_data["documents"]
        print(f"  - {prof_id}: patent={len(docs['patent'])}, "
              f"article={len(docs['article'])}, project={len(docs['project'])}")
    print("✓ 교수별 집계 완료\n")
    
    # 2. AHP 점수 계산 및 순위 매기기
    print("[2/2] AHP 점수 계산 및 순위 매기기 중...")
    ranker = ProfessorRanker()
    ranked_professors = ranker.rank_professors(professor_data, DEFAULT_TYPE_WEIGHTS)
    
    print(f"  - 상위 5명:")
    for prof in ranked_professors[:5]:
        prof_id = prof.get("professor_id", "N/A")
        score = prof.get("total_score", 0)
        scores_by_type = prof.get("scores_by_type", {})
        print(f"    {prof['rank']}. {prof_id}: {score:.4f} "
              f"(patent={scores_by_type.get('patent', 0):.4f}, "
              f"article={scores_by_type.get('article', 0):.4f}, "
              f"project={scores_by_type.get('project', 0):.4f})")
    print("✓ AHP 랭킹 완료\n")
    
    # 3. 결과 저장
    print("[3/3] 결과 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ahp_results_{timestamp}.json"
    
    result_data = {
        "query": query,
        "keywords": rag_results.get("keywords", {}),
        "timestamp": timestamp,
        "total_professors": len(ranked_professors),
        "type_weights": DEFAULT_TYPE_WEIGHTS,
        "ranked_professors": ranked_professors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 결과 저장: {output_file}")
    print("✓ 저장 완료\n")
    
    print("=" * 60)
    print("AHP 랭킹 완료!")
    print("=" * 60)
    print(f"총 {len(ranked_professors)}명의 교수 순위가 매겨졌습니다.")
    print(f"결과 파일: {output_file}")


if __name__ == "__main__":
    main()
