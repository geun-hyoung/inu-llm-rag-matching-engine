"""
리포트 생성 스크립트
AHP 결과를 기반으로 GPT-4o-mini를 사용하여 리포트 생성
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.reporting.report_generator import ReportGenerator
from config.settings import OPENAI_API_KEY
from src.utils.cost_tracker import get_cost_tracker


def main():
    """리포트 생성 실행"""
    # 비용 추적 시작
    tracker = get_cost_tracker()
    tracker.start_task("report", description="리포트 생성")

    parser = argparse.ArgumentParser(description="AHP 결과 기반 리포트 생성")
    parser.add_argument(
        "--ahp-file",
        type=str,
        default="results/test/ahp/ahp_results_20260127_220100.json",
        help="AHP 결과 JSON 파일 경로"
    )
    parser.add_argument(
        "--rag-file",
        type=str,
        default=None,
        help="RAG 결과 JSON 파일 경로 (선택사항, 엔티티/관계 정보 추출용)"
    )
    parser.add_argument(
        "--few-shot-file",
        type=str,
        default=None,
        help="보고서 생성용 Few-shot 예시 JSON 파일 경로 (선택사항, 기본값: data/report_few_shot_examples.json)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API 키 (None이면 config에서 가져옴)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/test/report",
        help="리포트 출력 디렉토리"
    )
    
    args = parser.parse_args()
    
    # API 키 확인
    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print("Error: OpenAI API 키가 설정되지 않았습니다.")
        print("  - config/settings.py에 OPENAI_API_KEY를 설정하거나")
        print("  - --api-key 옵션으로 직접 입력하세요.")
        return
    
    # 파일 경로 확인
    ahp_file = Path(args.ahp_file)
    if not ahp_file.exists():
        print(f"Error: AHP 결과 파일을 찾을 수 없습니다: {ahp_file}")
        return
    
    # AHP 결과 로드
    print("=" * 60)
    print("리포트 생성 시작")
    print("=" * 60)
    print(f"AHP 결과 파일: {ahp_file}")
    
    with open(ahp_file, 'r', encoding='utf-8') as f:
        ahp_results = json.load(f)
    
    query = ahp_results.get("query", "")
    print(f"Query: {query}")
    print()
    
    # RAG 결과 로드 (선택사항)
    rag_results = None
    if args.rag_file:
        rag_file = Path(args.rag_file)
        if rag_file.exists():
            print(f"RAG 결과 파일: {rag_file}")
            with open(rag_file, 'r', encoding='utf-8') as f:
                rag_results = json.load(f)
        else:
            print(f"Warning: RAG 결과 파일을 찾을 수 없습니다: {rag_file}")
    
    # 보고서 생성용 Few-shot 예시 로드 (선택사항)
    few_shot_examples = None
    if args.few_shot_file:
        few_shot_file = Path(args.few_shot_file)
    else:
        # 기본 경로 시도
        few_shot_file = Path("data/report_few_shot_examples.json")
    
    if few_shot_file.exists():
        print(f"보고서 생성용 Few-shot 예시 파일: {few_shot_file}")
        with open(few_shot_file, 'r', encoding='utf-8') as f:
            few_shot_data = json.load(f)
            if isinstance(few_shot_data, list):
                few_shot_examples = few_shot_data
            elif isinstance(few_shot_data, dict) and "examples" in few_shot_data:
                few_shot_examples = few_shot_data["examples"]
            elif isinstance(few_shot_data, dict) and "metadata" in few_shot_data:
                # 메타데이터가 있는 경우 examples 필드 확인
                if "examples" in few_shot_data:
                    few_shot_examples = few_shot_data["examples"]
        if few_shot_examples:
            print(f"  - {len(few_shot_examples)}개의 보고서 예시 로드됨")
    elif args.few_shot_file:
        print(f"Warning: 보고서 생성용 Few-shot 예시 파일을 찾을 수 없습니다: {few_shot_file}")
    
    print()
    
    # 리포트 생성기 초기화
    generator = ReportGenerator(output_dir=args.output_dir, api_key=api_key)
    
    # 리포트 생성
    print("[1/3] 리포트 생성 중...")
    try:
        report_data = generator.generate_report(
            ahp_results=ahp_results,
            rag_results=rag_results,
            few_shot_examples=few_shot_examples
        )
        print("[OK] 리포트 생성 완료\n")
    except Exception as e:
        print(f"Error: 리포트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 리포트 저장
    print("[2/3] 리포트 저장 중...")
    json_path = generator.save_json(report_data)
    text_path = generator.save_text(report_data)
    print(f"  [OK] JSON 저장: {json_path}")
    print(f"  [OK] TXT 저장: {text_path}\n")
    
    # 리포트 미리보기
    print("[3/3] 리포트 미리보기")
    print("=" * 60)
    report_text = report_data.get("report_text", "")
    print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
    print("=" * 60)
    print()
    
    print("=" * 60)
    print("리포트 생성 완료!")
    print("=" * 60)
    print(f"저장 위치: {json_path.parent}")
    print(f"JSON 파일: {json_path.name}")
    print(f"TXT 파일: {text_path.name}")

    # 비용 추적 종료
    cost_result = tracker.end_task()
    if cost_result and cost_result.get('total_cost_usd', 0) > 0:
        print(f"\nAPI Cost: ${cost_result['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
