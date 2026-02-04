"""
연구과제 데이터 필터링 스크립트
test 폴더의 project.json을 필터링하여 필요한 컬럼만 추출하고,
한국어 컬럼명을 영어로 변경하여 train 폴더에 저장합니다.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import re

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROJECT_DATA_FILE, DATA_TRAIN_PROJECT_FILE
from data_filtering.text_preprocessing import preprocess_text


def load_project_json(input_file: str = None) -> List[Dict]:
    """
    data 폴더의 project.json 파일을 읽어옵니다.
    
    Args:
        input_file: 입력 파일 경로 (None이면 설정 파일의 경로 사용)
        
    Returns:
        프로젝트 데이터 리스트
    """
    if input_file is None:
        input_file = PROJECT_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 프로젝트 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        print(f"  - 총 {len(project_data):,}개의 프로젝트 데이터 로드 완료")
        return project_data
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def parse_year_from_project(project: Dict) -> Any:
    """
    프로젝트 데이터에서 연도를 추출합니다.
    excel_base_year 또는 RCH_ST_DT에서 연도를 추출합니다.
    
    Args:
        project: 프로젝트 데이터 딕셔너리
        
    Returns:
        연도 (int 또는 None)
    """
    # 1. excel_base_year 확인
    excel_base_year = project.get('excel_기준년도')
    if excel_base_year:
        try:
            year = int(str(excel_base_year).strip())
            if 1900 <= year <= 2100:  # 유효한 연도 범위
                return year
        except:
            pass
    
    # 2. RCH_ST_DT에서 연도 추출
    rch_st_dt = project.get('RCH_ST_DT')
    if rch_st_dt:
        date_str = str(rch_st_dt).strip()
        # yyyymmdd 형식
        if len(date_str) >= 8 and date_str[:4].isdigit():
            try:
                year = int(date_str[:4])
                if 1900 <= year <= 2100:
                    return year
            except:
                pass
        # yyyy-mm-dd 형식
        elif '-' in date_str:
            parts = date_str.split('-')
            if len(parts) > 0 and parts[0].isdigit():
                try:
                    year = int(parts[0])
                    if 1900 <= year <= 2100:
                        return year
                except:
                    pass
    
    return None


def filter_project_data(projects: List[Dict]) -> tuple:
    """
    프로젝트 데이터에서 필요한 컬럼만 추출하고 한국어 컬럼명을 영어로 변경합니다.
    - 연도 필터: 2015년 이상 데이터만 유지 (excel_기준년도 또는 RCH_ST_DT 기준).
    - 텍스트 전처리(수식/기호 제거, 100자 이상 5000자 이하 필터링)를 적용합니다.
    
    Args:
        projects: 프로젝트 데이터 리스트
        
    Returns:
        (필터링된 프로젝트 데이터 리스트, 필터링 통계)
    """
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 프로젝트 수: {len(projects):,}개")
    
    filtered_projects = []
    filter_stats = {
        'total': len(projects),
        'year_filtered': 0,  # 2015년 미만으로 제외된 개수
        'text_preprocessing_passed': 0,
        'text_preprocessing_failed': 0
    }
    
    for idx, project in enumerate(projects, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(projects):,}개")
        
        # 연도 추출 (2015년 이상만 유지)
        year = parse_year_from_project(project)
        if year is None or year < 2015:
            filter_stats['year_filtered'] += 1
            continue
        
        # summary 필드 생성: excel_연구목표요약과 excel_연구내용요약 합치기
        objective = project.get('excel_연구목표요약', '')
        content = project.get('excel_연구내용요약', '')
        
        # 두 필드를 합치기 (공백으로 구분)
        summary_parts = []
        if objective and str(objective).strip():
            summary_parts.append(str(objective).strip())
        if content and str(content).strip():
            summary_parts.append(str(content).strip())
        
        if not summary_parts:
            filter_stats['text_preprocessing_failed'] += 1
            continue
        
        summary = ' '.join(summary_parts)
        
        # 텍스트 전처리 (수식/기호 제거, 100자 이상 5000자 이하 필터링)
        preprocessed_text, is_valid = preprocess_text(summary, min_length=100, max_length=5000)
        
        if not is_valid:
            filter_stats['text_preprocessing_failed'] += 1
            continue  # 최소 길이 조건을 만족하지 않으면 제외
        
        filter_stats['text_preprocessing_passed'] += 1
        
        # 공통 컬럼 구조로 데이터 생성
        filtered_project = {
            'data_type': 'project',
            'no': len(filtered_projects) + 1,  # 필터링된 데이터의 순번
            'text': preprocessed_text,  # 전처리된 텍스트
            'title': project.get('PRJ_NM'),  # 과제명
            'year': year,  # 연도
            'professor_info': project.get('professor_info'),  # 교수 정보
            'metadata': {  # 추가 메타데이터
                'PRJ_RSPR_EMP_ID': project.get('PRJ_RSPR_EMP_ID'),
                'TOT_RND_AMT': project.get('TOT_RND_AMT'),
                'RCH_ST_DT': project.get('RCH_ST_DT'),
                'excel_base_year': project.get('excel_기준년도'),
                'excel_project_name_kr': project.get('excel_과제명(국문)'),
                'excel_expected_effect_summary': project.get('excel_기대효과요약'),
                'excel_연구목표요약': project.get('excel_연구목표요약'),  # EDA 분석을 위해 추가
                'excel_연구내용요약': project.get('excel_연구내용요약')  # EDA 분석을 위해 추가
            }
        }
        
        filtered_projects.append(filtered_project)
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 프로젝트: {len(filtered_projects):,}개")
    
    return filtered_projects, filter_stats


def save_filtered_data(filtered_projects: List[Dict]):
    """
    필터링된 프로젝트 데이터를 JSON 파일로 저장합니다.
    data/train 폴더에 저장합니다.
    
    Args:
        filtered_projects: 필터링된 프로젝트 데이터 리스트
    """
    # data/train 폴더에 필터링 후 데이터 저장
    train_output_path = Path(DATA_TRAIN_PROJECT_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[저장] 필터링 후 데이터 저장 중: {train_output_path}")
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_projects, f, ensure_ascii=False, indent=2)
        
        print(f"[완료] 총 {len(filtered_projects):,}개의 필터링된 프로젝트 데이터를 저장했습니다.")
        print(f"[저장 위치] {train_output_path}")
    except Exception as e:
        print(f"[오류] 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def has_value(value: Any) -> bool:
    """
    값이 존재하는지 확인합니다 (None, 빈 문자열, 빈 리스트 제외)
    
    Args:
        value: 확인할 값
        
    Returns:
        값이 존재하면 True, 아니면 False
    """
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def main():
    """
    메인 함수: data 폴더의 project.json을 필터링하여 data/test와 data/train 폴더에 저장
    """
    # data 폴더의 project.json 파일 읽기
    print("\n[시작] data 폴더의 프로젝트 데이터 읽기 중...")
    projects = load_project_json()
    
    if not projects:
        print("[경고] 프로젝트 데이터가 없습니다.")
        return
    
    original_count = len(projects)
    
    # 데이터 필터링
    filtered_projects, filter_stats = filter_project_data(projects)
    
    filtered_count = len(filtered_projects)
    
    # train 폴더에 필터링 후 데이터 저장
    save_filtered_data(filtered_projects)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 프로젝트 수: {original_count:,}개")
    print(f"2. 필터링된 프로젝트 수: {filtered_count:,}개")
    
    # 교수 정보가 있는 프로젝트 수
    professor_matched = len([p for p in filtered_projects if p.get("professor_info")])
    print(f"3. 교수 정보가 있는 프로젝트 수: {professor_matched:,}개")
    
    # 연도 필터 통계
    print(f"\n[연도 필터 통계] (2015년 이상만 유지)")
    print(f"  - 연도 미충족 제외 (2015년 미만 또는 연도 없음): {filter_stats['year_filtered']:,}개")
    
    # 텍스트 전처리 통계
    print(f"\n[텍스트 전처리 통계]")
    print(f"  - 전처리 통과 (100자 이상 5000자 이하): {filter_stats['text_preprocessing_passed']:,}개")
    print(f"  - 전처리 실패 (100자 미만 또는 5000자 초과): {filter_stats['text_preprocessing_failed']:,}개")
    total_filtered = filter_stats['year_filtered'] + filter_stats['text_preprocessing_failed']
    print(f"  - 전체 제외: {total_filtered:,}개 (연도 {filter_stats['year_filtered']:,} + 전처리 실패 {filter_stats['text_preprocessing_failed']:,})")
    if filter_stats['total'] > 0:
        print(f"  - 필터링률: {total_filtered / filter_stats['total'] * 100:.1f}%")
    
    # 컬럼 정보 출력
    if filtered_projects:
        print(f"\n[컬럼] 공통 컬럼 구조:")
        for key in filtered_projects[0].keys():
            print(f"   - {key}")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
