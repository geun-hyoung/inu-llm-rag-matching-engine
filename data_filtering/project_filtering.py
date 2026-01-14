"""
연구과제 데이터 필터링 스크립트
test 폴더의 project.json을 필터링하여 필요한 컬럼만 추출하고,
한국어 컬럼명을 영어로 변경하여 train 폴더에 저장합니다.
"""

import json
from pathlib import Path
from typing import List, Dict
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROJECT_DATA_FILE, DATA_TRAIN_PROJECT_FILE


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


def filter_project_data(projects: List[Dict]) -> List[Dict]:
    """
    프로젝트 데이터에서 필요한 컬럼만 추출하고 한국어 컬럼명을 영어로 변경합니다.
    
    Args:
        projects: 프로젝트 데이터 리스트
        
    Returns:
        필터링된 프로젝트 데이터 리스트
    """
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 프로젝트 수: {len(projects):,}개")
    
    # 유지할 컬럼 목록
    keep_columns = {
        'PRJ_NM',
        'excel_과제명(국문)',
        'PRJ_RSPR_EMP_ID',
        'TOT_RND_AMT',
        'RCH_ST_DT',
        'excel_기준년도',
        'excel_연구목표요약',
        'excel_연구내용요약',
        'excel_기대효과요약',
        'professor_info'
    }
    
    # 컬럼명 매핑 (한국어 -> 영어)
    column_mapping = {
        'excel_과제명(국문)': 'excel_project_name_kr',
        'excel_기준년도': 'excel_base_year',
        'excel_연구목표요약': 'excel_research_objective_summary',
        'excel_연구내용요약': 'excel_research_content_summary',
        'excel_기대효과요약': 'excel_expected_effect_summary'
    }
    
    filtered_projects = []
    
    for idx, project in enumerate(projects, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(projects):,}개")
        
        # 필터링된 프로젝트 데이터 생성
        filtered_project = {}
        
        for key, value in project.items():
            # 유지할 컬럼만 포함
            if key in keep_columns:
                # 컬럼명 변경 (한국어 -> 영어)
                new_key = column_mapping.get(key, key)
                filtered_project[new_key] = value
        
        # summary 필드 생성: excel_research_objective_summary와 excel_research_content_summary 합치기
        objective = filtered_project.get('excel_research_objective_summary', '')
        content = filtered_project.get('excel_research_content_summary', '')
        
        # 두 필드를 합치기 (공백으로 구분)
        summary_parts = []
        if objective and str(objective).strip():
            summary_parts.append(str(objective).strip())
        if content and str(content).strip():
            summary_parts.append(str(content).strip())
        
        if summary_parts:
            filtered_project['summary'] = ' '.join(summary_parts)
        else:
            filtered_project['summary'] = None
        
        # data_type과 no 필드 추가
        filtered_project['data_type'] = 'project'
        filtered_project['no'] = idx
        
        filtered_projects.append(filtered_project)
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 프로젝트: {len(filtered_projects):,}개")
    
    return filtered_projects


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
    
    # 데이터 필터링
    filtered_projects = filter_project_data(projects)
    
    # train 폴더에 필터링 후 데이터 저장
    save_filtered_data(filtered_projects)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 프로젝트 수: {len(projects):,}개")
    print(f"2. 필터링된 프로젝트 수: {len(filtered_projects):,}개")
    
    # 교수 정보가 있는 프로젝트 수
    professor_matched = len([p for p in filtered_projects if p.get("professor_info")])
    print(f"3. 교수 정보가 있는 프로젝트 수: {professor_matched:,}개")
    
    # 컬럼 정보 출력
    if filtered_projects:
        print(f"\n[컬럼] 유지된 컬럼:")
        for key in filtered_projects[0].keys():
            print(f"   - {key}")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
