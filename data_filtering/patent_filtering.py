"""
특허 데이터 필터링 스크립트
data 폴더의 patent.json을 필터링하여 필요한 조건을 만족하는 데이터만 추출합니다.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PATENT_DATA_FILE, DATA_TRAIN_PATENT_FILE


def load_patent_json(input_file: str = None) -> List[Dict]:
    """
    data 폴더의 patent.json 파일을 읽어옵니다.
    
    Args:
        input_file: 입력 파일 경로 (None이면 설정 파일의 경로 사용)
        
    Returns:
        특허 데이터 리스트
    """
    if input_file is None:
        input_file = PATENT_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 특허 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            patent_data = json.load(f)
        
        print(f"  - 총 {len(patent_data):,}개의 특허 데이터 로드 완료")
        return patent_data
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


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


def parse_application_date(date_str: Any) -> tuple:
    """
    kipris_application_date를 파싱하여 연도를 추출합니다.
    다양한 형식을 지원합니다: yyyymmdd, yyyy-mm-dd, yyyy/mm/dd 등
    
    Args:
        date_str: 날짜 문자열
        
    Returns:
        (연도(int), 파싱 성공 여부(bool))
    """
    if not has_value(date_str):
        return None, False
    
    date_str = str(date_str).strip()
    
    # yyyymmdd 형식 (8자리)
    if len(date_str) >= 8 and date_str[:4].isdigit():
        try:
            year = int(date_str[:4])
            return year, True
        except:
            pass
    
    # yyyy-mm-dd 또는 yyyy/mm/dd 형식
    try:
        # 하이픈이나 슬래시로 구분된 경우
        if '-' in date_str or '/' in date_str:
            parts = date_str.replace('/', '-').split('-')
            if len(parts) > 0 and parts[0].isdigit():
                year = int(parts[0])
                return year, True
    except:
        pass
    
    return None, False


def filter_patent_data(patents: List[Dict]) -> tuple:
    """
    특허 데이터에서 필수 조건을 만족하는 데이터만 필터링합니다.
    
    필터링 조건:
    1. null 값이 있으면 안 되는 필드: kipris_application_name, mbr_sn, kipris_abstract, tech_invnt_se, kipris_register_status
    2. kipris_application_date가 2015년 이후인 데이터만
    
    Args:
        patents: 특허 데이터 리스트
        
    Returns:
        (필터링된 특허 데이터 리스트, 필터링 통계)
    """
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 특허 수: {len(patents):,}개")
    
    # 필수 필드 목록 (null이면 안 되는 필드)
    required_fields = {
        'kipris_application_name',
        'mbr_sn',
        'kipris_abstract',
        'tech_invnt_se',
        'kipris_register_status'
    }
    
    filtered_patents = []
    filter_stats = {
        'total': len(patents),
        'null_field_filtered': 0,  # 필수 필드 null로 제외된 개수
        'date_filtered': 0,  # 날짜 조건으로 제외된 개수
        'passed': 0  # 통과한 개수
    }
    
    for idx, patent in enumerate(patents, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(patents):,}개")
        
        # 1. 필수 필드 null 체크
        has_null_field = False
        for field in required_fields:
            if not has_value(patent.get(field)):
                has_null_field = True
                break
        
        if has_null_field:
            filter_stats['null_field_filtered'] += 1
            continue
        
        # 2. 날짜 조건 체크 (2015년 이후)
        application_date = patent.get('kipris_application_date')
        year, date_valid = parse_application_date(application_date)
        
        if not date_valid or year is None:
            filter_stats['date_filtered'] += 1
            continue
        
        if year < 2015:
            filter_stats['date_filtered'] += 1
            continue
        
        # 모든 조건 통과
        # data_type과 no 필드 추가
        patent['data_type'] = 'patent'
        patent['no'] = len(filtered_patents) + 1  # 필터링된 데이터의 순번
        
        filtered_patents.append(patent)
        filter_stats['passed'] += 1
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 특허: {len(filtered_patents):,}개")
    
    return filtered_patents, filter_stats


def save_filtered_data(filtered_patents: List[Dict]):
    """
    필터링된 특허 데이터를 JSON 파일로 저장합니다.
    data/train 폴더에 저장합니다.
    
    Args:
        filtered_patents: 필터링된 특허 데이터 리스트
    """
    # data/train 폴더에 필터링 후 데이터 저장
    train_output_path = Path(DATA_TRAIN_PATENT_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[저장] 필터링 후 데이터 저장 중: {train_output_path}")
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_patents, f, ensure_ascii=False, indent=2)
        
        print(f"[완료] 총 {len(filtered_patents):,}개의 필터링된 특허 데이터를 저장했습니다.")
        print(f"[저장 위치] {train_output_path}")
    except Exception as e:
        print(f"[오류] 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    메인 함수: data 폴더의 patent.json을 필터링하여 data/train 폴더에 저장
    """
    # data 폴더의 patent.json 파일 읽기
    print("\n[시작] data 폴더의 특허 데이터 읽기 중...")
    patents = load_patent_json()
    
    if not patents:
        print("[경고] 특허 데이터가 없습니다.")
        return
    
    original_count = len(patents)
    
    # 데이터 필터링
    filtered_patents, filter_stats = filter_patent_data(patents)
    
    filtered_count = len(filtered_patents)
    
    # train 폴더에 필터링 후 데이터 저장
    save_filtered_data(filtered_patents)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 특허 수: {original_count:,}개")
    print(f"2. 필터링된 특허 수: {filtered_count:,}개")
    
    # 교수 정보가 있는 특허 수
    professor_matched = len([p for p in filtered_patents if p.get("professor_info")])
    print(f"3. 교수 정보가 있는 특허 수: {professor_matched:,}개")
    
    # 필터링 상세 통계
    print(f"\n[필터링 상세 통계]")
    print(f"  - 필수 필드 null로 제외된 데이터: {filter_stats['null_field_filtered']:,}개")
    print(f"  - 날짜 조건(2015년 이전)으로 제외된 데이터: {filter_stats['date_filtered']:,}개")
    print(f"  - 최종 통과한 데이터: {filter_stats['passed']:,}개")
    
    # 필수 필드 정보 출력
    if filtered_patents:
        print(f"\n[필수 필드] null이면 안 되는 필드:")
        required_fields = ['kipris_application_name', 'mbr_sn', 'kipris_abstract', 'tech_invnt_se', 'kipris_register_status']
        for field in required_fields:
            print(f"   - {field}")
        
        print(f"\n[컬럼] 샘플 레코드의 컬럼 (처음 10개):")
        sample_keys = list(filtered_patents[0].keys())[:10]
        for key in sample_keys:
            print(f"   - {key}")
        if len(filtered_patents[0].keys()) > 10:
            print(f"   ... (총 {len(filtered_patents[0].keys())}개 컬럼)")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
