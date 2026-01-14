"""
논문 데이터 필터링 스크립트
test 폴더의 article.json을 필터링하여 필요한 컬럼만 추출하고,
abstract_description, abstract_translated, abstract를 우선순위에 따라 처리합니다.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import ARTICLE_DATA_FILE, DATA_TRAIN_ARTICLE_FILE


def load_article_json(input_file: str = None) -> List[Dict]:
    """
    data 폴더의 article.json 파일을 읽어옵니다.
    
    Args:
        input_file: 입력 파일 경로 (None이면 설정 파일의 경로 사용)
        
    Returns:
        논문 데이터 리스트
    """
    if input_file is None:
        input_file = ARTICLE_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 논문 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        
        print(f"  - 총 {len(article_data):,}개의 논문 데이터 로드 완료")
        return article_data
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


def process_abstract(article: Dict) -> Any:
    """
    abstract, abstract_description, abstract_translated를 우선순위에 따라 처리합니다.
    
    우선순위:
    1. abstract (기존 값)이 있으면 → 그대로 사용
    2. 없고 abstract_description이 있으면 → abstract에 사용
    3. 없고 abstract_translated가 있으면 → abstract에 사용
    4. 3개 컬럼 모두 없으면 → null
    
    Args:
        article: 논문 데이터 딕셔너리
        
    Returns:
        처리된 abstract 값
    """
    abstract = article.get("abstract")
    abstract_description = article.get("abstract_description")
    abstract_translated = article.get("abstract_translated")
    
    # 우선순위 1: abstract (기존 값)
    if has_value(abstract):
        return abstract
    
    # 우선순위 2: abstract_description
    if has_value(abstract_description):
        return abstract_description
    
    # 우선순위 3: abstract_translated
    if has_value(abstract_translated):
        return abstract_translated
    
    # 우선순위 4: 3개 컬럼 모두 없으면 null
    return None


def filter_article_data(articles: List[Dict]) -> tuple:
    """
    논문 데이터에서 필요한 컬럼만 추출하고 abstract를 처리합니다.
    
    Args:
        articles: 논문 데이터 리스트
        
    Returns:
        (필터링된 논문 데이터 리스트, abstract 처리 통계)
    """
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 논문 수: {len(articles):,}개")
    
    # 유지할 컬럼 목록
    keep_columns = {
        'THSS_NM',
        'abstract',
        'YY',
        'THSS_PATICP_GBN',
        'JRNL_GBN',
        'professor_info'
    }
    
    filtered_articles = []
    abstract_stats = {
        'from_original': 0,
        'from_description': 0,
        'from_translated': 0,
        'null': 0
    }
    
    for idx, article in enumerate(articles, 1):
        if idx % 10000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(articles):,}개")
        
        # 필터링된 논문 데이터 생성
        filtered_article = {}
        
        for key in keep_columns:
            if key in article:
                filtered_article[key] = article[key]
            else:
                filtered_article[key] = None
        
        # 원본 값 저장 (통계용) - abstract 처리 전에 원본 데이터에서 가져옴
        original_abstract = article.get("abstract")
        original_abstract_desc = article.get("abstract_description")
        original_abstract_trans = article.get("abstract_translated")
        
        # abstract 처리 (우선순위에 따라)
        processed_abstract = process_abstract(article)
        filtered_article['abstract'] = processed_abstract
        
        # abstract_description과 abstract_translated는 제거 (병합 완료 후)
        
        # 통계 업데이트
        if has_value(original_abstract):
            abstract_stats['from_original'] += 1
        elif has_value(original_abstract_desc):
            abstract_stats['from_description'] += 1
        elif has_value(original_abstract_trans):
            abstract_stats['from_translated'] += 1
        else:
            abstract_stats['null'] += 1
        
        # data_type과 no 필드 추가
        filtered_article['data_type'] = 'article'
        filtered_article['no'] = idx
        
        filtered_articles.append(filtered_article)
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 논문: {len(filtered_articles):,}개")
    
    return filtered_articles, abstract_stats


def save_filtered_data(filtered_articles: List[Dict]):
    """
    필터링된 논문 데이터를 JSON 파일로 저장합니다.
    data/train 폴더에 저장합니다.
    
    Args:
        filtered_articles: 필터링된 논문 데이터 리스트
    """
    # data/train 폴더에 필터링 후 데이터 저장
    train_output_path = Path(DATA_TRAIN_ARTICLE_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[저장] 필터링 후 데이터 저장 중: {train_output_path}")
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, ensure_ascii=False, indent=2)
        
        print(f"[완료] 총 {len(filtered_articles):,}개의 필터링된 논문 데이터를 저장했습니다.")
        print(f"[저장 위치] {train_output_path}")
    except Exception as e:
        print(f"[오류] 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    메인 함수: data 폴더의 article.json을 필터링하여 data/test와 data/train 폴더에 저장
    """
    # data 폴더의 article.json 파일 읽기
    print("\n[시작] data 폴더의 논문 데이터 읽기 중...")
    articles = load_article_json()
    
    if not articles:
        print("[경고] 논문 데이터가 없습니다.")
        return
    
    original_count = len(articles)
    
    # 데이터 필터링
    filtered_articles, abstract_stats = filter_article_data(articles)
    
    filtered_count = len(filtered_articles)
    
    # train 폴더에 필터링 후 데이터 저장
    save_filtered_data(filtered_articles)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 논문 수: {original_count:,}개")
    print(f"2. 필터링된 논문 수: {filtered_count:,}개")
    
    # 교수 정보가 있는 논문 수
    professor_matched = len([a for a in filtered_articles if a.get("professor_info")])
    print(f"3. 교수 정보가 있는 논문 수: {professor_matched:,}개")
    
    # abstract 처리 통계
    print(f"\n[Abstract 처리 통계]")
    print(f"  - 기존 abstract 컬럼에서 가져온 경우: {abstract_stats['from_original']:,}개")
    print(f"  - abstract_description에서 가져온 경우: {abstract_stats['from_description']:,}개")
    print(f"  - abstract_translated에서 가져온 경우: {abstract_stats['from_translated']:,}개")
    print(f"  - abstract가 null인 경우: {abstract_stats['null']:,}개")
    
    # 컬럼 정보 출력
    if filtered_articles:
        print(f"\n[컬럼] 유지된 컬럼:")
        for key in filtered_articles[0].keys():
            print(f"   - {key}")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
