"""
논문 데이터 필터링 스크립트
test 폴더의 article.json을 필터링하여 필요한 컬럼만 추출하고,
abstract_description, abstract_translated, abstract를 우선순위에 따라 처리합니다.
"""

import json
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import ARTICLE_DATA_FILE, DATA_TRAIN_ARTICLE_FILE
from data_filtering.text_preprocessing import preprocess_text
from langdetect import detect, LangDetectException, DetectorFactory

# 언어 감지 재현성 설정
DetectorFactory.seed = 0


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


def detect_language(text: str) -> Optional[str]:
    """
    텍스트의 언어를 감지합니다.
    
    Args:
        text: 언어를 감지할 텍스트
        
    Returns:
        언어 코드 (예: 'ko', 'en') 또는 None (감지 실패 시)
    """
    if not text or not text.strip():
        return None
    
    try:
        # 최소 길이 체크 (너무 짧으면 정확도가 떨어짐)
        if len(text.strip()) < 10:
            return None
        
        language = detect(text)
        return language
    except LangDetectException:
        return None
    except Exception:
        return None


def parse_list_string(text: str) -> Optional[List]:
    """
    문자열로 된 리스트를 파싱합니다.
    예: "['한국어...', 'English...']" -> ['한국어...', 'English...']
    
    Args:
        text: 리스트 형태의 문자열
        
    Returns:
        파싱된 리스트 또는 None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # 리스트 형태의 문자열인지 확인
    if text.startswith('[') and text.endswith(']'):
        try:
            # Python 리스트로 파싱
            parsed_list = ast.literal_eval(text)
            if isinstance(parsed_list, list):
                return parsed_list
        except (ValueError, SyntaxError):
            pass
    
    return None


def select_abstract_from_list(abstract_list: List) -> str:
    """
    리스트에서 언어를 감지하여 우선순위에 맞게 하나의 초록을 선택합니다.
    우선순위: 한국어 > 영어 > 기타 > 첫 번째 항목
    
    Args:
        abstract_list: 초록 리스트
        
    Returns:
        선택된 초록 문자열
    """
    if len(abstract_list) == 0:
        return ""
    
    # 한국어 초록 우선 찾기
    korean_abstract = None
    english_abstract = None
    other_abstracts = []
    
    for item in abstract_list:
        if item is None:
            continue
        
        item_str = str(item).strip()
        if not item_str:
            continue
        
        lang = detect_language(item_str)
        
        if lang == 'ko':
            korean_abstract = item_str
        elif lang == 'en':
            english_abstract = item_str
        else:
            other_abstracts.append(item_str)
    
    # 우선순위: 한국어 > 영어 > 기타
    if korean_abstract:
        return korean_abstract
    elif english_abstract:
        return english_abstract
    elif other_abstracts:
        return other_abstracts[0]  # 첫 번째 것 사용
    else:
        # 언어 감지 실패한 경우, 첫 번째 항목 반환
        first_item = abstract_list[0]
        return str(first_item).strip() if first_item else ""


def process_abstract(article: Dict) -> Any:
    """
    abstract, abstract_description, abstract_translated를 우선순위에 따라 처리합니다.
    리스트 형태의 문자열인 경우 파싱하여 한국어/영어/기타 언어 중 우선순위에 맞게 하나만 선택합니다.
    
    우선순위:
    1. abstract (기존 값)이 있으면 → 그대로 사용
    2. 없고 abstract_description이 있으면 → abstract에 사용
    3. 없고 abstract_translated가 있으면 → abstract에 사용
    4. 3개 컬럼 모두 없으면 → null
    
    리스트 형태의 문자열 처리:
    - 파싱하여 리스트로 변환
    - 한국어 > 영어 > 기타 순으로 하나만 선택
    
    Args:
        article: 논문 데이터 딕셔너리
        
    Returns:
        처리된 abstract 값 (단일 문자열)
    """
    abstract = article.get("abstract")
    abstract_description = article.get("abstract_description")
    abstract_translated = article.get("abstract_translated")
    
    selected_abstract = None
    
    # 우선순위 1: abstract (기존 값)
    if has_value(abstract):
        selected_abstract = abstract
    # 우선순위 2: abstract_description
    elif has_value(abstract_description):
        selected_abstract = abstract_description
    # 우선순위 3: abstract_translated
    elif has_value(abstract_translated):
        selected_abstract = abstract_translated
    
    if selected_abstract is None:
        return None
    
    # 이미 리스트인 경우
    if isinstance(selected_abstract, list):
        return select_abstract_from_list(selected_abstract)
    
    # 문자열인 경우
    if isinstance(selected_abstract, str):
        # 리스트 형태의 문자열인지 확인하고 파싱
        parsed_list = parse_list_string(selected_abstract)
        if parsed_list:
            # 파싱된 리스트에서 언어 우선순위로 선택
            return select_abstract_from_list(parsed_list)
        else:
            # 일반 문자열인 경우 그대로 반환
            return selected_abstract
    
    # 그 외의 경우 문자열로 변환
    return str(selected_abstract) if selected_abstract else None


def parse_year(year_value: Any) -> Optional[int]:
    """
    연도 값을 파싱하여 정수로 변환합니다.
    
    Args:
        year_value: 연도 값 (문자열, 정수, 또는 None)
        
    Returns:
        파싱된 연도 (int) 또는 None
    """
    if year_value is None:
        return None
    
    try:
        if isinstance(year_value, int):
            return year_value if 1900 <= year_value <= 2100 else None
        elif isinstance(year_value, str):
            year_str = year_value.strip()
            if year_str.isdigit():
                year_int = int(year_str)
                return year_int if 1900 <= year_int <= 2100 else None
    except (ValueError, TypeError):
        pass
    
    return None


def has_invalid_metadata(metadata: Dict) -> bool:
    """
    metadata에 결측값이 있거나 "기타학술지(비정기발행학술지)"인지 확인합니다.
    
    Args:
        metadata: 메타데이터 딕셔너리
        
    Returns:
        True: 결측값이 있거나 "기타학술지(비정기발행학술지)"인 경우
        False: 유효한 경우
    """
    if not metadata:
        return True
    
    for key, value in metadata.items():
        # 결측값 확인
        if value is None or (isinstance(value, str) and not value.strip()):
            return True
        
        # "기타학술지(비정기발행학술지)" 확인
        if isinstance(value, str) and value.strip() == "기타학술지(비정기발행학술지)":
            return True
    
    return False


def filter_article_data(articles: List[Dict]) -> tuple:
    """
    논문 데이터에서 필요한 컬럼만 추출하고 abstract를 처리합니다.
    텍스트 전처리(수식/기호 제거, 100자 이상 5000자 이하 필터링)를 적용합니다.
    연도 필터링(2015년 이상) 및 metadata 필터링을 적용합니다.
    
    Args:
        articles: 논문 데이터 리스트
        
    Returns:
        (필터링된 논문 데이터 리스트, 필터링 통계)
    """
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 논문 수: {len(articles):,}개")
    
    filtered_articles = []
    filter_stats = {
        'total': len(articles),
        'year_filtered': 0,  # 연도 필터링으로 제외된 개수
        'metadata_filtered': 0,  # metadata 필터링으로 제외된 개수
        'abstract_processed': 0,
        'text_preprocessing_passed': 0,
        'text_preprocessing_failed': 0,
        'abstract_from_original': 0,
        'abstract_from_description': 0,
        'abstract_from_translated': 0,
        'abstract_null': 0
    }
    
    for idx, article in enumerate(articles, 1):
        if idx % 10000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(articles):,}개")
        
        # 1. 연도 필터링 (2015년 이상)
        year_value = article.get('YY')
        year = parse_year(year_value)
        
        if year is None or year < 2015:
            filter_stats['year_filtered'] += 1
            continue
        
        # 2. metadata 필터링 (결측값 또는 "기타학술지(비정기발행학술지)" 확인)
        metadata = {
            'THSS_PATICP_GBN': article.get('THSS_PATICP_GBN'),
            'JRNL_GBN': article.get('JRNL_GBN')
        }
        
        if has_invalid_metadata(metadata):
            filter_stats['metadata_filtered'] += 1
            continue
        
        # 원본 값 저장 (통계용)
        original_abstract = article.get("abstract")
        original_abstract_desc = article.get("abstract_description")
        original_abstract_trans = article.get("abstract_translated")
        
        # abstract 처리 (우선순위에 따라)
        processed_abstract = process_abstract(article)
        
        # 통계 업데이트 (abstract 출처)
        if has_value(original_abstract):
            filter_stats['abstract_from_original'] += 1
        elif has_value(original_abstract_desc):
            filter_stats['abstract_from_description'] += 1
        elif has_value(original_abstract_trans):
            filter_stats['abstract_from_translated'] += 1
        else:
            filter_stats['abstract_null'] += 1
        
        # 텍스트 전처리 (수식/기호 제거, 100자 이상 5000자 이하 필터링)
        preprocessed_text, is_valid = preprocess_text(processed_abstract, min_length=100, max_length=5000)
        
        if not is_valid:
            filter_stats['text_preprocessing_failed'] += 1
            continue  # 최소 길이 조건을 만족하지 않으면 제외
        
        filter_stats['text_preprocessing_passed'] += 1
        
        # 공통 컬럼 구조로 데이터 생성
        filtered_article = {
            'data_type': 'article',
            'no': len(filtered_articles) + 1,  # 필터링된 데이터의 순번
            'text': preprocessed_text,  # 전처리된 텍스트
            'title': article.get('THSS_NM'),  # 논문 제목
            'year': year,  # 파싱된 연도
            'professor_info': article.get('professor_info'),  # 교수 정보
            'metadata': metadata  # 필터링된 메타데이터
        }
        
        filtered_articles.append(filtered_article)
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 논문: {len(filtered_articles):,}개")
    
    return filtered_articles, filter_stats


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
    filtered_articles, filter_stats = filter_article_data(articles)
    
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
    
    # 필터링 상세 통계
    print(f"\n[필터링 상세 통계]")
    print(f"  - 연도 필터링 (2015년 미만 또는 연도 없음): {filter_stats['year_filtered']:,}개")
    print(f"  - Metadata 필터링 (결측값 또는 '기타학술지(비정기발행학술지)'): {filter_stats['metadata_filtered']:,}개")
    
    # Abstract 처리 통계
    print(f"\n[Abstract 처리 통계]")
    print(f"  - 기존 abstract 컬럼에서 가져온 경우: {filter_stats['abstract_from_original']:,}개")
    print(f"  - abstract_description에서 가져온 경우: {filter_stats['abstract_from_description']:,}개")
    print(f"  - abstract_translated에서 가져온 경우: {filter_stats['abstract_from_translated']:,}개")
    print(f"  - abstract가 null인 경우: {filter_stats['abstract_null']:,}개")
    
    # 텍스트 전처리 통계
    print(f"\n[텍스트 전처리 통계]")
    print(f"  - 전처리 통과 (100자 이상 5000자 이하): {filter_stats['text_preprocessing_passed']:,}개")
    print(f"  - 전처리 실패 (100자 미만 또는 5000자 초과): {filter_stats['text_preprocessing_failed']:,}개")
    if filter_stats['total'] > 0:
        total_filtered = filter_stats['year_filtered'] + filter_stats['metadata_filtered'] + filter_stats['text_preprocessing_failed']
        print(f"  - 전체 필터링률: {total_filtered / filter_stats['total'] * 100:.1f}%")
        print(f"    * 연도 필터링률: {filter_stats['year_filtered'] / filter_stats['total'] * 100:.1f}%")
        print(f"    * Metadata 필터링률: {filter_stats['metadata_filtered'] / filter_stats['total'] * 100:.1f}%")
        print(f"    * 텍스트 전처리 필터링률: {filter_stats['text_preprocessing_failed'] / filter_stats['total'] * 100:.1f}%")
    
    # 컬럼 정보 출력
    if filtered_articles:
        print(f"\n[컬럼] 공통 컬럼 구조:")
        for key in filtered_articles[0].keys():
            print(f"   - {key}")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
