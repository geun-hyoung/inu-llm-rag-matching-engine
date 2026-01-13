"""
논문 데이터 탐색적 분석 (EDA)
교수와 논문의 관계를 중심으로 분석합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from langdetect import detect, DetectorFactory, LangDetectException

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_TRAIN_ARTICLE_FILE, EDA_RESULTS_DIR

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
sns.set_style("whitegrid")
sns.set_palette("husl")

# 언어 감지 재현성 설정
DetectorFactory.seed = 0


def load_article_data(file_path: str) -> List[Dict]:
    """논문 JSON 데이터를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[데이터 로드 완료] {len(data):,}개")
        return data
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 오류: {e}")
        return []


def analyze_basic_info(data: List[Dict]) -> Dict[str, Any]:
    """기본 정보 분석"""
    if not data:
        return {}
    
    return {
        "total_articles": len(data),
        "total_fields": len(data[0].keys()) if data else 0,
        "field_names": list(data[0].keys()) if data else [],
        "sample_record_keys": list(data[0].keys())[:10] if data else []
    }


def analyze_professor_article_relationship(data: List[Dict]) -> Dict[str, Any]:
    """교수-논문 관계 분석"""
    if not data:
        return {}
    
    # 교수별 논문 개수
    professor_article_count = defaultdict(int)
    professor_info_map = {}
    
    # 교수 정보별 통계
    professors_by_college = defaultdict(set)  # 대학별 교수 집합
    professors_by_department = defaultdict(set)  # 학과별 교수 집합
    professors_by_status = defaultdict(set)  # 재직 상태별 교수 집합
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        # 교수 식별자 (SQ 또는 EMP_NO)
        prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
        if prof_id:
            professor_article_count[prof_id] += 1
            professor_info_map[prof_id] = prof_info
        
        # 대학/학과/재직 상태별 집계
        college = prof_info.get("COLG_NM", "")
        department = prof_info.get("HG_NM", "")
        status = prof_info.get("HOOF_GBN", "")
        
        if prof_id:
            if college:
                professors_by_college[college].add(prof_id)
            if department:
                professors_by_department[department].add(prof_id)
            if status:
                professors_by_status[status].add(prof_id)
    
    # 교수별 논문 개수 분포
    article_counts = list(professor_article_count.values())
    
    return {
        "total_professors": len(professor_article_count),
        "total_articles": sum(professor_article_count.values()),
        "professor_article_distribution": {
            "min": min(article_counts) if article_counts else 0,
            "max": max(article_counts) if article_counts else 0,
            "mean": sum(article_counts) / len(article_counts) if article_counts else 0,
            "median": sorted(article_counts)[len(article_counts)//2] if article_counts else 0
        },
        "professors_by_article_count": {
            "1개": sum(1 for c in article_counts if c == 1),
            "2-5개": sum(1 for c in article_counts if 2 <= c <= 5),
            "6-10개": sum(1 for c in article_counts if 6 <= c <= 10),
            "11-20개": sum(1 for c in article_counts if 11 <= c <= 20),
            "21개 이상": sum(1 for c in article_counts if c >= 21)
        },
        "college_distribution": {college: len(profs) for college, profs in professors_by_college.items()},
        "department_distribution": {dept: len(profs) for dept, profs in professors_by_department.items()},
        "status_distribution": {status: len(profs) for status, profs in professors_by_status.items()}
    }


def analyze_article_type_by_professor(data: List[Dict]) -> Dict[str, Any]:
    """교수별 논문 유형 분석"""
    if not data:
        return {}
    
    # 교수별 유형 분포
    professor_type = defaultdict(lambda: defaultdict(int))
    type_overall = defaultdict(int)
    
    # 참여 구분별 분포
    participation_type = defaultdict(int)
    journal_type = defaultdict(int)
    
    for item in data:
        prof_info = item.get("professor_info", {})
        participation = item.get("THSS_PATICP_GBN", "")
        journal = item.get("JRNL_GBN", "")
        
        if participation:
            participation_type[participation] += 1
        if journal:
            journal_type[journal] += 1
        
        if not prof_info:
            continue
        
        prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
        if prof_id and participation:
            professor_type[prof_id][participation] += 1
            type_overall[participation] += 1
    
    # 교수별 주요 유형 (가장 많은 유형)
    professor_main_type = {}
    for prof_id, types in professor_type.items():
        if types:
            main_type = max(types.items(), key=lambda x: x[1])[0]
            professor_main_type[prof_id] = main_type
    
    main_type_distribution = Counter(professor_main_type.values())
    
    return {
        "overall_participation_distribution": dict(participation_type),
        "overall_journal_distribution": dict(journal_type),
        "professors_by_main_participation": dict(main_type_distribution),
        "participation_types": list(participation_type.keys()),
        "journal_types": list(journal_type.keys())
    }


def analyze_article_timeline(data: List[Dict]) -> Dict[str, Any]:
    """논문 발행 시기 분석 (연도별, 교수별)"""
    if not data:
        return {}
    
    # 연도별 논문 개수
    year_articles = defaultdict(int)
    year_professors = defaultdict(set)
    
    # 교수별 발행 연도
    professor_years = defaultdict(set)
    
    for item in data:
        year_str = item.get("YY", "")
        prof_info = item.get("professor_info", {})
        
        if not year_str or len(year_str) < 4:
            continue
        
        year = year_str[:4]
        year_articles[year] += 1
        
        prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", "")) if prof_info else ""
        if prof_id:
            year_professors[year].add(prof_id)
            professor_years[prof_id].add(year)
    
    # 교수별 활동 기간 (발행 연도 범위)
    professor_activity_periods = {}
    for prof_id, years in professor_years.items():
        if years:
            year_list = sorted([int(y) for y in years if y.isdigit()])
            if year_list:
                professor_activity_periods[prof_id] = {
                    "start_year": min(year_list),
                    "end_year": max(year_list),
                    "span_years": max(year_list) - min(year_list) + 1,
                    "total_years": len(years)
                }
    
    return {
        "year_distribution": dict(sorted(year_articles.items())),
        "professors_per_year": {year: len(profs) for year, profs in sorted(year_professors.items())},
        "activity_period_stats": {
            "avg_span": sum(p["span_years"] for p in professor_activity_periods.values()) / len(professor_activity_periods) if professor_activity_periods else 0,
            "max_span": max((p["span_years"] for p in professor_activity_periods.values()), default=0),
            "min_span": min((p["span_years"] for p in professor_activity_periods.values()), default=0)
        }
    }


def analyze_article_content(data: List[Dict]) -> Dict[str, Any]:
    """논문 내용 분석 (제목, 초록)"""
    if not data:
        return {}
    
    titles = []
    abstracts = []
    
    for item in data:
        title = item.get("THSS_NM", "")
        abstract = item.get("abstract", "")
        
        # abstract가 리스트인 경우 처리
        if isinstance(abstract, list):
            abstract = " ".join(str(a) for a in abstract if a)
        elif abstract is None:
            abstract = ""
        
        if title:
            titles.append(title)
        if abstract and abstract.strip():
            abstracts.append(abstract)
    
    # 길이 분석
    title_lengths = [len(t) for t in titles]
    abstract_lengths = [len(a) for a in abstracts]
    
    return {
        "titles": {
            "total": len(titles),
            "length_stats": {
                "min": min(title_lengths) if title_lengths else 0,
                "max": max(title_lengths) if title_lengths else 0,
                "mean": sum(title_lengths) / len(title_lengths) if title_lengths else 0,
                "median": sorted(title_lengths)[len(title_lengths)//2] if title_lengths else 0
            }
        },
        "abstracts": {
            "total": len(abstracts),
            "length_stats": {
                "min": min(abstract_lengths) if abstract_lengths else 0,
                "max": max(abstract_lengths) if abstract_lengths else 0,
                "mean": sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
                "median": sorted(abstract_lengths)[len(abstract_lengths)//2] if abstract_lengths else 0
            }
        },
        "content_completeness": {
            "has_title": len(titles),
            "has_abstract": len(abstracts),
            "has_both": sum(1 for item in data if item.get("THSS_NM") and item.get("abstract"))
        }
    }


def analyze_abstract_detailed(data: List[Dict]) -> Dict[str, Any]:
    """초록(abstract)에 대한 상세 분석"""
    if not data:
        return {}
    
    abstracts = []
    abstract_lengths = []
    
    for item in data:
        abstract = item.get("abstract", "")
        
        # abstract가 리스트인 경우 처리
        if isinstance(abstract, list):
            abstract = " ".join(str(a) for a in abstract if a)
        elif abstract is None:
            abstract = ""
        
        if abstract and abstract.strip():  # 비어있지 않은 경우만
            abstracts.append(abstract)
            abstract_lengths.append(len(abstract))
    
    total_items = len(data)
    missing_count = total_items - len(abstracts)
    missing_rate = (missing_count / total_items * 100) if total_items > 0 else 0
    
    if not abstract_lengths:
        return {
            "total_items": total_items,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "error": "초록 데이터가 없습니다."
        }
    
    # 기술 통계량 계산
    lengths_array = np.array(abstract_lengths)
    
    # 4분위수 계산
    q1 = np.percentile(lengths_array, 25)
    q2 = np.percentile(lengths_array, 50)  # 중앙값
    q3 = np.percentile(lengths_array, 75)
    iqr = q3 - q1  # 사분위 범위
    
    # 기술 통계량
    mean_length = np.mean(lengths_array)
    median_length = np.median(lengths_array)
    std_length = np.std(lengths_array)
    min_length = np.min(lengths_array)
    max_length = np.max(lengths_array)
    
    return {
        "total_items": total_items,
        "missing_count": missing_count,
        "missing_rate": round(missing_rate, 2),
        "valid_count": len(abstracts),
        "valid_rate": round((len(abstracts) / total_items * 100) if total_items > 0 else 0, 2),
        "descriptive_statistics": {
            "min": int(min_length),
            "max": int(max_length),
            "mean": round(mean_length, 2),
            "median": int(median_length),
            "std": round(std_length, 2),
            "q1": int(q1),
            "q2": int(q2),
            "q3": int(q3),
            "iqr": int(iqr)
        },
        "quartile_distribution": {
            "q1_under": int(np.sum(lengths_array <= q1)),
            "q1_to_q2": int(np.sum((lengths_array > q1) & (lengths_array <= q2))),
            "q2_to_q3": int(np.sum((lengths_array > q2) & (lengths_array <= q3))),
            "q3_over": int(np.sum(lengths_array > q3))
        },
        "length_summary": {
            "shortest": min(abstract_lengths),
            "longest": max(abstract_lengths),
            "shortest_text": abstracts[np.argmin(abstract_lengths)][:100] + "..." if abstracts else "",
            "longest_text": abstracts[np.argmax(abstract_lengths)][:100] + "..." if abstracts else ""
        }
    }


def visualize_abstract_distribution(data: List[Dict], output_dir: Path):
    """초록 길이 분포 시각화 - 간단하고 트렌드한 학술적 스타일"""
    if not data:
        return
    
    abstract_lengths = []
    for item in data:
        abstract = item.get("abstract", "")
        
        # abstract가 리스트인 경우 처리
        if isinstance(abstract, list):
            abstract = " ".join(str(a) for a in abstract if a)
        elif abstract is None:
            abstract = ""
        
        if abstract and abstract.strip():
            abstract_lengths.append(len(abstract))
    
    if not abstract_lengths:
        print("[경고] 시각화할 초록 데이터가 없습니다.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 깔끔한 단일 히스토그램 (학술적이고 트렌드한 스타일)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 히스토그램 (KDE 포함) - seaborn 스타일
    sns.histplot(abstract_lengths, bins=40, kde=True, 
                 color='#3498db', alpha=0.7, 
                 edgecolor='white', linewidth=0.5)
    
    # 통계 선 표시
    mean_val = np.mean(abstract_lengths)
    median_val = np.median(abstract_lengths)
    q1_val = np.percentile(abstract_lengths, 25)
    q3_val = np.percentile(abstract_lengths, 75)
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.0f}', zorder=5)
    ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.0f}', zorder=5)
    
    # 스타일링 - 학술적이고 트렌드한 느낌
    ax.set_xlabel('Abstract Length (characters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Article Abstract Lengths', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 그리드 (은은하게)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # 범례
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, fontsize=10)
    
    # 통계 정보를 텍스트로 간단히 추가 (하단에)
    stats_text = f'n = {len(abstract_lengths):,} | ' \
                 f'Q1: {q1_val:.0f} | Q3: {q3_val:.0f} | ' \
                 f'SD: {np.std(abstract_lengths):.1f}'
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            style='italic')
    
    plt.tight_layout()
    output_path = output_dir / "article_abstract_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[초록 분포 시각화 저장 완료] {output_path}")


def detect_language(text: str) -> str:
    """
    텍스트의 언어를 감지합니다.
    
    Args:
        text: 감지할 텍스트
        
    Returns:
        언어 코드 (예: 'ko', 'en', 'ja' 등) 또는 'unknown'
    """
    if not text or not text.strip():
        return 'unknown'
    
    try:
        # 짧은 텍스트는 정확도가 낮을 수 있음
        if len(text.strip()) < 10:
            return 'unknown'
        
        lang = detect(text)
        return lang
    except (LangDetectException, Exception):
        return 'unknown'


def preprocess_text_for_ngram(text: str) -> List[str]:
    """
    텍스트를 전처리하여 단어 리스트로 변환합니다.
    
    Args:
        text: 전처리할 텍스트
        
    Returns:
        단어 리스트
    """
    if not text:
        return []
    
    # 소문자 변환 및 특수문자 제거 (한글, 영문, 숫자만 유지)
    text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
    
    # 공백으로 분리
    words = text.split()
    
    # 빈 문자열 제거 및 최소 길이 필터링 (1글자 이상)
    words = [w.strip() for w in words if len(w.strip()) > 0]
    
    return words


def extract_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    단어 리스트에서 n-gram을 추출합니다.
    
    Args:
        words: 단어 리스트
        n: n-gram 크기 (1 또는 2)
        
    Returns:
        n-gram 리스트
    """
    if n < 1 or len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def analyze_language_distribution(data: List[Dict], text_field: str = "abstract") -> Dict[str, Any]:
    """
    텍스트 데이터의 언어 분포를 분석합니다.
    
    Args:
        data: 데이터 리스트
        text_field: 분석할 텍스트 필드명
        
    Returns:
        언어 분포 통계
    """
    if not data:
        return {}
    
    language_counts = defaultdict(int)
    language_samples = defaultdict(list)
    
    for item in data:
        text = item.get(text_field, "")
        
        # 리스트인 경우 처리
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        elif text is None:
            text = ""
        
        if text and text.strip():
            lang = detect_language(text)
            language_counts[lang] += 1
            
            # 각 언어별 샘플 저장 (최대 3개)
            if len(language_samples[lang]) < 3:
                language_samples[lang].append(text[:100] + "..." if len(text) > 100 else text)
    
    total = sum(language_counts.values())
    
    return {
        "total_texts": total,
        "language_distribution": dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)),
        "language_percentage": {
            lang: round((count / total * 100), 2) if total > 0 else 0
            for lang, count in language_counts.items()
        },
        "language_samples": {lang: samples for lang, samples in language_samples.items()}
    }


def create_wordcloud_from_ngrams(ngrams: List[Tuple], n: int, output_path: Path, 
                                  title: str = "Word Cloud", max_words: int = 100):
    """
    n-gram에서 워드 클라우드를 생성합니다.
    
    Args:
        ngrams: n-gram 리스트
        n: n-gram 크기 (1 또는 2)
        output_path: 저장 경로
        title: 제목
        max_words: 최대 단어 수
    """
    if not ngrams:
        print(f"[경고] {title} 생성할 n-gram이 없습니다.")
        return
    
    # n-gram 빈도 계산
    ngram_counter = Counter(ngrams)
    
    # 상위 n-gram 선택
    top_ngrams = ngram_counter.most_common(max_words)
    
    # n-gram을 문자열로 변환 (공백으로 연결)
    word_freq = {}
    for ngram, count in top_ngrams:
        if n == 1:
            word = ngram[0] if ngram else ""
        else:
            word = " ".join(ngram)
        
        if word:
            word_freq[word] = count
    
    if not word_freq:
        print(f"[경고] {title} 생성할 단어가 없습니다.")
        return
    
    # 워드 클라우드 생성 (세련된 스타일)
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',  # 세련된 색상 팔레트
        max_words=max_words,
        relative_scaling=0.5,
        font_path='malgun.ttf' if sys.platform == 'win32' else None,  # Windows 한글 폰트
        collocations=False
    ).generate_from_frequencies(word_freq)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[워드 클라우드 저장 완료] {output_path}")


def analyze_text_for_rag(data: List[Dict], text_field: str, data_type: str, output_dir: Path) -> Dict[str, Any]:
    """
    RAG 분석에 사용할 텍스트를 종합적으로 분석합니다.
    
    Args:
        data: 데이터 리스트
        text_field: 분석할 텍스트 필드명
        data_type: 데이터 유형 ('article', 'patent', 'project')
        output_dir: 출력 디렉토리
        
    Returns:
        분석 결과 딕셔너리
    """
    if not data:
        return {}
    
    print(f"\n[RAG 텍스트 분석 시작] {data_type} - {text_field}")
    
    # 텍스트 추출 및 전처리
    all_texts = []
    all_words = []
    
    for item in data:
        text = item.get(text_field, "")
        
        # 리스트인 경우 처리
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        elif text is None:
            text = ""
        
        if text and text.strip():
            all_texts.append(text)
            words = preprocess_text_for_ngram(text)
            all_words.extend(words)
    
    if not all_texts:
        print(f"[경고] 분석할 텍스트가 없습니다.")
        return {}
    
    # 1-gram 및 2-gram 추출
    print("  - n-gram 추출 중...")
    unigrams = all_words  # 1-gram은 단어 그대로
    bigrams = []
    
    for text in all_texts:
        words = preprocess_text_for_ngram(text)
        bigrams.extend(extract_ngrams(words, 2))
    
    # 언어 분포 분석
    print("  - 언어 감지 중...")
    language_stats = analyze_language_distribution(data, text_field)
    
    # 워드 클라우드 생성
    print("  - 워드 클라우드 생성 중...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1-gram 워드 클라우드
    unigram_path = output_dir / f"{data_type}_1gram_wordcloud.png"
    create_wordcloud_from_ngrams(
        [(w,) for w in unigrams], 
        n=1, 
        output_path=unigram_path,
        title=f"{data_type.upper()} 1-gram Word Cloud"
    )
    
    # 2-gram 워드 클라우드
    bigram_path = output_dir / f"{data_type}_2gram_wordcloud.png"
    create_wordcloud_from_ngrams(
        bigrams, 
        n=2, 
        output_path=bigram_path,
        title=f"{data_type.upper()} 2-gram Word Cloud"
    )
    
    # 통계 정보
    unigram_counter = Counter(unigrams)
    bigram_counter = Counter(bigrams)
    
    # top n-grams를 문자열 키로 변환
    top_unigrams_list = unigram_counter.most_common(20)
    top_bigrams_list = bigram_counter.most_common(20)
    
    top_unigrams_dict = {str(word): count for word, count in top_unigrams_list}
    top_bigrams_dict = {" ".join(ngram) if isinstance(ngram, tuple) else str(ngram): count 
                        for ngram, count in top_bigrams_list}
    
    return {
        "text_field": text_field,
        "total_texts": len(all_texts),
        "total_words": len(all_words),
        "unique_unigrams": len(unigram_counter),
        "unique_bigrams": len(bigram_counter),
        "top_unigrams": top_unigrams_dict,
        "top_bigrams": top_bigrams_dict,
        "language_distribution": language_stats
    }


def analyze_professor_info_completeness(data: List[Dict]) -> Dict[str, Any]:
    """교수 정보 완전성 분석"""
    if not data:
        return {}
    
    prof_fields = ["SQ", "EMP_NO", "NM", "GEN_GBN", "BIRTH_DT", "NAT_GBN", 
                   "RECHER_REG_NO", "WKGD_NM", "COLG_NM", "HG_NM", 
                   "HOOF_GBN", "HANDP_NO", "OFCE_TELNO", "EMAIL"]
    
    field_completeness = defaultdict(int)
    total_with_prof = 0
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        total_with_prof += 1
        for field in prof_fields:
            if prof_info.get(field):
                field_completeness[field] += 1
    
    completeness_rate = {
        field: (count / total_with_prof * 100) if total_with_prof > 0 else 0
        for field, count in field_completeness.items()
    }
    
    return {
        "total_with_professor_info": total_with_prof,
        "field_completeness_rate": dict(sorted(completeness_rate.items(), key=lambda x: x[1], reverse=True)),
        "most_complete_fields": list(sorted(completeness_rate.items(), key=lambda x: x[1], reverse=True))[:5],
        "least_complete_fields": list(sorted(completeness_rate.items(), key=lambda x: x[1]))[:5]
    }


def analyze_college_department_articles(data: List[Dict]) -> Dict[str, Any]:
    """대학/학과별 논문 분석 (교수-논문 매핑 중심)"""
    if not data:
        return {}
    
    college_articles = defaultdict(int)
    department_articles = defaultdict(int)
    college_department = defaultdict(lambda: defaultdict(int))
    college_professors = defaultdict(set)  # 단과대별 교수 집합
    department_professors = defaultdict(set)  # 학과별 교수 집합
    college_professor_article_count = defaultdict(lambda: defaultdict(int))  # 단과대별 교수별 논문수
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        college = prof_info.get("COLG_NM", "")
        department = prof_info.get("HG_NM", "")
        prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
        
        if college:
            college_articles[college] += 1
            if prof_id:
                college_professors[college].add(prof_id)
                college_professor_article_count[college][prof_id] += 1
        
        if department:
            department_articles[department] += 1
            if prof_id:
                department_professors[department].add(prof_id)
        
        if college and department:
            college_department[college][department] += 1
    
    # 단과대별 평균 논문 수 (교수당)
    college_avg_articles = {}
    for college, prof_articles in college_professor_article_count.items():
        prof_count = len(prof_articles)
        if prof_count > 0:
            avg_articles = sum(prof_articles.values()) / prof_count
            college_avg_articles[college] = round(avg_articles, 2)
    
    # 단과대별 교수당 평균 논문수 순위
    top_colleges_by_avg = sorted(college_avg_articles.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "college_article_distribution": dict(sorted(college_articles.items(), key=lambda x: x[1], reverse=True)),
        "department_article_distribution": dict(sorted(department_articles.items(), key=lambda x: x[1], reverse=True)),
        "top_colleges": list(sorted(college_articles.items(), key=lambda x: x[1], reverse=True))[:10],
        "top_departments": list(sorted(department_articles.items(), key=lambda x: x[1], reverse=True))[:10],
        "college_professor_count": {college: len(profs) for college, profs in college_professors.items()},
        "department_professor_count": {dept: len(profs) for dept, profs in department_professors.items()},
        "college_avg_articles_per_professor": college_avg_articles,
        "top_colleges_by_avg_articles": top_colleges_by_avg,
        "college_department_matrix": {
            college: dict(depts) 
            for college, depts in college_department.items()
        }
    }


def save_results(results: Dict[str, Any], output_path: Path):
    """결과를 JSON 파일로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[결과 저장 완료] {output_path}")


def print_summary(results: Dict[str, Any]):
    """요약 정보를 출력합니다."""
    print("\n" + "=" * 70)
    print("[논문 데이터 EDA 요약] 교수와 논문 관계 중심")
    print("=" * 70)
    
    # 기본 정보
    if "basic_info" in results:
        basic = results["basic_info"]
        print(f"\n[1] 기본 정보")
        print(f"   - 총 논문 개수: {basic.get('total_articles', 0):,}개")
        print(f"   - 필드 수: {basic.get('total_fields', 0)}개")
    
    # 교수-논문 관계
    if "professor_article_relationship" in results:
        rel = results["professor_article_relationship"]
        print(f"\n[2] 교수-논문 관계")
        print(f"   - 총 교수 수: {rel.get('total_professors', 0):,}명")
        print(f"   - 총 논문 수: {rel.get('total_articles', 0):,}개")
        
        dist = rel.get("professor_article_distribution", {})
        print(f"   - 교수당 평균 논문 수: {dist.get('mean', 0):.2f}개")
        print(f"   - 교수당 중앙값 논문 수: {dist.get('median', 0)}개")
        print(f"   - 최다 논문 보유 교수: {dist.get('max', 0)}개")
        
        count_dist = rel.get("professors_by_article_count", {})
        print(f"   - 논문 개수별 교수 분포:")
        for range_str, count in count_dist.items():
            print(f"     * {range_str}: {count:,}명")
        
        print(f"\n   - 대학별 교수 수:")
        for college, count in list(rel.get("college_distribution", {}).items())[:5]:
            print(f"     * {college}: {count:,}명")
        
        print(f"\n   - 학과별 교수 수:")
        for dept, count in list(rel.get("department_distribution", {}).items())[:5]:
            print(f"     * {dept}: {count:,}명")
    
    # 논문 유형
    if "article_type" in results:
        atype = results["article_type"]
        print(f"\n[3] 논문 유형 분석")
        participation = atype.get("overall_participation_distribution", {})
        print(f"   - 참여 구분별 분포:")
        for part, count in sorted(participation.items(), key=lambda x: x[1], reverse=True):
            print(f"     * {part}: {count:,}개")
        
        journal = atype.get("overall_journal_distribution", {})
        print(f"   - 저널 구분별 분포:")
        for jrnl, count in sorted(journal.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     * {jrnl}: {count:,}개")
    
    # 연도별 추이
    if "timeline" in results:
        timeline = results["timeline"]
        year_dist = timeline.get("year_distribution", {})
        if year_dist:
            print(f"\n[4] 연도별 발행 추이")
            recent_years = list(sorted(year_dist.items(), reverse=True))[:5]
            for year, count in recent_years:
                prof_count = timeline.get("professors_per_year", {}).get(year, 0)
                print(f"   - {year}년: {count:,}개 (교수 {prof_count:,}명)")
    
    # 대학/학과별 논문
    if "college_department" in results:
        cd = results["college_department"]
        print(f"\n[5] 대학/학과별 논문 분포 (교수-논문 매핑)")
        print(f"   - 상위 대학 (논문 수):")
        for college, count in cd.get("top_colleges", [])[:5]:
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            avg_articles = cd.get("college_avg_articles_per_professor", {}).get(college, 0)
            print(f"     * {college}: {count:,}개 (교수 {prof_count}명, 교수당 평균 {avg_articles:.2f}개)")
        print(f"   - 상위 학과 (논문 수):")
        for dept, count in cd.get("top_departments", [])[:5]:
            prof_count = cd.get("department_professor_count", {}).get(dept, 0)
            print(f"     * {dept}: {count:,}개 (교수 {prof_count}명)")
        print(f"   - 교수당 평균 논문수 상위 단과대:")
        for college, avg in cd.get("top_colleges_by_avg_articles", [])[:5]:
            total = cd.get("college_article_distribution", {}).get(college, 0)
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            print(f"     * {college}: 교수당 평균 {avg:.2f}개 (총 {total}개, 교수 {prof_count}명)")
    
    # 논문 내용
    if "content" in results:
        content = results["content"]
        print(f"\n[6] 논문 내용 분석")
        titles = content.get("titles", {})
        abstracts = content.get("abstracts", {})
        print(f"   - 제목: {titles.get('total', 0):,}개")
        if titles.get("length_stats"):
            print(f"     평균 길이: {titles['length_stats'].get('mean', 0):.1f}자")
        print(f"   - 초록: {abstracts.get('total', 0):,}개")
        if abstracts.get("length_stats"):
            print(f"     평균 길이: {abstracts['length_stats'].get('mean', 0):.1f}자")
    
    # 초록 상세 분석
    if "abstract_detailed" in results:
        abs_detail = results["abstract_detailed"]
        if "error" not in abs_detail:
            print(f"\n[8] 초록(Abstract) 상세 분석")
            print(f"   - 총 데이터: {abs_detail.get('total_items', 0):,}개")
            print(f"   - 유효 초록: {abs_detail.get('valid_count', 0):,}개 ({abs_detail.get('valid_rate', 0)}%)")
            print(f"   - 결측치: {abs_detail.get('missing_count', 0):,}개 ({abs_detail.get('missing_rate', 0)}%)")
            
            stats = abs_detail.get("descriptive_statistics", {})
            if stats:
                print(f"\n   - 기술 통계량:")
                print(f"     * 최소값: {stats.get('min', 0):,}자")
                print(f"     * 최대값: {stats.get('max', 0):,}자")
                print(f"     * 평균: {stats.get('mean', 0):.2f}자")
                print(f"     * 중앙값 (Q2): {stats.get('median', 0):,}자")
                print(f"     * 표준편차: {stats.get('std', 0):.2f}자")
                print(f"\n   - 4분위수:")
                print(f"     * Q1 (1사분위수): {stats.get('q1', 0):,}자")
                print(f"     * Q2 (2사분위수, 중앙값): {stats.get('q2', 0):,}자")
                print(f"     * Q3 (3사분위수): {stats.get('q3', 0):,}자")
                print(f"     * IQR (사분위 범위): {stats.get('iqr', 0):,}자")
            
            quartile = abs_detail.get("quartile_distribution", {})
            if quartile:
                print(f"\n   - 4분위수별 분포:")
                print(f"     * Q1 이하: {quartile.get('q1_under', 0):,}개")
                print(f"     * Q1~Q2: {quartile.get('q1_to_q2', 0):,}개")
                print(f"     * Q2~Q3: {quartile.get('q2_to_q3', 0):,}개")
                print(f"     * Q3 초과: {quartile.get('q3_over', 0):,}개")
    
    # 교수 정보 완전성
    if "professor_completeness" in results:
        comp = results["professor_completeness"]
        print(f"\n[7] 교수 정보 완전성")
        print(f"   - 교수 정보가 있는 논문: {comp.get('total_with_professor_info', 0):,}개")
        print(f"   - 가장 완전한 필드:")
        for field, rate in comp.get("most_complete_fields", [])[:3]:
            print(f"     * {field}: {rate:.1f}%")
    
    print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    print("[논문 데이터 EDA 시작] 교수-논문 관계 중심...")
    
    # 데이터 로드 (data/train 폴더의 필터링된 데이터 사용)
    data = load_article_data(DATA_TRAIN_ARTICLE_FILE)
    
    if not data:
        print("[오류] 분석할 데이터가 없습니다.")
        return
    
    # 분석 수행
    results = {
        "basic_info": analyze_basic_info(data),
        "professor_article_relationship": analyze_professor_article_relationship(data),
        "article_type": analyze_article_type_by_professor(data),
        "timeline": analyze_article_timeline(data),
        "content": analyze_article_content(data),
        "professor_completeness": analyze_professor_info_completeness(data),
        "college_department": analyze_college_department_articles(data),
        "abstract_detailed": analyze_abstract_detailed(data)
    }
    
    # RAG 텍스트 분석 (abstract)
    print("\n[RAG 텍스트 분석 시작]")
    rag_analysis = analyze_text_for_rag(
        data, 
        text_field="abstract", 
        data_type="article",
        output_dir=Path(EDA_RESULTS_DIR)
    )
    results["rag_text_analysis"] = rag_analysis
    
    # 결과 출력
    print_summary(results)
    
    # 결과 저장
    output_path = Path(EDA_RESULTS_DIR) / "article_eda_results.json"
    save_results(results, output_path)
    
    # 초록 분포 시각화
    visualize_abstract_distribution(data, Path(EDA_RESULTS_DIR))


if __name__ == "__main__":
    main()
