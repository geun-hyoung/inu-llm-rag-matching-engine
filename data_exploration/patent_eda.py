"""
특허 데이터 탐색적 분석 (EDA)
교수와 특허의 관계를 중심으로 분석합니다.
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
from config.settings import DATA_TRAIN_PATENT_FILE, EDA_RESULTS_DIR

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
sns.set_style("whitegrid")
sns.set_palette("husl")

# 언어 감지 재현성 설정
DetectorFactory.seed = 0


def load_patent_data(file_path: str) -> List[Dict]:
    """특허 JSON 데이터를 로드합니다."""
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
        "total_patents": len(data),
        "total_fields": len(data[0].keys()) if data else 0,
        "field_names": list(data[0].keys()) if data else [],
        "sample_record_keys": list(data[0].keys())[:10] if data else []
    }


def analyze_professor_patent_relationship(data: List[Dict]) -> Dict[str, Any]:
    """교수-특허 관계 분석"""
    if not data:
        return {}
    
    # 교수별 특허 개수
    professor_patent_count = defaultdict(int)
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
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "")
        if prof_id:
            professor_patent_count[prof_id] += 1
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
    
    # 교수별 특허 개수 분포
    patent_counts = list(professor_patent_count.values())
    
    return {
        "total_professors": len(professor_patent_count),
        "total_patents": sum(professor_patent_count.values()),
        "professor_patent_distribution": {
            "min": min(patent_counts) if patent_counts else 0,
            "max": max(patent_counts) if patent_counts else 0,
            "mean": sum(patent_counts) / len(patent_counts) if patent_counts else 0,
            "median": sorted(patent_counts)[len(patent_counts)//2] if patent_counts else 0
        },
        "professors_by_patent_count": {
            "1개": sum(1 for c in patent_counts if c == 1),
            "2-5개": sum(1 for c in patent_counts if 2 <= c <= 5),
            "6-10개": sum(1 for c in patent_counts if 6 <= c <= 10),
            "11-20개": sum(1 for c in patent_counts if 11 <= c <= 20),
            "21개 이상": sum(1 for c in patent_counts if c >= 21)
        },
        "college_distribution": {college: len(profs) for college, profs in professors_by_college.items()},
        "department_distribution": {dept: len(profs) for dept, profs in professors_by_department.items()},
        "status_distribution": {status: len(profs) for status, profs in professors_by_status.items()}
    }


def analyze_patent_status_by_professor(data: List[Dict]) -> Dict[str, Any]:
    """교수별 특허 상태 분석"""
    if not data:
        return {}
    
    # 교수별 상태 분포
    professor_status = defaultdict(lambda: defaultdict(int))
    status_overall = defaultdict(int)
    
    for item in data:
        prof_info = item.get("professor_info", {})
        status = item.get("kipris_register_status", "")
        
        if not prof_info or not status:
            continue
        
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "")
        if prof_id:
            professor_status[prof_id][status] += 1
            status_overall[status] += 1
    
    # 교수별 주요 상태 (가장 많은 상태)
    professor_main_status = {}
    for prof_id, statuses in professor_status.items():
        if statuses:
            main_status = max(statuses.items(), key=lambda x: x[1])[0]
            professor_main_status[prof_id] = main_status
    
    main_status_distribution = Counter(professor_main_status.values())
    
    return {
        "overall_status_distribution": dict(status_overall),
        "professors_by_main_status": dict(main_status_distribution),
        "status_types": list(status_overall.keys())
    }


def analyze_patent_timeline(data: List[Dict]) -> Dict[str, Any]:
    """특허 출원 시기 분석 (연도별, 교수별)"""
    if not data:
        return {}
    
    # 연도별 출원 개수
    year_patents = defaultdict(int)
    year_professors = defaultdict(set)
    
    # 교수별 출원 연도
    professor_years = defaultdict(set)
    
    for item in data:
        date_str = item.get("kipris_application_date", "")
        prof_info = item.get("professor_info", {})
        
        if not date_str or len(date_str) < 4:
            continue
        
        year = date_str[:4]
        year_patents[year] += 1
        
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "") if prof_info else ""
        if prof_id:
            year_professors[year].add(prof_id)
            professor_years[prof_id].add(year)
    
    # 교수별 활동 기간 (출원 연도 범위)
    professor_activity_periods = {}
    for prof_id, years in professor_years.items():
        if years:
            year_list = sorted([int(y) for y in years])
            professor_activity_periods[prof_id] = {
                "start_year": min(year_list),
                "end_year": max(year_list),
                "span_years": max(year_list) - min(year_list) + 1,
                "total_years": len(years)
            }
    
    return {
        "year_distribution": dict(sorted(year_patents.items())),
        "professors_per_year": {year: len(profs) for year, profs in sorted(year_professors.items())},
        "activity_period_stats": {
            "avg_span": sum(p["span_years"] for p in professor_activity_periods.values()) / len(professor_activity_periods) if professor_activity_periods else 0,
            "max_span": max((p["span_years"] for p in professor_activity_periods.values()), default=0),
            "min_span": min((p["span_years"] for p in professor_activity_periods.values()), default=0)
        }
    }


def analyze_patent_content(data: List[Dict]) -> Dict[str, Any]:
    """특허 내용 분석 (제목, 요약)"""
    if not data:
        return {}
    
    titles = []
    abstracts = []
    
    for item in data:
        title = item.get("title", "")
        text = item.get("text", "")
        
        if title:
            titles.append(title)
        if text:
            abstracts.append(text)
    
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
            "has_text": len(abstracts),
            "has_both": sum(1 for item in data if item.get("title") and item.get("text"))
        }
    }


def analyze_abstract_detailed(data: List[Dict]) -> Dict[str, Any]:
    """텍스트(text)에 대한 상세 분석"""
    if not data:
        return {}
    
    texts = []
    text_lengths = []
    
    for item in data:
        text = item.get("text", "")
        if text and text.strip():  # 비어있지 않은 경우만
            texts.append(text)
            text_lengths.append(len(text))
    
    total_items = len(data)
    missing_count = total_items - len(texts)
    missing_rate = (missing_count / total_items * 100) if total_items > 0 else 0
    
    if not text_lengths:
        return {
            "total_items": total_items,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "error": "텍스트 데이터가 없습니다."
        }
    
    # 기술 통계량 계산
    lengths_array = np.array(text_lengths)
    
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
    
    # 분위수별 개수
    q1_count = np.sum(lengths_array <= q1)
    q2_count = np.sum(lengths_array <= q2)
    q3_count = np.sum(lengths_array <= q3)
    
    return {
        "total_items": total_items,
        "missing_count": missing_count,
        "missing_rate": round(missing_rate, 2),
        "valid_count": len(texts),
        "valid_rate": round((len(texts) / total_items * 100) if total_items > 0 else 0, 2),
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
            "q1_under": int(q1_count),
            "q1_to_q2": int(np.sum((lengths_array > q1) & (lengths_array <= q2))),
            "q2_to_q3": int(np.sum((lengths_array > q2) & (lengths_array <= q3))),
            "q3_over": int(np.sum(lengths_array > q3))
        },
        "length_summary": {
            "shortest": min(text_lengths),
            "longest": max(text_lengths),
            "shortest_text": texts[np.argmin(text_lengths)][:100] + "..." if texts else "",
            "longest_text": texts[np.argmax(text_lengths)][:100] + "..." if texts else ""
        }
    }


def analyze_metadata(data: List[Dict]) -> Dict[str, Any]:
    """metadata 분석 (AHP를 위한 정보 수집)"""
    if not data:
        return {}
    
    # 발명 구분 (tech_invnt_se) 분석
    invention_type_dist = defaultdict(int)
    invention_type_by_professor = defaultdict(set)
    
    # 등록 상태 (kipris_register_status) 분석
    register_status_dist = defaultdict(int)
    register_status_by_professor = defaultdict(set)
    
    # 출원일자 (kipris_application_date) 분석
    application_years = []
    application_year_dist = defaultdict(int)
    
    # 교수별 metadata 통계
    professor_metadata = defaultdict(lambda: {
        'invention_types': set(),
        'register_statuses': set(),
        'total_patents': 0,
        'application_years': []
    })
    
    for item in data:
        metadata = item.get("metadata", {})
        prof_info = item.get("professor_info", {})
        
        if prof_info:
            prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
        else:
            prof_id = None
        
        # 발명 구분 분석
        invention_type = metadata.get("tech_invnt_se", "")
        if invention_type:
            invention_type_dist[invention_type] += 1
            if prof_id:
                invention_type_by_professor[invention_type].add(prof_id)
                professor_metadata[prof_id]['invention_types'].add(invention_type)
        
        # 등록 상태 분석
        register_status = metadata.get("kipris_register_status", "")
        if register_status:
            register_status_dist[register_status] += 1
            if prof_id:
                register_status_by_professor[register_status].add(prof_id)
                professor_metadata[prof_id]['register_statuses'].add(register_status)
        
        # 출원일자 분석
        application_date = metadata.get("kipris_application_date", "")
        if application_date:
            date_str = str(application_date).strip()
            # yyyymmdd 형식에서 연도 추출
            if len(date_str) >= 4 and date_str[:4].isdigit():
                year = int(date_str[:4])
                if 1900 <= year <= 2100:
                    application_years.append(year)
                    application_year_dist[year] += 1
                    if prof_id:
                        professor_metadata[prof_id]['application_years'].append(year)
        
        if prof_id:
            professor_metadata[prof_id]['total_patents'] += 1
    
    # 교수별 metadata 요약
    prof_metadata_summary = {}
    for prof_id, info in professor_metadata.items():
        years = info['application_years']
        prof_metadata_summary[prof_id] = {
            'total_patents': info['total_patents'],
            'invention_types_count': len(info['invention_types']),
            'register_statuses_count': len(info['register_statuses']),
            'invention_types': list(info['invention_types']),
            'register_statuses': list(info['register_statuses']),
            'application_years': sorted(set(years)) if years else [],
            'earliest_application_year': min(years) if years else None,
            'latest_application_year': max(years) if years else None
        }
    
    return {
        "invention_type_distribution": dict(sorted(invention_type_dist.items(), key=lambda x: x[1], reverse=True)),
        "invention_type_by_professor_count": {
            k: len(v) for k, v in sorted(invention_type_by_professor.items(), key=lambda x: len(x[1]), reverse=True)
        },
        "register_status_distribution": dict(sorted(register_status_dist.items(), key=lambda x: x[1], reverse=True)),
        "register_status_by_professor_count": {
            k: len(v) for k, v in sorted(register_status_by_professor.items(), key=lambda x: len(x[1]), reverse=True)
        },
        "application_year_distribution": dict(sorted(application_year_dist.items())),
        "application_year_stats": {
            "min": min(application_years) if application_years else None,
            "max": max(application_years) if application_years else None,
            "mean": sum(application_years) / len(application_years) if application_years else None,
            "median": sorted(application_years)[len(application_years)//2] if application_years else None
        },
        "professor_metadata_summary": prof_metadata_summary,
        "total_with_metadata": len([item for item in data if item.get("metadata")]),
        "metadata_completeness": {
            "has_invention_type": len([item for item in data if item.get("metadata", {}).get("tech_invnt_se")]),
            "has_register_status": len([item for item in data if item.get("metadata", {}).get("kipris_register_status")]),
            "has_application_date": len([item for item in data if item.get("metadata", {}).get("kipris_application_date")]),
            "has_all": len([item for item in data if item.get("metadata", {}).get("tech_invnt_se") and item.get("metadata", {}).get("kipris_register_status") and item.get("metadata", {}).get("kipris_application_date")])
        }
    }


def visualize_abstract_distribution(data: List[Dict], output_dir: Path):
    """텍스트 길이 분포 시각화 - 간단하고 트렌드한 학술적 스타일"""
    if not data:
        return
    
    text_lengths = []
    for item in data:
        text = item.get("text", "")
        if text and text.strip():
            text_lengths.append(len(text))
    
    if not text_lengths:
        print("[경고] 시각화할 텍스트 데이터가 없습니다.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 깔끔한 단일 히스토그램 (학술적이고 트렌드한 스타일)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 히스토그램 (KDE 포함) - seaborn 스타일
    sns.histplot(text_lengths, bins=40, kde=True, 
                 color='#3498db', alpha=0.7, 
                 edgecolor='white', linewidth=0.5)
    
    # 통계 선 표시
    mean_val = np.mean(text_lengths)
    median_val = np.median(text_lengths)
    q1_val = np.percentile(text_lengths, 25)
    q3_val = np.percentile(text_lengths, 75)
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.0f}', zorder=5)
    ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.0f}', zorder=5)
    
    # 스타일링 - 학술적이고 트렌드한 느낌
    ax.set_xlabel('Text Length (characters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Patent Text Lengths', 
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
    stats_text = f'n = {len(text_lengths):,} | ' \
                 f'Q1: {q1_val:.0f} | Q3: {q3_val:.0f} | ' \
                 f'SD: {np.std(text_lengths):.1f}'
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            style='italic')
    
    plt.tight_layout()
    output_path = output_dir / "patent_text_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[텍스트 분포 시각화 저장 완료] {output_path}")


def detect_language(text: str) -> str:
    """텍스트의 언어를 감지합니다."""
    if not text or not text.strip():
        return 'unknown'
    
    try:
        if len(text.strip()) < 10:
            return 'unknown'
        lang = detect(text)
        return lang
    except (LangDetectException, Exception):
        return 'unknown'


def preprocess_text_for_ngram(text: str) -> List[str]:
    """텍스트를 전처리하여 단어 리스트로 변환합니다."""
    if not text:
        return []
    
    text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
    words = text.split()
    words = [w.strip() for w in words if len(w.strip()) > 0]
    
    return words


def extract_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """단어 리스트에서 n-gram을 추출합니다."""
    if n < 1 or len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def analyze_language_distribution(data: List[Dict], text_field: str = "text") -> Dict[str, Any]:
    """텍스트 데이터의 언어 분포를 분석합니다."""
    if not data:
        return {}
    
    language_counts = defaultdict(int)
    language_samples = defaultdict(list)
    
    for item in data:
        text = item.get(text_field, "")
        
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        elif text is None:
            text = ""
        
        if text and text.strip():
            lang = detect_language(text)
            language_counts[lang] += 1
            
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
    """n-gram에서 워드 클라우드를 생성합니다."""
    if not ngrams:
        print(f"[경고] {title} 생성할 n-gram이 없습니다.")
        return
    
    ngram_counter = Counter(ngrams)
    top_ngrams = ngram_counter.most_common(max_words)
    
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
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=max_words,
        relative_scaling=0.5,
        font_path='malgun.ttf' if sys.platform == 'win32' else None,
        collocations=False
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[워드 클라우드 저장 완료] {output_path}")


def analyze_text_for_rag(data: List[Dict], text_field: str, data_type: str, output_dir: Path) -> Dict[str, Any]:
    """RAG 분석에 사용할 텍스트를 종합적으로 분석합니다."""
    if not data:
        return {}
    
    print(f"\n[RAG 텍스트 분석 시작] {data_type} - {text_field}")
    
    all_texts = []
    all_words = []
    
    for item in data:
        text = item.get(text_field, "")
        
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
    
    print("  - n-gram 추출 중...")
    unigrams = all_words
    bigrams = []
    
    for text in all_texts:
        words = preprocess_text_for_ngram(text)
        bigrams.extend(extract_ngrams(words, 2))
    
    print("  - 언어 감지 중...")
    language_stats = analyze_language_distribution(data, text_field)
    
    print("  - 워드 클라우드 생성 중...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unigram_path = output_dir / f"{data_type}_1gram_wordcloud.png"
    create_wordcloud_from_ngrams(
        [(w,) for w in unigrams], 
        n=1, 
        output_path=unigram_path,
        title=f"{data_type.upper()} 1-gram Word Cloud"
    )
    
    bigram_path = output_dir / f"{data_type}_2gram_wordcloud.png"
    create_wordcloud_from_ngrams(
        bigrams, 
        n=2, 
        output_path=bigram_path,
        title=f"{data_type.upper()} 2-gram Word Cloud"
    )
    
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


def analyze_college_department_patents(data: List[Dict]) -> Dict[str, Any]:
    """대학/학과별 특허 분석 (교수-특허 매핑 중심)"""
    if not data:
        return {}
    
    college_patents = defaultdict(int)
    department_patents = defaultdict(int)
    college_department = defaultdict(lambda: defaultdict(int))
    college_professors = defaultdict(set)  # 단과대별 교수 집합
    department_professors = defaultdict(set)  # 학과별 교수 집합
    college_professor_patent_count = defaultdict(lambda: defaultdict(int))  # 단과대별 교수별 특허수
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        college = prof_info.get("COLG_NM", "")
        department = prof_info.get("HG_NM", "")
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "")
        
        if college:
            college_patents[college] += 1
            if prof_id:
                college_professors[college].add(prof_id)
                college_professor_patent_count[college][prof_id] += 1
        
        if department:
            department_patents[department] += 1
            if prof_id:
                department_professors[department].add(prof_id)
        
        if college and department:
            college_department[college][department] += 1
    
    # 단과대별 평균 특허 수 (교수당)
    college_avg_patents = {}
    for college, prof_patents in college_professor_patent_count.items():
        prof_count = len(prof_patents)
        if prof_count > 0:
            avg_patents = sum(prof_patents.values()) / prof_count
            college_avg_patents[college] = round(avg_patents, 2)
    
    # 단과대별 교수당 평균 특허수 순위
    top_colleges_by_avg = sorted(college_avg_patents.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "college_patent_distribution": dict(sorted(college_patents.items(), key=lambda x: x[1], reverse=True)),
        "department_patent_distribution": dict(sorted(department_patents.items(), key=lambda x: x[1], reverse=True)),
        "top_colleges": list(sorted(college_patents.items(), key=lambda x: x[1], reverse=True))[:10],
        "top_departments": list(sorted(department_patents.items(), key=lambda x: x[1], reverse=True))[:10],
        "college_professor_count": {college: len(profs) for college, profs in college_professors.items()},
        "department_professor_count": {dept: len(profs) for dept, profs in department_professors.items()},
        "college_avg_patents_per_professor": college_avg_patents,
        "top_colleges_by_avg_patents": top_colleges_by_avg,
        "college_department_matrix": {
            college: dict(depts) 
            for college, depts in college_department.items()
        }
    }


def analyze_invention_type(data: List[Dict]) -> Dict[str, Any]:
    """발명 구분(tech_invnt_se) 분석"""
    if not data:
        return {}
    
    invention_type_count = defaultdict(int)
    invention_type_by_professor = defaultdict(set)
    invention_type_by_college = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        inv_type = item.get("tech_invnt_se", "")
        prof_info = item.get("professor_info", {})
        
        if inv_type:
            invention_type_count[inv_type] += 1
            
            prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "") if prof_info else ""
            if prof_id:
                invention_type_by_professor[inv_type].add(prof_id)
            
            college = prof_info.get("COLG_NM", "") if prof_info else ""
            if college:
                invention_type_by_college[college][inv_type] += 1
    
    return {
        "invention_type_distribution": dict(sorted(invention_type_count.items(), key=lambda x: x[1], reverse=True)),
        "professors_by_invention_type": {inv_type: len(profs) for inv_type, profs in invention_type_by_professor.items()},
        "invention_type_by_college": {
            college: dict(types) 
            for college, types in invention_type_by_college.items()
        }
    }


def analyze_status_timeline(data: List[Dict]) -> Dict[str, Any]:
    """등록 상태별 연도 추이 분석"""
    if not data:
        return {}
    
    status_year = defaultdict(lambda: defaultdict(int))
    year_status = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        status = item.get("kipris_register_status", "")
        date_str = item.get("kipris_application_date", "")
        
        if not status or not date_str or len(date_str) < 4:
            continue
        
        year = date_str[:4]
        status_year[status][year] += 1
        year_status[year][status] += 1
    
    # 연도별 주요 상태
    year_main_status = {}
    for year, statuses in year_status.items():
        if statuses:
            main_status = max(statuses.items(), key=lambda x: x[1])[0]
            year_main_status[year] = main_status
    
    return {
        "status_by_year": {
            status: dict(sorted(years.items())) 
            for status, years in status_year.items()
        },
        "year_by_status": {
            year: dict(statuses) 
            for year, statuses in sorted(year_status.items())
        },
        "main_status_by_year": dict(sorted(year_main_status.items()))
    }


def analyze_application_number(data: List[Dict]) -> Dict[str, Any]:
    """출원번호 분석"""
    if not data:
        return {}
    
    has_application_number = 0
    application_number_prefixes = defaultdict(int)  # 출원번호 앞자리 패턴
    
    for item in data:
        app_number = item.get("kipris_application_number", "")
        if app_number and app_number.strip():
            has_application_number += 1
            # 출원번호 앞 2-3자리로 분류 (예: "10", "102")
            if len(app_number) >= 2:
                prefix = app_number[:2]
                application_number_prefixes[prefix] += 1
    
    return {
        "total_with_application_number": has_application_number,
        "application_number_rate": (has_application_number / len(data) * 100) if data else 0,
        "prefix_distribution": dict(sorted(application_number_prefixes.items(), key=lambda x: x[1], reverse=True))
    }


def analyze_register_number(data: List[Dict]) -> Dict[str, Any]:
    """등록번호 분석"""
    if not data:
        return {}
    
    has_register_number = 0
    register_numbers = []
    
    for item in data:
        reg_number = item.get("kipris_register_number", "")
        if reg_number and reg_number.strip():
            has_register_number += 1
            register_numbers.append(reg_number)
    
    return {
        "total_with_register_number": has_register_number,
        "register_number_rate": (has_register_number / len(data) * 100) if data else 0,
        "sample_register_numbers": register_numbers[:10] if register_numbers else []
    }


def analyze_title_detailed(data: List[Dict]) -> Dict[str, Any]:
    """특허명(kipris_application_name) 상세 분석"""
    if not data:
        return {}
    
    titles = []
    title_lengths = []
    
    for item in data:
        title = item.get("kipris_application_name", "")
        if title and title.strip():
            titles.append(title)
            title_lengths.append(len(title))
    
    if not title_lengths:
        return {
            "total_items": len(data),
            "valid_count": 0,
            "error": "특허명 데이터가 없습니다."
        }
    
    lengths_array = np.array(title_lengths)
    
    return {
        "total_items": len(data),
        "valid_count": len(titles),
        "valid_rate": round((len(titles) / len(data) * 100) if data else 0, 2),
        "length_statistics": {
            "min": int(np.min(lengths_array)),
            "max": int(np.max(lengths_array)),
            "mean": round(np.mean(lengths_array), 2),
            "median": int(np.median(lengths_array)),
            "std": round(np.std(lengths_array), 2),
            "q1": int(np.percentile(lengths_array, 25)),
            "q2": int(np.percentile(lengths_array, 50)),
            "q3": int(np.percentile(lengths_array, 75))
        },
        "length_distribution": {
            "short_under_20": int(np.sum(lengths_array < 20)),
            "medium_20_40": int(np.sum((lengths_array >= 20) & (lengths_array < 40))),
            "long_40_60": int(np.sum((lengths_array >= 40) & (lengths_array < 60))),
            "very_long_60_over": int(np.sum(lengths_array >= 60))
        }
    }


def analyze_professor_yearly_activity(data: List[Dict]) -> Dict[str, Any]:
    """교수별 연도별 특허 활동 분석"""
    if not data:
        return {}
    
    professor_yearly = defaultdict(lambda: defaultdict(int))
    professor_yearly_status = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for item in data:
        prof_info = item.get("professor_info", {})
        date_str = item.get("kipris_application_date", "")
        status = item.get("kipris_register_status", "")
        
        if not prof_info or not date_str or len(date_str) < 4:
            continue
        
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "")
        year = date_str[:4]
        
        if prof_id:
            professor_yearly[prof_id][year] += 1
            if status:
                professor_yearly_status[prof_id][year][status] += 1
    
    # 교수별 연도별 활동 통계
    prof_activity_stats = {}
    for prof_id, years in professor_yearly.items():
        year_list = sorted([int(y) for y in years.keys() if y.isdigit()])
        if year_list:
            prof_activity_stats[prof_id] = {
                "first_year": min(year_list),
                "last_year": max(year_list),
                "total_years": len(year_list),
                "total_patents": sum(years.values()),
                "avg_per_year": round(sum(years.values()) / len(years), 2) if years else 0
            }
    
    # 가장 활발한 교수 (연도별 평균 특허 수)
    top_active_professors = sorted(
        prof_activity_stats.items(), 
        key=lambda x: x[1]["avg_per_year"], 
        reverse=True
    )[:10]
    
    return {
        "total_active_professors": len(professor_yearly),
        "professor_yearly_patents": {
            prof_id: dict(sorted(years.items())) 
            for prof_id, years in professor_yearly.items()
        },
        "professor_activity_statistics": prof_activity_stats,
        "top_active_professors": [
            {
                "prof_id": prof_id,
                "name": "",  # 나중에 professor_info에서 가져올 수 있음
                **stats
            }
            for prof_id, stats in top_active_professors
        ]
    }


def analyze_status_year_relationship(data: List[Dict]) -> Dict[str, Any]:
    """등록 상태와 연도의 관계 분석"""
    if not data:
        return {}
    
    status_year_matrix = defaultdict(lambda: defaultdict(int))
    year_status_ratio = defaultdict(lambda: defaultdict(float))
    
    for item in data:
        status = item.get("kipris_register_status", "")
        date_str = item.get("kipris_application_date", "")
        
        if not status or not date_str or len(date_str) < 4:
            continue
        
        year = date_str[:4]
        status_year_matrix[status][year] += 1
    
    # 연도별 상태 비율 계산
    for year in set(y for status_dict in status_year_matrix.values() for y in status_dict.keys()):
        total = sum(status_year_matrix[status].get(year, 0) for status in status_year_matrix.keys())
        if total > 0:
            for status in status_year_matrix.keys():
                count = status_year_matrix[status].get(year, 0)
                year_status_ratio[year][status] = round((count / total * 100), 2)
    
    return {
        "status_year_matrix": {
            status: dict(sorted(years.items())) 
            for status, years in status_year_matrix.items()
        },
        "year_status_ratio": {
            year: dict(statuses) 
            for year, statuses in sorted(year_status_ratio.items())
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
    print("[특허 데이터 EDA 요약] 교수와 특허 관계 중심")
    print("=" * 70)
    
    # 기본 정보
    if "basic_info" in results:
        basic = results["basic_info"]
        print(f"\n[1] 기본 정보")
        print(f"   - 총 특허 개수: {basic.get('total_patents', 0):,}개")
        print(f"   - 필드 수: {basic.get('total_fields', 0)}개")
    
    # 교수-특허 관계
    if "professor_patent_relationship" in results:
        rel = results["professor_patent_relationship"]
        print(f"\n[2] 교수-특허 관계")
        print(f"   - 총 교수 수: {rel.get('total_professors', 0):,}명")
        print(f"   - 총 특허 수: {rel.get('total_patents', 0):,}개")
        
        dist = rel.get("professor_patent_distribution", {})
        print(f"   - 교수당 평균 특허 수: {dist.get('mean', 0):.2f}개")
        print(f"   - 교수당 중앙값 특허 수: {dist.get('median', 0)}개")
        print(f"   - 최다 특허 보유 교수: {dist.get('max', 0)}개")
        
        count_dist = rel.get("professors_by_patent_count", {})
        print(f"   - 특허 개수별 교수 분포:")
        for range_str, count in count_dist.items():
            print(f"     * {range_str}: {count:,}명")
        
        print(f"\n   - 대학별 교수 수:")
        for college, count in list(rel.get("college_distribution", {}).items())[:5]:
            print(f"     * {college}: {count:,}명")
        
        print(f"\n   - 학과별 교수 수:")
        for dept, count in list(rel.get("department_distribution", {}).items())[:5]:
            print(f"     * {dept}: {count:,}명")
    
    # 특허 상태
    if "patent_status" in results:
        status = results["patent_status"]
        print(f"\n[3] 특허 상태 분석")
        overall = status.get("overall_status_distribution", {})
        for stat, count in sorted(overall.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {stat}: {count:,}개")
    
    # 연도별 추이
    if "timeline" in results:
        timeline = results["timeline"]
        year_dist = timeline.get("year_distribution", {})
        if year_dist:
            print(f"\n[4] 연도별 출원 추이")
            recent_years = list(sorted(year_dist.items(), reverse=True))[:5]
            for year, count in recent_years:
                prof_count = timeline.get("professors_per_year", {}).get(year, 0)
                print(f"   - {year}년: {count:,}개 (교수 {prof_count:,}명)")
    
    # 대학/학과별 특허
    if "college_department" in results:
        cd = results["college_department"]
        print(f"\n[5] 대학/학과별 특허 분포 (교수-특허 매핑)")
        print(f"   - 상위 대학 (특허 수):")
        for college, count in cd.get("top_colleges", [])[:5]:
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            avg_patents = cd.get("college_avg_patents_per_professor", {}).get(college, 0)
            print(f"     * {college}: {count:,}개 (교수 {prof_count}명, 교수당 평균 {avg_patents:.2f}개)")
        print(f"   - 상위 학과 (특허 수):")
        for dept, count in cd.get("top_departments", [])[:5]:
            prof_count = cd.get("department_professor_count", {}).get(dept, 0)
            print(f"     * {dept}: {count:,}개 (교수 {prof_count}명)")
        print(f"   - 교수당 평균 특허수 상위 단과대:")
        for college, avg in cd.get("top_colleges_by_avg_patents", [])[:5]:
            total = cd.get("college_patent_distribution", {}).get(college, 0)
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            print(f"     * {college}: 교수당 평균 {avg:.2f}개 (총 {total}개, 교수 {prof_count}명)")
    
    # 특허 내용
    if "content" in results:
        content = results["content"]
        print(f"\n[6] 특허 내용 분석")
        titles = content.get("titles", {})
        abstracts = content.get("abstracts", {})
        print(f"   - 제목: {titles.get('total', 0):,}개")
        if titles.get("length_stats"):
            print(f"     평균 길이: {titles['length_stats'].get('mean', 0):.1f}자")
        print(f"   - 요약: {abstracts.get('total', 0):,}개")
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
        print(f"   - 교수 정보가 있는 특허: {comp.get('total_with_professor_info', 0):,}개")
        print(f"   - 가장 완전한 필드:")
        for field, rate in comp.get("most_complete_fields", [])[:3]:
            print(f"     * {field}: {rate:.1f}%")
    
    # 발명 구분 분석
    if "invention_type" in results:
        inv_type = results["invention_type"]
        print(f"\n[9] 발명 구분(tech_invnt_se) 분석")
        dist = inv_type.get("invention_type_distribution", {})
        for inv_t, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            prof_count = inv_type.get("professors_by_invention_type", {}).get(inv_t, 0)
            print(f"   - {inv_t}: {count:,}개 (교수 {prof_count:,}명)")
    
    # 등록 상태별 연도 추이
    if "status_timeline" in results:
        status_tl = results["status_timeline"]
        print(f"\n[10] 등록 상태별 연도 추이")
        main_status = status_tl.get("main_status_by_year", {})
        recent_years = list(sorted(main_status.items(), reverse=True))[:5]
        for year, status in recent_years:
            print(f"   - {year}년: 주요 상태 '{status}'")
    
    # 출원번호/등록번호 분석
    if "application_number" in results:
        app_num = results["application_number"]
        print(f"\n[11] 출원번호 분석")
        print(f"   - 출원번호가 있는 특허: {app_num.get('total_with_application_number', 0):,}개")
        print(f"   - 출원번호 보유율: {app_num.get('application_number_rate', 0):.1f}%")
    
    if "register_number" in results:
        reg_num = results["register_number"]
        print(f"\n[12] 등록번호 분석")
        print(f"   - 등록번호가 있는 특허: {reg_num.get('total_with_register_number', 0):,}개")
        print(f"   - 등록번호 보유율: {reg_num.get('register_number_rate', 0):.1f}%")
    
    # 특허명 상세 분석
    if "title_detailed" in results:
        title_detail = results["title_detailed"]
        if "error" not in title_detail:
            print(f"\n[13] 특허명 상세 분석")
            print(f"   - 유효 특허명: {title_detail.get('valid_count', 0):,}개 ({title_detail.get('valid_rate', 0)}%)")
            stats = title_detail.get("length_statistics", {})
            if stats:
                print(f"   - 평균 길이: {stats.get('mean', 0):.2f}자")
                print(f"   - 중앙값: {stats.get('median', 0):,}자")
                dist = title_detail.get("length_distribution", {})
                print(f"   - 길이 분포:")
                print(f"     * 20자 미만: {dist.get('short_under_20', 0):,}개")
                print(f"     * 20-40자: {dist.get('medium_20_40', 0):,}개")
                print(f"     * 40-60자: {dist.get('long_40_60', 0):,}개")
                print(f"     * 60자 이상: {dist.get('very_long_60_over', 0):,}개")
    
    # 교수별 연도별 활동
    if "professor_yearly_activity" in results:
        prof_yearly = results["professor_yearly_activity"]
        print(f"\n[14] 교수별 연도별 특허 활동")
        print(f"   - 활동 교수 수: {prof_yearly.get('total_active_professors', 0):,}명")
        top_profs = prof_yearly.get("top_active_professors", [])[:5]
        if top_profs:
            print(f"   - 연도당 평균 특허 수 상위 교수:")
            for i, prof in enumerate(top_profs, 1):
                print(f"     [{i}] 교수 ID: {prof.get('prof_id', '')}, 연도당 평균: {prof.get('avg_per_year', 0):.2f}개")
    
    # 등록 상태와 연도 관계
    if "status_year_relationship" in results:
        status_year = results["status_year_relationship"]
        print(f"\n[15] 등록 상태와 연도 관계")
        ratio = status_year.get("year_status_ratio", {})
        recent_years = list(sorted(ratio.items(), reverse=True))[:3]
        for year, statuses in recent_years:
            print(f"   - {year}년 상태 비율:")
            for status, ratio_val in sorted(statuses.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     * {status}: {ratio_val}%")
    
    print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    print("[특허 데이터 EDA 시작] 교수-특허 관계 중심...")
    
    # 데이터 로드 (필터링된 데이터 사용)
    data = load_patent_data(DATA_TRAIN_PATENT_FILE)
    
    if not data:
        print("[오류] 분석할 데이터가 없습니다.")
        return
    
    # 분석 수행
    print("\n[분석 시작] 다양한 EDA 분석 수행 중...")
    results = {
        "basic_info": analyze_basic_info(data),
        "professor_patent_relationship": analyze_professor_patent_relationship(data),
        "patent_status": analyze_patent_status_by_professor(data),
        "timeline": analyze_patent_timeline(data),
        "content": analyze_patent_content(data),
        "professor_completeness": analyze_professor_info_completeness(data),
        "college_department": analyze_college_department_patents(data),
        "abstract_detailed": analyze_abstract_detailed(data),
        "invention_type": analyze_invention_type(data),
        "status_timeline": analyze_status_timeline(data),
        "application_number": analyze_application_number(data),
        "register_number": analyze_register_number(data),
        "title_detailed": analyze_title_detailed(data),
        "professor_yearly_activity": analyze_professor_yearly_activity(data),
        "status_year_relationship": analyze_status_year_relationship(data),
        "metadata_analysis": analyze_metadata(data)
    }
    print("[분석 완료] 모든 EDA 분석 완료")
    
    # RAG 텍스트 분석 (text)
    print("\n[RAG 텍스트 분석 시작]")
    rag_analysis = analyze_text_for_rag(
        data, 
        text_field="text", 
        data_type="patent",
        output_dir=Path(EDA_RESULTS_DIR)
    )
    results["rag_text_analysis"] = rag_analysis
    
    # 결과 출력
    print_summary(results)
    
    # 결과 저장
    output_path = Path(EDA_RESULTS_DIR) / "patent_eda_results.json"
    save_results(results, output_path)
    
    # 초록 분포 시각화
    visualize_abstract_distribution(data, Path(EDA_RESULTS_DIR))


if __name__ == "__main__":
    main()
