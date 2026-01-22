"""
연구과제 데이터 탐색적 분석 (EDA)
교수와 연구과제의 관계를 중심으로 분석합니다.
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
from datetime import datetime
import re
from wordcloud import WordCloud
from langdetect import detect, DetectorFactory, LangDetectException

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_TRAIN_PROJECT_FILE, EDA_RESULTS_DIR

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
sns.set_style("whitegrid")
sns.set_palette("husl")

# 언어 감지 재현성 설정
DetectorFactory.seed = 0


def load_project_data(file_path: str) -> List[Dict]:
    """연구과제 JSON 데이터를 로드합니다."""
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
        "total_projects": len(data),
        "total_fields": len(data[0].keys()) if data else 0,
        "field_names": list(data[0].keys()) if data else [],
        "sample_record_keys": list(data[0].keys())[:10] if data else []
    }


def analyze_professor_project_relationship(data: List[Dict]) -> Dict[str, Any]:
    """교수-연구과제 관계 분석"""
    if not data:
        return {}
    
    # 교수별 연구과제 개수
    professor_project_count = defaultdict(int)
    professor_info_map = {}
    
    # 교수별 총 연구비
    professor_total_amount = defaultdict(float)
    
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
            professor_project_count[prof_id] += 1
            professor_info_map[prof_id] = prof_info
            
            # 연구비 합계
            amount = item.get("TOT_RND_AMT", 0)
            if isinstance(amount, (int, float)) and amount > 0:
                professor_total_amount[prof_id] += float(amount)
        
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
    
    # 교수별 연구과제 개수 분포
    project_counts = list(professor_project_count.values())
    
    # 교수별 총 연구비 분포
    total_amounts = list(professor_total_amount.values())
    
    return {
        "total_professors": len(professor_project_count),
        "total_projects": sum(professor_project_count.values()),
        "professor_project_distribution": {
            "min": min(project_counts) if project_counts else 0,
            "max": max(project_counts) if project_counts else 0,
            "mean": sum(project_counts) / len(project_counts) if project_counts else 0,
            "median": sorted(project_counts)[len(project_counts)//2] if project_counts else 0
        },
        "professors_by_project_count": {
            "1개": sum(1 for c in project_counts if c == 1),
            "2-5개": sum(1 for c in project_counts if 2 <= c <= 5),
            "6-10개": sum(1 for c in project_counts if 6 <= c <= 10),
            "11-20개": sum(1 for c in project_counts if 11 <= c <= 20),
            "21개 이상": sum(1 for c in project_counts if c >= 21)
        },
        "professor_total_amount_distribution": {
            "min": min(total_amounts) if total_amounts else 0,
            "max": max(total_amounts) if total_amounts else 0,
            "mean": sum(total_amounts) / len(total_amounts) if total_amounts else 0,
            "median": sorted(total_amounts)[len(total_amounts)//2] if total_amounts else 0
        },
        "college_distribution": {college: len(profs) for college, profs in professors_by_college.items()},
        "department_distribution": {dept: len(profs) for dept, profs in professors_by_department.items()},
        "status_distribution": {status: len(profs) for status, profs in professors_by_status.items()}
    }


def analyze_project_amount(data: List[Dict]) -> Dict[str, Any]:
    """연구비 규모 분석"""
    if not data:
        return {}
    
    amounts = []
    amount_by_professor = defaultdict(list)
    
    for item in data:
        amount = item.get("TOT_RND_AMT", 0)
        if isinstance(amount, (int, float)) and amount > 0:
            amounts.append(float(amount))
            
            prof_info = item.get("professor_info", {})
            if prof_info:
                prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
                if prof_id:
                    amount_by_professor[prof_id].append(float(amount))
    
    if not amounts:
        return {
            "error": "연구비 데이터가 없습니다."
        }
    
    amounts_array = np.array(amounts)
    
    # 연구비 구간별 분포
    amount_ranges = {
        "5천만원 이하": sum(1 for a in amounts if a <= 50_000_000),
        "5천만원 초과 3억원 이하": sum(1 for a in amounts if 50_000_000 < a <= 300_000_000),
        "3억원 초과 5억원 이하": sum(1 for a in amounts if 300_000_000 < a <= 500_000_000),
        "5억원 초과": sum(1 for a in amounts if a > 500_000_000)
    }
    
    # 교수별 평균 연구비
    professor_avg_amounts = {}
    for prof_id, prof_amounts in amount_by_professor.items():
        if prof_amounts:
            professor_avg_amounts[prof_id] = sum(prof_amounts) / len(prof_amounts)
    
    return {
        "total_projects_with_amount": len(amounts),
        "descriptive_statistics": {
            "min": int(np.min(amounts_array)),
            "max": int(np.max(amounts_array)),
            "mean": round(np.mean(amounts_array), 2),
            "median": int(np.median(amounts_array)),
            "std": round(np.std(amounts_array), 2),
            "q1": int(np.percentile(amounts_array, 25)),
            "q2": int(np.percentile(amounts_array, 50)),
            "q3": int(np.percentile(amounts_array, 75)),
            "iqr": int(np.percentile(amounts_array, 75) - np.percentile(amounts_array, 25))
        },
        "amount_range_distribution": amount_ranges,
        "professor_avg_amount_stats": {
            "min": min(professor_avg_amounts.values()) if professor_avg_amounts else 0,
            "max": max(professor_avg_amounts.values()) if professor_avg_amounts else 0,
            "mean": sum(professor_avg_amounts.values()) / len(professor_avg_amounts) if professor_avg_amounts else 0,
            "median": sorted(professor_avg_amounts.values())[len(professor_avg_amounts)//2] if professor_avg_amounts else 0
        }
    }


def analyze_project_timeline(data: List[Dict]) -> Dict[str, Any]:
    """연구과제 시기 분석 (연도별, 기준년도별, 교수별)"""
    if not data:
        return {}
    
    # 기준년도별 연구과제 개수
    base_year_projects = defaultdict(int)
    base_year_professors = defaultdict(set)
    
    # 연구 시작일별 분석
    start_date_projects = defaultdict(int)
    start_date_professors = defaultdict(set)
    
    # 교수별 기준년도
    professor_years = defaultdict(set)
    
    for item in data:
        base_year = item.get("excel_base_year")
        start_date = item.get("RCH_ST_DT", "")
        prof_info = item.get("professor_info", {})
        
        if base_year:
            year_str = str(base_year)
            base_year_projects[year_str] += 1
            
            prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", "")) if prof_info else ""
            if prof_id:
                base_year_professors[year_str].add(prof_id)
                professor_years[prof_id].add(year_str)
        
        if start_date and len(start_date) >= 4:
            year = start_date[:4]
            start_date_projects[year] += 1
            
            prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", "")) if prof_info else ""
            if prof_id:
                start_date_professors[year].add(prof_id)
    
    # 교수별 활동 기간 (기준년도 범위)
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
        "base_year_distribution": dict(sorted(base_year_projects.items())),
        "start_date_year_distribution": dict(sorted(start_date_projects.items())),
        "professors_per_base_year": {year: len(profs) for year, profs in sorted(base_year_professors.items())},
        "professors_per_start_year": {year: len(profs) for year, profs in sorted(start_date_professors.items())},
        "activity_period_stats": {
            "avg_span": sum(p["span_years"] for p in professor_activity_periods.values()) / len(professor_activity_periods) if professor_activity_periods else 0,
            "max_span": max((p["span_years"] for p in professor_activity_periods.values()), default=0),
            "min_span": min((p["span_years"] for p in professor_activity_periods.values()), default=0)
        }
    }


def analyze_project_content(data: List[Dict]) -> Dict[str, Any]:
    """연구과제 내용 분석 (제목, 연구목표, 연구내용, 기대효과)"""
    if not data:
        return {}
    
    titles = []
    objectives = []
    contents = []
    effects = []
    
    for item in data:
        title = item.get("PRJ_NM", "") or item.get("excel_project_name_kr", "")
        objective = item.get("excel_research_objective_summary", "")
        content = item.get("excel_research_content_summary", "")
        effect = item.get("excel_expected_effect_summary", "")
        
        if title:
            titles.append(title)
        if objective and objective.strip():
            objectives.append(objective)
        if content and content.strip():
            contents.append(content)
        if effect and effect.strip():
            effects.append(effect)
    
    # 길이 분석
    title_lengths = [len(t) for t in titles]
    objective_lengths = [len(o) for o in objectives]
    content_lengths = [len(c) for c in contents]
    effect_lengths = [len(e) for e in effects]
    
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
        "objectives": {
            "total": len(objectives),
            "length_stats": {
                "min": min(objective_lengths) if objective_lengths else 0,
                "max": max(objective_lengths) if objective_lengths else 0,
                "mean": sum(objective_lengths) / len(objective_lengths) if objective_lengths else 0,
                "median": sorted(objective_lengths)[len(objective_lengths)//2] if objective_lengths else 0
            }
        },
        "contents": {
            "total": len(contents),
            "length_stats": {
                "min": min(content_lengths) if content_lengths else 0,
                "max": max(content_lengths) if content_lengths else 0,
                "mean": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
                "median": sorted(content_lengths)[len(content_lengths)//2] if content_lengths else 0
            }
        },
        "effects": {
            "total": len(effects),
            "length_stats": {
                "min": min(effect_lengths) if effect_lengths else 0,
                "max": max(effect_lengths) if effect_lengths else 0,
                "mean": sum(effect_lengths) / len(effect_lengths) if effect_lengths else 0,
                "median": sorted(effect_lengths)[len(effect_lengths)//2] if effect_lengths else 0
            }
        },
        "content_completeness": {
            "has_title": len(titles),
            "has_objective": len(objectives),
            "has_content": len(contents),
            "has_effect": len(effects),
            "has_all": sum(1 for item in data 
                          if (item.get("PRJ_NM") or item.get("excel_project_name_kr")) 
                          and item.get("excel_research_objective_summary")
                          and item.get("excel_research_content_summary")
                          and item.get("excel_expected_effect_summary"))
        }
    }


def visualize_amount_distribution(data: List[Dict], output_dir: Path):
    """연구비 분포 시각화"""
    if not data:
        return
    
    amounts = []
    for item in data:
        amount = item.get("TOT_RND_AMT", 0)
        if isinstance(amount, (int, float)) and amount > 0:
            amounts.append(float(amount) / 1_000_000)  # 백만원 단위로 변환
    
    if not amounts:
        print("[경고] 시각화할 연구비 데이터가 없습니다.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 히스토그램
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(amounts, bins=40, kde=True, 
                 color='#3498db', alpha=0.7, 
                 edgecolor='white', linewidth=0.5)
    
    # 통계 선 표시
    mean_val = np.mean(amounts)
    median_val = np.median(amounts)
    q1_val = np.percentile(amounts, 25)
    q3_val = np.percentile(amounts, 75)
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.0f}백만원', zorder=5)
    ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.0f}백만원', zorder=5)
    
    ax.set_xlabel('Research Amount (Million KRW)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Research Project Amounts', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, fontsize=10)
    
    stats_text = f'n = {len(amounts):,} | ' \
                 f'Q1: {q1_val:.0f} | Q3: {q3_val:.0f} | ' \
                 f'SD: {np.std(amounts):.1f}'
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            style='italic')
    
    plt.tight_layout()
    output_path = output_dir / "project_amount_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[연구비 분포 시각화 저장 완료] {output_path}")


def visualize_summary_distribution(data: List[Dict], output_dir: Path):
    """텍스트 길이 분포 시각화 - 특허 스타일 참고"""
    if not data:
        return
    
    text_lengths = []
    for item in data:
        text = item.get("text", "")
        
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        elif text is None:
            text = ""
        
        if text and text.strip():
            text_lengths.append(len(text))
    
    if not text_lengths:
        print("[경고] 시각화할 텍스트 데이터가 없습니다.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(text_lengths, bins=40, kde=True, 
                 color='#3498db', alpha=0.7, 
                 edgecolor='white', linewidth=0.5)
    
    mean_val = np.mean(text_lengths)
    median_val = np.median(text_lengths)
    q1_val = np.percentile(text_lengths, 25)
    q3_val = np.percentile(text_lengths, 75)
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.0f}', zorder=5)
    ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.0f}', zorder=5)
    
    ax.set_xlabel('Text Length (characters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Project Text Lengths', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, fontsize=10)
    
    stats_text = f'n = {len(text_lengths):,} | ' \
                 f'Q1: {q1_val:.0f} | Q3: {q3_val:.0f} | ' \
                 f'SD: {np.std(text_lengths):.1f}'
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            style='italic')
    
    plt.tight_layout()
    output_path = output_dir / "project_text_length_distribution.png"
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


def analyze_college_department_projects(data: List[Dict]) -> Dict[str, Any]:
    """대학/학과별 연구과제 분석 (교수-연구과제 매핑 중심)"""
    if not data:
        return {}
    
    college_projects = defaultdict(int)
    department_projects = defaultdict(int)
    college_department = defaultdict(lambda: defaultdict(int))
    college_professors = defaultdict(set)  # 단과대별 교수 집합
    department_professors = defaultdict(set)  # 학과별 교수 집합
    college_professor_project_count = defaultdict(lambda: defaultdict(int))  # 단과대별 교수별 연구과제수
    college_total_amount = defaultdict(float)  # 단과대별 총 연구비
    department_total_amount = defaultdict(float)  # 학과별 총 연구비
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        college = prof_info.get("COLG_NM", "")
        department = prof_info.get("HG_NM", "")
        prof_id = str(prof_info.get("SQ", "")) or str(prof_info.get("EMP_NO", ""))
        amount = item.get("TOT_RND_AMT", 0)
        if isinstance(amount, (int, float)):
            amount = float(amount)
        else:
            amount = 0
        
        if college:
            college_projects[college] += 1
            college_total_amount[college] += amount
            if prof_id:
                college_professors[college].add(prof_id)
                college_professor_project_count[college][prof_id] += 1
        
        if department:
            department_projects[department] += 1
            department_total_amount[department] += amount
            if prof_id:
                department_professors[department].add(prof_id)
        
        if college and department:
            college_department[college][department] += 1
    
    # 단과대별 평균 연구과제 수 (교수당)
    college_avg_projects = {}
    for college, prof_projects in college_professor_project_count.items():
        prof_count = len(prof_projects)
        if prof_count > 0:
            avg_projects = sum(prof_projects.values()) / prof_count
            college_avg_projects[college] = round(avg_projects, 2)
    
    # 단과대별 평균 연구비 (교수당)
    college_avg_amount = {}
    for college in college_professors:
        prof_count = len(college_professors[college])
        if prof_count > 0 and college_total_amount[college] > 0:
            college_avg_amount[college] = round(college_total_amount[college] / prof_count, 2)
    
    # 단과대별 교수당 평균 연구과제수 순위
    top_colleges_by_avg = sorted(college_avg_projects.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 단과대별 총 연구비 순위
    top_colleges_by_amount = sorted(college_total_amount.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "college_project_distribution": dict(sorted(college_projects.items(), key=lambda x: x[1], reverse=True)),
        "department_project_distribution": dict(sorted(department_projects.items(), key=lambda x: x[1], reverse=True)),
        "college_total_amount": dict(sorted(college_total_amount.items(), key=lambda x: x[1], reverse=True)),
        "department_total_amount": dict(sorted(department_total_amount.items(), key=lambda x: x[1], reverse=True)),
        "top_colleges": list(sorted(college_projects.items(), key=lambda x: x[1], reverse=True))[:10],
        "top_departments": list(sorted(department_projects.items(), key=lambda x: x[1], reverse=True))[:10],
        "top_colleges_by_amount": top_colleges_by_amount,
        "college_professor_count": {college: len(profs) for college, profs in college_professors.items()},
        "department_professor_count": {dept: len(profs) for dept, profs in department_professors.items()},
        "college_avg_projects_per_professor": college_avg_projects,
        "college_avg_amount_per_professor": college_avg_amount,
        "top_colleges_by_avg_projects": top_colleges_by_avg,
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
    print("[연구과제 데이터 EDA 요약] 교수와 연구과제 관계 중심")
    print("=" * 70)
    
    # 기본 정보
    if "basic_info" in results:
        basic = results["basic_info"]
        print(f"\n[1] 기본 정보")
        print(f"   - 총 연구과제 개수: {basic.get('total_projects', 0):,}개")
        print(f"   - 필드 수: {basic.get('total_fields', 0)}개")
    
    # 교수-연구과제 관계
    if "professor_project_relationship" in results:
        rel = results["professor_project_relationship"]
        print(f"\n[2] 교수-연구과제 관계")
        print(f"   - 총 교수 수: {rel.get('total_professors', 0):,}명")
        print(f"   - 총 연구과제 수: {rel.get('total_projects', 0):,}개")
        
        dist = rel.get("professor_project_distribution", {})
        print(f"   - 교수당 평균 연구과제 수: {dist.get('mean', 0):.2f}개")
        print(f"   - 교수당 중앙값 연구과제 수: {dist.get('median', 0)}개")
        print(f"   - 최다 연구과제 보유 교수: {dist.get('max', 0)}개")
        
        count_dist = rel.get("professors_by_project_count", {})
        print(f"   - 연구과제 개수별 교수 분포:")
        for range_str, count in count_dist.items():
            print(f"     * {range_str}: {count:,}명")
        
        amount_dist = rel.get("professor_total_amount_distribution", {})
        print(f"\n   - 교수별 총 연구비 통계:")
        print(f"     * 평균: {amount_dist.get('mean', 0):,.0f}원")
        print(f"     * 중앙값: {amount_dist.get('median', 0):,.0f}원")
        print(f"     * 최대: {amount_dist.get('max', 0):,.0f}원")
        
        print(f"\n   - 대학별 교수 수:")
        for college, count in list(rel.get("college_distribution", {}).items())[:5]:
            print(f"     * {college}: {count:,}명")
        
        print(f"\n   - 학과별 교수 수:")
        for dept, count in list(rel.get("department_distribution", {}).items())[:5]:
            print(f"     * {dept}: {count:,}명")
    
    # 연구비 분석
    if "project_amount" in results:
        amount = results["project_amount"]
        if "error" not in amount:
            print(f"\n[3] 연구비 규모 분석")
            stats = amount.get("descriptive_statistics", {})
            print(f"   - 총 연구과제 수 (연구비 있음): {amount.get('total_projects_with_amount', 0):,}개")
            print(f"   - 평균 연구비: {stats.get('mean', 0):,.0f}원")
            print(f"   - 중앙값 연구비: {stats.get('median', 0):,.0f}원")
            print(f"   - 최대 연구비: {stats.get('max', 0):,.0f}원")
            
            ranges = amount.get("amount_range_distribution", {})
            print(f"\n   - 연구비 구간별 분포:")
            for range_str, count in ranges.items():
                print(f"     * {range_str}: {count:,}개")
    
    # 연도별 추이
    if "timeline" in results:
        timeline = results["timeline"]
        base_year_dist = timeline.get("base_year_distribution", {})
        if base_year_dist:
            print(f"\n[4] 기준년도별 연구과제 추이")
            recent_years = list(sorted(base_year_dist.items(), reverse=True))[:5]
            for year, count in recent_years:
                prof_count = timeline.get("professors_per_base_year", {}).get(year, 0)
                print(f"   - {year}년: {count:,}개 (교수 {prof_count:,}명)")
    
    # 대학/학과별 연구과제
    if "college_department" in results:
        cd = results["college_department"]
        print(f"\n[5] 대학/학과별 연구과제 분포 (교수-연구과제 매핑)")
        print(f"   - 상위 대학 (연구과제 수):")
        for college, count in cd.get("top_colleges", [])[:5]:
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            avg_projects = cd.get("college_avg_projects_per_professor", {}).get(college, 0)
            total_amount = cd.get("college_total_amount", {}).get(college, 0)
            print(f"     * {college}: {count:,}개 (교수 {prof_count}명, 교수당 평균 {avg_projects:.2f}개, 총 연구비 {total_amount:,.0f}원)")
        
        print(f"   - 상위 대학 (총 연구비):")
        for college, amount in cd.get("top_colleges_by_amount", [])[:5]:
            count = cd.get("college_project_distribution", {}).get(college, 0)
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            print(f"     * {college}: {amount:,.0f}원 (연구과제 {count}개, 교수 {prof_count}명)")
        
        print(f"   - 상위 학과 (연구과제 수):")
        for dept, count in cd.get("top_departments", [])[:5]:
            prof_count = cd.get("department_professor_count", {}).get(dept, 0)
            print(f"     * {dept}: {count:,}개 (교수 {prof_count}명)")
        
        print(f"   - 교수당 평균 연구과제수 상위 단과대:")
        for college, avg in cd.get("top_colleges_by_avg_projects", [])[:5]:
            total = cd.get("college_project_distribution", {}).get(college, 0)
            prof_count = cd.get("college_professor_count", {}).get(college, 0)
            print(f"     * {college}: 교수당 평균 {avg:.2f}개 (총 {total}개, 교수 {prof_count}명)")
    
    # 연구과제 내용
    if "content" in results:
        content = results["content"]
        print(f"\n[6] 연구과제 내용 분석")
        titles = content.get("titles", {})
        objectives = content.get("objectives", {})
        contents = content.get("contents", {})
        effects = content.get("effects", {})
        print(f"   - 제목: {titles.get('total', 0):,}개")
        if titles.get("length_stats"):
            print(f"     평균 길이: {titles['length_stats'].get('mean', 0):.1f}자")
        print(f"   - 연구목표: {objectives.get('total', 0):,}개")
        if objectives.get("length_stats"):
            print(f"     평균 길이: {objectives['length_stats'].get('mean', 0):.1f}자")
        print(f"   - 연구내용: {contents.get('total', 0):,}개")
        if contents.get("length_stats"):
            print(f"     평균 길이: {contents['length_stats'].get('mean', 0):.1f}자")
        print(f"   - 기대효과: {effects.get('total', 0):,}개")
        if effects.get("length_stats"):
            print(f"     평균 길이: {effects['length_stats'].get('mean', 0):.1f}자")
        
        completeness = content.get("content_completeness", {})
        print(f"   - 모든 내용이 있는 연구과제: {completeness.get('has_all', 0):,}개")
    
    # 교수 정보 완전성
    if "professor_completeness" in results:
        comp = results["professor_completeness"]
        print(f"\n[7] 교수 정보 완전성")
        print(f"   - 교수 정보가 있는 연구과제: {comp.get('total_with_professor_info', 0):,}개")
        print(f"   - 가장 완전한 필드:")
        for field, rate in comp.get("most_complete_fields", [])[:3]:
            print(f"     * {field}: {rate:.1f}%")
    
    print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    print("[연구과제 데이터 EDA 시작] 교수-연구과제 관계 중심...")
    
    # 데이터 로드 (data/train 폴더의 필터링된 데이터 사용)
    data = load_project_data(DATA_TRAIN_PROJECT_FILE)
    
    if not data:
        print("[오류] 분석할 데이터가 없습니다.")
        return
    
    # 2015년 이전 데이터 제거 (전처리 단계)
    original_count = len(data)
    filtered_data = []
    for item in data:
        # excel_base_year 우선 확인
        base_year = item.get("excel_base_year")
        if base_year is not None:
            # 숫자형이면 직접 비교, 문자열이면 변환
            try:
                year = int(base_year) if isinstance(base_year, (int, float)) else int(str(base_year).strip())
                if year >= 2015:
                    filtered_data.append(item)
                continue
            except (ValueError, TypeError):
                pass
        
        # excel_base_year가 없거나 유효하지 않으면 RCH_ST_DT 확인
        start_date = item.get("RCH_ST_DT", "")
        if isinstance(start_date, str) and len(start_date) >= 4:
            try:
                year = int(start_date[:4])
                if year >= 2015:
                    filtered_data.append(item)
                continue
            except (ValueError, TypeError):
                pass
        
        # 연도 정보가 없거나 유효하지 않으면 포함 (안전을 위해)
        filtered_data.append(item)
    
    data = filtered_data
    filtered_count = len(data)
    if original_count != filtered_count:
        print(f"[전처리] 2015년 이전 데이터 제거: {original_count - filtered_count}개 제거됨 (전체: {original_count}개 → 필터링 후: {filtered_count}개)")
    
    # 분석 수행
    results = {
        "basic_info": analyze_basic_info(data),
        "professor_project_relationship": analyze_professor_project_relationship(data),
        "project_amount": analyze_project_amount(data),
        "timeline": analyze_project_timeline(data),
        "content": analyze_project_content(data),
        "professor_completeness": analyze_professor_info_completeness(data),
        "college_department": analyze_college_department_projects(data)
    }
    
    # RAG 텍스트 분석 (text)
    print("\n[RAG 텍스트 분석 시작]")
    rag_analysis = analyze_text_for_rag(
        data, 
        text_field="text", 
        data_type="project",
        output_dir=Path(EDA_RESULTS_DIR)
    )
    results["rag_text_analysis"] = rag_analysis
    
    # 결과 출력
    print_summary(results)
    
    # 결과 저장
    output_path = Path(EDA_RESULTS_DIR) / "project_eda_results.json"
    save_results(results, output_path)
    
    # 연구비 분포 시각화
    visualize_amount_distribution(data, Path(EDA_RESULTS_DIR))
    
    # Summary 길이 분포 시각화
    visualize_summary_distribution(data, Path(EDA_RESULTS_DIR))


if __name__ == "__main__":
    main()
