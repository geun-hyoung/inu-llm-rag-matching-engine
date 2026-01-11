"""
특허 데이터 탐색적 분석 (EDA)
교수와 특허의 관계를 중심으로 분석합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PATENT_DATA_FILE, EDA_RESULTS_DIR

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_patent_data(file_path: str) -> List[Dict]:
    """특허 JSON 데이터를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 데이터 로드 완료: {len(data):,}개")
        return data
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
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
        title = item.get("kipris_application_name", "")
        abstract = item.get("kipris_abstract", "")
        
        if title:
            titles.append(title)
        if abstract:
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
            "has_both": sum(1 for item in data if item.get("kipris_application_name") and item.get("kipris_abstract"))
        }
    }


def analyze_abstract_detailed(data: List[Dict]) -> Dict[str, Any]:
    """초록(kipris_abstract)에 대한 상세 분석"""
    if not data:
        return {}
    
    abstracts = []
    abstract_lengths = []
    
    for item in data:
        abstract = item.get("kipris_abstract", "")
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
    
    # 분위수별 개수
    q1_count = np.sum(lengths_array <= q1)
    q2_count = np.sum(lengths_array <= q2)
    q3_count = np.sum(lengths_array <= q3)
    
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
            "q1_under": int(q1_count),
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
        abstract = item.get("kipris_abstract", "")
        if abstract and abstract.strip():
            abstract_lengths.append(len(abstract))
    
    if not abstract_lengths:
        print("⚠️ 시각화할 초록 데이터가 없습니다.")
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
    ax.set_title('Distribution of Patent Abstract Lengths', 
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
    output_path = output_dir / "abstract_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 초록 분포 시각화 저장 완료: {output_path}")


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


def save_results(results: Dict[str, Any], output_path: Path):
    """결과를 JSON 파일로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 결과 저장 완료: {output_path}")


def print_summary(results: Dict[str, Any]):
    """요약 정보를 출력합니다."""
    print("\n" + "=" * 70)
    print("📊 특허 데이터 EDA 요약 - 교수와 특허 관계 중심")
    print("=" * 70)
    
    # 기본 정보
    if "basic_info" in results:
        basic = results["basic_info"]
        print(f"\n1️⃣  기본 정보")
        print(f"   - 총 특허 개수: {basic.get('total_patents', 0):,}개")
        print(f"   - 필드 수: {basic.get('total_fields', 0)}개")
    
    # 교수-특허 관계
    if "professor_patent_relationship" in results:
        rel = results["professor_patent_relationship"]
        print(f"\n2️⃣  교수-특허 관계")
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
        print(f"\n3️⃣  특허 상태 분석")
        overall = status.get("overall_status_distribution", {})
        for stat, count in sorted(overall.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {stat}: {count:,}개")
    
    # 연도별 추이
    if "timeline" in results:
        timeline = results["timeline"]
        year_dist = timeline.get("year_distribution", {})
        if year_dist:
            print(f"\n4️⃣  연도별 출원 추이")
            recent_years = list(sorted(year_dist.items(), reverse=True))[:5]
            for year, count in recent_years:
                prof_count = timeline.get("professors_per_year", {}).get(year, 0)
                print(f"   - {year}년: {count:,}개 (교수 {prof_count:,}명)")
    
    # 대학/학과별 특허
    if "college_department" in results:
        cd = results["college_department"]
        print(f"\n5️⃣  대학/학과별 특허 분포 (교수-특허 매핑)")
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
        print(f"\n6️⃣  특허 내용 분석")
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
            print(f"\n8️⃣  초록(Abstract) 상세 분석")
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
        print(f"\n7️⃣  교수 정보 완전성")
        print(f"   - 교수 정보가 있는 특허: {comp.get('total_with_professor_info', 0):,}개")
        print(f"   - 가장 완전한 필드:")
        for field, rate in comp.get("most_complete_fields", [])[:3]:
            print(f"     * {field}: {rate:.1f}%")
    
    print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    print("🔍 특허 데이터 EDA 시작 (교수-특허 관계 중심)...")
    
    # 데이터 로드
    data = load_patent_data(PATENT_DATA_FILE)
    
    if not data:
        print("❌ 분석할 데이터가 없습니다.")
        return
    
    # 분석 수행
    results = {
        "basic_info": analyze_basic_info(data),
        "professor_patent_relationship": analyze_professor_patent_relationship(data),
        "patent_status": analyze_patent_status_by_professor(data),
        "timeline": analyze_patent_timeline(data),
        "content": analyze_patent_content(data),
        "professor_completeness": analyze_professor_info_completeness(data),
        "college_department": analyze_college_department_patents(data),
        "abstract_detailed": analyze_abstract_detailed(data)
    }
    
    # 결과 출력
    print_summary(results)
    
    # 결과 저장
    output_path = Path(EDA_RESULTS_DIR) / "patent_eda_results.json"
    save_results(results, output_path)
    
    # 초록 분포 시각화
    visualize_abstract_distribution(data, Path(EDA_RESULTS_DIR))


if __name__ == "__main__":
    main()
