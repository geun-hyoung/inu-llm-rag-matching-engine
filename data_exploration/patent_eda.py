"""
특허 데이터 탐색적 분석 (EDA)
교수와 특허의 관계를 중심으로 분석합니다.
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List
from collections import Counter, defaultdict

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PATENT_DATA_FILE, EDA_RESULTS_DIR


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
    """대학/학과별 특허 분석"""
    if not data:
        return {}
    
    college_patents = defaultdict(int)
    department_patents = defaultdict(int)
    college_department = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        prof_info = item.get("professor_info", {})
        if not prof_info:
            continue
        
        college = prof_info.get("COLG_NM", "")
        department = prof_info.get("HG_NM", "")
        
        if college:
            college_patents[college] += 1
        if department:
            department_patents[department] += 1
        if college and department:
            college_department[college][department] += 1
    
    return {
        "college_patent_distribution": dict(sorted(college_patents.items(), key=lambda x: x[1], reverse=True)),
        "department_patent_distribution": dict(sorted(department_patents.items(), key=lambda x: x[1], reverse=True)),
        "top_colleges": list(sorted(college_patents.items(), key=lambda x: x[1], reverse=True))[:10],
        "top_departments": list(sorted(department_patents.items(), key=lambda x: x[1], reverse=True))[:10],
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
        print(f"\n5️⃣  대학/학과별 특허 분포")
        print(f"   - 상위 대학 (특허 수):")
        for college, count in cd.get("top_colleges", [])[:5]:
            print(f"     * {college}: {count:,}개")
        print(f"   - 상위 학과 (특허 수):")
        for dept, count in cd.get("top_departments", [])[:5]:
            print(f"     * {dept}: {count:,}개")
    
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
        "college_department": analyze_college_department_patents(data)
    }
    
    # 결과 출력
    print_summary(results)
    
    # 결과 저장
    output_path = Path(EDA_RESULTS_DIR) / "patent_eda_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
