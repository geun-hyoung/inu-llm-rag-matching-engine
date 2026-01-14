"""
연구과제 데이터 수집기
MariaDB의 vw_inu_prj_info 테이블 데이터와 엑셀 파일 데이터를 결합하여
연구과제 데이터를 수집하고 JSON 파일로 저장합니다.
"""

import mariadb
import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.database import (
    get_db_connection, 
    close_db_connection, 
    get_project_statistics, 
    get_project_data,
    get_project_with_professor_info,
    TABLE_PROJECT,
    COL_PROJECT_PRJ_NM,
    COL_PROJECT_RSPR_EMP_ID
)
from config.settings import PROJECT_DATA_FILE


class ProjectCollector:
    """연구과제 데이터를 수집하는 클래스"""
    
    def __init__(self, json_file1: str = "data/project/project_source_1.json",
                 json_file2: str = "data/project/project_source_2.json"):
        """
        연구과제 수집기 초기화
        
        Args:
            json_file1: 첫 번째 JSON 파일 경로
            json_file2: 두 번째 JSON 파일 경로
        """
        self.json_file1 = Path(json_file1)
        self.json_file2 = Path(json_file2)
    
    def get_statistics(self, conn: mariadb.Connection) -> Dict[str, int]:
        """
        데이터베이스에서 통계 정보를 수집합니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            
        Returns:
            통계 정보 딕셔너리
        """
        return get_project_statistics(conn)
    
    def print_statistics(self, stats: Dict[str, int], collected_count: int = 0, professor_matched_count: int = 0):
        """
        통계 정보를 단계적으로 출력합니다.
        
        Args:
            stats: 통계 정보 딕셔너리
            collected_count: 최종 수집된 데이터 개수 (매핑된 데이터만)
            professor_matched_count: 교수 정보가 매칭된 데이터 개수
        """
        print("\n" + "=" * 60)
        print("[데이터 수집 통계]")
        print("=" * 60)
        print(f"[1] 연구과제 테이블 전체 row 수: {stats['total_records']:,}개")
        print(f"[2] 매핑 후 사용가능한 데이터 row 수: {collected_count:,}개")
        print(f"[3] 교수 정보 매칭된 데이터 row 수: {professor_matched_count:,}개")
        print("=" * 60)
        print()
    
    def load_json_files(self) -> pd.DataFrame:
        """
        두 개의 JSON 파일을 읽어서 하나의 DataFrame으로 병합합니다.
        
        Returns:
            병합된 DataFrame
        """
        json_data_list = []
        
        # 첫 번째 JSON 파일 읽기
        if self.json_file1.exists():
            print(f"[JSON 파일 읽기] {self.json_file1.name}")
            try:
                with open(self.json_file1, 'r', encoding='utf-8') as f:
                    data1 = json.load(f)
                df1 = pd.DataFrame(data1)
                print(f"  - 행 수: {len(df1):,}개")
                print(f"  - 컬럼 수: {len(df1.columns)}개")
                json_data_list.append(df1)
            except Exception as e:
                print(f"  - 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  - 파일이 존재하지 않습니다: {self.json_file1}")
        
        # 두 번째 JSON 파일 읽기
        if self.json_file2.exists():
            print(f"[JSON 파일 읽기] {self.json_file2.name}")
            try:
                with open(self.json_file2, 'r', encoding='utf-8') as f:
                    data2 = json.load(f)
                df2 = pd.DataFrame(data2)
                print(f"  - 행 수: {len(df2):,}개")
                print(f"  - 컬럼 수: {len(df2.columns)}개")
                json_data_list.append(df2)
            except Exception as e:
                print(f"  - 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  - 파일이 존재하지 않습니다: {self.json_file2}")
        
        if not json_data_list:
            print("[경고] 읽을 수 있는 JSON 파일이 없습니다.")
            return pd.DataFrame()
        
        # 두 DataFrame 병합
        merged_df = pd.concat(json_data_list, ignore_index=True)
        print(f"\n[완료] JSON 파일 병합 완료: 총 {len(merged_df):,}개 행")
        
        return merged_df
    
    def get_db_projects(self, conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
        """
        데이터베이스에서 연구과제 데이터를 가져옵니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            limit: 가져올 최대 개수 (None이면 전체)
            
        Returns:
            연구과제 데이터 리스트
        """
        return get_project_data(conn, limit)
    
    def get_db_projects_with_professor(self, conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
        """
        데이터베이스에서 연구과제 데이터와 교수 정보를 함께 가져옵니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            limit: 가져올 최대 개수 (None이면 전체)
            
        Returns:
            [{"project_data": {...}, "professor_info": {...}}, ...] 형태의 리스트
        """
        return get_project_with_professor_info(conn, limit)
    
    def merge_data(self, db_projects: List[Dict], json_df: pd.DataFrame, projects_with_professor: List[Dict] = None) -> tuple:
        """
        데이터베이스 데이터와 JSON 데이터를 병합합니다.
        PRJ_NM과 '과제명(국문)' 컬럼으로 매핑하고,
        RCH_ST_DT의 연도와 JSON의 기준년도도 함께 매핑합니다.
        교수 정보도 PRJ_RSPR_EMP_ID와 EMP_NO로 매핑하여 추가합니다.
        
        Args:
            db_projects: 데이터베이스 연구과제 데이터 리스트
            json_df: JSON 데이터 DataFrame
            projects_with_professor: 교수 정보가 포함된 프로젝트 데이터 리스트
            
        Returns:
            (병합된 데이터 리스트, 교수 정보 매칭 개수)
        """
        merged_data = []
        
        # JSON 데이터에서 '과제명(국문)' 컬럼 찾기
        json_project_name_col = None
        for col in json_df.columns:
            if '과제명' in str(col) and '국문' in str(col):
                json_project_name_col = col
                break
        
        # JSON 데이터에서 '기준년도' 컬럼 찾기
        json_year_col = None
        for col in json_df.columns:
            if '기준년도' in str(col) or '년도' in str(col):
                json_year_col = col
                break
        
        if json_project_name_col is None:
            print("[경고] JSON 파일에서 '과제명(국문)' 컬럼을 찾을 수 없습니다.")
            print(f"   사용 가능한 컬럼: {json_df.columns.tolist()[:10]}...")
            # 매핑 컬럼이 없으면 빈 리스트 반환 (매핑된 데이터만 저장)
            print("   매핑 컬럼이 없어 매핑된 데이터가 없습니다.")
            return [], 0
        
        if json_year_col is None:
            print("[경고] JSON 파일에서 '기준년도' 컬럼을 찾을 수 없습니다.")
            print(f"   사용 가능한 컬럼: {json_df.columns.tolist()[:10]}...")
            print("   연도 매핑 없이 과제명만으로 매핑합니다.")
        
        print(f"\n[데이터 병합 시작]")
        print(f"   - 데이터베이스 프로젝트 수: {len(db_projects):,}개")
        print(f"   - JSON 데이터 행 수: {len(json_df):,}개")
        print(f"   - 매핑 키 1: DB.PRJ_NM <-> JSON.{json_project_name_col}")
        if json_year_col:
            print(f"   - 매핑 키 2: DB.RCH_ST_DT(연도) <-> JSON.{json_year_col}")
        
        # 데이터베이스 프로젝트를 PRJ_NM으로 인덱싱 (중복 허용)
        db_projects_by_name = {}
        for project in db_projects:
            prj_nm = project.get(COL_PROJECT_PRJ_NM)
            if prj_nm and str(prj_nm).strip():
                prj_nm_clean = str(prj_nm).strip()
                if prj_nm_clean not in db_projects_by_name:
                    db_projects_by_name[prj_nm_clean] = []
                db_projects_by_name[prj_nm_clean].append(project)
        
        print(f"   - 고유한 PRJ_NM 개수: {len(db_projects_by_name):,}개")
        
        # 교수 정보를 PRJ_RSPR_EMP_ID로 인덱싱
        professor_info_by_emp_id = {}
        if projects_with_professor:
            for item in projects_with_professor:
                project_data = item.get("project_data", {})
                professor_info = item.get("professor_info")
                prj_rspr_emp_id = project_data.get(COL_PROJECT_RSPR_EMP_ID)
                if prj_rspr_emp_id and professor_info and professor_info.get("SQ"):
                    emp_id_key = str(prj_rspr_emp_id).strip()
                    if emp_id_key not in professor_info_by_emp_id:
                        professor_info_by_emp_id[emp_id_key] = []
                    professor_info_by_emp_id[emp_id_key].append({
                        "project_data": project_data,
                        "professor_info": professor_info
                    })
        
        # JSON 데이터와 매핑 (매핑된 데이터만 수집)
        matched_count = 0
        skipped_by_year = 0
        professor_matched_count = 0
        
        for idx, json_row in json_df.iterrows():
            json_project_name = json_row.get(json_project_name_col)
            
            # 과제명이 없는 JSON 행은 건너뛰기 (매핑된 데이터만 저장)
            if pd.isna(json_project_name) or not str(json_project_name).strip():
                continue
            
            json_project_name_clean = str(json_project_name).strip()
            
            # JSON의 기준년도 추출
            json_year = None
            if json_year_col:
                json_year_value = json_row.get(json_year_col)
                if pd.notna(json_year_value):
                    # 숫자로 변환 시도
                    try:
                        if isinstance(json_year_value, (int, float)):
                            json_year = str(int(json_year_value))
                        else:
                            json_year = str(json_year_value).strip()
                    except:
                        json_year = None
            
            # 매칭되는 데이터베이스 프로젝트 찾기
            matched_db_projects = db_projects_by_name.get(json_project_name_clean, [])
            
            if matched_db_projects:
                # 매칭된 경우: 각 DB 프로젝트와 JSON 데이터를 병합
                for db_project in matched_db_projects:
                    # 연도 매칭 확인 (기준년도 컬럼이 있는 경우)
                    if json_year_col and json_year:
                        # RCH_ST_DT에서 연도 추출 (yyyymmdd 형태에서 yyyy 추출)
                        rch_st_dt = db_project.get('RCH_ST_DT')
                        db_year = None
                        if rch_st_dt:
                            rch_st_dt_str = str(rch_st_dt).strip()
                            if len(rch_st_dt_str) >= 4:
                                db_year = rch_st_dt_str[:4]  # 처음 4자리 (yyyy)
                        
                        # 연도가 매칭되지 않으면 건너뛰기
                        if db_year != json_year:
                            skipped_by_year += 1
                            continue
                    
                    # 매핑 전 데이터베이스 row 출력
                    print(f"\n[매핑 발견] {json_project_name_clean}")
                    if json_year:
                        print(f"  [연도 매칭] DB={db_year}, JSON={json_year}")
                    print(f"  [매핑 전 DB row]")
                    for key, value in db_project.items():
                        if value is not None:
                            print(f"    {key}: {value}")
                    
                    # JSON 데이터 추출
                    json_data = {}
                    for col in json_df.columns:
                        value = json_row[col]
                        if pd.notna(value):
                            if isinstance(value, (int, float)):
                                json_data[col] = value
                            else:
                                json_data[col] = str(value)
                        else:
                            json_data[col] = None
                    
                    # 병합: DB 데이터를 기본으로 하고 JSON 데이터는 excel_ prefix 추가
                    merged_item = {}
                    # 먼저 DB 데이터 추가
                    merged_item.update(db_project)
                    # JSON 데이터는 excel_ prefix로 추가 (기존 호환성 유지)
                    for k, v in json_data.items():
                        merged_item[f"excel_{k}"] = v
                    
                    # 교수 정보 추가 (PRJ_RSPR_EMP_ID로 매핑)
                    prj_rspr_emp_id = db_project.get(COL_PROJECT_RSPR_EMP_ID)
                    if prj_rspr_emp_id and professor_info_by_emp_id:
                        emp_id_key = str(prj_rspr_emp_id).strip()
                        matched_professors = professor_info_by_emp_id.get(emp_id_key, [])
                        if matched_professors:
                            # 첫 번째 매칭된 교수 정보 사용
                            professor_info = matched_professors[0]["professor_info"]
                            merged_item["professor_info"] = professor_info
                            professor_matched_count += 1
                            print(f"  [교수 정보 매칭] {professor_info.get('NM', '알 수 없음')} (EMP_NO: {professor_info.get('EMP_NO', '')})")
                    
                    # 매핑 후 병합된 row 출력
                    print(f"  [매핑 후 병합된 row] (일부 컬럼만 표시):")
                    for key in list(merged_item.keys())[:10]:  # 처음 10개만
                        print(f"    {key}: {merged_item[key]}")
                    if len(merged_item) > 10:
                        print(f"    ... (총 {len(merged_item)}개 컬럼)")
                    
                    merged_data.append(merged_item)
                    matched_count += 1
        
        print(f"\n[완료] 데이터 병합 완료")
        print(f"   - 매핑된 데이터: {matched_count:,}개 (저장 대상)")
        print(f"   - 교수 정보 매칭된 데이터: {professor_matched_count:,}개")
        if json_year_col:
            print(f"   - 연도 불일치로 제외된 데이터: {skipped_by_year:,}개")
        
        return merged_data, professor_matched_count
    
    def collect_and_save(self, limit: Optional[int] = None):
        """
        연구과제 데이터를 수집하고 JSON 파일로 저장합니다.
        
        Args:
            limit: 처리할 최대 개수 (None이면 전체)
        """
        conn = None
        collected_data = []
        
        try:
            conn = get_db_connection()
            
            # 통계 정보 수집
            print("\n📈 통계 정보 수집 중...")
            stats = self.get_statistics(conn)
            
            # 초기 통계 출력 (수집 전)
            self.print_statistics(stats, collected_count=0, professor_matched_count=0)
            
            # JSON 파일 읽기
            print("\n[JSON 파일 읽기]")
            json_df = self.load_json_files()
            
            if json_df.empty:
                print("[경고] JSON 데이터가 없습니다. 데이터베이스 데이터만 수집합니다.")
            
            # 데이터베이스에서 연구과제 데이터 가져오기
            print("\n🔍 데이터베이스에서 연구과제 데이터 조회 중...")
            db_projects = self.get_db_projects(conn, limit)
            
            if not db_projects:
                print("⚠️ 데이터베이스에 연구과제 데이터가 없습니다.")
                return
            
            # 교수 정보가 포함된 연구과제 데이터 가져오기
            print("\n👤 교수 정보 조회 중...")
            projects_with_professor = self.get_db_projects_with_professor(conn, limit)
            print(f"   - 교수 정보가 포함된 프로젝트: {len([p for p in projects_with_professor if p.get('professor_info')]):,}개")
            
            # 데이터 병합
            print("\n[데이터 병합 중]")
            collected_data, professor_matched_count = self.merge_data(db_projects, json_df, projects_with_professor)
            
            # JSON 파일로 저장 (교수 정보가 매핑된 데이터만)
            filtered_data = []
            if collected_data:
                # 교수 정보가 있는 데이터만 필터링
                for item in collected_data:
                    if "professor_info" in item and item["professor_info"]:
                        # professor_info가 비어있지 않은 경우만 저장
                        prof_info = item["professor_info"]
                        if prof_info and prof_info.get("SQ"):
                            filtered_data.append(item)
                
                if filtered_data:
                    # 프로젝트 데이터 저장 (data 폴더)
                    project_output_file = Path(PROJECT_DATA_FILE)
                    project_output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(project_output_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"\n[완료] 총 {len(filtered_data):,}개의 연구과제 데이터를 수집하여 저장했습니다.")
                    print(f"[저장 위치] {project_output_file}")
                    print(f"   (교수 정보가 매핑된 데이터만 저장)")
                else:
                    print("\n[경고] 교수 정보가 매핑된 데이터가 없습니다.")
            else:
                print("\n[경고] 수집된 데이터가 없습니다.")
            
            # 최종 통계 출력 (수집 후)
            self.print_statistics(stats, collected_count=len(filtered_data), professor_matched_count=professor_matched_count)
            
        except Exception as e:
            print(f"\n[오류] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시에도 현재까지의 통계 출력
            try:
                if conn:
                    stats = self.get_statistics(conn)
                    filtered_data = [item for item in collected_data if item.get("professor_info") and item["professor_info"].get("SQ")]
                    professor_matched_count = len(filtered_data)
                    self.print_statistics(stats, collected_count=len(filtered_data), professor_matched_count=professor_matched_count)
            except:
                pass
        finally:
            close_db_connection(conn)


if __name__ == "__main__":
    collector = ProjectCollector()
    
    # JSON 파일로 저장 (limit=None이면 전체 수집)
    collector.collect_and_save(limit=None)
