"""
KIPRIS 특허 데이터 수집기
MariaDB의 tb_inu_tech 테이블에서 ptnt_rgstr_id(특허 등록번호)를 기반으로 
KIPRIS에서 특허 데이터를 수집하고 JSON 파일로 저장합니다.
등록번호에서 하이픈(-)을 제거한 후 검색합니다.
"""

import mariadb
import pandas as pd
import requests
import time
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.database import (
    get_db_connection, 
    close_db_connection, 
    get_patent_statistics, 
    get_patent_register_ids,
    TARGET_TABLE, 
    COL_PATENT_REGISTER_ID
)
from config.settings import KIPRIS_API_KEY, PATENT_DATA_FILE


class KIPRISCollector:
    """KIPRIS API를 사용하여 특허 데이터를 수집하는 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        KIPRIS 수집기 초기화
        
        Args:
            api_key: KIPRIS API 키 (선택사항)
        """
        self.api_key = api_key
    
    def get_statistics(self, conn: mariadb.Connection) -> Dict[str, int]:
        """
        데이터베이스에서 통계 정보를 수집합니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            
        Returns:
            통계 정보 딕셔너리
        """
        return get_patent_statistics(conn)
    
    def print_statistics(self, stats: Dict[str, int], collected_count: int = 0, 
                        api_success_count: int = 0, api_fail_count: int = 0):
        """
        통계 정보를 단계적으로 출력합니다.
        
        Args:
            stats: 통계 정보 딕셔너리
            collected_count: 최종 수집된 데이터 개수 (필터링 후)
            api_success_count: API 검색 성공 개수
            api_fail_count: API 검색 실패 개수
        """
        print("\n" + "=" * 70)
        print("[데이터 수집 통계]")
        print("=" * 70)
        print(f"[1] 특허 테이블 전체 데이터 개수: {stats['total_records']:,}개")
        print(f"[2] 등록번호(ptnt_rgstr_id)가 있는 데이터: {stats.get('records_with_register_id', 0):,}개")
        print(f"[3] 교수 사번 매칭된 데이터: {stats.get('records_matched_with_professor', 0):,}개")
        if api_success_count > 0 or api_fail_count > 0:
            print(f"[4] API 검색 성공: {api_success_count:,}개")
            print(f"[5] API 검색 실패: {api_fail_count:,}개")
            print(f"[6] API 성공률: {(api_success_count / (api_success_count + api_fail_count) * 100):.1f}%")
        print(f"[최종] 필터링 후 저장된 데이터: {collected_count:,}개")
        if api_success_count > 0:
            print(f"[필터링률] {collected_count / api_success_count * 100:.1f}% (API 성공 대비 저장 비율)")
        print("=" * 70)
        print()
    
    def get_patent_register_ids(self, conn: mariadb.Connection, limit: Optional[int] = None, verbose: bool = False) -> List[Dict]:
        """
        데이터베이스에서 특허 등록번호(ptnt_rgstr_id)와 교수 정보를 가져옵니다.
        (ptnt_rgstr_id가 있고, v_emp1 테이블의 SQ와 매칭되는 것만)
        
        Args:
            conn: 데이터베이스 연결 객체
            limit: 가져올 최대 개수 (None이면 전체)
            verbose: 쿼리 정보를 출력할지 여부
            
        Returns:
            [{"ptnt_rgstr_id": "...", "ptnt_rgstr_id_clean": "...", "tech_nm": "...", "mbr_sn": "...", "professor_info": {...}}, ...] 형태의 리스트
        """
        return get_patent_register_ids(conn, limit, verbose)
    
    def get_original_patent_data(self, conn: mariadb.Connection, register_id: str) -> Optional[Dict]:
        """
        원본 특허 테이블에서 해당 등록번호의 모든 데이터를 가져옵니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            register_id: 특허 등록번호 (ptnt_rgstr_id)
            
        Returns:
            원본 테이블 데이터 딕셔너리 또는 None
        """
        try:
            # register_id는 이미 검증된 값이므로 안전하게 사용
            query = f"SELECT * FROM {TARGET_TABLE} WHERE {COL_PATENT_REGISTER_ID} = '{register_id}' LIMIT 1"
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return None
            
            # 첫 번째 행을 딕셔너리로 변환 (NaN 값 처리)
            row = df.iloc[0]
            original_data = {}
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    # 숫자 타입은 그대로, 문자열은 str로 변환
                    if isinstance(value, (int, float)):
                        original_data[col] = value
                    else:
                        original_data[col] = str(value)
                else:
                    original_data[col] = None
            
            return original_data
        except Exception as e:
            print(f"  - 원본 데이터 조회 실패 ({register_id}): {e}")
            return None
    
    def fetch_patent_data(self, register_id: str, mbr_sn: str = "", professor_info: Dict = None) -> Optional[Dict]:
        """
        KIPRIS API에서 특정 등록번호의 특허 데이터를 가져옵니다.
        
        Args:
            register_id: 특허 등록번호 (ptnt_rgstr_id, - 제거된 버전)
            mbr_sn: 교수 사번
            professor_info: 교수 정보 딕셔너리
            
        Returns:
            특허 데이터 딕셔너리 또는 None
        """
        if not self.api_key:
            print("API 키가 설정되지 않았습니다.")
            return None
        
        try:
            print(f"[특허 데이터 수집] 등록번호: {register_id}")
            
            # 등록번호로 검색 (registerNumber 파라미터 사용)
            import urllib.parse
            encoded_register_id = urllib.parse.quote(register_id, safe='')
            url = (
                f"https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
                f"?registerNumber={encoded_register_id}"
                f"&ServiceKey={self.api_key}"
                f"&numOfRows=10"
                f"&pageNo=1"
            )
            
            print(f"  [API 요청 URL] {url[:150]}...")  # URL 일부만 출력
            print(f"  [검색 필드] registerNumber (등록번호)")
            print(f"  [검색어] {register_id}")
            print(f"  [인코딩된 검색어] {encoded_register_id[:100]}...")
            
            try:
                response = requests.get(url, timeout=30)
                print(f"  [응답 상태 코드] {response.status_code}")
                
                # HTML 응답인지 확인
                if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                    print(f"  [오류] HTML 응답 (API 오류)")
                    print(f"  [응답 내용 일부] {response.text[:300]}")
                    return None
                
                # XML 파싱 (<?xml 선언이 없어도 <response> 등으로 시작할 수 있음)
                try:
                    root = ET.fromstring(response.content)
                except ET.ParseError:
                    # XML 선언이 없을 수 있으므로 다시 시도
                    root = ET.fromstring(response.text)
                
                # 에러 응답 확인
                success_yn = root.findtext(".//successYN", default="")
                result_msg = root.findtext(".//resultMsg", default="")
                result_code = root.findtext(".//resultCode", default="")
                total_count = root.findtext(".//totalCount", default="0")
                
                print(f"  [API 응답 정보]")
                print(f"    - successYN: {success_yn}")
                print(f"    - resultCode: {result_code}")
                print(f"    - resultMsg: {result_msg}")
                print(f"    - totalCount: {total_count}")
                
                # successYN이 "N"이거나 에러 메시지가 있는 경우만 에러 처리
                if success_yn == "N" or (result_msg and "ERROR" in result_msg.upper()):
                    print(f"  [오류] API 오류: {result_msg} (코드: {result_code})")
                    # 호출 제한 에러인 경우 예외 발생
                    if result_code in ["20", "21", "22"] or "LIMIT" in result_msg.upper() or "QUOTA" in result_msg.upper():
                        raise Exception(f"API 호출 제한 도달: {result_msg}")
                    return None
                
                print(f"  [XML 파싱 완료]")
                
            except ET.ParseError as e:
                print(f"  [오류] XML 파싱 실패: {str(e)[:100]}")
                print(f"  [응답 내용] (처음 500자): {response.text[:500]}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"  [오류] 요청 실패: {str(e)[:100]}")
                return None
            
            # XML에서 필요한 정보 추출 (예시 코드처럼 findtext 사용)
            items = root.findall(".//item")
            
            print(f"  [검색 결과] 총 {len(items)}개의 결과 발견")
            
            if items:
                # 등록번호로 검색했으므로 첫 번째 결과 사용
                item = items[0]
                
                # 검색된 등록번호 확인
                found_register_number = item.findtext("registerNumber", default="")
                found_title = item.findtext("inventionTitle", default="")
                print(f"  [검색된 등록번호] {found_register_number}")
                print(f"  [검색된 발명의 명칭] {found_title[:100]}...")
                
                # 등록번호 일치 확인
                if register_id not in found_register_number.replace("-", ""):
                    print(f"  [주의] 검색된 등록번호가 원본과 일치하지 않을 수 있습니다.")
                    print(f"    원본: {register_id}")
                    print(f"    검색결과: {found_register_number}")
                
                # 예시 코드 구조를 참고하여 findtext 사용
                result_data = {
                    "ptnt_rgstr_id": register_id,  # 원본 등록번호 (- 제거된 버전)
                    "mbr_sn": mbr_sn,  # 교수 사번
                    "kipris_index_no": item.findtext("indexNo", default=""),
                    "kipris_register_status": item.findtext("registerStatus", default=""),
                    "kipris_register_number": item.findtext("registerNumber", default=""),  # 등록번호
                    "kipris_application_date": item.findtext("applicationDate", default=""),
                    "kipris_abstract": item.findtext("astrtCont", default="").strip(),
                    "kipris_application_name": item.findtext("inventionTitle", default=""),
                    "kipris_application_number": item.findtext("applicationNumber", default=""),  # 출원번호도 저장
                }
                
                # 교수 정보 추가
                if professor_info:
                    result_data["professor_info"] = professor_info
                
                # totalCount 확인
                if total_count:
                    result_data["kipris_total_count"] = total_count
                
                print(f"  [추출된 데이터]")
                print(f"    - indexNo: {result_data.get('kipris_index_no')}")
                print(f"    - registerStatus: {result_data.get('kipris_register_status')}")
                print(f"    - registerNumber: {result_data.get('kipris_register_number')}")
                print(f"    - applicationDate: {result_data.get('kipris_application_date')}")
                print(f"    - applicationName: {result_data.get('kipris_application_name')[:50]}...")
                print(f"    - applicationNumber: {result_data.get('kipris_application_number')}")
                
                return result_data
            else:
                print(f"  [결과 없음] 등록번호 '{register_id}'에 대한 데이터를 찾을 수 없습니다.")
                print(f"  [디버깅 정보] API는 정상 응답했지만 검색 결과가 없습니다.")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패 ({register_id}): {e}")
            return None
        except ET.ParseError as e:
            print(f"XML 파싱 실패 ({register_id}): {e}")
            # 디버깅을 위해 응답 내용 출력
            try:
                print(f"응답 내용 (처음 500자): {response.text[:500]}")
            except:
                pass
            return None
        except Exception as e:
            print(f"특허 데이터 수집 실패 ({register_id}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_columns_to_table(self, conn: mariadb.Connection, columns: List[Dict[str, str]]):
        """
        테이블에 새로운 컬럼을 추가합니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            columns: 추가할 컬럼 정보 리스트 [{"name": "컬럼명", "type": "VARCHAR(255)"}, ...]
        """
        cursor = conn.cursor()
        
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            
            try:
                # 컬럼이 이미 존재하는지 확인
                check_query = f"""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = 'indigo' 
                    AND TABLE_NAME = '{TARGET_TABLE}' 
                    AND COLUMN_NAME = '{col_name}'
                """
                cursor.execute(check_query)
                exists = cursor.fetchone()[0] > 0
                
                if not exists:
                    alter_query = f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {col_name} {col_type}"
                    cursor.execute(alter_query)
                    print(f"컬럼 추가 완료: {col_name}")
                else:
                    print(f"컬럼이 이미 존재합니다: {col_name}")
                    
            except mariadb.Error as e:
                print(f"컬럼 추가 실패 ({col_name}): {e}")
        
        conn.commit()
    
    def update_patent_data(self, conn: mariadb.Connection, patent_data: Dict):
        """
        수집한 특허 데이터를 테이블에 업데이트합니다.
        
        Args:
            conn: 데이터베이스 연결 객체
            patent_data: 업데이트할 특허 데이터
        """
        cursor = conn.cursor()
        
        register_id = patent_data.get("ptnt_rgstr_id")
        if not register_id:
            print("등록번호가 없어 업데이트할 수 없습니다.")
            return
        
        # 업데이트할 컬럼들 (ptnt_rgstr_id 제외)
        update_fields = {k: v for k, v in patent_data.items() if k != "ptnt_rgstr_id"}
        
        if not update_fields:
            print("업데이트할 데이터가 없습니다.")
            return
        
        # 컬럼이 존재하는지 확인하고 없으면 추가
        cursor.execute(f"DESCRIBE {TARGET_TABLE}")
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        new_columns = []
        for col_name in update_fields.keys():
            if col_name not in existing_columns:
                new_columns.append({
                    "name": col_name,
                    "type": "TEXT"  # 기본 타입, 필요에 따라 수정
                })
        
        if new_columns:
            self.add_columns_to_table(conn, new_columns)
        
        # 데이터 업데이트
        set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [register_id]
        
        update_query = f"""
            UPDATE {TARGET_TABLE} 
            SET {set_clause} 
            WHERE {COL_PATENT_REGISTER_ID} = ?
        """
        
        try:
            cursor.execute(update_query, values)
            conn.commit()
            print(f"데이터 업데이트 완료: {register_id}")
        except mariadb.Error as e:
            print(f"데이터 업데이트 실패 ({register_id}): {e}")
            conn.rollback()
    
    def collect_and_save(self, limit: Optional[int] = None):
        """
        특허 데이터를 수집하고 JSON 파일로 저장합니다.
        
        Args:
            limit: 처리할 최대 개수 (None이면 전체)
        """
        conn = None
        collected_data = []
        api_success_count = 0
        api_fail_count = 0
        
        try:
            conn = get_db_connection()
            
            # 통계 정보 수집
            print("\n" + "=" * 70)
            print("[1단계: 통계 정보 수집]")
            print("=" * 70)
            stats = self.get_statistics(conn)
            
            # 초기 통계 출력 (수집 전)
            self.print_statistics(stats, collected_count=0, api_success_count=0, api_fail_count=0)
            
            # 특허 등록번호 목록 가져오기 (ptnt_rgstr_id가 있는 것만)
            print("\n" + "=" * 70)
            print("[2단계: 데이터베이스 쿼리]")
            print("=" * 70)
            
            register_id_list = self.get_patent_register_ids(conn, limit, verbose=True)
            
            if not register_id_list:
                print("[경고] 처리할 특허 등록번호가 없습니다.")
                return
            
            print(f"[쿼리 결과] 총 {len(register_id_list):,}개의 등록번호 조회됨")
            
            # 데이터 수집 시작
            total = len(register_id_list)
            print("\n" + "=" * 70)
            print(f"[3단계: KIPRIS API 데이터 수집]")
            print("=" * 70)
            print(f"[처리 대상] 총 {total:,}개의 등록번호를 처리합니다.\n")
            
            for idx, register_info in enumerate(register_id_list, 1):
                original_register_id = register_info["ptnt_rgstr_id"]  # 원본 (하이픈 포함)
                clean_register_id = register_info["ptnt_rgstr_id_clean"]  # 하이픈 제거된 버전
                tech_nm = register_info.get("tech_nm", "")  # 특허명 (확인용)
                mbr_sn = register_info["mbr_sn"]
                professor_info = register_info.get("professor_info", {})
                prof_name = professor_info.get("NM", "알 수 없음")
                print(f"\n[{idx}/{total}] ========================================")
                print(f"[처리 중] 등록번호 (원본): {original_register_id}")
                print(f"[처리 중] 등록번호 (검색용): {clean_register_id}")
                if tech_nm:
                    print(f"[확인용] 특허명: {tech_nm}")
                print(f"[교수 정보] 이름: {prof_name}, 사번: {mbr_sn}")
                
                try:
                    # 원본 테이블 데이터 가져오기 (등록번호로 조회)
                    original_data = self.get_original_patent_data(conn, original_register_id)
                    if original_data:
                        print(f"[원본 데이터] 조회 성공 (컬럼 수: {len(original_data)}개)")
                    else:
                        print(f"[원본 데이터] 조회 실패 또는 데이터 없음")
                    
                    # KIPRIS API에서 특허 데이터 수집 (하이픈 제거된 등록번호 사용)
                    kipris_data = self.fetch_patent_data(clean_register_id, mbr_sn, professor_info)
                    
                    if kipris_data:
                        api_success_count += 1
                        print(f"[결과] API 검색 성공")
                        
                        # 원본 데이터와 KIPRIS 데이터 병합 (KIPRIS 데이터가 우선)
                        merged_data = {}
                        
                        # 먼저 원본 데이터 추가
                        if original_data:
                            merged_data.update(original_data)
                        
                        # KIPRIS 데이터 추가 (같은 키가 있으면 덮어씀)
                        merged_data.update(kipris_data)
                        
                        # 교수 정보는 항상 포함 (kipris_data에 이미 포함되어 있지만 확실하게)
                        if professor_info:
                            merged_data["professor_info"] = professor_info
                        
                        collected_data.append(merged_data)
                    else:
                        api_fail_count += 1
                        print(f"[결과] API 검색 실패")
                
                except Exception as e:
                    api_fail_count += 1
                    error_msg = str(e)
                    if "호출 제한" in error_msg or "LIMIT" in error_msg.upper() or "QUOTA" in error_msg.upper():
                        print(f"\n[중단] API 호출 제한에 도달했습니다. 수집을 중단합니다.")
                        print(f"[현재까지 수집] {len(collected_data):,}개의 데이터")
                        break
                    else:
                        print(f"[오류] {error_msg}")
                        continue
                
                # API 호출 제한을 위한 대기 (1초)
                try:
                    if idx < total:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n[중단] 사용자에 의해 중단되었습니다.")
                    break
            
            # JSON 파일로 저장 (특허 정보와 교수 정보가 모두 있는 것만)
            print("\n" + "=" * 70)
            print("[4단계: 데이터 필터링 및 저장]")
            print("=" * 70)
            print(f"[필터링 전] API 검색 성공 데이터: {len(collected_data):,}개")
            
            filtered_data = []
            if collected_data:
                # 특허 정보와 교수 정보가 모두 있는 데이터만 필터링
                for item in collected_data:
                    if "professor_info" in item and item["professor_info"]:
                        # professor_info가 비어있지 않은 경우만 저장
                        prof_info = item["professor_info"]
                        if prof_info and prof_info.get("SQ"):
                            filtered_data.append(item)
                
                print(f"[필터링 조건] professor_info가 있고 SQ가 있는 데이터만 저장")
                print(f"[필터링 후] 저장 대상 데이터: {len(filtered_data):,}개")
                print(f"[필터링률] {len(filtered_data) / len(collected_data) * 100:.1f}%")
                
                if filtered_data:
                    # 특허 데이터 저장 (data 폴더)
                    patent_output_file = Path(PATENT_DATA_FILE)
                    patent_output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(patent_output_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"\n[저장 완료]")
                    print(f"  - 저장된 데이터: {len(filtered_data):,}개")
                    print(f"  - 저장 위치: {patent_output_file}")
                    print(f"  - 파일 크기: {patent_output_file.stat().st_size / 1024 / 1024:.2f} MB")
                else:
                    print("\n[경고] 필터링 후 저장할 데이터가 없습니다.")
            else:
                print("[경고] 수집된 데이터가 없습니다.")
            
            # 최종 통계 출력 (수집 후)
            self.print_statistics(stats, collected_count=len(filtered_data), 
                                api_success_count=api_success_count, api_fail_count=api_fail_count)
            
        except Exception as e:
            print(f"\n[오류] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시에도 현재까지의 통계 출력
            try:
                if conn:
                    stats = self.get_statistics(conn)
                    self.print_statistics(stats, collected_count=len(collected_data),
                                        api_success_count=api_success_count, api_fail_count=api_fail_count)
            except:
                pass
        finally:
            close_db_connection(conn)


if __name__ == "__main__":
    # config에서 API 키 가져오기
    print(f"사용 중인 API 키: {KIPRIS_API_KEY[:20]}... (처음 20자만 표시)")
    collector = KIPRISCollector(api_key=KIPRIS_API_KEY)
    
    # JSON 파일로 저장 (limit=None이면 전체 수집, 호출 제한까지)
    collector.collect_and_save(limit=None)

