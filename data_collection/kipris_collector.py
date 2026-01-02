"""
KIPRIS 특허 데이터 수집기
MariaDB의 tb_inu_tech 테이블에서 tech_aplct_id를 기반으로 
KIPRIS에서 특허 데이터를 수집하고 테이블에 추가합니다.
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
from config.database import get_db_connection, close_db_connection, TARGET_TABLE, TARGET_ID_COLUMN
from config.settings import KIPRIS_API_KEY


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
        stats = {}
        
        # 1. 특허 테이블 전체 데이터 개수
        query_total = f"SELECT COUNT(*) as total_count FROM {TARGET_TABLE}"
        df_total = pd.read_sql(query_total, conn)
        stats['total_records'] = int(df_total.iloc[0]['total_count'])
        
        # 2. 출원 아이디(tech_aplct_id)가 있는 데이터 개수
        query_with_id = f"""
            SELECT COUNT(*) as count_with_id
            FROM {TARGET_TABLE}
            WHERE {TARGET_ID_COLUMN} IS NOT NULL 
                AND {TARGET_ID_COLUMN} != ''
        """
        df_with_id = pd.read_sql(query_with_id, conn)
        stats['records_with_application_id'] = int(df_with_id.iloc[0]['count_with_id'])
        
        # 3. 교수 사번 매칭된 데이터 개수
        query_matched = f"""
            SELECT COUNT(DISTINCT t.{TARGET_ID_COLUMN}) as matched_count
            FROM {TARGET_TABLE} t
            INNER JOIN v_emp1 e ON CAST(t.inpt_mbr_id AS CHAR) = CAST(e.EMP_NO AS CHAR)
            WHERE t.{TARGET_ID_COLUMN} IS NOT NULL 
                AND t.{TARGET_ID_COLUMN} != ''
                AND t.inpt_mbr_id IS NOT NULL
                AND t.inpt_mbr_id != ''
        """
        df_matched = pd.read_sql(query_matched, conn)
        stats['records_matched_with_professor'] = int(df_matched.iloc[0]['matched_count'])
        
        return stats
    
    def print_statistics(self, stats: Dict[str, int], collected_count: int = 0):
        """
        통계 정보를 단계적으로 출력합니다.
        
        Args:
            stats: 통계 정보 딕셔너리
            collected_count: 최종 수집된 데이터 개수
        """
        print("\n" + "=" * 60)
        print("📊 데이터 수집 통계")
        print("=" * 60)
        print(f"1️⃣  특허 테이블 전체 데이터 개수: {stats['total_records']:,}개")
        print(f"2️⃣  출원 아이디(tech_aplct_id)가 있는 데이터: {stats['records_with_application_id']:,}개")
        print(f"3️⃣  교수 사번 매칭된 데이터: {stats['records_matched_with_professor']:,}개")
        print(f"4️⃣  최종 수집된 데이터: {collected_count:,}개")
        print("=" * 60)
        print()
    
    def get_application_ids(self, conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
        """
        데이터베이스에서 특허 출원번호와 교수 정보를 가져옵니다.
        (tech_aplct_id가 있고, v_emp1 테이블의 EMP_NO와 매칭되는 것만)
        
        Args:
            conn: 데이터베이스 연결 객체
            limit: 가져올 최대 개수 (None이면 전체)
            
        Returns:
            [{"tech_aplct_id": "...", "inpt_mbr_id": "...", "professor_info": {...}}, ...] 형태의 리스트
        """
        # v_emp1과 조인하여 EMP_NO가 매칭되는 것만 가져오기
        # Collation 문제 해결을 위해 CAST 사용
        query = f"""
            SELECT DISTINCT 
                t.{TARGET_ID_COLUMN}, 
                t.inpt_mbr_id,
                e.EMP_NO,
                e.NM,
                e.GEN_GBN,
                e.BIRTH_DT,
                e.NAT_GBN,
                e.RECHER_REG_NO,
                e.WKGD_NM,
                e.COLG_NM,
                e.HG_NM,
                e.HOOF_GBN,
                e.HANDP_NO,
                e.OFCE_TELNO,
                e.EMAIL
            FROM {TARGET_TABLE} t
            INNER JOIN v_emp1 e ON CAST(t.inpt_mbr_id AS CHAR) = CAST(e.EMP_NO AS CHAR)
            WHERE t.{TARGET_ID_COLUMN} IS NOT NULL 
                AND t.{TARGET_ID_COLUMN} != ''
                AND t.inpt_mbr_id IS NOT NULL
                AND t.inpt_mbr_id != ''
        """
        if limit:
            query += f" LIMIT {limit}"
        query += ";"
        
        df = pd.read_sql(query, conn)
        
        # 딕셔너리 리스트로 변환
        application_list = []
        for _, row in df.iterrows():
            if pd.notna(row[TARGET_ID_COLUMN]):
                # 교수 정보 추출
                professor_info = {
                    "EMP_NO": str(row["EMP_NO"]) if pd.notna(row["EMP_NO"]) else "",
                    "NM": str(row["NM"]) if pd.notna(row["NM"]) else "",
                    "GEN_GBN": str(row["GEN_GBN"]) if pd.notna(row["GEN_GBN"]) else "",
                    "BIRTH_DT": str(row["BIRTH_DT"]) if pd.notna(row["BIRTH_DT"]) else "",
                    "NAT_GBN": str(row["NAT_GBN"]) if pd.notna(row["NAT_GBN"]) else "",
                    "RECHER_REG_NO": str(row["RECHER_REG_NO"]) if pd.notna(row["RECHER_REG_NO"]) else "",
                    "WKGD_NM": str(row["WKGD_NM"]) if pd.notna(row["WKGD_NM"]) else "",
                    "COLG_NM": str(row["COLG_NM"]) if pd.notna(row["COLG_NM"]) else "",
                    "HG_NM": str(row["HG_NM"]) if pd.notna(row["HG_NM"]) else "",
                    "HOOF_GBN": str(row["HOOF_GBN"]) if pd.notna(row["HOOF_GBN"]) else "",
                    "HANDP_NO": str(row["HANDP_NO"]) if pd.notna(row["HANDP_NO"]) else "",
                    "OFCE_TELNO": str(row["OFCE_TELNO"]) if pd.notna(row["OFCE_TELNO"]) else "",
                    "EMAIL": str(row["EMAIL"]) if pd.notna(row["EMAIL"]) else "",
                }
                
                application_list.append({
                    "tech_aplct_id": str(row[TARGET_ID_COLUMN]),
                    "inpt_mbr_id": str(row["inpt_mbr_id"]) if pd.notna(row["inpt_mbr_id"]) else "",
                    "professor_info": professor_info
                })
        
        return application_list
    
    def fetch_patent_data(self, application_id: str, inpt_mbr_id: str = "", professor_info: Dict = None) -> Optional[Dict]:
        """
        KIPRIS API에서 특정 출원번호의 특허 데이터를 가져옵니다.
        
        Args:
            application_id: 특허 출원번호 (tech_aplct_id)
            
        Returns:
            특허 데이터 딕셔너리 또는 None
        """
        if not self.api_key:
            print("API 키가 설정되지 않았습니다.")
            return None
        
        try:
            print(f"특허 데이터 수집 중: {application_id}")
            
            # word 파라미터로 applicationNumber 검색
            # KIPRIS API는 API 키를 그대로 사용 (URL 인코딩 불필요)
            url = (
                f"https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
                f"?word={application_id}"
                f"&ServiceKey={self.api_key}"
                f"&numOfRows=10"
                f"&pageNo=1"
            )
            
            try:
                response = requests.get(url, timeout=30)
                print(f"  - 응답 상태 코드: {response.status_code}")
                
                # HTML 응답인지 확인
                if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                    print(f"  - HTML 응답 (API 오류)")
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
                
                # successYN이 "N"이거나 에러 메시지가 있는 경우만 에러 처리
                if success_yn == "N" or (result_msg and "ERROR" in result_msg.upper()):
                    print(f"  - API 오류: {result_msg} (코드: {result_code})")
                    # 호출 제한 에러인 경우 예외 발생
                    if result_code in ["20", "21", "22"] or "LIMIT" in result_msg.upper() or "QUOTA" in result_msg.upper():
                        raise Exception(f"API 호출 제한 도달: {result_msg}")
                    return None
                
                print(f"  - XML 파싱 완료")
                
            except ET.ParseError as e:
                print(f"  - XML 파싱 실패: {str(e)[:100]}")
                print(f"  - 응답 내용 (처음 500자): {response.text[:500]}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"  - 요청 실패: {str(e)[:100]}")
                return None
            
            # XML에서 필요한 정보 추출 (예시 코드처럼 findtext 사용)
            items = root.findall(".//item")
            
            if items:
                item = items[0]  # 첫 번째 결과 사용
                
                # 예시 코드 구조를 참고하여 findtext 사용
                result_data = {
                    "tech_aplct_id": application_id,
                    "inpt_mbr_id": inpt_mbr_id,  # 교수 사번
                    "kipris_index_no": item.findtext("indexNo", default=""),
                    "kipris_register_status": item.findtext("registerStatus", default=""),
                    "kipris_application_date": item.findtext("applicationDate", default=""),
                    "kipris_abstract": item.findtext("astrtCont", default="").strip(),  # 예시에서는 astrtCont
                    "kipris_application_name": item.findtext("inventionTitle", default=""),  # 예시에서는 inventionTitle
                }
                
                # 교수 정보 추가
                if professor_info:
                    result_data["professor_info"] = professor_info
                
                # totalCount 확인
                total_count = root.findtext(".//totalCount", default="")
                if total_count:
                    result_data["kipris_total_count"] = total_count
                
                print(f"  - indexNo: {result_data.get('kipris_index_no')}")
                print(f"  - registerStatus: {result_data.get('kipris_register_status')}")
                print(f"  - applicationDate: {result_data.get('kipris_application_date')}")
                print(f"  - applicationName: {result_data.get('kipris_application_name')}")
                
                return result_data
            else:
                print(f"  - 데이터를 찾을 수 없습니다: {application_id}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패 ({application_id}): {e}")
            return None
        except ET.ParseError as e:
            print(f"XML 파싱 실패 ({application_id}): {e}")
            # 디버깅을 위해 응답 내용 출력
            print(f"응답 내용 (처음 500자): {response.text[:500]}")
            return None
        except Exception as e:
            print(f"특허 데이터 수집 실패 ({application_id}): {e}")
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
        
        application_id = patent_data.get("tech_aplct_id")
        if not application_id:
            print("출원번호가 없어 업데이트할 수 없습니다.")
            return
        
        # 업데이트할 컬럼들 (tech_aplct_id 제외)
        update_fields = {k: v for k, v in patent_data.items() if k != "tech_aplct_id"}
        
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
        values = list(update_fields.values()) + [application_id]
        
        update_query = f"""
            UPDATE {TARGET_TABLE} 
            SET {set_clause} 
            WHERE {TARGET_ID_COLUMN} = ?
        """
        
        try:
            cursor.execute(update_query, values)
            conn.commit()
            print(f"데이터 업데이트 완료: {application_id}")
        except mariadb.Error as e:
            print(f"데이터 업데이트 실패 ({application_id}): {e}")
            conn.rollback()
    
    def collect_and_save(self, limit: Optional[int] = None):
        """
        특허 데이터를 수집하고 JSON 파일로 저장합니다.
        
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
            self.print_statistics(stats, collected_count=0)
            
            # 출원번호 목록 가져오기 (tech_aplct_id가 있는 것만)
            application_list = self.get_application_ids(conn, limit)
            
            if not application_list:
                print("처리할 출원번호가 없습니다.")
                return
            
            # 데이터 수집 시작
            total = len(application_list)
            print(f"\n🔍 총 {total:,}개의 출원번호를 처리합니다.\n")
            
            for idx, app_info in enumerate(application_list, 1):
                app_id = app_info["tech_aplct_id"]
                inpt_mbr_id = app_info["inpt_mbr_id"]
                professor_info = app_info.get("professor_info", {})
                prof_name = professor_info.get("NM", "알 수 없음")
                print(f"[{idx}/{total}] 처리 중: {app_id} (교수: {prof_name}, 사번: {inpt_mbr_id})")
                
                try:
                    # 특허 데이터 수집
                    patent_data = self.fetch_patent_data(app_id, inpt_mbr_id, professor_info)
                    
                    if patent_data:
                        collected_data.append(patent_data)
                
                except Exception as e:
                    error_msg = str(e)
                    if "호출 제한" in error_msg or "LIMIT" in error_msg.upper() or "QUOTA" in error_msg.upper():
                        print(f"\n⚠️ API 호출 제한에 도달했습니다. 수집을 중단합니다.")
                        print(f"현재까지 {len(collected_data)}개의 데이터를 수집했습니다.")
                        break
                    else:
                        print(f"  - 오류 발생: {error_msg}")
                        continue
                
                # API 호출 제한을 위한 대기 (1초)
                try:
                    if idx < total:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n사용자에 의해 중단되었습니다.")
                    break
            
            # JSON 파일로 저장 (특허 정보와 교수 정보가 모두 있는 것만)
            filtered_data = []
            if collected_data:
                # 특허 정보와 교수 정보가 모두 있는 데이터만 필터링
                for item in collected_data:
                    if "professor_info" in item and item["professor_info"]:
                        # professor_info가 비어있지 않은 경우만 저장
                        prof_info = item["professor_info"]
                        if prof_info and prof_info.get("EMP_NO"):
                            filtered_data.append(item)
                
                if filtered_data:
                    # 특허 데이터 저장 (data/patent 폴더)
                    patent_output_file = Path("data/patent/kipris_data.json")
                    patent_output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(patent_output_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"\n✅ 총 {len(filtered_data):,}개의 특허 데이터를 수집하여 저장했습니다.")
                    print(f"📁 저장 위치: {patent_output_file}")
                    print(f"   (특허 정보와 교수 정보가 모두 포함된 데이터만 저장)")
                else:
                    print("\n⚠️ 특허 정보와 교수 정보가 모두 있는 데이터가 없습니다.")
            
            # 최종 통계 출력 (수집 후)
            self.print_statistics(stats, collected_count=len(filtered_data))
            
            if not collected_data:
                print("\n❌ 수집된 데이터가 없습니다.")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시에도 현재까지의 통계 출력
            try:
                if conn:
                    stats = self.get_statistics(conn)
                    self.print_statistics(stats, collected_count=len(collected_data))
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

