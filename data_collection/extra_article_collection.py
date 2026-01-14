"""
논문 데이터에 교수 정보 추가
paper.json 파일에서 데이터를 읽어서 EMP_NO로 교수 정보를 매핑하여
article.json 파일로 저장합니다.
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
    TABLE_EMPLOYEE,
    COL_EMP_SQ,
    COL_EMP_NO,
    COL_EMP_NM,
    COL_EMP_GEN_GBN,
    COL_EMP_BIRTH_DT,
    COL_EMP_NAT_GBN,
    COL_EMP_RECHER_REG_NO,
    COL_EMP_WKGD_NM,
    COL_EMP_COLG_NM,
    COL_EMP_HG_NM,
    COL_EMP_HOOF_GBN,
    COL_EMP_HANDP_NO,
    COL_EMP_OFCE_TELNO,
    COL_EMP_EMAIL
)
from config.settings import ARTICLE_DATA_FILE


def get_professor_info_by_emp_no(conn: mariadb.Connection, emp_no: str) -> Optional[Dict]:
    """
    EMP_NO로 교수 정보를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        emp_no: 교수 사번
        
    Returns:
        교수 정보 딕셔너리 또는 None
    """
    if not emp_no or not str(emp_no).strip():
        return None
    
    emp_no_clean = str(emp_no).strip()
    
    query = f"""
        SELECT 
            {COL_EMP_SQ},
            {COL_EMP_NO},
            {COL_EMP_NM},
            {COL_EMP_GEN_GBN},
            {COL_EMP_BIRTH_DT},
            {COL_EMP_NAT_GBN},
            {COL_EMP_RECHER_REG_NO},
            {COL_EMP_WKGD_NM},
            {COL_EMP_COLG_NM},
            {COL_EMP_HG_NM},
            {COL_EMP_HOOF_GBN},
            {COL_EMP_HANDP_NO},
            {COL_EMP_OFCE_TELNO},
            {COL_EMP_EMAIL}
        FROM {TABLE_EMPLOYEE}
        WHERE CAST({COL_EMP_NO} AS CHAR) = '{emp_no_clean}'
        LIMIT 1
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        if df.empty:
            return None
        
        row = df.iloc[0]
        
        professor_info = {
            "SQ": str(row[COL_EMP_SQ]) if pd.notna(row[COL_EMP_SQ]) else "",
            "EMP_NO": str(row[COL_EMP_NO]) if pd.notna(row[COL_EMP_NO]) else "",
            "NM": str(row[COL_EMP_NM]) if pd.notna(row[COL_EMP_NM]) else "",
            "GEN_GBN": str(row[COL_EMP_GEN_GBN]) if pd.notna(row[COL_EMP_GEN_GBN]) else "",
            "BIRTH_DT": str(row[COL_EMP_BIRTH_DT]) if pd.notna(row[COL_EMP_BIRTH_DT]) else "",
            "NAT_GBN": str(row[COL_EMP_NAT_GBN]) if pd.notna(row[COL_EMP_NAT_GBN]) else "",
            "RECHER_REG_NO": str(row[COL_EMP_RECHER_REG_NO]) if pd.notna(row[COL_EMP_RECHER_REG_NO]) else "",
            "WKGD_NM": str(row[COL_EMP_WKGD_NM]) if pd.notna(row[COL_EMP_WKGD_NM]) else "",
            "COLG_NM": str(row[COL_EMP_COLG_NM]) if pd.notna(row[COL_EMP_COLG_NM]) else "",
            "HG_NM": str(row[COL_EMP_HG_NM]) if pd.notna(row[COL_EMP_HG_NM]) else "",
            "HOOF_GBN": str(row[COL_EMP_HOOF_GBN]) if pd.notna(row[COL_EMP_HOOF_GBN]) else "",
            "HANDP_NO": str(row[COL_EMP_HANDP_NO]) if pd.notna(row[COL_EMP_HANDP_NO]) else "",
            "OFCE_TELNO": str(row[COL_EMP_OFCE_TELNO]) if pd.notna(row[COL_EMP_OFCE_TELNO]) else "",
            "EMAIL": str(row[COL_EMP_EMAIL]) if pd.notna(row[COL_EMP_EMAIL]) else "",
        }
        
        return professor_info
    except Exception as e:
        print(f"교수 정보 조회 실패 (EMP_NO: {emp_no}): {e}")
        return None


def load_paper_json(paper_file: str = "data/article/paper_no_professor.json") -> List[Dict]:
    """
    paper_no_professor.json 파일을 읽어옵니다.
    
    Args:
        paper_file: paper_no_professor.json 파일 경로
        
    Returns:
        논문 데이터 리스트
    """
    paper_path = Path(paper_file)
    
    if not paper_path.exists():
        print(f"⚠️ 파일이 존재하지 않습니다: {paper_file}")
        return []
    
    print(f"📂 paper_no_professor.json 파일 읽기 중: {paper_path}")
    
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        print(f"  - 총 {len(paper_data):,}개의 논문 데이터 로드 완료")
        return paper_data
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def add_professor_info_to_articles(articles: List[Dict], conn: mariadb.Connection) -> List[Dict]:
    """
    논문 데이터에 교수 정보를 추가합니다.
    
    Args:
        articles: 논문 데이터 리스트
        conn: 데이터베이스 연결 객체
        
    Returns:
        교수 정보가 추가된 논문 데이터 리스트
    """
    print(f"\n👤 교수 정보 매핑 시작...")
    print(f"   - 총 논문 수: {len(articles):,}개")
    
    # EMP_NO별 교수 정보 캐시 (중복 조회 방지)
    professor_cache = {}
    
    articles_with_professor = []
    matched_count = 0
    unmatched_count = 0
    
    for idx, article in enumerate(articles, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(articles):,}개 (매칭: {matched_count:,}개, 미매칭: {unmatched_count:,}개)")
        
        # 기존 데이터 복사
        article_with_prof = article.copy()
        
        # EMP_NO 추출
        emp_no = article.get("EMP_NO")
        
        if emp_no:
            emp_no_str = str(emp_no).strip()
            
            # 캐시에서 먼저 확인
            if emp_no_str in professor_cache:
                professor_info = professor_cache[emp_no_str]
            else:
                # 데이터베이스에서 조회
                professor_info = get_professor_info_by_emp_no(conn, emp_no_str)
                professor_cache[emp_no_str] = professor_info
            
            if professor_info:
                article_with_prof["professor_info"] = professor_info
                matched_count += 1
            else:
                article_with_prof["professor_info"] = None
                unmatched_count += 1
        else:
            article_with_prof["professor_info"] = None
            unmatched_count += 1
        
        articles_with_professor.append(article_with_prof)
    
    print(f"\n✅ 교수 정보 매핑 완료")
    print(f"   - 매칭된 논문: {matched_count:,}개")
    print(f"   - 미매칭된 논문: {unmatched_count:,}개")
    print(f"   - 교수 정보 캐시 크기: {len(professor_cache):,}개")
    
    return articles_with_professor


def save_article_json(articles: List[Dict], output_file: str = None):
    """
    논문 데이터를 JSON 파일로 저장합니다.
    
    Args:
        articles: 논문 데이터 리스트
        output_file: 출력 파일 경로 (None이면 설정 파일의 경로 사용)
    """
    if output_file is None:
        output_file = ARTICLE_DATA_FILE
    
    # data 폴더에 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 JSON 파일 저장 중: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 총 {len(articles):,}개의 논문 데이터를 저장했습니다.")
        print(f"📁 저장 위치: {output_path}")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    메인 함수: paper_no_professor.json에서 데이터를 읽어서 교수 정보를 추가하고 article.json으로 저장
    """
    conn = None
    
    try:
        # 데이터베이스 연결
        print("\n🔌 데이터베이스 연결 중...")
        conn = get_db_connection()
        print("✅ 데이터베이스 연결 성공")
        
        # paper_no_professor.json 파일 읽기
        print("\n📂 paper_no_professor.json 파일 읽기 중...")
        articles = load_paper_json()
        
        if not articles:
            print("⚠️ 논문 데이터가 없습니다.")
            return
        
        # 교수 정보 추가
        articles_with_professor = add_professor_info_to_articles(articles, conn)
        
        # JSON 파일로 저장
        save_article_json(articles_with_professor)
        
        # 통계 출력
        print("\n" + "=" * 60)
        print("📊 최종 통계")
        print("=" * 60)
        print(f"1️⃣  전체 논문 수: {len(articles_with_professor):,}개")
        professor_matched = len([a for a in articles_with_professor if a.get("professor_info")])
        print(f"2️⃣  교수 정보 매칭된 논문 수: {professor_matched:,}개")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
