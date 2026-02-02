"""
AHP 설정
AHP 가중치 및 비교 행렬 설정
"""

# ============================================================
# 1차 기준: 데이터 타입별 가중치
# ============================================================
DEFAULT_TYPE_WEIGHTS = {
    "article": 0.4,     # 논문
    "patent": 0.2,      # 특허
    "project": 0.4      # 연구과제
}

# ============================================================
# 2차 기준: L1 - 시간 (모든 데이터 타입 공통)
# ============================================================
TIME_WEIGHTS = {
    "0-2": 0.56,    # 0~2년
    "3-5": 0.26,    # 3~5년
    "6-8": 0.12,    # 6~8년
    "9-11": 0.06    # 9~11년
}

# ============================================================
# 2차 기준: L2 - 기여도 (논문만)
# ============================================================
ARTICLE_CONTRIBUTION_WEIGHTS = {
    "단독": 0.52,
    "공동_교신": 0.20,
    "공동_제1저자": 0.20,
    "공동_참여": 0.08
}

# ============================================================
# 2차 기준: L3 - 규모
# ============================================================

# 논문 규모 가중치 (학술지 등급)
ARTICLE_SCALE_WEIGHTS = {
    "국제공인학술지(SCI급)": 0.52,
    "국제_일반_학술지": 0.20,
    "국내_전문_학술지": 0.20,
    "국내_일반_학술지": 0.08
}

# 연구과제 규모 가중치 (연구비 규모)
PROJECT_SCALE_WEIGHTS = {
    "5억원_초과": 0.56,
    "3억원_초과_5억원_이하": 0.26,
    "5천만원_초과_3억원_이하": 0.12,
    "5천만원_이하": 0.06
}

# 특허는 L3 규모 기준 없음

# ============================================================
# 2차 기준: L4 - 권리상태 (특허만)
# ============================================================
PATENT_STATUS_WEIGHTS = {
    "등록": 0.75,
    "소멸": 0.25
}

# ============================================================
# 메타데이터 필드 매핑 (data/train 폴더의 원본 데이터 구조)
# ============================================================

# 논문 (Article) 메타데이터 필드
ARTICLE_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 기준)
    "contribution": "metadata.THSS_PATICP_GBN",       # 기여도 (L2 기준)
    "journal_type": "metadata.JRNL_GBN"                # 학술지 등급 (L3 규모 기준)
}

# 특허 (Patent) 메타데이터 필드
PATENT_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 기준)
    "status": "metadata.kipris_register_status"       # 권리상태 (L4 기준)
}

# 연구과제 (Project) 메타데이터 필드
PROJECT_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 기준)
    "budget": "metadata.TOT_RND_AMT"                   # 연구비 총액 (L3 규모 기준)
}

# ============================================================
# 각 데이터 타입별 종합 가중치 구조
# ============================================================

# 논문 (Article) 평가 기준 가중치
# L1(시간) + L2(기여도) + L3(규모)
ARTICLE_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "contribution": 1.0,      # L2 기여도 가중치 (ARTICLE_CONTRIBUTION_WEIGHTS 사용)
    "scale": 1.0              # L3 규모 가중치 (ARTICLE_SCALE_WEIGHTS 사용)
}

# 특허 (Patent) 평가 기준 가중치
# L1(시간) + L4(권리상태)
PATENT_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "status": 1.0             # L4 권리상태 가중치 (PATENT_STATUS_WEIGHTS 사용)
}

# 연구과제 (Project) 평가 기준 가중치
# L1(시간) + L3(규모)
PROJECT_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "scale": 1.0              # L3 규모 가중치 (PROJECT_SCALE_WEIGHTS 사용)
}

# ============================================================
# 메타데이터 값 매핑 함수
# ============================================================

def map_article_contribution(value: str) -> str:
    """
    논문 기여도 값을 AHP 가중치 키로 매핑
    
    Args:
        value: metadata.THSS_PATICP_GBN 값
            - "단독"
            - "공동(제1)"
            - "공동(교신)"
            - "공동(참여)"
    
    Returns:
        AHP 가중치 키
    """
    mapping = {
        "단독": "단독",
        "공동(제1)": "공동_제1저자",
        "공동(교신)": "공동_교신",
        "공동(참여)": "공동_참여"
    }
    return mapping.get(value, "공동_참여")  # 기본값: 가장 낮은 가중치


def map_article_journal_type(value: str) -> str:
    """
    논문 학술지 등급을 AHP 가중치 키로 매핑
    
    Args:
        value: metadata.JRNL_GBN 값
            - "국제공인학술지(SCI급)"
            - "국제 일반 학술지"
            - "국내전문학술지(학진등재[후보])" -> "국내_전문_학술지"
            - "국내 일반 학술지" -> "국내_일반_학술지"
    
    Returns:
        AHP 가중치 키
    """
    # 정확한 매칭
    if "국제공인학술지(SCI급)" in value or "SCI" in value:
        return "국제공인학술지(SCI급)"
    elif "국제" in value and "일반" in value:
        return "국제_일반_학술지"
    elif "국내전문학술지" in value or "국내 전문" in value:
        return "국내_전문_학술지"
    elif "국내" in value and "일반" in value:
        return "국내_일반_학술지"
    else:
        # 기본값: 가장 낮은 가중치
        return "국내_일반_학술지"


def map_patent_status(value: str) -> str:
    """
    특허 권리상태를 AHP 가중치 키로 매핑
    
    Args:
        value: metadata.kipris_register_status 값
            - "등록"
            - "소멸"
    
    Returns:
        AHP 가중치 키
    """
    mapping = {
        "등록": "등록",
        "소멸": "소멸"
    }
    return mapping.get(value, "소멸")  # 기본값: 가장 낮은 가중치


def map_project_budget(amount: float) -> str:
    """
    연구과제 연구비를 AHP 가중치 키로 매핑
    
    Args:
        amount: metadata.TOT_RND_AMT 값 (원 단위)
    
    Returns:
        AHP 가중치 키
    """
    if amount > 500000000:  # 5억원 초과
        return "5억원_초과"
    elif amount > 300000000:  # 3억원 초과 ~ 5억원 이하
        return "3억원_초과_5억원_이하"
    elif amount > 50000000:  # 5천만원 초과 ~ 3억원 이하
        return "5천만원_초과_3억원_이하"
    else:  # 5천만원 이하
        return "5천만원_이하"


def calculate_time_weight(year: int, current_year: int = None) -> str:
    """
    연도를 기반으로 시간 가중치 키 계산
    
    Args:
        year: 문서 연도
        current_year: 현재 연도 (None이면 2026으로 가정)
    
    Returns:
        AHP 가중치 키
    """
    if current_year is None:
        from datetime import datetime
        current_year = datetime.now().year
    
    age = current_year - year
    
    if age <= 2:
        return "0-2"
    elif age <= 5:
        return "3-5"
    elif age <= 8:
        return "6-8"
    elif age <= 11:
        return "9-11"
    else:
        return "9-11"  # 11년 이상도 가장 낮은 가중치


# ============================================================
# 일관성 검증 임계값
# ============================================================
CONSISTENCY_THRESHOLD = 0.1  # CR < 0.1이면 일관성 통과
