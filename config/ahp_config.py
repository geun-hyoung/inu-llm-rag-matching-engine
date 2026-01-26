"""
AHP 설정
AHP 가중치 및 비교 행렬 설정
"""

# 데이터 타입별 기본 가중치 (쌍대 비교 행렬로부터 계산된 값)
DEFAULT_TYPE_WEIGHTS = {
    "patent": 0.40,      # 특허
    "article": 0.35,     # 논문
    "project": 0.25      # 연구과제
}

# 데이터 타입별 쌍대 비교 행렬 (1~9 척도)
# 값의 의미: 1=동등, 3=약간 중요, 5=중요, 7=매우 중요, 9=절대적으로 중요
TYPE_COMPARISON_MATRIX = {
    ("patent", "article"): 1.5,      # 특허가 논문보다 약간 중요
    ("patent", "project"): 2.0,       # 특허가 연구과제보다 중요
    ("article", "project"): 1.3       # 논문이 연구과제보다 약간 중요
}

# 각 데이터 타입 내부 평가 기준 가중치
# 문서 수, 평균 유사도, 최고 유사도 등의 가중치
PATENT_CRITERIA_WEIGHTS = {
    "count": 0.3,           # 문서 수
    "avg_similarity": 0.4,  # 평균 유사도
    "max_similarity": 0.3   # 최고 유사도
}

ARTICLE_CRITERIA_WEIGHTS = {
    "count": 0.3,
    "avg_similarity": 0.4,
    "max_similarity": 0.3
}

PROJECT_CRITERIA_WEIGHTS = {
    "count": 0.3,
    "avg_similarity": 0.4,
    "max_similarity": 0.3
}

# 일관성 검증 임계값
CONSISTENCY_THRESHOLD = 0.1  # CR < 0.1이면 일관성 통과
