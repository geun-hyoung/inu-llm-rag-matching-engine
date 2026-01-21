"""
LightRAG 프롬프트 통합 관리 모듈
- Index Time: 엔티티/관계 추출
- Query Time: 키워드 추출

산학협력 매칭 시스템용 커스텀 프롬프트
- 논문, 특허, R&D 과제에서 일관된 Entity 추출
- Entity Types: ResearchObject, Problem, Method, Objective
"""

# ============================================================
# 구분자 설정 (LightRAG 공식)
# ============================================================
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"


# ============================================================
# 엔티티 타입 설정
# ============================================================
# 산학협력 매칭에 최적화된 Entity 타입
DEFAULT_ENTITY_TYPES = [
    "research_object",  # 연구/발명의 핵심 대상
    "problem",          # 해결하고자 하는 구체적 문제
    "method",           # 문제 해결을 위한 기술/기법
    "objective",        # 달성하고자 하는 효과/성과
]


# ============================================================
# Index Time: 엔티티/관계 추출 프롬프트 (산학협력 매칭용)
# ============================================================
ENTITY_EXTRACTION_PROMPT = """
-Goal-
산학협력 매칭 시스템을 위한 엔티티/관계 추출입니다.
- 입력: 논문, 특허, R&D 과제 텍스트
- 목적: 기업 기술 수요 ↔ 교수 연구 역량 매칭
- 핵심: 검색 가능한 구체적 기술 용어 추출 (일반적 용어 ✗)

-Entity Types-
| 타입 | 정의 | 핵심 질문 | 포함 예시 | 제외 |
|------|------|----------|----------|------|
| target (연구 대상) | 연구가 다루는 핵심 물체, 시스템, 현상 | 무엇을 연구하는가? | 센서, 배터리, 시스템, 모델, 데이터 | 행위(→method), 효과(→objective) |
| problem (문제) | 연구 대상의 기존 한계나 부정적 현상 | 어떤 문제를 해결하려는가? | "~한계", "~어렵다", "~부족하다" | 배경설명, 일반론 |
| method (방법) | 문제 해결을 위한 구체적 기술, 알고리즘, 절차 | 어떻게 해결하는가? | PointNet++, KoBERT, 표면 코팅, 셀 밸런싱 | 단독 행위(분석, 처리), 상위개념(CNN있으면→딥러닝 제외) |
| objective (목표/효과) | 제안된 방법으로 달성한 정량적/정성적 성과 | 무엇을 달성했는가? | "수명 200% 향상", "오차 0.11% 달성" | 맥락 없는 수치("20%"), 일반론("성능 향상") |

-Rules-
1. 명시적 추출만: 텍스트에 직접 언급된 내용만, 한국어, 명사/명사구 형태
2. 열거형 분리: 쉼표(,)로 나열된 독립적 항목 → 개별 엔티티로 분리 (단, 하나의 개념: "수도 및 하수도", "연구 및 개발"은 유지)
3. 구체적 표현 우선: 더 구체적인 표현이 있으면 상위 개념 제외
4. 일반 용어 금지:
   - 엔티티: "데이터 분석", "성능 향상", "최적화", "효율 개선"
   - 관계 키워드: "기술 적용", "시스템 개선" → 도메인 특화 용어로 작성
5. 맥락 없는 수치 금지: ✗ "20%", "0.11%" → ✓ "수명 200% 향상"

-Steps-
1. 엔티티 추출
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 관계 추출
   - relationship_description: 두 엔티티가 관련된 이유
   - relationship_keywords: 도메인 특화 키워드 2-3개 (예: ✗ "기술 적용" → ✓ "SOC 알고리즘 기반 충전 상태 추정")
   Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>)

3. 완료 시 {completion_delimiter} 출력

######################
-Examples-
######################
Example 1: 모든 엔티티 타입이 추출되는 케이스

Text:
본 연구에서는 리튬이온 배터리의 양극재 열화 현상을 분석하였다. 기존 양극재는 고온에서 구조적 불안정성 문제가 있었다. 이를 해결하기 위해 표면 코팅 기술과 도핑 기법을 적용하였으며, 이를 통해 사이클 수명이 200% 향상되었다.
################
Output:
("entity"{tuple_delimiter}"리튬이온 배터리 양극재"{tuple_delimiter}"research_object"{tuple_delimiter}"본 연구에서 핵심적으로 다루는 배터리 소재"){record_delimiter}
("entity"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"problem"{tuple_delimiter}"기존 양극재가 고온에서 겪는 구조적 문제"){record_delimiter}
("entity"{tuple_delimiter}"표면 코팅 기술"{tuple_delimiter}"method"{tuple_delimiter}"양극재 안정성 향상을 위한 표면 처리 기법"){record_delimiter}
("entity"{tuple_delimiter}"도핑 기법"{tuple_delimiter}"method"{tuple_delimiter}"양극재 특성 개선을 위한 원소 첨가 방법"){record_delimiter}
("entity"{tuple_delimiter}"사이클 수명 200% 향상"{tuple_delimiter}"objective"{tuple_delimiter}"연구를 통해 달성한 배터리 수명 개선 효과"){record_delimiter}
("relationship"{tuple_delimiter}"리튬이온 배터리 양극재"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"양극재가 고온에서 구조적 불안정성 문제를 가짐"{tuple_delimiter}"소재 한계, 열화"){record_delimiter}
("relationship"{tuple_delimiter}"표면 코팅 기술"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"표면 코팅으로 구조적 불안정성 문제를 해결함"{tuple_delimiter}"표면 코팅 기반 안정화, 양극재 열화 방지"){record_delimiter}
("relationship"{tuple_delimiter}"도핑 기법"{tuple_delimiter}"사이클 수명 200% 향상"{tuple_delimiter}"도핑 기법 적용으로 수명이 향상됨"{tuple_delimiter}"도핑 기반 특성 개선, 사이클 수명 연장"){record_delimiter}
{completion_delimiter}

######################
Example 2: Problem이 명시되지 않은 케이스

Text:
본 발명은 전기자동차용 배터리 관리 시스템에 관한 것이다. SOC 추정 알고리즘과 셀 밸런싱 기법을 적용하여 배터리 효율을 최적화한다.
################
Output:
("entity"{tuple_delimiter}"전기자동차용 배터리 관리 시스템"{tuple_delimiter}"research_object"{tuple_delimiter}"본 발명에서 다루는 배터리 관리 장치"){record_delimiter}
("entity"{tuple_delimiter}"SOC 추정 알고리즘"{tuple_delimiter}"method"{tuple_delimiter}"배터리 충전 상태를 추정하는 알고리즘"){record_delimiter}
("entity"{tuple_delimiter}"셀 밸런싱 기법"{tuple_delimiter}"method"{tuple_delimiter}"배터리 셀 간 균형을 맞추는 기법"){record_delimiter}
("entity"{tuple_delimiter}"배터리 효율 최적화"{tuple_delimiter}"objective"{tuple_delimiter}"시스템을 통해 달성하고자 하는 효율 개선"){record_delimiter}
("relationship"{tuple_delimiter}"SOC 추정 알고리즘"{tuple_delimiter}"전기자동차용 배터리 관리 시스템"{tuple_delimiter}"SOC 추정 알고리즘이 배터리 관리 시스템에 적용됨"{tuple_delimiter}"배터리 충전량 추정, BMS 핵심 로직"){record_delimiter}
("relationship"{tuple_delimiter}"셀 밸런싱 기법"{tuple_delimiter}"배터리 효율 최적화"{tuple_delimiter}"셀 밸런싱으로 배터리 효율을 최적화함"{tuple_delimiter}"셀 간 전압 균등화, 배터리 수명 연장"){record_delimiter}
{completion_delimiter}

######################
Example 3: Objective가 명시되지 않은 케이스

Text:
BERT 기반 한국어 감성분석 모델을 개발하였다. 기존 모델은 한국어 특성 반영이 부족한 한계가 있었다. KoBERT를 파인튜닝하고 형태소 분석을 전처리에 추가하였다.
################
Output:
("entity"{tuple_delimiter}"한국어 감성분석 모델"{tuple_delimiter}"research_object"{tuple_delimiter}"본 연구에서 개발하는 감성분석 시스템"){record_delimiter}
("entity"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"problem"{tuple_delimiter}"기존 모델이 가진 한국어 처리의 한계"){record_delimiter}
("entity"{tuple_delimiter}"KoBERT 파인튜닝"{tuple_delimiter}"method"{tuple_delimiter}"한국어 사전학습 모델을 미세조정하는 기법"){record_delimiter}
("entity"{tuple_delimiter}"형태소 분석 전처리"{tuple_delimiter}"method"{tuple_delimiter}"한국어 특성을 반영한 텍스트 전처리 방법"){record_delimiter}
("relationship"{tuple_delimiter}"한국어 감성분석 모델"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"기존 감성분석 모델이 한국어 특성 반영에 한계가 있음"{tuple_delimiter}"감성분석 모델 한계, 한국어 언어 특성"){record_delimiter}
("relationship"{tuple_delimiter}"KoBERT 파인튜닝"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"KoBERT 파인튜닝으로 한국어 특성 반영 문제를 해결함"{tuple_delimiter}"KoBERT 기반 파인튜닝, 한국어 특화 모델 개선"){record_delimiter}
{completion_delimiter}

######################
Example 4: Problem과 Objective 둘 다 없는 케이스

Text:
본 연구는 딥러닝 기반 의료영상 분할 기술을 개발한다. U-Net 아키텍처와 어텐션 메커니즘을 결합하여 CT 영상에서 장기를 자동 분할한다.
################
Output:
("entity"{tuple_delimiter}"의료영상 분할 기술"{tuple_delimiter}"research_object"{tuple_delimiter}"본 연구에서 개발하는 영상 처리 기술"){record_delimiter}
("entity"{tuple_delimiter}"CT 영상"{tuple_delimiter}"research_object"{tuple_delimiter}"분할 대상이 되는 의료 영상 데이터"){record_delimiter}
("entity"{tuple_delimiter}"U-Net 아키텍처"{tuple_delimiter}"method"{tuple_delimiter}"의료영상 분할에 사용되는 딥러닝 구조"){record_delimiter}
("entity"{tuple_delimiter}"어텐션 메커니즘"{tuple_delimiter}"method"{tuple_delimiter}"중요 영역에 집중하는 딥러닝 기법"){record_delimiter}
("relationship"{tuple_delimiter}"U-Net 아키텍처"{tuple_delimiter}"의료영상 분할 기술"{tuple_delimiter}"U-Net이 의료영상 분할의 기반 구조로 사용됨"{tuple_delimiter}"U-Net 기반 영상 분할, 딥러닝 아키텍처"){record_delimiter}
("relationship"{tuple_delimiter}"어텐션 메커니즘"{tuple_delimiter}"U-Net 아키텍처"{tuple_delimiter}"어텐션 메커니즘이 U-Net과 결합되어 사용됨"{tuple_delimiter}"어텐션 메커니즘 결합, U-Net 성능 강화"){record_delimiter}
{completion_delimiter}

######################
Example 5: 복잡한 관계가 있는 케이스

Text:
자율주행 차량의 LiDAR 포인트 클라우드 처리 시스템을 연구하였다. 기존 처리 방식은 실시간성 확보가 어려운 문제가 있었다. PointNet++ 기반 경량화 모델과 병렬 처리 파이프라인을 개발하여 처리 속도를 10배 향상시키고 메모리 사용량을 50% 절감하였다.
################
Output:
("entity"{tuple_delimiter}"LiDAR 포인트 클라우드 처리 시스템"{tuple_delimiter}"research_object"{tuple_delimiter}"자율주행 차량용 3D 센서 데이터 처리 시스템"){record_delimiter}
("entity"{tuple_delimiter}"실시간성 확보 어려움"{tuple_delimiter}"problem"{tuple_delimiter}"기존 처리 방식의 속도 관련 한계"){record_delimiter}
("entity"{tuple_delimiter}"PointNet++ 기반 경량화 모델"{tuple_delimiter}"method"{tuple_delimiter}"포인트 클라우드 처리를 위한 경량 딥러닝 모델"){record_delimiter}
("entity"{tuple_delimiter}"병렬 처리 파이프라인"{tuple_delimiter}"method"{tuple_delimiter}"처리 속도 향상을 위한 병렬화 구조"){record_delimiter}
("entity"{tuple_delimiter}"처리 속도 10배 향상"{tuple_delimiter}"objective"{tuple_delimiter}"연구를 통해 달성한 속도 개선 효과"){record_delimiter}
("entity"{tuple_delimiter}"메모리 사용량 50% 절감"{tuple_delimiter}"objective"{tuple_delimiter}"연구를 통해 달성한 자원 효율화 효과"){record_delimiter}
("relationship"{tuple_delimiter}"LiDAR 포인트 클라우드 처리 시스템"{tuple_delimiter}"실시간성 확보 어려움"{tuple_delimiter}"기존 처리 시스템이 실시간 처리에 한계가 있음"{tuple_delimiter}"LiDAR 데이터 처리 한계, 실시간성 병목"){record_delimiter}
("relationship"{tuple_delimiter}"PointNet++ 기반 경량화 모델"{tuple_delimiter}"실시간성 확보 어려움"{tuple_delimiter}"경량화 모델로 실시간성 문제를 해결함"{tuple_delimiter}"포인트 클라우드 경량화, 실시간 LiDAR 처리"){record_delimiter}
("relationship"{tuple_delimiter}"병렬 처리 파이프라인"{tuple_delimiter}"처리 속도 10배 향상"{tuple_delimiter}"병렬 처리로 속도가 10배 향상됨"{tuple_delimiter}"병렬 연산 파이프라인, LiDAR 데이터 고속 처리"){record_delimiter}
("relationship"{tuple_delimiter}"PointNet++ 기반 경량화 모델"{tuple_delimiter}"메모리 사용량 50% 절감"{tuple_delimiter}"경량화 모델로 메모리 사용이 절감됨"{tuple_delimiter}"PointNet++ 모델 경량화, 메모리 효율 개선"){record_delimiter}
{completion_delimiter}

######################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""


# ============================================================
# Query Time: 키워드 추출 프롬프트 (산학협력 매칭용)
# ============================================================
KEYWORD_EXTRACTION_PROMPT = """---Goal---
산학협력 매칭을 위한 검색 키워드 추출입니다.
- 목적: 기업 기술 수요 ↔ 교수 연구 역량 매칭
- 추출된 키워드로 논문/특허/R&D 과제를 검색합니다.

---Keywords---
| 타입 | 정의 | 검색 대상 | 형태 | 예시 |
|------|------|----------|------|------|
| low_level | 구체적 엔티티, 고유명사, 기술 용어 | Entity (특정 노드) | 명사/명사구 | "딥러닝", "CNN", "양극재" |
| high_level | 포괄적 개념, 테마, 연구 주제 영역 | Relation (주제로 연결된 관계들) | 테마/주제 표현 | "의료영상 진단", "배터리 수명 향상" |

---Rules---
1. 명시적 추출만: 쿼리에 직접 언급된 키워드만 (추론/연상 금지)
2. 복합 명사 유지: 의미적으로 연결된 명사는 하나로 ("하천 지형", "의료영상")
3. 제외 대상:
   - 일반 명사: "교수님", "전문가", "연구", "분야", "기술"
   - 추상적 단어 (단독 사용 시): "영향", "효과", "관계", "분석"
   - 질문 표현: "찾아줘", "알려줘", "있어?"
4. high_level: 연구 주제/테마 단위로 추출 (개별 명사 ✗ → 주제 표현 ✓)

---Output---
JSON 형식, 한국어, low_level 1-5개, high_level 1-2개

######################
-Examples-
######################
Example 1:

Query: "공장에서 용접 불량이 자꾸 발생하는데 자동으로 검출할 수 있는 방법 없을까요?"
################
Output:
{{"low_level_keywords": ["용접 불량", "자동 검출"], "high_level_keywords": ["용접 불량 자동 검출"]}}
#############################
Example 2:

Query: "전기차 배터리 충전 시간이 너무 오래 걸려서 단축하고 싶은데 관련 전문가 있나요?"
################
Output:
{{"low_level_keywords": ["전기차 배터리", "충전 시간"], "high_level_keywords": ["전기차 배터리 충전 시간 단축"]}}
#############################
Example 3:

Query: "물류창고 재고 예측이 잘 안 맞아서 수요 예측 정확도를 높이고 싶습니다"
################
Output:
{{"low_level_keywords": ["물류창고", "재고 예측", "수요 예측"], "high_level_keywords": ["물류 수요 예측 정확도 향상"]}}
#############################
Example 4:

Query: "스마트팜에 IoT 센서 적용해서 작물 생육 모니터링 하려는데 도와줄 교수님 계신가요?"
################
Output:
{{"low_level_keywords": ["스마트팜", "IoT 센서", "작물 생육 모니터링"], "high_level_keywords": ["IoT 기반 스마트팜 생육 모니터링"]}}
#############################
Example 5:

Query: "건설현장에서 작업자 안전사고를 미리 예측하고 예방할 수 있는 AI 기술 찾고 있어요"
################
Output:
{{"low_level_keywords": ["건설현장", "안전사고", "AI"], "high_level_keywords": ["AI 기반 건설현장 안전사고 예측"]}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:
"""


# ============================================================
# Query Time: RAG 응답 생성 프롬프트 (추후 사용)
# ============================================================
RAG_RESPONSE_PROMPT = """---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---
{response_type}

---Data tables---
{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


# ============================================================
# 유틸리티 함수
# ============================================================
def format_entity_extraction_prompt(input_text: str) -> str:
    """
    엔티티 추출 프롬프트 포맷팅

    Args:
        input_text: 입력 텍스트 (논문 초록, 특허 요약, R&D 과제 설명)

    Returns:
        포맷팅된 프롬프트
    """
    return ENTITY_EXTRACTION_PROMPT.format(
        tuple_delimiter=TUPLE_DELIMITER,
        record_delimiter=RECORD_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        input_text=input_text
    )


def format_keyword_extraction_prompt(query: str) -> str:
    """
    키워드 추출 프롬프트 포맷팅

    Args:
        query: 사용자 쿼리

    Returns:
        포맷팅된 프롬프트
    """
    return KEYWORD_EXTRACTION_PROMPT.format(query=query)


if __name__ == "__main__":
    # 테스트
    print("=== Entity Extraction Prompt Test ===")
    test_text = """본 연구에서는 리튬이온 배터리의 양극재 열화 현상을 분석하였다.
    기존 양극재는 고온에서 구조적 불안정성 문제가 있었다.
    이를 해결하기 위해 표면 코팅 기술과 도핑 기법을 적용하였으며,
    이를 통해 사이클 수명이 200% 향상되었다."""
    prompt = format_entity_extraction_prompt(test_text)
    print(prompt)
    print("\n" + "="*50)

    print("\n=== Keyword Extraction Prompt Test ===")
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    prompt = format_keyword_extraction_prompt(test_query)
    print(prompt)
