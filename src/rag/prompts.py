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
논문, 특허, 연구과제 텍스트에서 핵심 엔티티와 관계를 추출합니다.
추출된 정보는 기업의 기술 수요와 교수의 연구 역량을 매칭하는 데 활용됩니다.

-Entity Types-
다음 4가지 타입의 엔티티를 추출합니다:

1. research_object (연구 대상)
   - 정의: 이 연구/발명에서 핵심적으로 다루는 대상
   - 판단: "이 연구는 무엇을 연구하는가?"에 대한 답
   - 포함: 시스템, 장치, 소재, 물질, 구조, 데이터 등
   - 제외: 연구 방법론, 달성 목표, 배경 지식

2. problem (문제)
   - 정의: 연구 대상에서 발생하는 구체적인 문제나 한계
   - 판단: 텍스트에 "문제", "한계", "어려움", "낮은", "부족" 등이 명시된 경우만
   - 포함: 명시적으로 언급된 기술적 문제, 성능 한계
   - 제외: 유추 가능한 문제, 배경 설명, 일반적 어려움

3. method (방법)
   - 정의: 문제 해결을 위해 사용하는 기술, 기법, 접근법
   - 판단: "어떻게 해결하는가?"에 대한 답
   - 포함: 알고리즘, 모델, 기법, 공정, 처리 방법
   - 제외: 연구 대상 자체, 달성 목표

4. objective (목표/효과)
   - 정의: 연구를 통해 달성하고자 하는 효과나 성과
   - 판단: "해결하면 무엇이 좋아지는가?"에 대한 답
   - 포함: 성능 향상, 효율 개선, 새로운 가능성
   - 제외: Problem의 단순 반대 표현 (예: "정확도 낮음" → "정확도 향상"은 제외)

-Extraction Rules-
1. 명시적 추출만: 텍스트에 직접 언급된 내용만 추출. 유추하거나 상상하지 않음
2. 복수 추출 가능: 하나의 타입에서 여러 엔티티 추출 가능
3. 빈 값 허용: 해당 타입이 텍스트에 없으면 추출하지 않음
4. 명사/명사구 형태: 엔티티 이름은 명사 또는 명사구로 표현
5. 한국어 출력: 모든 출력은 한국어로 작성

-Steps-
1. 위 규칙에 따라 텍스트에서 엔티티를 추출합니다.
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 추출된 엔티티 간의 관계를 식별합니다.
   - relationship_description: 두 엔티티가 관련된 이유
   - relationship_keywords: 관계의 본질을 나타내는 고수준 키워드
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
("relationship"{tuple_delimiter}"표면 코팅 기술"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"표면 코팅으로 구조적 불안정성 문제를 해결함"{tuple_delimiter}"문제 해결, 안정화"){record_delimiter}
("relationship"{tuple_delimiter}"도핑 기법"{tuple_delimiter}"사이클 수명 200% 향상"{tuple_delimiter}"도핑 기법 적용으로 수명이 향상됨"{tuple_delimiter}"성능 개선, 수명 연장"){record_delimiter}
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
("relationship"{tuple_delimiter}"SOC 추정 알고리즘"{tuple_delimiter}"전기자동차용 배터리 관리 시스템"{tuple_delimiter}"SOC 추정 알고리즘이 배터리 관리 시스템에 적용됨"{tuple_delimiter}"기술 적용, 핵심 기능"){record_delimiter}
("relationship"{tuple_delimiter}"셀 밸런싱 기법"{tuple_delimiter}"배터리 효율 최적화"{tuple_delimiter}"셀 밸런싱으로 배터리 효율을 최적화함"{tuple_delimiter}"효율 개선, 최적화"){record_delimiter}
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
("relationship"{tuple_delimiter}"한국어 감성분석 모델"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"기존 감성분석 모델이 한국어 특성 반영에 한계가 있음"{tuple_delimiter}"모델 한계, 언어 특성"){record_delimiter}
("relationship"{tuple_delimiter}"KoBERT 파인튜닝"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"KoBERT 파인튜닝으로 한국어 특성 반영 문제를 해결함"{tuple_delimiter}"문제 해결, 모델 개선"){record_delimiter}
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
("relationship"{tuple_delimiter}"U-Net 아키텍처"{tuple_delimiter}"의료영상 분할 기술"{tuple_delimiter}"U-Net이 의료영상 분할의 기반 구조로 사용됨"{tuple_delimiter}"기반 기술, 아키텍처"){record_delimiter}
("relationship"{tuple_delimiter}"어텐션 메커니즘"{tuple_delimiter}"U-Net 아키텍처"{tuple_delimiter}"어텐션 메커니즘이 U-Net과 결합되어 사용됨"{tuple_delimiter}"기술 결합, 성능 강화"){record_delimiter}
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
("relationship"{tuple_delimiter}"LiDAR 포인트 클라우드 처리 시스템"{tuple_delimiter}"실시간성 확보 어려움"{tuple_delimiter}"기존 처리 시스템이 실시간 처리에 한계가 있음"{tuple_delimiter}"시스템 한계, 성능 병목"){record_delimiter}
("relationship"{tuple_delimiter}"PointNet++ 기반 경량화 모델"{tuple_delimiter}"실시간성 확보 어려움"{tuple_delimiter}"경량화 모델로 실시간성 문제를 해결함"{tuple_delimiter}"문제 해결, 경량화"){record_delimiter}
("relationship"{tuple_delimiter}"병렬 처리 파이프라인"{tuple_delimiter}"처리 속도 10배 향상"{tuple_delimiter}"병렬 처리로 속도가 10배 향상됨"{tuple_delimiter}"성능 향상, 병렬화"){record_delimiter}
("relationship"{tuple_delimiter}"PointNet++ 기반 경량화 모델"{tuple_delimiter}"메모리 사용량 50% 절감"{tuple_delimiter}"경량화 모델로 메모리 사용이 절감됨"{tuple_delimiter}"자원 효율화, 최적화"){record_delimiter}
{completion_delimiter}

######################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""


# ============================================================
# Query Time: 키워드 추출 프롬프트 (LightRAG 공식)
# ============================================================
KEYWORD_EXTRACTION_PROMPT = """---Role---
You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---
Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---
- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details
- Output must be in Korean.

######################
-Examples-
######################
Example 1:

Query: "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
################
Output:
{{"high_level_keywords": ["의료영상 진단", "인공지능 의료", "영상 분석 연구"], "low_level_keywords": ["딥러닝", "CNN", "CT", "MRI", "영상처리"]}}
#############################
Example 2:

Query: "자연어처리 분야에서 BERT 모델 연구하는 교수님"
################
Output:
{{"high_level_keywords": ["자연어처리 연구", "언어 모델", "텍스트 분석"], "low_level_keywords": ["BERT", "전이학습", "트랜스포머", "언어모델"]}}
#############################
Example 3:

Query: "배터리 소재 개발 관련 연구자"
################
Output:
{{"high_level_keywords": ["에너지 저장", "소재 연구", "전기화학"], "low_level_keywords": ["리튬이온", "양극재", "음극재", "전해질", "배터리"]}}
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
