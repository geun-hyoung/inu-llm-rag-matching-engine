"""
LightRAG 프롬프트 통합 관리 모듈
- Index Time: 엔티티/관계 추출
- Query Time: 키워드 추출
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
# 특허/논문 도메인에 맞게 커스터마이징
DEFAULT_ENTITY_TYPES = [
    "organization",  # 기관, 회사, 대학
    "person",        # 인물, 연구자
    "technology",    # 기술, 알고리즘
    "method",        # 방법론, 프로세스
    "product",       # 제품, 시스템
    "material",      # 소재, 재료
    "concept",       # 개념, 이론
    "event",         # 이벤트, 행사
]


# ============================================================
# Index Time: 엔티티/관계 추출 프롬프트
# ============================================================
ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>)

3. Return output in Korean. All entity names, descriptions, and relationship descriptions must be in Korean.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, organization, location]
Text:
본 발명은 딥러닝 기반 의료영상 분석 시스템에 관한 것이다. 인천대학교 의공학과에서 개발한 CNN 알고리즘은 CT 영상에서 암 병변을 자동으로 검출한다.
################
Output:
("entity"{tuple_delimiter}"딥러닝"{tuple_delimiter}"technology"{tuple_delimiter}"심층 신경망 기반의 기계학습 기술로, 의료영상 분석에 활용됨"){record_delimiter}
("entity"{tuple_delimiter}"의료영상 분석 시스템"{tuple_delimiter}"product"{tuple_delimiter}"의료 영상을 분석하여 진단을 보조하는 시스템"){record_delimiter}
("entity"{tuple_delimiter}"인천대학교"{tuple_delimiter}"organization"{tuple_delimiter}"의공학과에서 CNN 알고리즘을 개발한 교육 기관"){record_delimiter}
("entity"{tuple_delimiter}"CNN 알고리즘"{tuple_delimiter}"method"{tuple_delimiter}"합성곱 신경망 기반 알고리즘으로 암 병변 검출에 사용됨"){record_delimiter}
("entity"{tuple_delimiter}"CT 영상"{tuple_delimiter}"concept"{tuple_delimiter}"컴퓨터 단층촬영으로 얻은 의료 영상 데이터"){record_delimiter}
("relationship"{tuple_delimiter}"딥러닝"{tuple_delimiter}"의료영상 분석 시스템"{tuple_delimiter}"딥러닝 기술이 의료영상 분석 시스템의 핵심 기술로 활용됨"{tuple_delimiter}"기술 적용, 핵심 기술"){record_delimiter}
("relationship"{tuple_delimiter}"인천대학교"{tuple_delimiter}"CNN 알고리즘"{tuple_delimiter}"인천대학교 의공학과에서 CNN 알고리즘을 개발함"{tuple_delimiter}"개발, 연구"){record_delimiter}
("relationship"{tuple_delimiter}"CNN 알고리즘"{tuple_delimiter}"CT 영상"{tuple_delimiter}"CNN 알고리즘이 CT 영상에서 암 병변을 검출하는데 사용됨"{tuple_delimiter}"영상 분석, 병변 검출"){record_delimiter}
{completion_delimiter}

######################
Example 2:

Entity_types: [person, technology, organization, concept]
Text:
김철수 교수는 전이학습 기법을 활용한 자연어처리 연구를 수행하고 있다. BERT 모델을 기반으로 한국어 감성분석 시스템을 개발하였다.
################
Output:
("entity"{tuple_delimiter}"김철수 교수"{tuple_delimiter}"person"{tuple_delimiter}"전이학습 기법을 활용한 자연어처리 연구를 수행하는 연구자"){record_delimiter}
("entity"{tuple_delimiter}"전이학습"{tuple_delimiter}"method"{tuple_delimiter}"사전 학습된 모델을 새로운 태스크에 적용하는 기계학습 기법"){record_delimiter}
("entity"{tuple_delimiter}"자연어처리"{tuple_delimiter}"technology"{tuple_delimiter}"인간의 언어를 컴퓨터가 이해하고 처리하는 기술 분야"){record_delimiter}
("entity"{tuple_delimiter}"BERT 모델"{tuple_delimiter}"method"{tuple_delimiter}"Google에서 개발한 사전학습 언어 모델"){record_delimiter}
("entity"{tuple_delimiter}"한국어 감성분석 시스템"{tuple_delimiter}"product"{tuple_delimiter}"한국어 텍스트의 감성을 분석하는 시스템"){record_delimiter}
("relationship"{tuple_delimiter}"김철수 교수"{tuple_delimiter}"전이학습"{tuple_delimiter}"김철수 교수가 전이학습 기법을 연구에 활용함"{tuple_delimiter}"연구 방법론, 활용"){record_delimiter}
("relationship"{tuple_delimiter}"전이학습"{tuple_delimiter}"자연어처리"{tuple_delimiter}"전이학습이 자연어처리 연구에 적용됨"{tuple_delimiter}"기술 적용"){record_delimiter}
("relationship"{tuple_delimiter}"BERT 모델"{tuple_delimiter}"한국어 감성분석 시스템"{tuple_delimiter}"BERT 모델을 기반으로 한국어 감성분석 시스템이 개발됨"{tuple_delimiter}"기반 기술, 개발"){record_delimiter}
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: [{entity_types}]
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
def format_entity_extraction_prompt(
    input_text: str,
    entity_types: list = None
) -> str:
    """
    엔티티 추출 프롬프트 포맷팅

    Args:
        input_text: 입력 텍스트
        entity_types: 엔티티 타입 리스트 (기본값: DEFAULT_ENTITY_TYPES)

    Returns:
        포맷팅된 프롬프트
    """
    types = entity_types or DEFAULT_ENTITY_TYPES
    return ENTITY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(types),
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
    test_text = "본 발명은 딥러닝 기반 의료영상 분석 시스템에 관한 것이다."
    prompt = format_entity_extraction_prompt(test_text)
    print(prompt[:500] + "...")

    print("\n=== Keyword Extraction Prompt Test ===")
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    prompt = format_keyword_extraction_prompt(test_query)
    print(prompt)
