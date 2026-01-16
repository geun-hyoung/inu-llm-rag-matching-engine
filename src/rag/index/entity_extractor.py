"""
엔티티/관계 추출 모듈
LightRAG 원본 프롬프트를 사용하여 GPT로 직접 추출
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL
from src.rag.prompts import (
    TUPLE_DELIMITER,
    RECORD_DELIMITER,
    COMPLETION_DELIMITER,
    format_entity_extraction_prompt,
)


@dataclass
class Entity:
    """추출된 엔티티"""
    name: str
    entity_type: str
    description: str
    source_doc_id: str


@dataclass
class Relation:
    """추출된 관계"""
    source_entity: str
    target_entity: str
    keywords: str  # LightRAG 원본: 관계를 설명하는 키워드들
    description: str
    source_doc_id: str


class EntityRelationExtractor:
    """LightRAG 스타일 엔티티/관계 추출기"""

    def __init__(self):
        """추출기 초기화"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

        print(f"EntityRelationExtractor initialized with model: {self.model}")

    def _build_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return format_entity_extraction_prompt(text)

    def _call_llm(self, prompt: str) -> str:
        """GPT API 호출"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call error: {e}")
            return ""

    def _parse_response(
        self,
        response: str,
        doc_id: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        LLM 응답 파싱

        Args:
            response: LLM 응답 텍스트
            doc_id: 문서 ID

        Returns:
            (엔티티 리스트, 관계 리스트)
        """
        entities = []
        relations = []

        # 레코드별로 분리
        records = response.split(RECORD_DELIMITER)

        for record in records:
            record = record.strip()
            if not record or COMPLETION_DELIMITER in record:
                continue

            # 괄호 안의 내용 추출
            match = re.search(r'\("([^"]+)"' + re.escape(TUPLE_DELIMITER) + r'(.+)\)', record)
            if not match:
                continue

            record_type = match.group(1).lower()
            fields_str = match.group(2)

            # 필드 분리 (TUPLE_DELIMITER로 단순 분리)
            fields = [f.strip().strip('"') for f in fields_str.split(TUPLE_DELIMITER)]

            try:
                if record_type == "entity" and len(fields) >= 3:
                    entity = Entity(
                        name=fields[0].strip().upper(),
                        entity_type=fields[1].strip().upper(),
                        description=fields[2].strip() if len(fields) > 2 else "",
                        source_doc_id=doc_id
                    )
                    entities.append(entity)

                elif record_type == "relationship" and len(fields) >= 4:
                    relation = Relation(
                        source_entity=fields[0].strip().upper(),
                        target_entity=fields[1].strip().upper(),
                        description=fields[2].strip() if len(fields) > 2 else "",
                        keywords=fields[3].strip() if len(fields) > 3 else "",
                        source_doc_id=doc_id
                    )
                    relations.append(relation)

            except Exception as e:
                print(f"Parse error for record: {e}")
                continue

        return entities, relations

    def extract_from_document(
        self,
        doc_id: str,
        text: str,
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        단일 문서에서 엔티티와 관계 추출

        Args:
            doc_id: 문서 ID
            text: 문서 텍스트
            doc_type: 문서 타입 (patent/article/project)

        Returns:
            (엔티티 리스트, 관계 리스트)
        """
        # 텍스트가 너무 길면 자르기
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        # 프롬프트 생성 및 LLM 호출
        prompt = self._build_prompt(text)
        response = self._call_llm(prompt)

        if not response:
            return [], []

        # 응답 파싱
        entities, relations = self._parse_response(response, doc_id)

        return entities, relations

    def extract_batch(
        self,
        documents: List[Dict],
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        여러 문서에서 배치로 엔티티/관계 추출

        Args:
            documents: 문서 리스트 (doc_id, text 포함)
            doc_type: 문서 타입

        Returns:
            (전체 엔티티 리스트, 전체 관계 리스트)
        """
        all_entities = []
        all_relations = []

        for idx, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc_{idx}")
            text = doc.get("text", "")

            if not text:
                continue

            try:
                entities, relations = self.extract_from_document(
                    doc_id=doc_id,
                    text=text,
                    doc_type=doc_type
                )
                all_entities.extend(entities)
                all_relations.extend(relations)

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(documents)} documents")

            except Exception as e:
                print(f"Error processing doc {doc_id}: {e}")
                continue

        print(f"Total extracted: {len(all_entities)} entities, {len(all_relations)} relations")
        return all_entities, all_relations


if __name__ == "__main__":
    # 테스트
    print("Testing EntityRelationExtractor...")

    extractor = EntityRelationExtractor()

    # 테스트 텍스트 (특허 스타일)
    test_text = """
    본 발명은 딥러닝 기반 의료영상 분석 시스템에 관한 것이다.
    특히 CT 및 MRI 영상에서 암 병변을 자동으로 검출하는 CNN 알고리즘을 제안한다.
    제안된 방법은 ResNet 아키텍처를 기반으로 하며, 전이학습을 통해
    적은 양의 학습 데이터로도 높은 정확도를 달성한다.
    인천대학교 의공학과에서 개발한 이 기술은 실제 병원 환경에서
    방사선과 전문의의 진단을 보조하는데 활용될 수 있다.
    """

    entities, relations = extractor.extract_from_document(
        doc_id="test_patent_001",
        text=test_text,
        doc_type="patent"
    )

    print(f"\n추출된 엔티티 ({len(entities)}개):")
    for e in entities:
        print(f"  - {e.name} ({e.entity_type}): {e.description[:50]}...")

    print(f"\n추출된 관계 ({len(relations)}개):")
    for r in relations:
        print(f"  - {r.source_entity} --[{r.keywords}]--> {r.target_entity}")

    print("\nTest completed!")