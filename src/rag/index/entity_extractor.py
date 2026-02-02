"""
엔티티/관계 추출 모듈
LightRAG 원본 프롬프트를 사용하여 GPT로 직접 추출

동기/비동기 버전 모두 제공:
- EntityRelationExtractor: 동기 버전 (기존)
- AsyncEntityRelationExtractor: 비동기 버전 (asyncio 기반, 병렬 처리)
"""

import sys
import re
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI, AsyncOpenAI

# 로깅 설정
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL
from src.utils.cost_tracker import log_chat_usage
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

    def _call_llm(self, prompt: str, doc_id: str = "", max_retries: int = 3) -> str:
        """GPT API 호출 (재시도 로직 포함)"""
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4096
                )

                # 비용 추적
                log_chat_usage(
                    component="entity_extraction",
                    model=self.model,
                    response=response
                )

                # 토큰 사용량 로깅
                usage = response.usage
                logger.debug(f"[{doc_id}] API OK - tokens: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}")

                return response.choices[0].message.content

            except Exception as e:
                wait_time = (attempt + 1) * 5  # 5초, 10초, 15초
                logger.warning(f"[{doc_id}] API ERROR (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")

                if attempt < max_retries - 1:
                    logger.info(f"[{doc_id}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[{doc_id}] All retries failed")
                    return ""

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
        response = self._call_llm(prompt, doc_id=doc_id)

        if not response:
            logger.warning(f"[{doc_id}] Empty response - no entities extracted")
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
                logger.warning(f"[{doc_id}] Empty text - skipped")
                continue

            try:
                entities, relations = self.extract_from_document(
                    doc_id=doc_id,
                    text=text,
                    doc_type=doc_type
                )
                all_entities.extend(entities)
                all_relations.extend(relations)

                # 추출 결과 로깅
                logger.info(f"[{doc_id}] Extracted {len(entities)} entities, {len(relations)} relations")

            except Exception as e:
                logger.error(f"[{doc_id}] Exception: {type(e).__name__}: {e}")
                continue

        logger.info(f"Batch total: {len(all_entities)} entities, {len(all_relations)} relations")
        return all_entities, all_relations


class AsyncEntityRelationExtractor:
    """
    비동기 엔티티/관계 추출기 (asyncio 기반)

    OpenAI API를 병렬로 호출하여 처리 속도를 높임.
    Semaphore를 사용하여 동시 요청 수를 제한 (rate limit 준수).
    체크포인트 기능으로 중단 시 재개 가능.
    """

    def __init__(
        self,
        concurrency: int = 5,
        max_retries: int = 3,
        checkpoint_dir: str = "data/checkpoints",
        checkpoint_interval: int = 20
    ):
        """
        추출기 초기화

        Args:
            concurrency: 동시 요청 수 (기본값: 5, Tier 1 TPM 기준 안전한 값)
            max_retries: API 호출 실패 시 재시도 횟수
            checkpoint_dir: 체크포인트 저장 디렉토리
            checkpoint_interval: 체크포인트 저장 간격 (N개 문서마다)
        """
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.semaphore: Optional[asyncio.Semaphore] = None

        # 체크포인트 설정
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval

        # 통계 추적
        self.stats = {
            "success": 0,
            "failed": 0,
            "retries": 0
        }

        print(f"AsyncEntityRelationExtractor initialized")
        print(f"  Model: {self.model}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Max retries: {max_retries}")
        print(f"  Checkpoint interval: {checkpoint_interval} docs")

    def _get_checkpoint_path(self, doc_type: str) -> Path:
        """체크포인트 파일 경로 반환"""
        return self.checkpoint_dir / f"extraction_{doc_type}_checkpoint.json"

    def _save_checkpoint(
        self,
        doc_type: str,
        processed_doc_ids: List[str],
        failed_doc_ids: List[str],
        entities: List[Entity],
        relations: List[Relation],
        total_docs: int
    ):
        """체크포인트 저장 (단일 파일 덮어쓰기)"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._get_checkpoint_path(doc_type)

        checkpoint_data = {
            "doc_type": doc_type,
            "total_docs": total_docs,
            "processed_count": len(processed_doc_ids),
            "failed_count": len(failed_doc_ids),
            "processed_doc_ids": processed_doc_ids,
            "failed_doc_ids": failed_doc_ids,
            "entities": [asdict(e) if hasattr(e, '__dataclass_fields__') else e for e in entities],
            "relations": [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in relations],
            "last_saved_at": datetime.now().isoformat(),
            "stats": self.stats.copy()
        }

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved: {len(processed_doc_ids)}/{total_docs} docs ({len(processed_doc_ids)/total_docs*100:.1f}%)")

    def load_checkpoint(self, doc_type: str) -> Optional[Dict]:
        """체크포인트 로드 (있으면 반환, 없으면 None)"""
        checkpoint_path = self._get_checkpoint_path(doc_type)

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        logger.info(f"Checkpoint loaded: {checkpoint_data['processed_count']}/{checkpoint_data['total_docs']} docs")
        logger.info(f"  Last saved: {checkpoint_data['last_saved_at']}")
        logger.info(f"  Entities: {len(checkpoint_data['entities'])}")
        logger.info(f"  Relations: {len(checkpoint_data['relations'])}")

        return checkpoint_data

    def clear_checkpoint(self, doc_type: str):
        """체크포인트 삭제"""
        checkpoint_path = self._get_checkpoint_path(doc_type)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint cleared: {checkpoint_path}")

    def _build_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return format_entity_extraction_prompt(text)

    def _parse_response(
        self,
        response: str,
        doc_id: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """LLM 응답 파싱 (동기 버전과 동일)"""
        entities = []
        relations = []

        records = response.split(RECORD_DELIMITER)

        for record in records:
            record = record.strip()
            if not record or COMPLETION_DELIMITER in record:
                continue

            match = re.search(r'\("([^"]+)"' + re.escape(TUPLE_DELIMITER) + r'(.+)\)', record)
            if not match:
                continue

            record_type = match.group(1).lower()
            fields_str = match.group(2)
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
                logger.debug(f"[{doc_id}] Parse error: {e}")
                continue

        return entities, relations

    async def _call_llm_async(self, prompt: str, doc_id: str = "") -> str:
        """
        비동기 GPT API 호출 (지수 백오프 재시도)
        """
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=4096
                    )

                # 비용 추적
                log_chat_usage(
                    component="entity_extraction_async",
                    model=self.model,
                    response=response
                )

                usage = response.usage
                logger.debug(f"[{doc_id}] API OK - tokens: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}")

                self.stats["success"] += 1
                return response.choices[0].message.content

            except Exception as e:
                self.stats["retries"] += 1
                wait_time = (2 ** attempt) * 2  # 지수 백오프: 2초, 4초, 8초

                error_type = type(e).__name__
                logger.warning(f"[{doc_id}] API ERROR (attempt {attempt+1}/{self.max_retries}): {error_type}: {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"[{doc_id}] Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[{doc_id}] All retries failed")
                    self.stats["failed"] += 1
                    return ""

        return ""

    async def extract_from_document_async(
        self,
        doc_id: str,
        text: str,
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        단일 문서에서 엔티티와 관계 비동기 추출
        """
        # 텍스트가 너무 길면 자르기
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        prompt = self._build_prompt(text)
        response = await self._call_llm_async(prompt, doc_id=doc_id)

        if not response:
            logger.warning(f"[{doc_id}] Empty response - no entities extracted")
            return [], []

        entities, relations = self._parse_response(response, doc_id)
        return entities, relations

    async def _process_single_doc(
        self,
        doc: Dict,
        doc_type: str,
        progress_callback=None
    ) -> Tuple[str, List[Entity], List[Relation], bool]:
        """
        단일 문서 처리 (내부 헬퍼)

        Returns:
            (doc_id, entities, relations, success)
        """
        doc_id = doc.get("doc_id", "unknown")
        text = doc.get("text", "")

        if not text:
            logger.warning(f"[{doc_id}] Empty text - skipped")
            return doc_id, [], [], False

        try:
            entities, relations = await self.extract_from_document_async(
                doc_id=doc_id,
                text=text,
                doc_type=doc_type
            )

            if progress_callback:
                progress_callback(doc_id, len(entities), len(relations))

            if entities:
                logger.info(f"[{doc_id}] Extracted {len(entities)} entities, {len(relations)} relations")
                return doc_id, entities, relations, True
            else:
                return doc_id, [], [], False

        except Exception as e:
            logger.error(f"[{doc_id}] Exception: {type(e).__name__}: {e}")
            return doc_id, [], [], False

    async def extract_batch_async(
        self,
        documents: List[Dict],
        doc_type: str = "patent",
        progress_callback=None,
        resume: bool = True
    ) -> Tuple[List[Entity], List[Relation], List[str]]:
        """
        여러 문서에서 비동기 병렬 추출 (체크포인트 지원)

        Args:
            documents: 문서 리스트 (doc_id, text 포함)
            doc_type: 문서 타입
            progress_callback: 진행상황 콜백 함수 (선택)
            resume: 체크포인트에서 재개할지 여부 (기본: True)

        Returns:
            (전체 엔티티 리스트, 전체 관계 리스트, 실패한 doc_id 리스트)
        """
        # Semaphore 초기화 (동시 요청 수 제한)
        self.semaphore = asyncio.Semaphore(self.concurrency)

        total_docs = len(documents)

        # 체크포인트에서 재개
        all_entities: List[Entity] = []
        all_relations: List[Relation] = []
        processed_doc_ids: List[str] = []
        failed_doc_ids: List[str] = []

        if resume:
            checkpoint = self.load_checkpoint(doc_type)
            if checkpoint:
                processed_doc_ids = checkpoint.get("processed_doc_ids", [])
                failed_doc_ids = checkpoint.get("failed_doc_ids", [])
                # Entity/Relation 복원
                for e_dict in checkpoint.get("entities", []):
                    all_entities.append(Entity(**e_dict))
                for r_dict in checkpoint.get("relations", []):
                    all_relations.append(Relation(**r_dict))
                self.stats = checkpoint.get("stats", {"success": 0, "failed": 0, "retries": 0})
                logger.info(f"Resuming from checkpoint: {len(processed_doc_ids)} already processed")
            else:
                self.stats = {"success": 0, "failed": 0, "retries": 0}
        else:
            self.stats = {"success": 0, "failed": 0, "retries": 0}

        # 이미 처리된 문서 제외
        processed_set = set(processed_doc_ids)
        remaining_docs = [d for d in documents if d.get("doc_id", "") not in processed_set]

        logger.info(f"Starting async batch extraction:")
        logger.info(f"  Total: {total_docs} docs")
        logger.info(f"  Already processed: {len(processed_doc_ids)}")
        logger.info(f"  Remaining: {len(remaining_docs)}")
        logger.info(f"  Concurrency: {self.concurrency}")
        logger.info(f"  Checkpoint interval: {self.checkpoint_interval}")

        if not remaining_docs:
            logger.info("All documents already processed!")
            return all_entities, all_relations, failed_doc_ids

        # 처리 카운터 (체크포인트용)
        processed_count = [0]

        async def process_with_checkpoint(doc: Dict) -> Tuple[str, List[Entity], List[Relation], bool]:
            """단일 문서 처리 후 체크포인트 저장"""
            result = await self._process_single_doc(doc, doc_type, progress_callback)
            doc_id, entities, relations, success = result

            # 결과 저장 (동시성 고려하여 락 사용)
            if success:
                all_entities.extend(entities)
                all_relations.extend(relations)
                processed_doc_ids.append(doc_id)
            else:
                failed_doc_ids.append(doc_id)

            # 카운터 증가 및 체크포인트 저장
            processed_count[0] += 1
            if processed_count[0] % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    doc_type=doc_type,
                    processed_doc_ids=processed_doc_ids,
                    failed_doc_ids=failed_doc_ids,
                    entities=all_entities,
                    relations=all_relations,
                    total_docs=total_docs
                )

            return result

        # 모든 문서를 병렬로 처리
        tasks = [process_with_checkpoint(doc) for doc in remaining_docs]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 최종 체크포인트 저장
        self._save_checkpoint(
            doc_type=doc_type,
            processed_doc_ids=processed_doc_ids,
            failed_doc_ids=failed_doc_ids,
            entities=all_entities,
            relations=all_relations,
            total_docs=total_docs
        )

        logger.info(f"Async batch complete:")
        logger.info(f"  Total: {total_docs} docs")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {len(failed_doc_ids)}")
        logger.info(f"  Retries: {self.stats['retries']}")
        logger.info(f"  Entities: {len(all_entities)}")
        logger.info(f"  Relations: {len(all_relations)}")

        return all_entities, all_relations, failed_doc_ids

    def extract_batch(
        self,
        documents: List[Dict],
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation], List[str]]:
        """
        동기 래퍼 - asyncio.run()으로 비동기 메서드 호출

        기존 코드와 호환성을 위해 동기 인터페이스 제공
        """
        return asyncio.run(
            self.extract_batch_async(documents, doc_type)
        )


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