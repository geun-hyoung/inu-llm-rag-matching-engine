"""
Index Time 파이프라인
특허/논문/연구과제 데이터를 처리하여 벡터DB + 그래프DB에 저장
"""

import argparse
import json
import sys
import logging
import pickle
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

# tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PROJECT_FILE
)
from src.rag.preprocessing.text_processor import TextProcessor
from src.rag.index.entity_extractor import EntityRelationExtractor, AsyncEntityRelationExtractor
from src.rag.store.vector_store import ChromaVectorStore
from src.rag.store.graph_store import GraphStore
from src.rag.embedding.embedder import Embedder
from src.utils.cost_tracker import get_cost_tracker


# 로깅 설정 (콘솔 + 파일)
def setup_logging(doc_type: str = "index"):
    """로깅 설정 - 콘솔과 파일에 동시 출력"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 타임스탬프 포함 로그 파일명
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"build_index_{doc_type}_{timestamp}.log"

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    root_logger.handlers.clear()

    # 포맷터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Index Time 파이프라인 실행기"""

    def __init__(
        self,
        doc_type: str = "patent",
        force_api: bool = True,
        store_dir: str = None,
        concurrency: int = 1,
        checkpoint_interval: int = 20
    ):
        """
        Args:
            doc_type: 문서 타입 (patent/article/project)
            force_api: OpenAI API 강제 사용 여부
            store_dir: RAG 저장소 경로 (None이면 기본값 사용)
            concurrency: 동시 API 요청 수 (1=동기, 2+=비동기)
            checkpoint_interval: 체크포인트 저장 간격 (N개 문서마다)
        """
        self.doc_type = doc_type
        self.store_dir = store_dir
        self.concurrency = concurrency

        logger.info(f"Initializing IndexBuilder for {doc_type}...")
        if store_dir:
            logger.info(f"Using custom store_dir: {store_dir}")

        # 모듈 초기화
        self.text_processor = TextProcessor()

        # concurrency에 따라 동기/비동기 추출기 선택
        if concurrency > 1:
            logger.info(f"Using AsyncEntityRelationExtractor with concurrency={concurrency}")
            self.extractor = AsyncEntityRelationExtractor(
                concurrency=concurrency,
                checkpoint_interval=checkpoint_interval
            )
            self.use_async = True
        else:
            logger.info("Using synchronous EntityRelationExtractor")
            self.extractor = EntityRelationExtractor()
            self.use_async = False

        self.embedder = Embedder(force_api=force_api)
        self.vector_store = ChromaVectorStore(persist_dir=store_dir)
        self.graph_store = GraphStore(store_dir=store_dir, doc_type=doc_type)

        # 처리 통계
        self.stats = {
            "docs_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "entities_after_merge": 0,
            "relations_after_merge": 0,
            "chunks_stored": 0,
            "errors": 0
        }

    def load_data(self, file_path: str = None) -> List[Dict]:
        """데이터 파일 로드 (train 데이터 사용)"""
        if file_path is None:
            if self.doc_type == "patent":
                file_path = DATA_TRAIN_PATENT_FILE
            elif self.doc_type == "article":
                file_path = DATA_TRAIN_ARTICLE_FILE
            elif self.doc_type == "project":
                file_path = DATA_TRAIN_PROJECT_FILE
            else:
                raise ValueError(f"Unknown doc_type: {self.doc_type}")

        logger.info(f"Loading data from {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} documents")
        return data

    def process_documents(self, raw_data: List[Dict]) -> List[Dict]:
        """텍스트 전처리"""
        logger.info("Processing documents...")

        if self.doc_type == "patent":
            processed = self.text_processor.process_patents(raw_data)
        elif self.doc_type == "article":
            processed = self.text_processor.process_articles(raw_data)
        elif self.doc_type == "project":
            processed = self.text_processor.process_projects(raw_data)
        else:
            raise NotImplementedError(f"{self.doc_type} processing not implemented")

        # ProcessedDocument를 dict로 변환
        docs = [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata
            }
            for doc in processed
        ]

        logger.info(f"Processed {len(docs)} documents")
        logger.info(f"Stats: {self.text_processor.get_stats()}")

        return docs

    def extract_entities_relations(
        self,
        docs: List[Dict],
        batch_size: int = 10
    ) -> tuple[List[Dict], List[Dict], List[str]]:
        """엔티티/관계 추출

        Returns:
            (엔티티 리스트, 관계 리스트, 실패한 doc_id 리스트)
        """
        logger.info(f"Extracting entities and relations...")

        # 비동기 모드: 전체 문서를 한 번에 병렬 처리
        if self.use_async:
            return self._extract_async(docs)

        # 동기 모드: 기존 배치 처리
        return self._extract_sync(docs, batch_size)

    def _extract_async(
        self,
        docs: List[Dict]
    ) -> tuple[List[Dict], List[Dict], List[str]]:
        """비동기 엔티티/관계 추출 (asyncio 기반)"""
        import asyncio
        import time

        logger.info(f"Async extraction: {len(docs)} docs, concurrency={self.concurrency}")

        # 진행상황 카운터 및 시간 추적
        processed_count = [0]
        total_docs = len(docs)
        start_time = [time.time()]
        last_log_time = [time.time()]

        def progress_callback(doc_id: str, entity_count: int, relation_count: int):
            processed_count[0] += 1
            current_time = time.time()

            # 10초마다 또는 100개마다 또는 완료 시 로그 출력
            should_log = (
                current_time - last_log_time[0] >= 10 or
                processed_count[0] % 100 == 0 or
                processed_count[0] == total_docs
            )

            if should_log:
                last_log_time[0] = current_time
                elapsed = current_time - start_time[0]
                pct = processed_count[0] / total_docs * 100

                # ETA 계산
                if processed_count[0] > 0:
                    avg_time = elapsed / processed_count[0]
                    remaining = total_docs - processed_count[0]
                    eta_seconds = avg_time * remaining

                    # 시간 포맷팅
                    if eta_seconds >= 3600:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    elif eta_seconds >= 60:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"

                    elapsed_str = f"{elapsed/60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
                    speed = processed_count[0] / elapsed * 60  # docs/min

                    logger.info(
                        f"Progress: {processed_count[0]}/{total_docs} ({pct:.1f}%) | "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str} | Speed: {speed:.1f} docs/min"
                    )
                else:
                    logger.info(f"Progress: {processed_count[0]}/{total_docs} ({pct:.1f}%)")

        # 비동기 배치 추출 실행 (progress_callback 연결)
        entities, relations, failed_doc_ids = asyncio.run(
            self.extractor.extract_batch_async(
                documents=docs,
                doc_type=self.doc_type,
                progress_callback=progress_callback
            )
        )

        # dataclass를 dict로 변환
        all_entities = []
        all_relations = []
        extracted_doc_ids = set()

        for e in entities:
            entity_dict = asdict(e) if hasattr(e, '__dataclass_fields__') else e
            all_entities.append(entity_dict)
            extracted_doc_ids.add(entity_dict.get('source_doc_id', ''))

        for r in relations:
            all_relations.append(asdict(r) if hasattr(r, '__dataclass_fields__') else r)

        self.stats["docs_processed"] = len(docs) - len(failed_doc_ids)
        self.stats["entities_extracted"] = len(all_entities)
        self.stats["relations_extracted"] = len(all_relations)
        self.stats["failed_docs"] = len(failed_doc_ids)

        logger.info(f"Extracted {len(all_entities)} entities, {len(all_relations)} relations")
        logger.info(f"Failed docs: {len(failed_doc_ids)} / {len(docs)} ({len(failed_doc_ids)/len(docs)*100:.1f}%)")

        return all_entities, all_relations, failed_doc_ids

    def _extract_sync(
        self,
        docs: List[Dict],
        batch_size: int = 10
    ) -> tuple[List[Dict], List[Dict], List[str]]:
        """동기 엔티티/관계 추출 (기존 방식)"""
        all_entities = []
        all_relations = []

        # 엔티티가 추출된 doc_id 추적
        extracted_doc_ids = set()

        # 배치 처리
        for i in tqdm(range(0, len(docs), batch_size), desc="Extracting"):
            batch = docs[i:i + batch_size]

            try:
                entities, relations = self.extractor.extract_batch(
                    documents=batch,
                    doc_type=self.doc_type
                )

                # dataclass를 dict로 변환
                for e in entities:
                    entity_dict = asdict(e) if hasattr(e, '__dataclass_fields__') else e
                    all_entities.append(entity_dict)
                    extracted_doc_ids.add(entity_dict.get('source_doc_id', ''))

                for r in relations:
                    all_relations.append(asdict(r) if hasattr(r, '__dataclass_fields__') else r)

                self.stats["docs_processed"] += len(batch)

            except Exception as e:
                logger.error(f"Extraction error at batch {i}: {e}")
                self.stats["errors"] += 1
                continue

        # 실패한 문서 ID 계산
        all_doc_ids = set(d.get('doc_id', '') for d in docs)
        failed_doc_ids = list(all_doc_ids - extracted_doc_ids)

        self.stats["entities_extracted"] = len(all_entities)
        self.stats["relations_extracted"] = len(all_relations)
        self.stats["failed_docs"] = len(failed_doc_ids)

        logger.info(f"Extracted {len(all_entities)} entities, {len(all_relations)} relations")
        logger.info(f"Failed docs: {len(failed_doc_ids)} / {len(docs)} ({len(failed_doc_ids)/len(docs)*100:.1f}%)")

        return all_entities, all_relations, failed_doc_ids

    def merge_duplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        중복 관계 병합 (LightRAG 방식)

        동일한 source-target 쌍의 관계를 병합:
        - keywords: 중복 제거 후 합침
        - description: 연결
        - weight: 최대값 유지
        """
        if not relations:
            return relations

        merged = {}

        for r in relations:
            # 키 생성: source|target (정렬로 방향 무관하게 처리)
            src = r.get("source_entity", "")
            tgt = r.get("target_entity", "")
            normalized_src, normalized_tgt = sorted([src, tgt])
            key = f"{normalized_src}|{normalized_tgt}"

            if key not in merged:
                merged[key] = r.copy()
            else:
                existing = merged[key]

                # keywords 병합 (중복 제거 후 합침)
                old_kw = set(k.strip() for k in existing.get("keywords", "").split(",") if k.strip())
                new_kw = set(k.strip() for k in r.get("keywords", "").split(",") if k.strip())
                existing["keywords"] = ",".join(sorted(old_kw | new_kw))

                # description 연결 (중복 제거)
                old_desc = existing.get("description", "")
                new_desc = r.get("description", "")
                if new_desc and new_desc not in old_desc:
                    existing["description"] = f"{old_desc}\n{new_desc}".strip()

                # weight 최대값 유지
                existing["weight"] = max(
                    existing.get("weight", 1),
                    r.get("weight", 1)
                )

                # source_doc_id 병합 (여러 문서에서 추출된 경우)
                old_doc_ids = set(existing.get("source_doc_id", "").split(","))
                new_doc_id = r.get("source_doc_id", "")
                if new_doc_id:
                    old_doc_ids.add(new_doc_id)
                existing["source_doc_id"] = ",".join(sorted(old_doc_ids - {""}))

        merged_relations = list(merged.values())

        logger.info(f"Merged relations: {len(relations)} -> {len(merged_relations)} (removed {len(relations) - len(merged_relations)} duplicates)")

        return merged_relations

    def merge_duplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        중복 엔티티 병합 (LightRAG 방식)

        동일한 name의 엔티티를 병합:
        - description: 연결
        - entity_type: 첫 번째 유지
        """
        if not entities:
            return entities

        merged = {}

        for e in entities:
            name = e.get("name", "").strip()
            if not name:
                continue

            if name not in merged:
                merged[name] = e.copy()
            else:
                existing = merged[name]

                # description 연결 (중복 제거)
                old_desc = existing.get("description", "")
                new_desc = e.get("description", "")
                if new_desc and new_desc not in old_desc:
                    existing["description"] = f"{old_desc}\n{new_desc}".strip()

                # source_doc_id 병합
                old_doc_ids = set(existing.get("source_doc_id", "").split(","))
                new_doc_id = e.get("source_doc_id", "")
                if new_doc_id:
                    old_doc_ids.add(new_doc_id)
                existing["source_doc_id"] = ",".join(sorted(old_doc_ids - {""}))

        merged_entities = list(merged.values())

        logger.info(f"Merged entities: {len(entities)} -> {len(merged_entities)} (removed {len(entities) - len(merged_entities)} duplicates)")

        return merged_entities

    def generate_embeddings(
        self,
        entities: List[Dict],
        relations: List[Dict],
        docs: List[Dict]
    ) -> tuple:
        """임베딩 생성 (엔티티, 관계, 청크)"""
        logger.info("Generating embeddings...")

        # 엔티티 임베딩 (LightRAG 방식: name + description)
        entity_texts = [
            f"{e.get('name', '')}\n{e.get('description', '')}"
            for e in entities
        ]

        # 관계 임베딩 (LightRAG 방식: keywords를 맨 앞에 배치)
        relation_texts = [
            f"{r.get('keywords', '')}\t{r['source_entity']}\n{r['target_entity']}\n{r.get('description', '')}"
            for r in relations
        ]

        # 청크 임베딩 (원본 문서 텍스트)
        chunk_texts = [doc["text"] for doc in docs]

        entity_embeddings = None
        relation_embeddings = None
        chunk_embeddings = None

        if entity_texts:
            logger.info(f"Encoding {len(entity_texts)} entities...")
            entity_embeddings = self.embedder.encode(entity_texts)

        if relation_texts:
            logger.info(f"Encoding {len(relation_texts)} relations...")
            relation_embeddings = self.embedder.encode(relation_texts)

        if chunk_texts:
            logger.info(f"Encoding {len(chunk_texts)} chunks...")
            chunk_embeddings = self.embedder.encode(chunk_texts)

        logger.info("Embeddings generated")

        return entity_embeddings, relation_embeddings, chunk_embeddings

    def store_to_vector_db(
        self,
        entities: List[Dict],
        relations: List[Dict],
        docs: List[Dict],
        entity_embeddings,
        relation_embeddings,
        chunk_embeddings
    ):
        """ChromaDB에 저장 (엔티티, 관계, 청크)"""
        logger.info("Storing to ChromaDB...")

        if entities and entity_embeddings is not None:
            self.vector_store.add_entities(
                entities=entities,
                embeddings=entity_embeddings,
                doc_type=self.doc_type
            )

        if relations and relation_embeddings is not None:
            self.vector_store.add_relations(
                relations=relations,
                embeddings=relation_embeddings,
                doc_type=self.doc_type
            )

        # 청크 저장 (Naive RAG용)
        if docs and chunk_embeddings is not None:
            chunks = [
                {
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "title": doc.get("metadata", {}).get("title", "")
                }
                for doc in docs
            ]
            self.vector_store.add_chunks(
                chunks=chunks,
                embeddings=chunk_embeddings,
                doc_type=self.doc_type
            )
            self.stats["chunks_stored"] = len(chunks)

        logger.info(f"ChromaDB stats: {self.vector_store.get_stats()}")

    def store_to_graph_db(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ):
        """GraphStore에 저장"""
        logger.info("Storing to GraphStore...")

        self.graph_store.add_entities_batch(entities)
        self.graph_store.add_relations_batch(relations)
        self.graph_store.save()

        logger.info(f"GraphStore stats: {self.graph_store.get_stats()}")

    def run(self, data_file: str = None, clear: bool = False, resume: bool = False):
        """전체 파이프라인 실행"""
        logger.info("=" * 50)
        logger.info(f"Starting Index Build for {self.doc_type}")
        logger.info("=" * 50)

        # 비용 추적 시작
        tracker = get_cost_tracker()
        tracker.start_task("indexing", description=f"{self.doc_type} 문서 인덱싱")

        checkpoint_file = Path(f"data/checkpoint_{self.doc_type}.pkl")

        # 체크포인트에서 복원
        if resume and checkpoint_file.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            docs = checkpoint['docs']
            entities = checkpoint['entities']
            relations = checkpoint['relations']
            logger.info(f"Loaded {len(docs)} docs, {len(entities)} entities, {len(relations)} relations")
        else:
            # 기존 데이터 초기화
            if clear:
                logger.info("Clearing existing data...")
                self.vector_store.clear_all()
                self.graph_store.clear()

            # 1. 데이터 로드
            raw_data = self.load_data(data_file)

            # 2. 텍스트 전처리
            docs = self.process_documents(raw_data)

            # 3. 엔티티/관계 추출
            entities, relations, failed_doc_ids = self.extract_entities_relations(docs)

            # 실패한 문서 ID 저장
            if failed_doc_ids:
                failed_file = Path(f"logs/failed_docs_{self.doc_type}.json")
                failed_file.parent.mkdir(exist_ok=True)
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "doc_type": self.doc_type,
                        "total_docs": len(docs),
                        "failed_count": len(failed_doc_ids),
                        "failed_doc_ids": failed_doc_ids
                    }, f, ensure_ascii=False, indent=2)
                logger.info(f"Failed doc IDs saved to: {failed_file}")

            if not entities:
                logger.warning("No entities extracted. Stopping.")
                return

            # 3.5 중복 병합 (LightRAG 방식)
            entities = self.merge_duplicate_entities(entities)
            relations = self.merge_duplicate_relations(relations)

            # 병합 후 통계 업데이트
            self.stats["entities_after_merge"] = len(entities)
            self.stats["relations_after_merge"] = len(relations)

            # 체크포인트 저장 (비용 정보 포함)
            logger.info(f"Saving checkpoint to {checkpoint_file}...")
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

            # 현재까지의 비용 정보 가져오기
            cost_summary = tracker.get_current_task_summary()

            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'docs': docs,
                    'entities': entities,
                    'relations': relations,
                    'failed_doc_ids': failed_doc_ids,
                    'cost_summary': cost_summary,
                    'stats': self.stats.copy()
                }, f)

            # cost_history.json에도 저장
            tracker._save_history()

            logger.info(f"Checkpoint saved with cost: ${cost_summary.get('total_cost_usd', 0):.4f}")

        # 4. 임베딩 생성 (엔티티, 관계, 청크)
        entity_embeddings, relation_embeddings, chunk_embeddings = self.generate_embeddings(
            entities, relations, docs
        )

        # 5. ChromaDB 저장 (엔티티, 관계, 청크)
        self.store_to_vector_db(
            entities, relations, docs,
            entity_embeddings, relation_embeddings, chunk_embeddings
        )

        # 6. GraphStore 저장
        self.store_to_graph_db(entities, relations)

        # 비용 추적 종료 (문서 처리 통계 포함)
        cost_result = tracker.end_task(**self.stats)
        if cost_result:
            logger.info(f"API Cost: ${cost_result.get('total_cost_usd', 0):.6f}")
            logger.info(f"Documents: {cost_result.get('metadata', {}).get('docs_processed', 0)}")

        # 결과 출력
        logger.info("=" * 50)
        logger.info("Index Build Complete!")
        logger.info(f"Stats: {self.stats}")
        logger.info("=" * 50)

        return self.stats

    def retry_failed(self, max_docs: int = None):
        """실패한 문서 재처리"""
        logger.info("=" * 50)
        logger.info(f"Retrying failed documents for {self.doc_type}")
        logger.info("=" * 50)

        # 1. 실패한 문서 ID 로드
        failed_file = Path(f"logs/failed_docs_{self.doc_type}.json")
        if not failed_file.exists():
            logger.error(f"Failed docs file not found: {failed_file}")
            return

        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_data = json.load(f)

        failed_ids = set(failed_data.get('failed_doc_ids', []))
        logger.info(f"Loaded {len(failed_ids)} failed doc IDs")

        if not failed_ids:
            logger.info("No failed documents to retry")
            return

        # 2. 체크포인트에서 문서 데이터 로드
        possible_paths = [
            Path(f"data/train/checkpoint_{self.doc_type}.pkl"),
            Path(f"data/checkpoint_{self.doc_type}.pkl"),
        ]

        doc_map = {}
        for ckpt_path in possible_paths:
            if ckpt_path.exists():
                with open(ckpt_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                docs = checkpoint.get('docs', [])
                doc_map = {d.get('doc_id', ''): d for d in docs}
                logger.info(f"Loaded {len(docs)} docs from {ckpt_path}")
                break

        if not doc_map:
            logger.error("No checkpoint file found")
            return

        # 3. 실패한 문서만 필터링
        failed_docs = [doc_map[doc_id] for doc_id in failed_ids if doc_id in doc_map]
        logger.info(f"Found {len(failed_docs)} failed documents to retry")

        if max_docs:
            failed_docs = failed_docs[:max_docs]
            logger.info(f"Limited to {max_docs} documents")

        # 4. 엔티티/관계 추출
        all_entities = []
        all_relations = []
        still_failed = []

        for doc in tqdm(failed_docs, desc="Retrying"):
            doc_id = doc.get('doc_id', '')
            text = doc.get('text', '')

            if not text:
                still_failed.append(doc_id)
                continue

            try:
                entities, relations = self.extractor.extract_from_document(
                    doc_id=doc_id,
                    text=text,
                    doc_type=self.doc_type
                )

                if entities:
                    for e in entities:
                        all_entities.append(asdict(e) if hasattr(e, '__dataclass_fields__') else e)
                    for r in relations:
                        all_relations.append(asdict(r) if hasattr(r, '__dataclass_fields__') else r)
                    logger.info(f"[{doc_id}] SUCCESS: {len(entities)} entities, {len(relations)} relations")
                else:
                    still_failed.append(doc_id)
                    logger.warning(f"[{doc_id}] No entities extracted")

            except Exception as e:
                still_failed.append(doc_id)
                logger.error(f"[{doc_id}] FAILED: {type(e).__name__}: {e}")

        logger.info(f"Extraction complete: {len(all_entities)} entities, {len(all_relations)} relations")
        logger.info(f"Still failed: {len(still_failed)} / {len(failed_docs)}")

        if not all_entities:
            logger.warning("No new entities extracted")
            return

        # 5. 임베딩 생성
        logger.info("Generating embeddings...")

        entity_texts = [f"{e.get('name', '')}\n{e.get('description', '')}" for e in all_entities]
        relation_texts = [
            f"{r.get('keywords', '')}\t{r['source_entity']}\n{r['target_entity']}\n{r.get('description', '')}"
            for r in all_relations
        ]

        entity_embeddings = self.embedder.encode(entity_texts) if entity_texts else None
        relation_embeddings = self.embedder.encode(relation_texts) if relation_texts else None

        # 6. ChromaDB에 추가
        logger.info("Storing to ChromaDB...")

        if all_entities and entity_embeddings is not None:
            self.vector_store.add_entities(
                entities=all_entities,
                embeddings=entity_embeddings,
                doc_type=self.doc_type
            )

        if all_relations and relation_embeddings is not None:
            self.vector_store.add_relations(
                relations=all_relations,
                embeddings=relation_embeddings,
                doc_type=self.doc_type
            )

        # 7. GraphStore에 추가
        logger.info("Storing to GraphStore...")
        self.graph_store.add_entities_batch(all_entities)
        self.graph_store.add_relations_batch(all_relations)
        self.graph_store.save()

        # 8. 여전히 실패한 문서 저장
        if still_failed:
            still_failed_file = Path(f"logs/still_failed_docs_{self.doc_type}.json")
            with open(still_failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "doc_type": self.doc_type,
                    "retry_count": len(failed_docs),
                    "still_failed_count": len(still_failed),
                    "still_failed_doc_ids": still_failed
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Still failed docs saved to: {still_failed_file}")

        # 9. 결과 요약
        logger.info("=" * 50)
        logger.info("Retry Complete!")
        logger.info(f"  Retried: {len(failed_docs)}")
        logger.info(f"  Success: {len(failed_docs) - len(still_failed)}")
        logger.info(f"  Still failed: {len(still_failed)}")
        logger.info(f"  New entities: {len(all_entities)}")
        logger.info(f"  New relations: {len(all_relations)}")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Build index for RAG system")
    parser.add_argument(
        "--doc-type",
        type=str,
        default="patent",
        choices=["patent", "article", "project"],
        help="Document type to process"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to data file (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before building"
    )
    parser.add_argument(
        "--force-api",
        action="store_true",
        default=False,
        help="Force use OpenAI API for embeddings (default: auto-detect GPU)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip entity extraction)"
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=None,
        help="Custom RAG store directory (default: data/rag_store)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed documents from logs/failed_docs_{doc_type}.json"
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        default=None,
        help="Maximum number of failed documents to retry (for testing)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent API requests (1=sync, 2+=async). Recommended: 10 for Tier 1"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Save checkpoint every N documents (default: 20)"
    )

    args = parser.parse_args()

    # 로깅 설정 (파일 + 콘솔)
    log_file = setup_logging(args.doc_type)
    logger.info(f"Log file: {log_file}")

    # 파이프라인 실행
    builder = IndexBuilder(
        doc_type=args.doc_type,
        force_api=args.force_api,
        store_dir=args.store_dir,
        concurrency=args.concurrency,
        checkpoint_interval=args.checkpoint_interval
    )

    if args.retry_failed:
        # 실패한 문서 재처리 모드
        builder.retry_failed(max_docs=args.max_retry)
    else:
        # 일반 인덱싱 모드
        builder.run(
            data_file=args.data_file,
            clear=args.clear,
            resume=args.resume
        )


if __name__ == "__main__":
    main()