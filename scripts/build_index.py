"""
Index Time 파이프라인
특허/논문/연구과제 데이터를 처리하여 벡터DB + 그래프DB에 저장
"""

import argparse
import json
import sys
import logging
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
    PATENT_DATA_FILE,
    ARTICLE_DATA_FILE,
    PROJECT_DATA_FILE
)
from src.rag.preprocessing.text_processor import TextProcessor
from src.rag.index.entity_extractor import EntityRelationExtractor
from src.rag.store.vector_store import ChromaVectorStore
from src.rag.store.graph_store import GraphStore
from src.rag.embedding.embedder import Embedder


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Index Time 파이프라인 실행기"""

    def __init__(self, doc_type: str = "patent", force_api: bool = True):
        """
        Args:
            doc_type: 문서 타입 (patent/article/project)
            force_api: OpenAI API 강제 사용 여부
        """
        self.doc_type = doc_type

        logger.info(f"Initializing IndexBuilder for {doc_type}...")

        # 모듈 초기화
        self.text_processor = TextProcessor()
        self.extractor = EntityRelationExtractor()
        self.embedder = Embedder(force_api=force_api)
        self.vector_store = ChromaVectorStore()
        self.graph_store = GraphStore(doc_type=doc_type)

        # 처리 통계
        self.stats = {
            "docs_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "chunks_stored": 0,
            "errors": 0
        }

    def load_data(self, file_path: str = None) -> List[Dict]:
        """데이터 파일 로드"""
        if file_path is None:
            if self.doc_type == "patent":
                file_path = PATENT_DATA_FILE
            elif self.doc_type == "article":
                # article_sample.json 파일이 있으면 그것을 사용, 없으면 기본 파일 사용
                sample_file = "data/article/article_sample.json"
                if Path(sample_file).exists():
                    file_path = sample_file
                    logger.info(f"Using article sample file: {sample_file}")
                else:
                    file_path = ARTICLE_DATA_FILE
                    logger.info(f"Sample file not found. Using full article data: {file_path}")
            elif self.doc_type == "project":
                file_path = PROJECT_DATA_FILE
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
    ) -> tuple[List[Dict], List[Dict]]:
        """엔티티/관계 추출"""
        logger.info(f"Extracting entities and relations...")

        all_entities = []
        all_relations = []

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
                    all_entities.append(asdict(e) if hasattr(e, '__dataclass_fields__') else e)
                for r in relations:
                    all_relations.append(asdict(r) if hasattr(r, '__dataclass_fields__') else r)

                self.stats["docs_processed"] += len(batch)

            except Exception as e:
                logger.error(f"Extraction error at batch {i}: {e}")
                self.stats["errors"] += 1
                continue

        self.stats["entities_extracted"] = len(all_entities)
        self.stats["relations_extracted"] = len(all_relations)

        logger.info(f"Extracted {len(all_entities)} entities, {len(all_relations)} relations")

        return all_entities, all_relations

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

    def run(self, data_file: str = None, clear: bool = False):
        """전체 파이프라인 실행"""
        logger.info("=" * 50)
        logger.info(f"Starting Index Build for {self.doc_type}")
        logger.info("=" * 50)

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
        entities, relations = self.extract_entities_relations(docs)

        if not entities:
            logger.warning("No entities extracted. Stopping.")
            return

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

        # 결과 출력
        logger.info("=" * 50)
        logger.info("Index Build Complete!")
        logger.info(f"Stats: {self.stats}")
        logger.info("=" * 50)

        return self.stats


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

    args = parser.parse_args()

    # 파이프라인 실행
    builder = IndexBuilder(
        doc_type=args.doc_type,
        force_api=args.force_api
    )
    builder.run(
        data_file=args.data_file,
        clear=args.clear
    )


if __name__ == "__main__":
    main()