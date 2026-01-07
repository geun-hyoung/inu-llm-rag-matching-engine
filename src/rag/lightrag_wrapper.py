"""
LightRAG 래퍼 모듈
엔티티/관계 추출을 위한 LightRAG 라이브러리 래퍼
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL, RAG_STORE_DIR
from src.embedding.encoder import Embedder


@dataclass
class Entity:
    """추출된 엔티티"""
    name: str
    entity_type: str  # TECHNOLOGY, METHOD, DOMAIN, APPLICATION 등
    description: str
    source_doc_id: str
    emp_no: str


@dataclass
class Relation:
    """추출된 관계"""
    source_entity: str
    target_entity: str
    relation_type: str  # USES, APPLIES_TO, RELATED_TO 등
    description: str
    source_doc_id: str
    emp_no: str


class LightRAGExtractor:
    """LightRAG를 활용한 엔티티/관계 추출기"""

    def __init__(self, working_dir: str = None):
        """
        LightRAG 추출기 초기화

        Args:
            working_dir: LightRAG 작업 디렉토리
        """
        self.working_dir = Path(working_dir or RAG_STORE_DIR) / "lightrag_work"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # OpenAI API 키 설정
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        self.rag = None
        self._init_lightrag()

    def _init_lightrag(self):
        """LightRAG 초기화"""
        try:
            from lightrag import LightRAG, QueryParam

            # 커스텀 임베딩 함수 생성
            self.embedder = Embedder()

            async def custom_embedding_func(texts: list[str]) -> np.ndarray:
                """우리 Embedder를 LightRAG에서 사용할 수 있도록 래핑"""
                return self.embedder.encode(texts)

            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_name=LLM_MODEL,
                embedding_func=custom_embedding_func,
                embedding_dim=self.embedder.dimension,
            )
            print(f"LightRAG initialized with model: {LLM_MODEL}")
            print(f"Using custom embedder: {self.embedder.model_name} (dim={self.embedder.dimension})")

        except ImportError as e:
            print(f"LightRAG import error: {e}")
            print("Please install: pip install lightrag-hku")
            raise
        except Exception as e:
            print(f"LightRAG init error: {e}")
            raise

    def extract_from_document(
        self,
        doc_id: str,
        text: str,
        emp_no: str,
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        단일 문서에서 엔티티와 관계 추출

        Args:
            doc_id: 문서 ID
            text: 문서 텍스트
            emp_no: 교수 사번
            doc_type: 문서 타입 (patent/article/project)

        Returns:
            (엔티티 리스트, 관계 리스트)
        """
        # LightRAG에 문서 삽입 (내부적으로 엔티티/관계 추출)
        self.rag.insert(text)

        # LightRAG 내부 그래프에서 엔티티/관계 추출
        entities, relations = self._extract_from_graph(doc_id, emp_no)

        return entities, relations

    def extract_batch(
        self,
        documents: List[Dict],
        doc_type: str = "patent"
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        여러 문서에서 배치로 엔티티/관계 추출

        Args:
            documents: 문서 리스트 (doc_id, text, emp_no 포함)
            doc_type: 문서 타입

        Returns:
            (전체 엔티티 리스트, 전체 관계 리스트)
        """
        all_entities = []
        all_relations = []

        for idx, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc_{idx}")
            text = doc.get("text", "")
            emp_no = doc.get("emp_no", "")

            if not text:
                continue

            try:
                entities, relations = self.extract_from_document(
                    doc_id=doc_id,
                    text=text,
                    emp_no=emp_no,
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

    def _extract_from_graph(
        self,
        doc_id: str,
        emp_no: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        LightRAG 내부 그래프에서 엔티티/관계 추출

        Args:
            doc_id: 문서 ID
            emp_no: 교수 사번

        Returns:
            (엔티티 리스트, 관계 리스트)
        """
        entities = []
        relations = []

        # LightRAG 내부 저장소 접근
        try:
            # 엔티티 추출
            if hasattr(self.rag, 'chunk_entity_relation_graph'):
                graph = self.rag.chunk_entity_relation_graph

                # 노드 (엔티티) 추출
                for node_id, node_data in graph.nodes(data=True):
                    entity = Entity(
                        name=node_id,
                        entity_type=node_data.get("entity_type", "UNKNOWN"),
                        description=node_data.get("description", ""),
                        source_doc_id=doc_id,
                        emp_no=emp_no
                    )
                    entities.append(entity)

                # 엣지 (관계) 추출
                for source, target, edge_data in graph.edges(data=True):
                    relation = Relation(
                        source_entity=source,
                        target_entity=target,
                        relation_type=edge_data.get("relation_type", "RELATED_TO"),
                        description=edge_data.get("description", ""),
                        source_doc_id=doc_id,
                        emp_no=emp_no
                    )
                    relations.append(relation)

        except Exception as e:
            print(f"Warning: Could not extract from graph: {e}")

        return entities, relations

    def query(self, query_text: str, mode: str = "hybrid") -> str:
        """
        LightRAG 쿼리 (테스트용)

        Args:
            query_text: 쿼리 텍스트
            mode: 검색 모드 (naive/local/global/hybrid)

        Returns:
            검색 결과
        """
        from lightrag import QueryParam

        return self.rag.query(query_text, param=QueryParam(mode=mode))


if __name__ == "__main__":
    # LightRAG 초기화 테스트
    print("Testing LightRAGExtractor...")

    try:
        extractor = LightRAGExtractor()
        print("LightRAG initialized successfully")
    except Exception as e:
        print(f"LightRAG init failed: {e}")