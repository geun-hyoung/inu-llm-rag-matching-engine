"""
ChromaDB 벡터 저장소
6개 컬렉션으로 엔티티/관계를 문서 타입별로 저장
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import asdict
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAG_STORE_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    raise


# 컬렉션 이름 상수
COLLECTIONS = {
    "patent_entities": "patent_entities",
    "patent_relations": "patent_relations",
    "article_entities": "article_entities",
    "article_relations": "article_relations",
    "project_entities": "project_entities",
    "project_relations": "project_relations",
}


class ChromaVectorStore:
    """ChromaDB 기반 벡터 저장소 - 6개 컬렉션 관리"""

    def __init__(self, persist_dir: str = None):
        """
        ChromaDB 벡터 저장소 초기화

        Args:
            persist_dir: 영구 저장 디렉토리
        """
        self.persist_dir = Path(persist_dir or RAG_STORE_DIR) / "chromadb"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화 (영구 저장)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # 6개 컬렉션 초기화
        self.collections = {}
        self._init_collections()

        print(f"ChromaDB initialized at: {self.persist_dir}")

    def _init_collections(self):
        """6개 컬렉션 생성/로드"""
        for collection_name in COLLECTIONS.values():
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
            )
            count = self.collections[collection_name].count()
            print(f"  - {collection_name}: {count} items")

    def _get_collection_name(self, doc_type: str, item_type: str) -> str:
        """
        문서 타입과 아이템 타입으로 컬렉션 이름 결정

        Args:
            doc_type: patent / article / project
            item_type: entities / relations

        Returns:
            컬렉션 이름
        """
        return f"{doc_type}_{item_type}"

    def add_entities(
        self,
        entities: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        엔티티를 컬렉션에 추가

        Args:
            entities: 엔티티 정보 리스트 (name, entity_type, description, source_doc_id, emp_no)
            embeddings: 엔티티 임베딩 벡터 (numpy array)
            doc_type: 문서 타입 (patent/article/project)
        """
        if len(entities) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "entities")
        collection = self.collections[collection_name]

        # ChromaDB 형식으로 변환
        ids = []
        documents = []
        metadatas = []

        for i, entity in enumerate(entities):
            # 고유 ID 생성: doc_type_entity_name_source_doc_id
            entity_id = f"{doc_type}_e_{entity['name']}_{entity['source_doc_id']}"
            entity_id = entity_id.replace(" ", "_")[:100]  # 길이 제한

            ids.append(entity_id)
            documents.append(entity.get("description", entity["name"]))
            metadatas.append({
                "name": entity["name"],
                "entity_type": entity.get("entity_type", "UNKNOWN"),
                "source_doc_id": entity.get("source_doc_id", ""),
                "emp_no": entity.get("emp_no", ""),
                "doc_type": doc_type
            })

        # 임베딩을 리스트로 변환
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # ChromaDB에 추가 (upsert로 중복 방지)
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Added {len(entities)} entities to {collection_name}")

    def add_relations(
        self,
        relations: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        관계를 컬렉션에 추가

        Args:
            relations: 관계 정보 리스트 (source_entity, target_entity, relation_type, description, source_doc_id, emp_no)
            embeddings: 관계 임베딩 벡터
            doc_type: 문서 타입
        """
        if len(relations) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "relations")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for i, relation in enumerate(relations):
            # 고유 ID 생성
            rel_id = f"{doc_type}_r_{relation['source_entity']}_{relation['target_entity']}_{relation['source_doc_id']}"
            rel_id = rel_id.replace(" ", "_")[:100]

            ids.append(rel_id)

            # 관계 설명 텍스트 (임베딩 대상)
            rel_text = relation.get("description", "")
            if not rel_text:
                rel_text = f"{relation['source_entity']} {relation['relation_type']} {relation['target_entity']}"
            documents.append(rel_text)

            metadatas.append({
                "source_entity": relation["source_entity"],
                "target_entity": relation["target_entity"],
                "relation_type": relation.get("relation_type", "RELATED_TO"),
                "source_doc_id": relation.get("source_doc_id", ""),
                "emp_no": relation.get("emp_no", ""),
                "doc_type": doc_type
            })

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Added {len(relations)} relations to {collection_name}")

    def search_entities(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None,
        filter_emp_no: str = None
    ) -> List[Dict]:
        """
        엔티티 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트 (None이면 전체)
            top_k: 반환할 결과 수
            filter_emp_no: 특정 교수만 필터링

        Returns:
            검색 결과 리스트
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        # numpy array를 리스트로 변환
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        all_results = []

        for doc_type in doc_types:
            collection_name = self._get_collection_name(doc_type, "entities")
            if collection_name not in self.collections:
                continue

            collection = self.collections[collection_name]

            if collection.count() == 0:
                continue

            # 필터 조건 구성
            where_filter = None
            if filter_emp_no:
                where_filter = {"emp_no": filter_emp_no}

            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )

                # 결과 정리
                if results and results["ids"][0]:
                    for i, id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        similarity = 1 - distance  # cosine distance를 similarity로 변환

                        all_results.append({
                            "id": id,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "doc_type": doc_type,
                            "item_type": "entity"
                        })

            except Exception as e:
                print(f"Search error in {collection_name}: {e}")
                continue

        # 유사도 기준 정렬 후 상위 k개 반환
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def search_relations(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None,
        filter_emp_no: str = None
    ) -> List[Dict]:
        """
        관계 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트
            top_k: 반환할 결과 수
            filter_emp_no: 특정 교수만 필터링

        Returns:
            검색 결과 리스트
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        all_results = []

        for doc_type in doc_types:
            collection_name = self._get_collection_name(doc_type, "relations")
            if collection_name not in self.collections:
                continue

            collection = self.collections[collection_name]

            if collection.count() == 0:
                continue

            where_filter = None
            if filter_emp_no:
                where_filter = {"emp_no": filter_emp_no}

            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results["ids"][0]:
                    for i, id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        similarity = 1 - distance

                        all_results.append({
                            "id": id,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "doc_type": doc_type,
                            "item_type": "relation"
                        })

            except Exception as e:
                print(f"Search error in {collection_name}: {e}")
                continue

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def search_all(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None,
        filter_emp_no: str = None
    ) -> Dict[str, List[Dict]]:
        """
        엔티티와 관계 모두 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트
            top_k: 각 타입별 반환할 결과 수
            filter_emp_no: 특정 교수만 필터링

        Returns:
            {"entities": [...], "relations": [...]}
        """
        return {
            "entities": self.search_entities(query_embedding, doc_types, top_k, filter_emp_no),
            "relations": self.search_relations(query_embedding, doc_types, top_k, filter_emp_no)
        }

    def get_stats(self) -> Dict[str, int]:
        """컬렉션별 통계 반환"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = collection.count()
        return stats

    def get_emp_no_stats(self, doc_type: str = "patent") -> Dict[str, int]:
        """
        교수별 엔티티/관계 수 통계

        Args:
            doc_type: 문서 타입

        Returns:
            {emp_no: count} 딕셔너리
        """
        entity_collection = self.collections.get(f"{doc_type}_entities")
        if not entity_collection or entity_collection.count() == 0:
            return {}

        # 모든 데이터 가져오기 (대용량에서는 pagination 필요)
        all_data = entity_collection.get(include=["metadatas"])

        emp_counts = {}
        for metadata in all_data["metadatas"]:
            emp_no = metadata.get("emp_no", "unknown")
            emp_counts[emp_no] = emp_counts.get(emp_no, 0) + 1

        return emp_counts

    def delete_by_doc_id(self, doc_id: str, doc_type: str = "patent"):
        """
        특정 문서의 엔티티/관계 삭제

        Args:
            doc_id: 삭제할 문서 ID
            doc_type: 문서 타입
        """
        for item_type in ["entities", "relations"]:
            collection_name = self._get_collection_name(doc_type, item_type)
            collection = self.collections.get(collection_name)

            if collection:
                collection.delete(where={"source_doc_id": doc_id})

        print(f"Deleted all items for doc_id: {doc_id}")

    def clear_all(self):
        """모든 컬렉션 초기화 (주의!)"""
        for name in COLLECTIONS.values():
            self.client.delete_collection(name)

        self._init_collections()
        print("All collections cleared")


if __name__ == "__main__":
    import numpy as np

    # 테스트
    print("Testing ChromaVectorStore...")
    store = ChromaVectorStore()

    # 샘플 엔티티
    test_entities = [
        {
            "name": "딥러닝",
            "entity_type": "TECHNOLOGY",
            "description": "심층 신경망 기반 기계학습 기술",
            "source_doc_id": "patent_001",
            "emp_no": "P12345"
        },
        {
            "name": "의료영상분석",
            "entity_type": "DOMAIN",
            "description": "CT, MRI 등 의료 영상 분석 분야",
            "source_doc_id": "patent_001",
            "emp_no": "P12345"
        }
    ]

    # 샘플 임베딩 (실제로는 Embedder 사용)
    test_embeddings = np.random.rand(2, 1536)  # OpenAI 임베딩 차원

    # 엔티티 추가
    store.add_entities(test_entities, test_embeddings, doc_type="patent")

    # 통계 확인
    print("\nCollection stats:")
    for name, count in store.get_stats().items():
        print(f"  - {name}: {count}")

    # 검색 테스트
    query_embedding = np.random.rand(1536)
    results = store.search_entities(query_embedding, doc_types=["patent"], top_k=5)

    print(f"\nSearch results: {len(results)} items")
    for r in results:
        print(f"  - {r['metadata'].get('name')}: {r['similarity']:.4f}")