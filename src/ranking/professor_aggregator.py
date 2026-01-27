"""
Professor Aggregator
RAG 검색 결과를 교수별로 집계하는 모듈
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import (
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PROJECT_FILE
)


class ProfessorAggregator:
    """교수별 문서 집계 클래스"""
    
    def __init__(self):
        """초기화"""
        # 원본 데이터 캐시 (성능 향상을 위해)
        self._data_cache = {
            "patent": None,
            "article": None,
            "project": None
        }
    
    def aggregate_by_professor(
        self,
        rag_results: Dict[str, Any],
        doc_types: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        RAG 검색 결과를 교수별로 집계
        
        Args:
            rag_results: test_rag.json 형태의 RAG 결과
                {
                    "query": "...",
                    "keywords": {...},
                    "retrieved_docs": [
                        {"no": "253", "data_type": "patent", "matches": [...]},
                        ...
                    ]
                }
            doc_types: 문서 타입 리스트 (["patent", "article", "project"])
            
        Returns:
            교수별 집계 데이터 딕셔너리
            {
                "professor_id": {
                    "professor_info": {...},
                    "documents": {
                        "patent": [원본 문서 데이터들],
                        "article": [원본 문서 데이터들],
                        "project": [원본 문서 데이터들]
                    }
                }
            }
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]
        
        professor_data = defaultdict(lambda: {
            "professor_info": None,
            "documents": {
                "patent": [],
                "article": [],
                "project": []
            }
        })
        
        retrieved_docs = rag_results.get("retrieved_docs", [])
        
        # 각 문서에 대해 원본 데이터 로드 및 교수별 집계
        for doc in retrieved_docs:
            doc_no = str(doc.get("no", ""))
            doc_type = doc.get("data_type", "")
            
            if not doc_no or doc_type not in doc_types:
                continue
            
            # 원본 문서 로드
            original_doc = self._load_original_document(doc_type, doc_no)
            if not original_doc:
                continue
            
            # 교수 정보 추출
            prof_info = self._extract_professor_info(original_doc)
            if not prof_info:
                continue
            
            # 교수 ID 생성
            prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO", "")
            if not prof_id:
                continue
            
            prof_id = str(prof_id)
            
            # 교수 정보 저장 (첫 번째 문서의 정보 사용)
            if professor_data[prof_id]["professor_info"] is None:
                professor_data[prof_id]["professor_info"] = prof_info
            
            # 문서 추가
            professor_data[prof_id]["documents"][doc_type].append(original_doc)
        
        return dict(professor_data)
    
    def _extract_professor_info(self, doc: Dict) -> Optional[Dict[str, Any]]:
        """
        문서 메타데이터에서 교수 정보 추출
        
        Args:
            doc: 원본 문서 데이터
            
        Returns:
            교수 정보 딕셔너리 또는 None
        """
        prof_info = doc.get("professor_info")
        if not prof_info:
            return None
        
        # SQ 또는 EMP_NO가 있어야 함
        prof_id = prof_info.get("SQ") or prof_info.get("EMP_NO")
        if not prof_id:
            return None
        
        return prof_info
    
    def _load_original_document(self, doc_type: str, doc_id: str) -> Optional[Dict]:
        """
        원본 문서 데이터 로드
        
        Args:
            doc_type: 문서 타입 ("patent", "article", "project")
            doc_id: 문서 ID (no 필드 값)
            
        Returns:
            원본 문서 데이터 또는 None
        """
        # 파일 경로 결정
        file_paths = {
            "patent": DATA_TRAIN_PATENT_FILE,
            "article": DATA_TRAIN_ARTICLE_FILE,
            "project": DATA_TRAIN_PROJECT_FILE
        }
        
        file_path = file_paths.get(doc_type)
        if not file_path or not Path(file_path).exists():
            return None
        
        # 캐시 확인
        if self._data_cache[doc_type] is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._data_cache[doc_type] = json.load(f)
            except Exception as e:
                print(f"Error loading {doc_type} data: {e}")
                return None
        
        # 문서 ID로 검색
        data = self._data_cache[doc_type]
        if isinstance(data, list):
            # doc_id를 정수로 변환 시도
            try:
                doc_id_int = int(doc_id)
                for item in data:
                    if item.get("no") == doc_id_int or str(item.get("no")) == doc_id:
                        return item
            except ValueError:
                # 정수 변환 실패 시 문자열로만 비교
                for item in data:
                    if str(item.get("no")) == doc_id:
                        return item
        
        return None
