"""
Report Generator
AHP 결과를 기반으로 GPT-4o-mini를 사용하여 리포트 생성
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL
from src.utils.cost_tracker import log_chat_usage, get_cost_tracker


class ReportGenerator:
    """산학 매칭 추천 보고서 생성 클래스"""
    
    def __init__(self, output_dir: str = None, api_key: str = None):
        """
        초기화
        
        Args:
            output_dir: 보고서 출력 디렉토리 (None이면 results/test/report 사용)
            api_key: OpenAI API 키 (None이면 config에서 가져옴)
        """
        if output_dir is None:
            self.output_dir = Path("results/test/report")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenAI 클라이언트 초기화
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. config/settings.py에 OPENAI_API_KEY를 설정하세요.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = LLM_MODEL
    
    def generate_report_from_query(
        self,
        query: str,
        doc_types: List[str] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        retrieval_top_k: int = None,
        retriever = None
    ) -> Dict[str, Any]:
        """
        쿼리를 기반으로 전체 파이프라인 실행 (RAG → AHP → 리포트 생성)

        Args:
            query: 검색 쿼리
            doc_types: 검색할 문서 타입 리스트 (기본: ["patent", "article", "project"])
            few_shot_examples: 보고서 생성용 Few-shot 예시 리스트
            retrieval_top_k: Local/Global 검색 시 각각 가져올 개수 (기본: 5)
            retriever: 외부에서 주입할 HybridRetriever 인스턴스 (None이면 내부 생성)

        Returns:
            생성된 리포트 데이터 딕셔너리
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        # 비용 추적 시작
        tracker = get_cost_tracker()
        tracker.start_task("full_pipeline", description=f"전체 파이프라인: {query[:30]}...")

        # RAG 검색 → 교수 집계 → AHP 랭킹 → 리포트 생성
        from src.rag.query.retriever import HybridRetriever
        from src.ranking.professor_aggregator import ProfessorAggregator
        from src.ranking.ranker import ProfessorRanker
        from config.settings import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
        from config.ahp_config import DEFAULT_TYPE_WEIGHTS

        # 1. RAG 검색 (외부 주입 또는 내부 생성)
        print("RAG 검색 수행 중...")
        if retriever is None:
            retriever = HybridRetriever(doc_types=doc_types)
        raw_rag_results = retriever.retrieve(
            query=query,
            retrieval_top_k=retrieval_top_k or RETRIEVAL_TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
            mode="hybrid"
        )

        # RAG 결과를 test_rag.json 형식으로 변환 (엔티티/관계 정보 보존)
        rag_results = self._convert_rag_results(raw_rag_results)

        # 2. 교수별 집계
        print("교수별 문서 집계 중...")
        aggregator = ProfessorAggregator()
        professor_data = aggregator.aggregate_by_professor(
            rag_results=rag_results,
            doc_types=doc_types
        )
        
        # 3. AHP 랭킹
        print("AHP 기반 교수 순위 평가 중...")
        ranker = ProfessorRanker()
        ranked_professors = ranker.rank_professors(professor_data, DEFAULT_TYPE_WEIGHTS)
        
        # 4. AHP 결과 형식으로 변환
        ahp_results = {
            "query": query,
            "keywords": rag_results.get("keywords", {}),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_professors": len(ranked_professors),
            "type_weights": DEFAULT_TYPE_WEIGHTS,
            "ranked_professors": ranked_professors
        }
        
        # 5. 리포트 생성
        result = self.generate_report(
            ahp_results=ahp_results,
            rag_results=rag_results,
            few_shot_examples=few_shot_examples
        )

        # 비용 추적 종료
        cost_result = tracker.end_task()
        if cost_result:
            result["api_cost"] = cost_result

        return result
    
    def generate_report(
        self,
        ahp_results: Dict[str, Any],
        rag_results: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        AHP 결과를 기반으로 리포트 생성
        
        Args:
            ahp_results: AHP 결과 JSON (ahp_results_*.json 파일 내용)
            rag_results: RAG 검색 결과 (엔티티/관계 정보 추출용, None이면 ahp_results에서 추출 시도)
            few_shot_examples: 보고서 생성용 Few-shot 예시 리스트
                - 형식: [{"input": {...}, "output": "..."}, ...]
                - 또는: {"examples": [{"input": {...}, "output": "..."}]}
                - 예시 파일: data/report_few_shot_examples.json
                - None이면 기본 프롬프트만 사용
            
        Returns:
            생성된 리포트 데이터 딕셔너리
        """
        # 입력 JSON 준비
        input_json = self._prepare_input_json(ahp_results, rag_results)
        
        # 프롬프트 생성
        prompt = self._build_prompt(input_json, few_shot_examples)
        
        # GPT-4o-mini 호출
        print("GPT-4o-mini를 사용하여 리포트 생성 중...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 문서 기반 교수 추천 보고서를 생성하는 보고서 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # 비용 추적
        log_chat_usage(
            component="report_generation",
            model=self.model,
            response=response
        )

        report_text = response.choices[0].message.content
        
        # 결과 구조화
        report_data = {
            "query": ahp_results.get("query", ""),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "report_text": report_text,
            "input_data": input_json,
            "model": self.model
        }
        
        return report_data
    
    def _convert_rag_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        HybridRetriever 결과를 test_rag.json 형식으로 변환
        (query.py의 save_query_result 로직과 동일)

        Args:
            raw_results: HybridRetriever.retrieve() 결과

        Returns:
            test_rag.json 형식의 딕셔너리
        """
        docs_dict = {}

        # local_results 처리
        for r in raw_results.get('local_results', []):
            no = str(r.get('metadata', {}).get('source_doc_id', ''))
            if not no:
                continue

            doc_type = r.get('doc_type', 'unknown')

            if no not in docs_dict:
                docs_dict[no] = {
                    "no": no,
                    "data_type": doc_type,
                    "matches": []
                }

            match_info = {
                "search_type": "local",
                "similarity": r.get('similarity', 0),
                "matched_entity": {
                    "name": r.get('metadata', {}).get('name', ''),
                    "entity_type": r.get('metadata', {}).get('entity_type', ''),
                    "description": r.get('document', '')
                },
                "neighbors_1hop": [
                    {
                        "name": n.get('name', ''),
                        "entity_type": n.get('entity_type', ''),
                        "relation_keywords": n.get('relation_keywords', []),
                        "relation_description": n.get('relation_description', '')
                    }
                    for n in r.get('neighbors', [])
                ]
            }
            docs_dict[no]["matches"].append(match_info)

        # global_results 처리
        for r in raw_results.get('global_results', []):
            no = str(r.get('metadata', {}).get('source_doc_id', ''))
            if not no:
                continue

            doc_type = r.get('doc_type', 'unknown')

            if no not in docs_dict:
                docs_dict[no] = {
                    "no": no,
                    "data_type": doc_type,
                    "matches": []
                }

            match_info = {
                "search_type": "global",
                "similarity": r.get('similarity', 0),
                "matched_relation": {
                    "source_entity": r.get('metadata', {}).get('source_entity', ''),
                    "target_entity": r.get('metadata', {}).get('target_entity', ''),
                    "keywords": r.get('metadata', {}).get('keywords', ''),
                    "description": r.get('document', '')
                },
                "source_entity_info": r.get('source_entity_info'),
                "target_entity_info": r.get('target_entity_info')
            }
            docs_dict[no]["matches"].append(match_info)

        # matches 내부 similarity 기준 정렬
        for doc in docs_dict.values():
            doc['matches'] = sorted(
                doc['matches'],
                key=lambda m: m.get('similarity', 0),
                reverse=True
            )

        # dict → list 변환 후 정렬
        retrieved_docs = sorted(
            docs_dict.values(),
            key=lambda doc: max((m.get('similarity', 0) for m in doc['matches']), default=0),
            reverse=True
        )

        return {
            "query": raw_results.get('query', ''),
            "keywords": {
                "high_level": raw_results.get('high_level_keywords', []),
                "low_level": raw_results.get('low_level_keywords', [])
            },
            "retrieved_docs": retrieved_docs
        }

    def _prepare_input_json(
        self,
        ahp_results: Dict[str, Any],
        rag_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        리포트 생성을 위한 입력 JSON 준비
        
        Args:
            ahp_results: AHP 결과
            rag_results: RAG 결과 (None이면 엔티티/관계 정보 없이 생성)
            
        Returns:
            리포트 생성용 입력 JSON
        """
        query = ahp_results.get("query", "")
        keywords = ahp_results.get("keywords", {})
        
        # 키워드 정보 준비 (high_level, low_level 활용)
        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])
        
        # 엔티티와 관계 추출 (문서별 엔티티/관계는 유지, 전체 추출은 키워드로 대체)
        extracted_relationships = []
        
        if rag_results:
            # RAG 결과에서 관계만 추출 (엔티티는 키워드로 대체)
            retrieved_docs = rag_results.get("retrieved_docs", [])
            relation_set = set()
            
            for doc in retrieved_docs:
                matches = doc.get("matches", [])
                for match in matches:
                    # 관계 추출
                    relation = match.get("matched_relation", {})
                    if relation:
                        relation_key = f"{relation.get('source_entity', '')} -> {relation.get('target_entity', '')}"
                        if relation_key not in relation_set:
                            extracted_relationships.append({
                                "source": relation.get("source_entity", ""),
                                "target": relation.get("target_entity", ""),
                                "description": relation.get("description", ""),
                                "keywords": relation.get("keywords", "")
                            })
                            relation_set.add(relation_key)
        
        # 교수별 문서 정보 준비 (최대 3명)
        professors_data = []
        ranked_professors = ahp_results.get("ranked_professors", [])[:3]  # 상위 3명만
        
        for prof in ranked_professors:
            prof_info = prof.get("professor_info", {})
            documents = prof.get("documents", {})
            document_scores = prof.get("document_scores", {})
            scores_by_type = prof.get("scores_by_type", {})
            total_score = prof.get("total_score", 0.0)
            
            prof_docs = []
            
            # 각 문서 타입별로 문서 정보 수집 (최대 3개씩)
            for doc_type in ["patent", "article", "project"]:
                docs = documents.get(doc_type, [])
                doc_scores_list = document_scores.get(doc_type, [])
                
                # 문서 점수 정보를 딕셔너리로 변환 (빠른 조회용)
                score_dict = {}
                for doc_score in doc_scores_list:
                    doc_no = str(doc_score.get("no", ""))
                    score_dict[doc_no] = doc_score.get("score", 0.0)
                
                # 점수 기준으로 정렬 (높은 점수 순)
                docs_with_scores = []
                for doc in docs:
                    doc_no = str(doc.get("no", ""))
                    score = score_dict.get(doc_no, 0.0)
                    docs_with_scores.append((doc, score))
                
                # 점수 기준 내림차순 정렬 후 상위 3개만 선택
                docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                selected_docs = docs_with_scores[:3]
                
                for doc, score in selected_docs:
                    # 문서 요약 (text 필드의 일부)
                    text = doc.get("text", "")
                    summary = text[:200] + "..." if len(text) > 200 else text
                    
                    # 엔티티와 관계 추출 (해당 문서의 matches에서)
                    doc_entities = []
                    doc_relations = []
                    
                    # 문서 번호로 RAG 결과에서 매칭 정보 찾기
                    doc_no = str(doc.get("no", ""))
                    if rag_results:
                        retrieved_docs = rag_results.get("retrieved_docs", [])
                        for rag_doc in retrieved_docs:
                            # 문서 번호 비교 (문자열로 통일)
                            rag_doc_no = str(rag_doc.get("no", ""))
                            if rag_doc_no == doc_no:
                                matches = rag_doc.get("matches", [])
                                for match in matches:
                                    # source_entity_info와 target_entity_info에서 엔티티 추출
                                    source_info = match.get("source_entity_info", {})
                                    target_info = match.get("target_entity_info", {})
                                    
                                    if source_info and source_info.get("name"):
                                        doc_entities.append(source_info["name"])
                                    if target_info and target_info.get("name"):
                                        doc_entities.append(target_info["name"])
                                    
                                    # matched_relation에서 관계 추출
                                    relation = match.get("matched_relation", {})
                                    if relation:
                                        source_entity = relation.get("source_entity", "")
                                        target_entity = relation.get("target_entity", "")
                                        if source_entity and target_entity:
                                            doc_relations.append(f"{source_entity} -> {target_entity}")
                                break
                    
                    prof_docs.append({
                        "type": doc_type,
                        "title": doc.get("title", ""),
                        "summary": summary,
                        "year": doc.get("year", ""),
                        "score": score,  # AHP 점수 추가
                        "entities": list(set(doc_entities)),
                        "relationships": list(set(doc_relations))
                    })
            
            professors_data.append({
                "name": prof_info.get("NM", ""),
                "department": f"{prof_info.get('COLG_NM', '')} {prof_info.get('HG_NM', '')}".strip(),
                "total_score": total_score,  # 교수 종합 점수
                "scores_by_type": scores_by_type,  # 타입별 점수
                "documents": prof_docs
            })
        
        input_json = {
            "query": query,
            "keywords": {
                "high_level": high_level_keywords,
                "low_level": low_level_keywords
            },
            "extracted_relationships": extracted_relationships,
            "professors": professors_data
        }
        
        return input_json
    
    def _build_prompt(
        self,
        input_json: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        리포트 생성을 위한 프롬프트 빌드
        
        Args:
            input_json: 입력 JSON 데이터
            few_shot_examples: 보고서 생성용 Few-shot 예시 리스트
                - 각 예시는 {"input": {...}, "output": "..."} 형식
                - 보고서 생성 형식을 학습하기 위한 예시 데이터
                - 참고 파일: data/report_few_shot_examples.json
            
        Returns:
            완성된 프롬프트 문자열
        """
        base_prompt = """당신은 문서 기반 교수 추천 보고서를 생성하는 보고서 요약 전문가입니다.
당신의 역할은 사용자의 검색 질의와 추천된 교수 및 관련 문서 데이터를 바탕으로,
아래에 명시된 보고서 형식을 그대로 따라 구조화된 보고서를 작성하는 것입니다.

반드시 다음 지침을 따르세요:
- 입력 JSON 정보만을 기반으로 작성하며, 외부 지식이나 해석을 추가하지 마세요.
- 새로운 정보를 생성하거나 추론하지 마세요.
- 주관적인 평가, 추천 문장, 해석 표현을 포함하지 마세요.
- 표와 문장은 간결하고 명확하게 작성하세요.
- 마크다운 형식으로 작성하세요 (Streamlit에서 표시됩니다).
- 교수는 최대 3명까지만 표시하세요 (이미 입력 JSON에 상위 3명만 포함되어 있습니다).
- 각 교수의 문서는 데이터 유형(patent/article/project)별로 최대 3개씩만 표시하세요 (이미 입력 JSON에 상위 3개씩만 포함되어 있습니다).

---

### [📄 보고서 출력 형식 지침]

## 1. 사용자 검색 정보

**입력 질의:** 입력 JSON의 "query" 필드 값을 그대로 사용하세요.

**고수준 키워드:** 입력 JSON의 "keywords.high_level" 배열을 쉼표로 구분하여 나열하세요. 예: "딥러닝 의료영상 분석, 의료영상 처리 시스템 개발"

**저수준 키워드:** 입력 JSON의 "keywords.low_level" 배열을 쉼표로 구분하여 나열하세요. 예: "딥러닝, 의료영상, 영상 분석, 시스템"

**추출 관계:** 입력 JSON의 "extracted_relationships" 배열에서 각 관계의 "source"와 "target"을 "A -> B" 형식으로 나열하세요. 
- 예: "자연스러운 화질 복원 -> 베이어 디모자이크 방법, 베이어 디모자이크 방법 -> 베이어 CFA 패턴"
- 배열이 비어있거나 없으면 "없음" 또는 생략하세요.

## 2. 교수별 관련 문서 목록

(아래 형식을 교수 1명당 반복해서 작성하세요. 최대 3명까지만)

입력 JSON의 "professors" 배열에서 각 교수 정보를 사용하세요:

### 교수명: [professors[].name]
**소속:** [professors[].department]  
**종합 점수:** [professors[].total_score] (AHP 종합 점수)  
**타입별 점수:** 특허=[professors[].scores_by_type.patent], 논문=[professors[].scores_by_type.article], 연구과제=[professors[].scores_by_type.project]

| 문서 유형 | 제목 | 연도 | 요약 | AHP점수 | 개체 | 관계 |
|-----------|------|------|------|---------|------|------|
| [documents[].type] | [documents[].title] | [documents[].year] | [documents[].summary] | [documents[].score] | [documents[].entities] | [documents[].relationships] |

**주의사항:**
- 각 교수마다 patent, article, project 각각 최대 3개씩만 표시하세요.
- 문서는 AHP 점수가 높은 순서대로 이미 정렬되어 있습니다.
- 개체(entities)는 해당 문서의 "entities" 배열을 쉼표로 구분하여 나열하세요. 비어있으면 빈 칸.
- 관계(relationships)는 해당 문서의 "relationships" 배열을 쉼표로 구분하여 나열하세요. 비어있으면 빈 칸.

## 3. 보고서 설명

모든 정보는 입력된 문서 요약 및 구조화된 정보에서 추출되었으며, 주관적인 해석이나 판단은 포함되지 않았습니다.
교수는 AHP 점수 기준 상위 3명만 포함되었으며, 각 교수의 문서는 데이터 유형별로 점수가 높은 상위 3개씩만 포함되었습니다.

---
"""
        
        # Few-shot 예시 추가
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples, 1):
                example_input = example.get("input", {})
                example_output = example.get("output", "")
                
                base_prompt += f"""
### [✅ Few-shot 예시 {i}]

입력 JSON:
{json.dumps(example_input, ensure_ascii=False, indent=2)}

출력 보고서:
{example_output}

---
"""
        
        # 최종 입력 JSON 추가
        base_prompt += f"""
### [🧾 새롭게 작성해야 할 보고서 대상 JSON]

다음 JSON 데이터를 위와 동일한 형식(예시 1~2 참조)에 따라 구조화된 보고서로 작성하세요:

{json.dumps(input_json, ensure_ascii=False, indent=2)}
"""
        
        return base_prompt
    
    def save_json(
        self,
        report_data: Dict[str, Any],
        filename: str = None
    ) -> Path:
        """
        JSON 형식으로 보고서 저장
        
        Args:
            report_data: 보고서 데이터
            filename: 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"report_{timestamp}.json"
        
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def save_text(
        self,
        report_data: Dict[str, Any],
        filename: str = None
    ) -> Path:
        """
        텍스트 형식으로 보고서 저장
        
        Args:
            report_data: 보고서 데이터
            filename: 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"report_{timestamp}.txt"
        
        file_path = self.output_dir / filename
        
        report_text = report_data.get("report_text", "")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return file_path
