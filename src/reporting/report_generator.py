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
            output_dir: 보고서 출력 디렉토리 (None이면 results/runs/report 사용)
            api_key: OpenAI API 키 (None이면 config에서 가져옴)
        """
        if output_dir is None:
            self.output_dir = Path("results/runs/report")
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
        
        # 4. AHP 결과 형식으로 변환 (한 번의 실행에서 RAG/AHP/REPORT 로그용 동일 타임스탬프 사용)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ahp_results = {
            "query": query,
            "keywords": rag_results.get("keywords", {}),
            "timestamp": run_ts,
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
        result["timestamp"] = run_ts
        result["rag_results"] = rag_results
        result["ahp_results"] = ahp_results

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
        리포트 생성을 위한 입력 JSON 준비.

        보고서에 표시되는 항목 출처:
        - query, keywords.high_level/low_level: RAG 검색 결과(쿼리·키워드 추출)
        - extracted_relationships: RAG retrieved_docs의 matches에서 추출
          (global: matched_relation, local: matched_entity + neighbors_1hop). 상위 N개만 사용.
        - professors: AHP ranked_professors 상위 3명
        - 각 교수 documents: AHP documents(patent/article/project) 유형별 상위 3개,
          entities/relationships: 해당 문서 no와 일치하는 RAG matches에서만 추출. 문서당 상위 N개만.

        Args:
            ahp_results: AHP 결과
            rag_results: RAG 결과 (None이면 엔티티/관계 정보 없이 생성)

        Returns:
            리포트 생성용 입력 JSON
        """
        # 보고서 항목 개수 제한 (무분별하게 많아지지 않도록)
        MAX_EXTRACTED_RELATIONSHIPS = 25
        MAX_ENTITIES_PER_DOC = 10
        MAX_RELATIONSHIPS_PER_DOC = 10

        query = ahp_results.get("query", "")
        keywords = ahp_results.get("keywords", {})

        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])

        # RAG 결과에서 추출된 관계 (상위 N개만)
        extracted_relationships = []
        if rag_results:
            retrieved_docs = rag_results.get("retrieved_docs", [])
            relation_set = set()
            for doc in retrieved_docs:
                for match in doc.get("matches", []):
                    if len(extracted_relationships) >= MAX_EXTRACTED_RELATIONSHIPS:
                        break
                    rel = match.get("matched_relation", {})
                    if rel:
                        relation_key = f"{rel.get('source_entity', '')} -> {rel.get('target_entity', '')}"
                        if relation_key not in relation_set:
                            extracted_relationships.append({
                                "source": rel.get("source_entity", ""),
                                "target": rel.get("target_entity", ""),
                                "description": rel.get("description", ""),
                                "keywords": rel.get("keywords", "")
                            })
                            relation_set.add(relation_key)
                    ent = match.get("matched_entity", {}) or {}
                    if ent.get("name"):
                        for n in match.get("neighbors_1hop", []):
                            if len(extracted_relationships) >= MAX_EXTRACTED_RELATIONSHIPS:
                                break
                            nname = n.get("name", "")
                            if nname:
                                relation_key = f"{ent['name']} -> {nname}"
                                if relation_key not in relation_set:
                                    extracted_relationships.append({
                                        "source": ent["name"],
                                        "target": nname,
                                        "description": n.get("relation_description", ""),
                                        "keywords": ", ".join(n.get("relation_keywords", []))
                                    })
                                    relation_set.add(relation_key)
                if len(extracted_relationships) >= MAX_EXTRACTED_RELATIONSHIPS:
                    break

        # 교수별 문서 정보 준비 (최대 3명)
        professors_data = []
        ranked_professors = ahp_results.get("ranked_professors", [])[:3]
        
        for idx, prof in enumerate(ranked_professors, 1):
            prof_info = prof.get("professor_info", {})
            documents = prof.get("documents", {})
            document_scores = prof.get("document_scores", {})
            
            prof_docs = []
            
            for doc_type in ["patent", "article", "project"]:
                docs = documents.get(doc_type, [])
                doc_scores_list = document_scores.get(doc_type, [])
                score_dict = {str(ds.get("no", "")): ds.get("score", 0.0) for ds in doc_scores_list}
                docs_with_scores = [(doc, score_dict.get(str(doc.get("no", "")), 0.0)) for doc in docs]
                docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                selected_docs = docs_with_scores[:3]
                
                for doc, _ in selected_docs:
                    text = doc.get("text", "")
                    summary = text[:200] + "..." if len(text) > 200 else text
                    doc_no = str(doc.get("no", ""))
                    
                    # 엔티티/관계: RAG retrieved_docs의 matches에서 추출 (global + local 모두)
                    doc_entities = []
                    doc_relations = []
                    if rag_results:
                        for rag_doc in rag_results.get("retrieved_docs", []):
                            if str(rag_doc.get("no", "")) != doc_no:
                                continue
                            for match in rag_doc.get("matches", []):
                                rel = match.get("matched_relation", {})
                                if rel:
                                    s, t = rel.get("source_entity", ""), rel.get("target_entity", "")
                                    if s:
                                        doc_entities.append(s)
                                    if t:
                                        doc_entities.append(t)
                                    if s and t:
                                        doc_relations.append(f"{s} -> {t}")
                                for info in (match.get("source_entity_info"), match.get("target_entity_info")):
                                    if info and info.get("name"):
                                        doc_entities.append(info["name"])
                                ent = match.get("matched_entity", {}) or {}
                                if ent.get("name"):
                                    doc_entities.append(ent["name"])
                                    for n in match.get("neighbors_1hop", []):
                                        nname = n.get("name", "")
                                        if nname:
                                            doc_entities.append(nname)
                                            doc_relations.append(f"{ent['name']} -> {nname}")
                            break

                    prof_docs.append({
                        "type": doc_type,
                        "title": doc.get("title", ""),
                        "summary": summary,
                        "year": doc.get("year", ""),
                        "entities": list(set(doc_entities))[:MAX_ENTITIES_PER_DOC],
                        "relationships": list(set(doc_relations))[:MAX_RELATIONSHIPS_PER_DOC]
                    })
            
            professors_data.append({
                "number": idx,
                "name": prof_info.get("NM", ""),
                "department": f"{prof_info.get('COLG_NM', '')} {prof_info.get('HG_NM', '')}".strip(),
                "contact": prof_info.get("EMAIL", ""),
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
        base_prompt = """당신은 산학협력 매칭을 위한 **공식 추천 보고서**를 작성하는 전문가입니다.
입력된 검색 질의와 추천 교수·문서 데이터만 사용하여, 아래 형식에 맞춰 **정돈된 보고서**를 작성하세요.

[지침]
- 입력 JSON의 값만 사용하고, 추론·해석·평가 문장을 넣지 마세요.
- 교수는 반드시 "1. OOO 교수", "2. OOO 교수", "3. OOO 교수" 형식으로 번호와 함께 표기하세요.
- AHP 점수·종합 점수는 보고서에 포함하지 마세요.
- 마크다운으로 작성하고, 표는 반드시 파이프(|) 테이블 형식을 사용하세요.
- 각 교수당 문서는 patent/article/project 유형별 최대 3개씩만 표시하세요.

---

### [보고서 출력 형식]

보고서 **맨 위**에 다음 제목 블록을 넣으세요 (날짜는 보고서 생성일로 비슷하게):

---
# 산학 매칭 추천 보고서

**작성일:** (현재 연도-월-일 형식)  
**검색 질의:** (입력 JSON의 query 값)
---

그 다음 아래 섹션을 **순서대로** 작성하세요.

---

## 1. 검색 개요

- **고수준 키워드:** (keywords.high_level 배열을 쉼표로 나열)
- **저수준 키워드:** (keywords.low_level 배열을 쉼표로 나열)
- **추출된 개체·관계:** extracted_relationships에서 "source -> target" 형식으로 나열. 없으면 "해당 없음"

---

## 2. 추천 교수 및 관련 문서

professors 배열을 순서대로 사용하세요. 각 교수 블록은 아래와 같이 작성하세요.

### 1. [이름] 교수
- **소속:** (department)
- **연락 수단:** (contact 이메일, 없으면 "-")

| 문서 유형 | 제목 | 연도 | 요약 | 개체 | 관계 |
|:----------|------|:----:|------|------|------|
| (type) | (title) | (year) | (summary 일부) | (entities 쉼표 구분) | (relationships 쉼표 구분) |

(2번, 3번 교수도 동일 형식으로 반복)

---

## 3. 안내

본 보고서의 내용은 검색된 문서 요약 및 추출된 개체·관계를 바탕으로 구성되었습니다.

---

## 4. 산학협력단 문의

산학협력 관련 문의는 아래 연락처로 이용하시기 바랍니다.

| 구분 | 내용 |
|------|------|
| 담당자 | 김OO |
| 이메일 | oo@inu.ac.kr |
| 연락처 | 032-835-0000 |

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

    def save_pdf(
        self,
        report_data: Dict[str, Any],
        filename: str = None
    ) -> Optional[Path]:
        """
        PDF 형식으로 보고서 저장 (fpdf2 사용, GTK 불필요).
        
        Returns:
            저장된 파일 경로. 실패 시 None.
        """
        if filename is None:
            timestamp = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"report_{timestamp}.pdf"
        file_path = self.output_dir / filename
        report_text = report_data.get("report_text", "")
        return self._save_pdf_fpdf2(file_path, report_text)

    @staticmethod
    def _emoji_to_text(s: str) -> str:
        """PDF에서 이모티콘이 깨지지 않도록 흔한 이모티콘을 [텍스트]로 치환."""
        if not s:
            return s
        replace_map = {
            "✅": "[체크]", "✓": "[체크]", "✔": "[체크]",
            "❌": "[오류]", "✗": "[X]",
            "⚠️": "[주의]", "⚠": "[주의]",
            "📋": "[보고서]", "📄": "[문서]", "📁": "[폴더]",
            "🔍": "[검색]", "📥": "[다운로드]", "🚀": "[실행]",
            "📌": "[핀]", "💡": "[아이디어]", "📊": "[차트]",
            "👉": "[참고]", "•": "·", "–": "-", "—": "-",
        }
        out = s
        for emoji, text in replace_map.items():
            out = out.replace(emoji, text)
        # 나머지 이모티콘 범위(대략)는 공백 또는 [?]로 (선택)
        return out

    @staticmethod
    def _parse_md_blocks(lines: List[str]) -> List[Dict[str, Any]]:
        """마크다운 줄을 일반 줄 / 표 블록으로 나눔. 표는 |...| 형태 연속 줄."""
        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                blocks.append({"type": "line", "content": ""})
                i += 1
                continue
            # 표 행: 맨 앞이 | 이고 중간에 | 가 있음
            if line.strip().startswith("|") and line.count("|") >= 2:
                table_rows: List[List[str]] = []
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].count("|") >= 2:
                    row_line = lines[i]
                    parts = [p.strip() for p in row_line.split("|")]
                    if len(parts) > 2:
                        cells = parts[1:-1]
                    else:
                        cells = [p for p in parts if p]
                    if not cells:
                        i += 1
                        continue
                    is_sep = all(all(ch in " \t:-" for ch in cell) for cell in cells)
                    if not is_sep:
                        table_rows.append(cells)
                    i += 1
                if table_rows:
                    blocks.append({"type": "table", "rows": table_rows})
                continue
            blocks.append({"type": "line", "content": line})
            i += 1
        return blocks

    def _save_pdf_fpdf2(self, file_path: Path, report_text: str) -> Optional[Path]:
        """fpdf2로 PDF 생성. 마크다운(# 헤딩, **굵게), 표(테이블), 이모티콘→텍스트 반영."""
        try:
            from fpdf import FPDF  # type: ignore[import-untyped]
        except ImportError:
            print("PDF 저장을 위해 pip install fpdf2 를 실행해주세요.")
            return None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_margins(20, 15, 20)
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        if usable_w <= 0:
            pdf.set_margins(10, 10, 10)
            usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_xy(pdf.l_margin, pdf.t_margin)

        # 한글 폰트: 맑은 고딕 (일반 + 볼드). 마크다운 ** 굵게용.
        font_added = False
        font_has_bold = False
        for (reg, bold) in [
            ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
            (Path.home() / "AppData/Local/Microsoft/Windows/Fonts/malgun.ttf", Path.home() / "AppData/Local/Microsoft/Windows/Fonts/malgunbd.ttf"),
        ]:
            pr = Path(reg) if isinstance(reg, str) else reg
            pb = Path(bold) if isinstance(bold, str) else bold
            if pr.exists():
                try:
                    pdf.add_font("Malgun", "", str(pr))
                    if pb.exists():
                        pdf.add_font("Malgun", "B", str(pb))
                        font_has_bold = True
                    pdf.set_font("Malgun", "", 10)
                    font_added = True
                    break
                except Exception:
                    continue

        # 이모티콘 폴백 폰트 (가능하면 사용, 실패 시 _emoji_to_text로 치환)
        for emoji_path in [
            "C:/Windows/Fonts/seguiemj.ttf",
            Path.home() / "AppData/Local/Microsoft/Windows/Fonts/seguiemj.ttf",
        ]:
            pe = Path(emoji_path) if isinstance(emoji_path, str) else emoji_path
            if pe.exists():
                try:
                    pdf.add_font("SegoeEmoji", "", str(pe))
                    pdf.set_fallback_fonts(["segoeemoji"], exact_match=False)
                    break
                except Exception:
                    pass

        if not font_added:
            pdf.set_font("Helvetica", "", 10)

        cell_w = max(usable_w * 0.95, 50.0)
        raw_lines = [ln.rstrip() for ln in report_text.replace("\r", "").split("\n")]
        blocks = self._parse_md_blocks(raw_lines)

        def get_heading_style(line: str):
            """줄 앞 # 개수에 따라 (폰트 크기, 줄높이, 제거할 문자 수, 볼드 여부)."""
            if line.startswith("### "):
                return (11, 7.5, 4, True)
            if line.startswith("## "):
                return (13, 8.0, 3, True)
            if line.startswith("# "):
                return (16, 9.0, 2, True)
            return (10, 6.5, 0, False)

        def render_blocks(pdf_obj, use_helvetica_only: bool):
            base_size = 10
            if use_helvetica_only:
                pdf_obj.set_font("Helvetica", "", base_size)
            elif font_added:
                pdf_obj.set_font("Malgun", "", base_size)
            markdown_ok = use_helvetica_only or (font_added and font_has_bold)

            for blk in blocks:
                if blk["type"] == "line":
                    line = blk["content"]
                    if not line:
                        pdf_obj.ln(6)
                        continue
                    pdf_obj.set_x(pdf_obj.l_margin)

                    size, line_h, strip_len, is_heading = get_heading_style(line)
                    content = line[strip_len:].lstrip() if strip_len else line
                    content = self._emoji_to_text(content)
                    txt = (content.encode("latin-1", errors="replace").decode("latin-1") + "\n") if use_helvetica_only else (content + "\n")

                    if is_heading:
                        pdf_obj.ln(2)
                    if size != base_size or (is_heading and font_has_bold and not use_helvetica_only):
                        if use_helvetica_only:
                            pdf_obj.set_font("Helvetica", "B" if is_heading else "", size)
                        elif font_added:
                            pdf_obj.set_font("Malgun", "B" if (is_heading and font_has_bold) else "", size)

                    try:
                        pdf_obj.multi_cell(w=cell_w, h=line_h, txt=txt, new_x="LMARGIN", new_y="NEXT", markdown=markdown_ok)
                    except Exception:
                        pdf_obj.set_x(pdf_obj.l_margin)
                        pdf_obj.set_font("Helvetica", "B" if is_heading else "", size)
                        safe_txt = content.encode("latin-1", errors="replace").decode("latin-1") + "\n"
                        pdf_obj.multi_cell(w=cell_w, h=line_h, txt=safe_txt, new_x="LMARGIN", new_y="NEXT", markdown=False)
                        if font_added and not use_helvetica_only:
                            pdf_obj.set_font("Malgun", "B" if (is_heading and font_has_bold) else "", size)

                    if is_heading:
                        pdf_obj.ln(2)
                    if size != base_size or (is_heading and font_has_bold and not use_helvetica_only):
                        if use_helvetica_only:
                            pdf_obj.set_font("Helvetica", "", base_size)
                        elif font_added:
                            pdf_obj.set_font("Malgun", "", base_size)

                elif blk["type"] == "table":
                    rows = blk["rows"]
                    if not rows:
                        continue
                    ncols = max(len(r) for r in rows)
                    padded = [list(r) + [""] * (ncols - len(r)) for r in rows]
                    pdf_obj.set_x(pdf_obj.l_margin)
                    pdf_obj.ln(3)
                    try:
                        with pdf_obj.table(
                            width=cell_w,
                            first_row_as_headings=True,
                            markdown=False,
                            line_height=6.5,
                            padding=3,
                            num_heading_rows=1,
                        ) as table:
                            for row in padded:
                                cells = [self._emoji_to_text(str(c)) for c in row]
                                if use_helvetica_only:
                                    cells = [c.encode("latin-1", errors="replace").decode("latin-1") for c in cells]
                                table.row(cells=cells)
                    except Exception:
                        # 테이블 실패 시 일반 텍스트로 fallback
                        for row in padded:
                            fallback_line = " | ".join(self._emoji_to_text(str(c)) for c in row)
                            pdf_obj.set_x(pdf_obj.l_margin)
                            pdf_obj.multi_cell(w=cell_w, h=6, txt=fallback_line + "\n", new_x="LMARGIN", new_y="NEXT", markdown=False)
                    pdf_obj.ln(3)

        try:
            render_blocks(pdf, use_helvetica_only=False)
        except Exception as e:
            if "horizontal space" in str(e).lower():
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_margins(20, 15, 20)
                pdf.set_xy(pdf.l_margin, pdf.t_margin)
                pdf.set_font("Helvetica", "", 10)
                render_blocks(pdf, use_helvetica_only=True)
            else:
                raise

        try:
            pdf.output(str(file_path))
            return file_path
        except Exception as e:
            print(f"fpdf2 PDF 저장 실패: {e}")
            return None
