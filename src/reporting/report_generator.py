"""
Report Generator
AHP 결과를 기반으로 GPT-4o-mini를 사용하여 리포트 생성
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL
from src.utils.cost_tracker import log_chat_usage, get_cost_tracker


def _escape_html(s: str) -> str:
    """HTML 이스케이프 (fallback용)."""
    if not s:
        return ""
    import html as _html
    return _html.escape(s)


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
        
        # GPT-4o-mini 호출 (속도: max_tokens 제한, temperature 낮춤)
        print("GPT-4o-mini를 사용하여 리포트 생성 중...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 문서 기반 교수 추천 보고서를 생성하는 보고서 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
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
        HybridRetriever 결과를 test_rag.json 형식으로 변환.
        merged_results만 사용 (이미 문서당 1건, similarity_threshold 이상).

        Args:
            raw_results: HybridRetriever.retrieve() 결과 (merged_results 사용)

        Returns:
            test_rag.json 형식의 딕셔너리
        """
        docs_dict = {}
        for r in raw_results.get('merged_results', []):
            no = str(r.get('metadata', {}).get('source_doc_id', ''))
            if not no:
                continue
            doc_type = r.get('doc_type', 'unknown')
            meta = r.get('metadata', {})

            if meta.get('name') is not None:
                match_info = {
                    "search_type": "local",
                    "similarity": r.get('similarity', 0),
                    "matched_entity": {
                        "name": meta.get('name', ''),
                        "entity_type": meta.get('entity_type', ''),
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
            else:
                match_info = {
                    "search_type": "global",
                    "similarity": r.get('similarity', 0),
                    "matched_relation": {
                        "source_entity": meta.get('source_entity', ''),
                        "target_entity": meta.get('target_entity', ''),
                        "keywords": meta.get('keywords', ''),
                        "description": r.get('document', '')
                    },
                    "source_entity_info": r.get('source_entity_info'),
                    "target_entity_info": r.get('target_entity_info')
                }

            docs_dict[(no, doc_type)] = {
                "no": no,
                "data_type": doc_type,
                "matches": [match_info]
            }

        retrieved_docs = sorted(
            docs_dict.values(),
            key=lambda doc: doc['matches'][0].get('similarity', 0) if doc.get('matches') else 0,
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
        - query, keywords.high_level/low_level: RAG 검색 결과(1차/2차 검색 키워드)
        - professors: AHP ranked_professors 상위 3명
        - 각 교수 documents: AHP documents(patent/article/project) 유형별 상위 3개, type/title/summary/year만 사용.

        Args:
            ahp_results: AHP 결과
            rag_results: RAG 결과 (키워드·retrieved_docs용, 현재 보고서 템플릿에서는 개체/관계 미사용)

        Returns:
            리포트 생성용 입력 JSON
        """
        query = ahp_results.get("query", "")
        keywords = ahp_results.get("keywords", {})

        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])

        # 문서 유형 → 한국어 표기 (보고서에서 논문/특허/연구 과제 별로 구분·표기용)
        DOC_TYPE_KO = {"article": "논문", "patent": "특허", "project": "연구 과제"}

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
                type_ko = DOC_TYPE_KO.get(doc_type, doc_type)
                
                for doc, _ in selected_docs:
                    text = doc.get("text", "")
                    summary = text[:200] + "..." if len(text) > 200 else text
                    prof_docs.append({
                        "type": doc_type,
                        "type_ko": type_ko,
                        "title": doc.get("title", ""),
                        "summary": summary,
                        "year": doc.get("year", ""),
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
입력된 검색 질의와 추천 교수·문서 데이터만 사용하여, **Word 문서처럼 구조화·가독성 높은 보고서**를 작성하세요.

[지침]
- 입력 JSON의 값만 사용하고, 추론·해석·평가 문장을 넣지 마세요.
- **마크다운 활용**: 제목은 #(대제목), ##(섹션), ###(소제목)으로 계층을 나누고, **굵게**는 **키워드**처럼 반드시 사용하세요.
- **강조**: "사용자 검색어", "1차 검색 키워드", "2차 검색 키워드", "소속", "이메일", "문서 유형", "제목", "연도" 등 라벨은 **굵게** 처리하세요.
- **이모티콘**: 섹션 구분을 위해 각 섹션 제목 앞에 이모티콘을 하나씩 넣으세요. 예: 📋 제목, 🔍 검색 개요, 👤 추천 교수, 📌 유의사항 및 문의
- 교수는 반드시 "1. OOO 교수", "2. OOO 교수", "3. OOO 교수" 형식으로 번호와 함께 표기하세요.
- AHP 점수·종합 점수는 보고서에 포함하지 마세요.
- **관련 문서**: 반드시 **2단계 불릿**으로만 작성하세요. 1단계 불릿에는 유형(**논문**, **특허**, **연구 과제**)만 쓰고, 그 아래 2단계 불릿(들여쓰기)에 실제 문서를 `**[제목]** (연도): 요약` 형식으로 나열하세요. 유형은 한국어로만 표기하고, 각 문서 요약은 사용자 검색어와 관련지어 한두 문장으로 하세요.

---

### [보고서 출력 형식]

보고서 **맨 위**에 다음 제목 블록을 넣으세요 (제목·사용자 검색어는 본문보다 한 단계 작게):

---
# 📋 산학 매칭 추천 보고서

**사용자 검색어:** (입력 JSON의 query 값)
---

그 다음 아래 섹션을 **순서대로** 작성하세요. 각 섹션 제목 앞에 이모티콘을 붙이고, 라벨은 **굵게** 처리하세요.

---

### 🔍 사용자 검색어 (검색 개요)

- **1차 검색 키워드:** (keywords.high_level 배열을 쉼표로 나열)
- **2차 검색 키워드:** (keywords.low_level 배열을 쉼표로 나열)

---

### 👤 추천 교수 및 관련 문서

professors 배열을 순서대로 사용하세요. 각 교수 블록에서 **관련 문서**는 반드시 아래 형식처럼 **2단계 불릿**으로만 작성하세요.
- **1단계 불릿**: 유형 이름만 (**논문**, **특허**, **연구 과제** 중 해당하는 것만)
- **2단계 불릿**: 그 유형에 속한 실제 문서들을 들여쓰기한 세부 불릿으로, 각 줄은 `- **[제목]** (연도): 요약 한두 문장` 형식

(documents 배열의 type_ko 값 사용. 유형 아래에 문서가 없으면 해당 유형은 생략)

#### 1. [이름] 교수
- **소속:** (department)
- **이메일:** (contact, 없으면 "-")

**관련 문서**
- **논문**
  - **[제목1]** (연도): (사용자 검색어와 관련지어 한두 문장 요약)
  - **[제목2]** (연도): (요약)
- **특허**
  - **[제목]** (연도): (요약)
- **연구 과제**
  - **[제목]** (연도): (요약)

(2번, 3번 교수도 위와 동일한 2단계 불릿 구조로 반복)

---

### 📌 유의사항 및 산학협력단 연락처

다음 내용을 **그대로** 반영하세요.

제공되는 자료는 현재 데이터베이스 기반이며, 사용자 검색어에 따라 결과 값이 달라질 수 있습니다. 부정확성이나 오류의 가능성을 가지고 있습니다. 교수 순서는 사용자 검색어나 현재 데이터베이스에 따라 달라질 수 있어 참고용으로 이용하시기 바랍니다.

추가적인 정보가 필요한 경우 **산학협력단**에 연락을 취하시기 바랍니다.

| **구분** | **내용** |
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
    ) -> tuple:
        """
        PDF 형식으로 보고서 저장. Playwright(HTML→PDF) 한 경로만 사용.
        
        Returns:
            (저장된 파일 경로 또는 None, 성공 여부)
        """
        if filename is None:
            timestamp = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"report_{timestamp}.pdf"
        file_path = self.output_dir / filename
        report_text = report_data.get("report_text", "")
        report_html = report_data.get("report_html")
        pdf_path = self._save_pdf_html_playwright(file_path, report_text=report_text, report_html=report_html)
        return (pdf_path, pdf_path is not None)

    def _save_pdf_html_playwright(
        self,
        file_path: Path,
        report_text: str = None,
        report_html: str = None,
    ) -> Optional[Path]:
        """
        Streamlit에 보이는 HTML을 그대로 PDF로 변환 (Playwright).
        report_html이 있으면 그대로 사용, 없으면 report_text를 마크다운→HTML 변환.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None

        # Streamlit에서 넘긴 HTML 우선 사용 (화면과 100% 동일)
        if report_html and report_html.strip():
            body_html = report_html.strip()
        elif report_text and (report_text or "").strip():
            try:
                import markdown as md_lib
            except ImportError:
                return None
            text = (report_text or "").strip()
            body_html = md_lib.markdown(text, extensions=["extra", "nl2br"])
            if not body_html.strip():
                body_html = "<p>" + _escape_html(text[:5000]) + "</p>"
        else:
            return None

        # 표 헤더에 인라인 배경색 추가 (인쇄 시 CSS 미적용 환경 대비)
        body_html = re.sub(
            r"<th(\s[^>]*)?>",
            r'<th style="background-color:#e8eef4; border:1px solid rgba(30,58,95,0.3); padding:4px 8px;"\1>',
            body_html,
            flags=re.IGNORECASE,
        )
        # 관련 문서: "(연도):" 를 한 덩어리로 유지, "): " 뒤는 논리적 공백 (콜백 사용으로 re 이스케이프 오류 방지)
        def _year_span(match):
            return '<span class="doc-year">(' + match.group(1) + '):</span>' + chr(0x00A0)
        body_html = re.sub(r"\((\d{4})\):\s+", _year_span, body_html)
        # 빈/줄바꿈만 있는 p 태그 제거 → 불필요한 줄간격 축소
        body_html = re.sub(r"<p>\s*</p>", "", body_html, flags=re.IGNORECASE)
        body_html = re.sub(r"<p>\s*<br\s*/?>\s*</p>", "", body_html, flags=re.IGNORECASE)

        # PDF용 HTML: 잘림 방지(overflow 숨기지 않음), 줄간격·여백 축소, 표·리스트 줄바꿈 보장
        head = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>산학 매칭 추천 보고서</title>
<style>
  html, body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  * { box-sizing: border-box; }
  html { width: 100%; }
  body {
    font-family: "Malgun Gothic", "Segoe UI Emoji", "Apple Color Emoji", "Apple SD Gothic Neo", sans-serif;
    font-size: 0.85rem !important;
    line-height: 1.28 !important;
    color: #1e3a5f;
    margin: 0 !important;
    padding: 0.4rem 0.6rem !important;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    word-break: keep-all;
    overflow-wrap: break-word;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box {
    background: #ffffff;
    color: #1e3a5f;
    padding: 0.5rem 0.6rem !important;
    border-radius: 6px;
    border: 1px solid rgba(30, 58, 95, 0.2);
    font-size: 0.85rem !important;
    line-height: 1.28 !important;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    word-break: keep-all;
    overflow-wrap: break-word;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box h1, .report-content-box h2, .report-content-box h3, .report-content-box h4,
  .report-content-box p, .report-content-box li, .report-content-box span,
  .report-content-box td, .report-content-box strong { line-height: 1.28 !important; }
  .report-content-box h1 { font-size: 1rem !important; margin: 0.4em 0 0.25em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h2 { font-size: 0.95rem !important; margin: 0.35em 0 0.2em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h3 { font-size: 0.9rem !important; margin: 0.3em 0 0.18em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h4 { font-size: 0.88rem !important; margin: 0.28em 0 0.15em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box p { margin: 0.45em 0 !important; }
  .report-content-box ul {
    list-style-type: circle;
    list-style-position: outside;
    padding-left: 1.35rem;
    margin: 0.2rem 0 !important;
    line-height: 1.28 !important;
  }
  .report-content-box ul ul {
    list-style-type: disc;
    list-style-position: outside;
    padding-left: 1.5rem;
    margin: 0.12rem 0 0.2rem 0 !important;
    margin-top: 0 !important;
  }
  .report-content-box li { margin: 0.12rem 0 !important; padding-left: 0.25rem; word-break: keep-all; overflow-wrap: break-word; }
  .report-content-box li li { margin: 0.1rem 0 !important; padding-left: 0.2rem; }
  .report-content-box strong { font-weight: 700; color: #1e3a5f; }
  .report-content-box hr { border: none; border-top: 1px solid rgba(30, 58, 95, 0.25); margin: 0.5em 0 !important; }
  .report-content-box ul ul li { page-break-inside: avoid; break-inside: avoid; orphans: 2; widows: 2; }
  .report-content-box table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 100%;
    max-width: 100%;
    margin: 0.3em 0 !important;
    font-size: 0.78rem !important;
    line-height: 1.28 !important;
    color: #1e3a5f;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box th, .report-content-box td {
    border: 1px solid rgba(30, 58, 95, 0.3);
    padding: 3px 6px !important;
    text-align: left;
    color: #1e3a5f;
    line-height: 1.28 !important;
    word-break: keep-all;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .report-content-box th {
    background: #e8eef4 !important;
    font-weight: 600;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box tbody tr:nth-child(even) td {
    background: #f4f6f9;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  @page { size: A4; margin: 18mm; }
  @media print {
    html, body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
    .report-content-box th { background: #e8eef4 !important; }
    .report-content-box tbody tr:nth-child(even) td { background: #f4f6f9 !important; }
    .report-content-box ul ul li { page-break-inside: avoid !important; break-inside: avoid !important; orphans: 2 !important; widows: 2 !important; }
  }
</style>
</head>
<body>
"""
        tail = """
</body>
</html>"""
        # 본문: 인라인 스타일로 줄간격·여백 적용 (가독성 위해 1.28)
        box_inline = "line-height:1.28; font-size:0.85rem; margin:0; padding:0.5rem 0.75rem;"
        html_doc = head + "<div class=\"report-content-box\" style=\"" + box_inline + "\">" + body_html + "</div>" + tail

        try:
            # Windows: 서브프로세스(Chromium 실행)를 위해 Proactor 이벤트 루프 필요 (NotImplementedError 방지)
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            # A4(210mm) - 좌우 여백 18mm*2 = 174mm → 약 657px (96dpi). 이 너비로 레이아웃해 PDF에서 글자 잘림 방지.
            content_width_px = 657
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": content_width_px, "height": 900})
                page.goto("about:blank")
                page.set_content(html_doc, wait_until="load")
                page.wait_for_timeout(1500)
                page.emulate_media(media="print")
                page.wait_for_timeout(200)
                page.pdf(
                    path=str(file_path),
                    format="A4",
                    margin={"top": "18mm", "right": "18mm", "bottom": "18mm", "left": "18mm"},
                    print_background=True,
                )
                browser.close()
            return file_path if file_path.exists() else None
        except Exception as e:
            import warnings
            msg = f"Playwright PDF 실패(HTML→PDF 미적용). 화면과 동일한 PDF를 쓰려면: playwright install chromium. 오류: {e}"
            warnings.warn(msg)
            print(msg)
            return None
