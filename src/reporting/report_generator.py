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
try:
    from config.settings import (
        REPORT_FEW_SHOT_MAX_EXAMPLES,
        REPORT_SUMMARY_MAX_CHARS,
        REPORT_MAX_TOKENS,
    )
except ImportError:
    REPORT_FEW_SHOT_MAX_EXAMPLES = None
    REPORT_SUMMARY_MAX_CHARS = 500
    REPORT_MAX_TOKENS = 4096
from src.utils.cost_tracker import log_chat_usage, get_cost_tracker


def _escape_html(s: str) -> str:
    """HTML 이스케이프 (fallback용)."""
    if not s:
        return ""
    import html as _html
    return _html.escape(s)


def normalize_keywords_if_duplicate_query(keywords: Dict[str, Any], query: str) -> Dict[str, List[str]]:
    """
    retriever가 실패해 high_level/low_level 둘 다 [query]로 온 경우를 정규화.
    저수준은 질의에서 토큰을 추출하고, 고수준은 질의 1개만 유지해 중복 표시를 막음.
    """
    high = list(keywords.get("high_level") or [])
    low = list(keywords.get("low_level") or [])
    if not query or (len(high) != 1 or len(low) != 1):
        return {"high_level": high, "low_level": low}
    if high[0] != query or low[0] != query:
        return {"high_level": high, "low_level": low}

    # 둘 다 [query] → 저수준만 질의에서 토큰 분리 (2글자 이상, 종결어 제외)
    stop = {"찾고", "있어", "해요", "해주실", "있나요", "있어요", "싶어", "부탁", "드려요"}
    tokens = [t.strip() for t in re.split(r"[\s,]+", query) if len(t.strip()) >= 2]
    tokens = [t for t in tokens if t not in stop][:6]
    if not tokens:
        tokens = [query]
    return {"high_level": [query], "low_level": tokens}


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
        raw_kw = {
            "high_level": raw_rag_results.get("high_level_keywords", []),
            "low_level": raw_rag_results.get("low_level_keywords", []),
        }
        ahp_results = {
            "query": query,
            "keywords": normalize_keywords_if_duplicate_query(raw_kw, query),
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
        max_tokens = getattr(self, "_max_tokens", None) or REPORT_MAX_TOKENS
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 문서 기반 교수 추천 보고서를 생성하는 보고서 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        # 비용 추적
        log_chat_usage(
            component="report_generation",
            model=self.model,
            response=response
        )

        report_text = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None) or ""
        truncated = finish_reason == "length"

        if truncated:
            print(
                "[경고] 보고서가 출력 토큰 제한에 걸려 잘렸을 수 있습니다. "
                "config/settings.py의 REPORT_MAX_TOKENS(현재 최대 16384)를 확인하세요."
            )

        # 사용자 검색어 섹션에서 1차/2차 키워드 블록이 있으면 제거 (해당 정보는 보고서에 미표기)
        report_text = self._inject_keyword_section(report_text, input_json)
        # 교수/문서 형식 보정 (교수 번호·이름 굵게, 문서 번호·요약 줄 고정)
        report_text = self._normalize_report_format(report_text)

        # 결과 구조화
        report_data = {
            "query": ahp_results.get("query", ""),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "report_text": report_text,
            "input_data": input_json,
            "model": self.model,
            "report_truncated": truncated,
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
        - query: 사용자 검색어(표시). keywords(1차/2차)는 보고서에 미표기.
        - professors: AHP ranked_professors 상위 3명
        - 각 교수 documents: AHP documents(patent/article/project) 유형별 상위 3개, type/title/summary/year만 사용.

        Args:
            ahp_results: AHP 결과
            rag_results: RAG 결과 (키워드·retrieved_docs용, 현재 보고서 템플릿에서는 개체/관계 미사용)

        Returns:
            리포트 생성용 입력 JSON
        """
        query = ahp_results.get("query", "")
        keywords = normalize_keywords_if_duplicate_query(
            ahp_results.get("keywords", {}), query
        )
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
                    max_chars = REPORT_SUMMARY_MAX_CHARS if REPORT_SUMMARY_MAX_CHARS else 600
                    text_for_summary = text[:max_chars] + "..." if len(text) > max_chars else text
                    prof_docs.append({
                        "type": doc_type,
                        "type_ko": type_ko,
                        "title": doc.get("title", ""),
                        "summary": text_for_summary,
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

    def _inject_keyword_section(self, report_text: str, input_json: Dict[str, Any]) -> str:
        """
        보고서 본문의 '사용자 검색어' 섹션에서 1차/2차·저수준·고수준 키워드 블록이 있으면 제거.
        (해당 정보는 보고서에 표기하지 않음)
        """
        section_header = "### 🔍 사용자 검색어"
        replacement_block = section_header + "\n\n"

        # "### 🔍 사용자 검색어" (또는 예전 "검색 개요" 포함 제목) 부터 다음 "###" 또는 "---" 직전까지를 헤더만 남기고 치환
        pattern = re.compile(
            r"(### 🔍 사용자 검색어(?: \(검색 개요\))?)\s*\n.*?(?=\n### |\n---|\n# |\Z)",
            re.DOTALL,
        )
        if pattern.search(report_text):
            return pattern.sub(replacement_block, report_text, count=1)
        return report_text

    def _normalize_report_format(self, report_text: str) -> str:
        """
        LLM 출력에서 자주 틀리는 보고서 형식을 후처리로 정규화합니다.
        - 교수 헤더: "3. 구충완 교수" → "**3.** **구충완 교수**"
        - 소속/이메일: "소속:" → "**소속:**"
        - 문서 목록: "[유형]" 아래 제목만 있고 번호 없는 줄 → "  2. **제목** (연도)", "요약:" → "  - 요약: "
        """
        if not report_text or not report_text.strip():
            return report_text

        lines = report_text.split("\n")
        out: List[str] = []
        # 문서 블록 내 상태: 0=블록 밖, 1=**[논문]** 등 유형 블록 안
        in_doc_block = False
        doc_num = 0
        # "### 👤 추천 교수" 구간에서만 교수/소속/이메일 정규화 적용
        in_professor_section = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # 섹션 진입: 추천 교수 구간 (##### N. 이름 교수 포함해도 구간으로 인식)
            if "### 👤 추천 교수" in line or ("##### " in line and "교수" in line) or ("##### **" in line and "교수" in line):
                in_professor_section = True
            if stripped.startswith("### ") and "추천 교수" not in line:
                in_professor_section = False
            if line.strip() == "---" and in_professor_section:
                pass  # 유지

            # 문서 유형 블록 시작 (**[논문]** / [논문] / **특허** / 특허 등 모든 변형)
            def _is_doc_type_header(s: str) -> Optional[str]:
                for pattern in [
                    r"^\*\*\[(논문|특허|연구 ?과제)\]\*\*$",
                    r"^\[(논문|특허|연구 ?과제)\]$",
                    r"^\*\*(논문|특허|연구 ?과제)\*\*$",
                    r"^(논문|특허|연구 ?과제)$",
                ]:
                    m = re.match(pattern, s)
                    if m:
                        return m.group(1).replace(" ", "")
                return None

            doc_type = _is_doc_type_header(stripped)
            if doc_type:
                in_doc_block = True
                doc_num = 0
                # 유형 블록 앞에 빈 줄 한 줄 (이전이 비어 있지 않을 때)
                if out and out[-1].strip() and _is_doc_type_header(out[-1].strip()) is None:
                    out.append("")
                # 통일 표기: **[논문]** **[특허]** **[연구 과제]**
                label = "연구 과제" if doc_type == "연구과제" else doc_type
                out.append("**[" + label + "]**")
                i += 1
                continue

            # 문서 블록 끝: 다음 교수(#####) 또는 ### 또는 ---
            if in_doc_block and (
                re.match(r"^\s*---\s*$", line)
                or stripped.startswith("### ")
                or (stripped.startswith("##### ") and "교수" in line)
            ):
                in_doc_block = False

            # 문서 블록 안(또는 추천 교수 구간): 제목/요약 형식 보정 (마지막 교수 항목도 동일 적용)
            in_doc_region = in_doc_block or in_professor_section
            if in_doc_region:
                year_at_end = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", line.strip())
                has_number_prefix = re.match(r"^\s*(\d+)\.\s+(.+)$", line.strip())
                # 번호 없이 "제목 (연도)" 만 있는 경우 → "  N. **제목** (연도)"
                if year_at_end and not re.match(r"^\s*\d+\.\s+", stripped):
                    doc_num += 1
                    title_part = year_at_end.group(1).strip()
                    year_part = year_at_end.group(2)
                    if title_part.startswith("**") and title_part.endswith("**"):
                        new_line = f"  {doc_num}. {title_part} ({year_part})"
                    else:
                        new_line = f"  {doc_num}. **{title_part}** ({year_part})"
                    out.append(new_line)
                    i += 1
                    continue
                # 번호는 있으나 들여쓰기/굵게 누락 ("1. 제목 (연도)") → "  N. **제목** (연도)". "N. 이름 교수" 는 문서가 아니므로 제외
                if has_number_prefix:
                    num_str, rest = has_number_prefix.group(1), has_number_prefix.group(2).strip()
                    if rest.endswith("교수"):
                        # 교수 헤더 줄(예: 3. 전광길 교수) → 문서로 처리하지 않고 아래 교수 보정으로 넘김
                        pass
                    else:
                        year_match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", rest)
                        if year_match:
                            doc_num = int(num_str)
                            title_part = year_match.group(1).strip()
                            year_part = year_match.group(2)
                            if title_part.startswith("**") and title_part.endswith("**"):
                                new_line = f"  {doc_num}. {title_part} ({year_part})"
                            else:
                                new_line = f"  {doc_num}. **{title_part}** ({year_part})"
                            out.append(new_line)
                            i += 1
                            continue
                        else:
                            doc_num += 1

            # 문서 블록 또는 추천 교수 구간: "요약:" / "- 요약:" 등 → "  - 요약: " 형태로 통일 (마지막 교수 항목 포함)
            if in_doc_block or in_professor_section:
                s = line.strip()
                if s.startswith("요약:") and not s.startswith("  - 요약:"):
                    rest = s[3:].strip()
                    out.append("  - 요약: " + rest)
                    i += 1
                    continue
                if (s.startswith("- 요약:") or s.startswith("-요약:")) and not line.startswith("  - "):
                    rest = s.split("요약:", 1)[-1].strip()
                    out.append("  - 요약: " + rest)
                    i += 1
                    continue

            # 추천 교수 구간: "##### N. 이름 교수" 또는 "N. 이름 교수" → 굵게 보정 (이미 ** 있으면 스킵)
            if in_professor_section and "교수" in line and not stripped.startswith("**"):
                # "##### 3. 전광길 교수" 형태
                m_heading = re.match(r"^(#####\s+)(\d+)\.\s+(.+?)\s*교수\s*$", stripped)
                if m_heading:
                    prefix, num, name = m_heading.group(1), m_heading.group(2), m_heading.group(3).strip()
                    out.append(prefix + "**" + num + ".** **" + name + " 교수**")
                    i += 1
                    continue
                # "3. 전광길 교수" 단독 줄
                m_plain = re.match(r"^(\d+)\.\s+(.+?)\s*교수\s*$", stripped)
                if m_plain:
                    num, name = m_plain.group(1), m_plain.group(2).strip()
                    out.append("**" + num + ".** **" + name + " 교수**")
                    i += 1
                    continue

            # 추천 교수 구간: "소속:" / "이메일:" 앞에 ** 없으면 추가
            if in_professor_section:
                if re.match(r"^소속:\s*", stripped) and not stripped.startswith("**"):
                    out.append("**소속:** " + line.strip()[3:].strip())
                    i += 1
                    continue
                if re.match(r"^이메일:\s*", stripped) and not stripped.startswith("**"):
                    out.append("**이메일:** " + line.strip()[4:].strip())
                    i += 1
                    continue

            out.append(line)
            i += 1

        # 2차 패스: [특허]/[연구 과제] 등에서 빠진 제목·요약 형식 한 번 더 보정
        result = "\n".join(out)
        return self._normalize_doc_format_second_pass(result)

    def _normalize_doc_format_second_pass(self, text: str) -> str:
        """**[논문]** **[특허]** **[연구 과제]** 블록: 유형 헤더 통일, 제목/요약 들여쓰기(2칸), 유형 앞 빈 줄."""
        if not text or not text.strip():
            return text

        def _is_doc_type_header(s: str) -> Optional[str]:
            for pattern in [
                r"^\*\*\[(논문|특허|연구 ?과제)\]\*\*$",
                r"^\[(논문|특허|연구 ?과제)\]$",
                r"^\*\*(논문|특허|연구 ?과제)\*\*$",
                r"^(논문|특허|연구 ?과제)$",
            ]:
                m = re.match(pattern, s)
                if m:
                    return m.group(1).replace(" ", "")
            return None

        lines = text.split("\n")
        out: List[str] = []
        in_doc_block = False
        doc_num = 0
        for line in lines:
            s = line.strip()
            doc_type = _is_doc_type_header(s)
            if doc_type:
                in_doc_block = True
                doc_num = 0
                # 유형 블록 앞에 빈 줄 한 줄 (이전 줄이 비어 있지 않을 때)
                if out and out[-1].strip() and not _is_doc_type_header(out[-1].strip()):
                    out.append("")
                label = "연구 과제" if doc_type == "연구과제" else doc_type
                out.append("**[" + label + "]**")
                continue
            if in_doc_block and (
                re.match(r"^\s*---\s*$", line)
                or s.startswith("### ")
                or (s.startswith("##### ") and "교수" in line)
            ):
                in_doc_block = False
            if in_doc_block:
                year_m = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", s)
                # 이미 "  2. **제목** (연도)" 형태면 들여쓰기만 검사
                has_num_and_year = re.match(r"^\s*(\d+)\.\s+(.+)\s*\((\d{4})\)\s*$", s)
                if year_m and not has_num_and_year:
                    doc_num += 1
                    title_part = year_m.group(1).strip()
                    year_part = year_m.group(2)
                    if title_part.startswith("**") and title_part.endswith("**"):
                        out.append(f"  {doc_num}. {title_part} ({year_part})")
                    else:
                        out.append(f"  {doc_num}. **{title_part}** ({year_part})")
                    continue
                if has_num_and_year:
                    num_str, mid, year_part = has_num_and_year.group(1), has_num_and_year.group(2).strip(), has_num_and_year.group(3)
                    doc_num = int(num_str)
                    if mid.startswith("**") and mid.endswith("**"):
                        new_line = f"  {doc_num}. {mid} ({year_part})"
                    else:
                        new_line = f"  {doc_num}. **{mid}** ({year_part})"
                    out.append(new_line)
                    continue
                if s.startswith("요약:") and not s.startswith("  - 요약:"):
                    rest = s[3:].strip()
                    out.append("  - 요약: " + rest)
                    continue
                if (s.startswith("- 요약:") or s.startswith("-요약:")) and not line.startswith("  - "):
                    rest = s.split("요약:", 1)[-1].strip()
                    out.append("  - 요약: " + rest)
                    continue
            out.append(line)
        return "\n".join(out)

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
- **강조**: "사용자 검색어", **"교수명"**, "소속", "이메일", "문서 유형", "제목", "연도" 등 라벨은 **굵게** 처리하세요. 교수 표기는 **이름과 '교수'까지 통째로 굵게** 하세요. 예: **홍길동 교수**.
- **이모티콘**: 섹션 구분을 위해 각 섹션 제목 앞에 이모티콘을 하나씩 넣으세요. 예: 📋 제목, 🔍 사용자 검색어, 👤 추천 교수, 📌 유의사항 및 문의
- **추천 교수 순서**: 추천되는 교수는 반드시 순서대로 1, 2, 3 번호를 붙이되, **마크다운 리스트(1. 2. 3.)를 쓰지 마세요.** 가로줄(---) 때문에 리스트가 끊겨 모두 "1."로 보이는 문제가 있으므로, 교수 번호는 **굵은 숫자**로만 표기하세요. 교수 표기는 **이름 + " 교수"** 까지 통째로 굵게 쓰세요. 예: **1.** **홍길동 교수**, **2.** **김철수 교수**, **3.** **이영희 교수**. 다음 줄에 **소속:**, **이메일:** 은 불릿(-) 없이 한 줄씩만 표기하고, 교수 블록 사이에는 가로줄(---)로 구분하세요.
- AHP 점수·종합 점수는 보고서에 포함하지 마세요.
- **사용자 검색어 관련 자료**: (1) 유형은 **대괄호 [ ]** 로만. (2) 각 문서는 첫 줄에 `  1. **제목** (연도)` 형식, 둘째 줄은 **반드시** `  - 요약: ` 로 시작(들여쓰기 2칸 + 하이픈 + 공백 + 요약:). "요약:"만 단독으로 쓰거나 불릿(-) 없이 쓰지 마세요. 요약 내용은 2~3문장, 입력 JSON의 summary를 참고해 사용자 검색어와의 연관성을 설명하고 문체를 통일하세요.

---

### [보고서 출력 형식]

보고서 **맨 위**에 다음 제목 블록을 넣으세요 (제목·사용자 검색어는 본문보다 한 단계 작게):

# 📋 AI 기반 검색 결과

**사용자 검색어:** (입력 JSON의 query 값)
---

그 다음 아래 섹션을 **순서대로** 작성하세요. 각 섹션 제목 앞에 이모티콘을 붙이고, 라벨은 **굵게** 처리하세요.

---

### 👤 추천 교수 및 관련 정보

professors 배열을 **순서대로** 사용하세요. 교수 번호는 **1.** **2.** **3.** 처럼 굵은 숫자로만 쓰고, **교수 표기는 "이름 교수" 전체를 굵게** 표기하세요. 예: **1.** **홍길동 교수**. 교수명 한 줄 다음에 소속·이메일을 불릿 없이 한 줄씩 표기하고, 교수 블록 사이에는 가로줄(---)을 넣어 구분하세요. **사용자 검색어 관련 자료** 작성 규칙:
- **유형**: 대괄호 [ ] 만 사용. **[논문]** **[특허]** **[연구 과제]** 중 해당하는 것만 표기.
- **문서**: 유형 아래 **반드시** (1) 첫 줄: `  1. **제목** (연도)` (들여쓰기 2칸 + 번호 + **제목** + 공백 + (연도)), (2) 둘째 줄: `  - 요약: ` 로 시작한 뒤 2~3문장. **요약 줄은 예외 없이 반드시 "  - 요약: "으로 시작**하세요. "요약:"만 단독으로 쓰거나 불릿(-) 없이 쓰지 마세요.
- **잘못된 예(금지)**: `제목 (연도)\n요약: 내용` 또는 `제목\n요약: 내용` → 이렇게 하지 마세요. 반드시 번호(1. 2. 3.)와 `  - 요약: ` 형식을 지키세요.
- **유형 간 줄 간격**: **[논문]** / **[특허]** / **[연구 과제]** 블록 사이에는 빈 줄을 한 줄 이상 넣어 구분하세요.

(documents 배열의 type_ko, title, summary, year 사용. 유형 아래에 문서가 없으면 해당 유형은 생략)

##### **1.** **[이름] 교수**
(위 [이름]은 입력 JSON의 professors[].name. **이름 교수** 전체를 굵게. 예: **홍길동 교수**)
**소속:** (department)
**이메일:** (contact, 없으면 "-")

**사용자 검색어 관련 자료**
**[논문]**
  1. **제목1** (2024)
  - 요약: (해당 문서 summary를 바탕으로 사용자 검색어와의 연관성을 2~3문장으로 설명. 문체 통일.)
  2. **제목2** (2023)
  - 요약: (동일한 방식으로 2~3문장 설명. 위와 같이 반드시 "  - 요약: "으로 시작.)

**[특허]**
  1. **제목** (2024)
  - 요약: (2~3문장으로 연관성 설명.)

**[연구 과제]**
  1. **제목** (2024)
  - 요약: (2~3문장으로 연관성 설명.)

---
(2번, 3번 교수는 **2.** **[이름] 교수**, **3.** **[이름] 교수**처럼 번호와 이름만 바꿔 반복. "이름 교수" 전체를 **굵게**. 각 교수 블록 끝에 ---로 구분)

---

### 📌 유의사항 및 문의 안내

- 추천 결과는 입력하신 검색어와 시스템에 등록된 정보를 바탕으로 제공되며, 검색 조건에 따라 달라질 수 있습니다.

- 출력 순서는 교수 순위나 우선순위를 의미하지 않으며, 본 결과는 참고 자료로 활용해 주시기를 바랍니다.

- 보다 정확한 정보나 산학협력 관련 상담이 필요하신 경우, 아래 산학협력단 담당자에게 문의해주시기 바랍니다.

| **구분** | **내용** |
|------|------|
| 담당자 | 김OO |
| 이메일 | oo@inu.ac.kr |
| 연락처 | 032-000-0000 |

---
"""
        
        # Few-shot 예시 추가 (REPORT_FEW_SHOT_MAX_EXAMPLES로 개수 제한 시 속도 향상)
        if few_shot_examples:
            limit = REPORT_FEW_SHOT_MAX_EXAMPLES
            examples_to_use = (
                few_shot_examples[:limit]
                if limit is not None and isinstance(limit, int)
                else few_shot_examples
            )
            for i, example in enumerate(examples_to_use, 1):
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
        # 사용자 검색어 관련 자료: "(연도):" 또는 "연도:" 를 한 덩어리로 유지, "): " 뒤는 논리적 공백 (콜백 사용으로 re 이스케이프 오류 방지)
        def _year_span(match):
            return '<span class="doc-year">(' + match.group(1) + '):</span>' + chr(0x00A0)
        body_html = re.sub(r"\((\d{4})\):\s+", _year_span, body_html)
        # 완전히 비어 있는 p만 제거 (공백만 있는 p는 줄바꿈으로 대체해 단락 간격 유지)
        body_html = re.sub(r"<p>\s*</p>", "<br/>", body_html, flags=re.IGNORECASE)
        body_html = re.sub(r"<p>\s*<br\s*/?>\s*</p>", "<br/>", body_html, flags=re.IGNORECASE)

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
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
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
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    word-break: keep-all;
    overflow-wrap: break-word;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box h1, .report-content-box h2, .report-content-box h3, .report-content-box h4, .report-content-box h5,
  .report-content-box p, .report-content-box li, .report-content-box span,
  .report-content-box td, .report-content-box strong { line-height: 1.75 !important; }
  .report-content-box h1 { font-size: 1.15rem !important; margin: 0.6em 0 0.4em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h2 { font-size: 1.08rem !important; margin: 0.55em 0 0.35em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h3 { font-size: 1.02rem !important; margin: 0.5em 0 0.3em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h4 { font-size: 0.98rem !important; margin: 0.45em 0 0.1em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h5 { font-size: 0.96rem !important; margin: 0.4em 0 0.1em !important; color: #1e3a5f; font-weight: 700; }
  .report-content-box h5 strong { font-weight: 700 !important; }
  .report-content-box h4 + ul { margin-top: 0.1rem !important; }
  .report-content-box h4 + p { margin: 0.12em 0 !important; }
  .report-content-box h4 + p + p { margin: 0.12em 0 !important; }
  .report-content-box p { margin: 0.5em 0 !important; line-height: 1.75 !important; }
  .report-content-box ul {
    list-style-type: circle;
    list-style-position: outside;
    padding-left: 1.35rem;
    margin: 0.5rem 0 !important;
    line-height: 1.75 !important;
  }
  .report-content-box ul ul {
    list-style-type: disc;
    list-style-position: outside;
    padding-left: 1.5rem;
    margin: 0.35rem 0 0.4rem 0 !important;
    margin-top: 0.3rem !important;
  }
  .report-content-box li { margin: 0.35rem 0 !important; padding-left: 0.25rem; word-break: keep-all; overflow-wrap: break-word; line-height: 1.75 !important; }
  .report-content-box li li { margin: 0.28rem 0 !important; padding-left: 0.2rem; }
  .report-content-box strong { font-weight: 700; color: #1e3a5f; }
  .report-content-box ol + p { margin-top: 0.6em !important; }
  .report-content-box hr { border: none; border-top: 1px solid rgba(30, 58, 95, 0.25); margin: 0.6em 0 !important; }
  .report-content-box ul ul li { page-break-inside: avoid; break-inside: avoid; orphans: 2; widows: 2; }
  .report-content-box table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 100%;
    max-width: 100%;
    margin: 0.5em 0 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    color: #1e3a5f;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .report-content-box th, .report-content-box td {
    border: 1px solid rgba(30, 58, 95, 0.3);
    padding: 4px 8px !important;
    text-align: left;
    color: #1e3a5f;
    line-height: 1.6 !important;
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
        # 본문: 인라인 스타일로 줄간격·글자 크기 적용 (가독성)
        box_inline = "line-height:1.5; font-size:0.95rem; margin:0; padding:0.5rem 0.75rem;"
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
