"""
Streamlit 앱 - 산학 매칭 리포트 생성 시스템
1. 쿼리 입력 → RAG → AHP → 리포트 (자동 파이프라인)
2. 기존 파일 선택 → 리포트 생성
"""

import streamlit as st
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.reporting.report_generator import ReportGenerator
from config.settings import OPENAI_API_KEY


# 페이지 설정
st.set_page_config(
    page_title="INU LLM RAG Matching Engine - 리포트 생성",
    page_icon="📊",
    layout="wide"
)

# 제목
st.title("📊 산학 매칭 리포트 생성 시스템")
st.markdown("---")

# 사이드바 - 공통 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # API Key 입력
    api_key = st.text_input(
        "OpenAI API Key",
        value=OPENAI_API_KEY if OPENAI_API_KEY else "",
        type="password",
        help="config/settings.py에 설정하거나 여기에 직접 입력하세요"
    )

    st.markdown("---")

    # 문서 타입 선택
    st.header("📄 검색 문서 타입")
    doc_types = st.multiselect(
        "검색할 문서 타입 선택",
        options=["patent", "article", "project"],
        default=["patent", "article", "project"],
        help="RAG 검색에 포함할 문서 타입을 선택하세요"
    )

    st.markdown("---")

    # Few-shot 예시
    st.header("📝 Few-shot 예시")
    few_shot_file = st.file_uploader(
        "Few-shot 예시 JSON 업로드 (선택)",
        type=["json"],
        help="기본: data/report_few_shot_examples.json"
    )

    # 기본 Few-shot 파일 자동 로드
    default_few_shot_path = Path("data/report_few_shot_examples.json")
    few_shot_examples = None
    if default_few_shot_path.exists() and not few_shot_file:
        try:
            with open(default_few_shot_path, 'r', encoding='utf-8') as f:
                few_shot_data = json.load(f)
                if isinstance(few_shot_data, list):
                    few_shot_examples = few_shot_data
                elif isinstance(few_shot_data, dict) and "examples" in few_shot_data:
                    few_shot_examples = few_shot_data["examples"]
            st.info(f"✓ 기본 Few-shot 로드됨 ({len(few_shot_examples)}개)")
        except Exception as e:
            st.warning(f"Few-shot 로드 실패: {e}")


# 탭 생성
tab1, tab2 = st.tabs(["🔍 쿼리 검색", "📁 파일 선택"])


# ===== 탭 1: 쿼리 입력 → 자동 파이프라인 =====
with tab1:
    st.header("🔍 쿼리로 검색 & 리포트 생성")
    st.markdown("쿼리를 입력하면 **RAG 검색 → AHP 랭킹 → 리포트 생성**이 자동으로 실행됩니다.")

    # 쿼리 입력
    query = st.text_input(
        "검색 쿼리",
        placeholder="예: 딥러닝 의료영상 분석 기술 연구를 찾고 있어요",
        help="산학협력 매칭을 위한 검색 쿼리를 입력하세요"
    )

    if st.button("🚀 검색 & 리포트 생성", type="primary", key="query_search"):
        if not api_key:
            st.error("⚠️ OpenAI API Key를 입력해주세요.")
        elif not query:
            st.warning("⚠️ 검색 쿼리를 입력해주세요.")
        elif not doc_types:
            st.warning("⚠️ 검색할 문서 타입을 선택해주세요.")
        else:
            # Few-shot 업로드 파일 처리
            if few_shot_file:
                try:
                    few_shot_data = json.load(few_shot_file)
                    if isinstance(few_shot_data, list):
                        few_shot_examples = few_shot_data
                    elif isinstance(few_shot_data, dict) and "examples" in few_shot_data:
                        few_shot_examples = few_shot_data["examples"]
                except Exception as e:
                    st.warning(f"Few-shot 파일 로드 실패: {e}")

            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("ReportGenerator 초기화 중...")
                progress_bar.progress(10)

                generator = ReportGenerator(api_key=api_key)

                status_text.text("RAG 검색 수행 중...")
                progress_bar.progress(30)

                # 전체 파이프라인 실행
                report_data = generator.generate_report_from_query(
                    query=query,
                    doc_types=doc_types,
                    few_shot_examples=few_shot_examples
                )

                progress_bar.progress(90)
                status_text.text("리포트 저장 중...")

                # 리포트 저장
                json_path = generator.save_json(report_data)
                text_path = generator.save_text(report_data)

                progress_bar.progress(100)
                status_text.text("완료!")

                st.success(f"✅ 리포트 생성 완료!")
                st.info(f"저장 위치: {json_path.parent}")

                # 생성된 리포트 표시
                st.markdown("---")
                st.header("📄 생성된 리포트")

                report_text = report_data.get("report_text", "")
                st.markdown(report_text)

                # 원본 텍스트 보기
                with st.expander("📋 원본 텍스트 보기"):
                    st.text_area("리포트 원본", value=report_text, height=400, disabled=True)

                # 입력 데이터 확인
                with st.expander("🔍 입력 데이터 확인 (디버깅)"):
                    st.json(report_data.get("input_data", {}))

                # 다운로드 버튼
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 JSON 다운로드",
                        data=json.dumps(report_data, ensure_ascii=False, indent=2),
                        file_name=json_path.name,
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="📥 TXT 다운로드",
                        data=report_text,
                        file_name=text_path.name,
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()


# ===== 탭 2: 기존 파일 선택 =====
with tab2:
    st.header("📁 기존 파일로 리포트 생성")
    st.markdown("이미 생성된 AHP/RAG 결과 파일을 선택하여 리포트를 생성합니다.")

    # 파일 선택
    col1, col2 = st.columns(2)

    with col1:
        ahp_results_dir = Path("results/test/ahp")
        ahp_files = list(ahp_results_dir.glob("ahp_results_*.json")) if ahp_results_dir.exists() else []

        if ahp_files:
            ahp_files.sort(reverse=True)
            selected_ahp_file = st.selectbox(
                "AHP 결과 파일",
                options=[f.name for f in ahp_files],
                index=0,
                help="results/test/ahp/"
            )
        else:
            st.warning("AHP 결과 파일 없음")
            selected_ahp_file = None

    with col2:
        rag_results_dir = Path("results/test/rag")
        rag_files = list(rag_results_dir.glob("*.json")) if rag_results_dir.exists() else []

        if rag_files:
            rag_files.sort(reverse=True)
            selected_rag_file = st.selectbox(
                "RAG 결과 파일",
                options=[f.name for f in rag_files],
                index=0,
                help="results/test/rag/"
            )
        else:
            st.warning("RAG 결과 파일 없음")
            selected_rag_file = None

    if not api_key:
        st.error("⚠️ OpenAI API Key를 입력해주세요.")
    elif selected_ahp_file and selected_rag_file:
        # 파일 로드
        ahp_file_path = ahp_results_dir / selected_ahp_file
        rag_file_path = rag_results_dir / selected_rag_file

        try:
            with open(ahp_file_path, 'r', encoding='utf-8') as f:
                ahp_results = json.load(f)

            with open(rag_file_path, 'r', encoding='utf-8') as f:
                rag_results = json.load(f)

            # AHP 결과 요약
            st.subheader("📋 AHP 결과 요약")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("검색 쿼리", ahp_results.get("query", "N/A"))
            with col2:
                st.metric("총 교수 수", ahp_results.get("total_professors", 0))
            with col3:
                type_weights = ahp_results.get("type_weights", {})
                st.metric("가중치", f"P:{type_weights.get('patent', 0):.1f}, A:{type_weights.get('article', 0):.1f}, Pr:{type_weights.get('project', 0):.1f}")

            # 상위 교수 목록
            ranked_professors = ahp_results.get("ranked_professors", [])
            if ranked_professors:
                st.subheader("🏆 상위 교수 순위")

                prof_data = []
                for i, prof in enumerate(ranked_professors[:10], 1):
                    prof_info = prof.get("professor_info", {})
                    scores = prof.get("scores_by_type", {})
                    prof_data.append({
                        "순위": i,
                        "교수명": prof_info.get("NM", ""),
                        "소속": f"{prof_info.get('COLG_NM', '')} {prof_info.get('HG_NM', '')}".strip(),
                        "종합 점수": f"{prof.get('total_score', 0):.4f}",
                        "특허": f"{scores.get('patent', 0):.4f}",
                        "논문": f"{scores.get('article', 0):.4f}",
                        "연구과제": f"{scores.get('project', 0):.4f}"
                    })

                df = pd.DataFrame(prof_data)
                st.dataframe(df, use_container_width=True)

            st.markdown("---")

            # 리포트 생성 버튼
            if st.button("🚀 리포트 생성", type="primary", key="file_generate"):
                # Few-shot 업로드 파일 처리
                if few_shot_file:
                    try:
                        few_shot_data = json.load(few_shot_file)
                        if isinstance(few_shot_data, list):
                            few_shot_examples = few_shot_data
                        elif isinstance(few_shot_data, dict) and "examples" in few_shot_data:
                            few_shot_examples = few_shot_data["examples"]
                    except Exception as e:
                        st.warning(f"Few-shot 파일 로드 실패: {e}")

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    generator = ReportGenerator(api_key=api_key)

                    status_text.text("GPT-4o-mini를 사용하여 리포트 생성 중...")
                    progress_bar.progress(50)

                    report_data = generator.generate_report(
                        ahp_results=ahp_results,
                        rag_results=rag_results,
                        few_shot_examples=few_shot_examples
                    )

                    progress_bar.progress(100)
                    status_text.text("완료!")

                    json_path = generator.save_json(report_data)
                    text_path = generator.save_text(report_data)

                    st.success(f"✅ 리포트 생성 완료!")
                    st.info(f"저장 위치: {json_path.parent}")

                    # 생성된 리포트 표시
                    st.markdown("---")
                    st.header("📄 생성된 리포트")

                    report_text = report_data.get("report_text", "")
                    st.markdown(report_text)

                    with st.expander("📋 원본 텍스트 보기"):
                        st.text_area("리포트 원본", value=report_text, height=400, disabled=True)

                    with st.expander("🔍 입력 데이터 확인 (디버깅)"):
                        st.json(report_data.get("input_data", {}))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 JSON 다운로드",
                            data=json.dumps(report_data, ensure_ascii=False, indent=2),
                            file_name=json_path.name,
                            mime="application/json",
                            key="file_json_download"
                        )
                    with col2:
                        st.download_button(
                            label="📥 TXT 다운로드",
                            data=report_text,
                            file_name=text_path.name,
                            mime="text/plain",
                            key="file_txt_download"
                        )

                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
                    st.exception(e)
                finally:
                    progress_bar.empty()
                    status_text.empty()

        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {str(e)}")
            st.exception(e)


# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>INU LLM RAG Matching Engine - 리포트 생성 시스템</div>",
    unsafe_allow_html=True
)
