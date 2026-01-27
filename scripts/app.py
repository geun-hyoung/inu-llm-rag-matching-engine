"""
Streamlit 앱 - 쿼리 기반 산학 매칭 리포트 생성 시스템
쿼리 입력 → RAG 검색 → AHP 랭킹 → 리포트 생성
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

# 사이드바 - 설정
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
    st.header("📄 문서 타입 선택")
    doc_types = st.multiselect(
        "검색할 문서 타입",
        options=["patent", "article", "project"],
        default=["patent", "article", "project"],
        help="리포트에 포함할 문서 타입을 선택하세요"
    )
    
    st.markdown("---")
    
    # 보고서 생성용 Few-shot 예시 업로드 (선택사항)
    st.header("📝 보고서 생성용 Few-shot 예시 (선택사항)")
    few_shot_file = st.file_uploader(
        "보고서 생성용 Few-shot 예시 JSON 파일 업로드",
        type=["json"],
        help="보고서 생성 형식을 학습하기 위한 Few-shot 예시 파일을 업로드하세요. 기본 파일: data/report_few_shot_examples.json"
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
            st.info(f"✓ 기본 Few-shot 예시 로드됨 ({len(few_shot_examples)}개)")
        except Exception as e:
            st.warning(f"기본 Few-shot 파일 로드 실패: {e}")


# 메인 영역 - 쿼리 입력
st.header("🔍 검색 쿼리 입력")

query = st.text_area(
    "검색하고 싶은 내용을 입력하세요",
    placeholder="예: 카메라로 찍은 이미지 색상을 보정하는 기술 개발 연구를 찾고 있어요, 도움 주실 수 있나요?",
    height=100,
    help="산학 매칭을 위한 검색 쿼리를 입력하세요"
)

# 리포트 생성 버튼
if not api_key:
    st.error("⚠️ OpenAI API Key를 입력하거나 config/settings.py에 설정해주세요.")
elif not query or not query.strip():
    st.info("👆 위에 검색 쿼리를 입력해주세요.")
elif not doc_types:
    st.warning("⚠️ 최소 하나 이상의 문서 타입을 선택해주세요.")
else:
    if st.button("🚀 리포트 생성", type="primary", width='stretch'):
        # Few-shot 예시 로드 (업로드된 파일이 있으면 우선 사용)
        if few_shot_file:
            try:
                few_shot_data = json.load(few_shot_file)
                if isinstance(few_shot_data, list):
                    few_shot_examples = few_shot_data
                elif isinstance(few_shot_data, dict) and "examples" in few_shot_data:
                    few_shot_examples = few_shot_data["examples"]
                elif isinstance(few_shot_data, dict) and "metadata" in few_shot_data:
                    if "examples" in few_shot_data:
                        few_shot_examples = few_shot_data["examples"]
                st.success(f"✓ Few-shot 예시 로드됨 ({len(few_shot_examples)}개)")
            except Exception as e:
                st.warning(f"Few-shot 파일 로드 실패: {e}")
                few_shot_examples = None
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 리포트 생성기 초기화
            generator = ReportGenerator(api_key=api_key)
            
            # 전체 파이프라인 실행 (RAG → AHP → 리포트)
            status_text.text("RAG 검색 수행 중...")
            progress_bar.progress(25)
            
            status_text.text("교수별 문서 집계 중...")
            progress_bar.progress(50)
            
            status_text.text("AHP 기반 교수 순위 평가 중...")
            progress_bar.progress(75)
            
            status_text.text("GPT-4o-mini를 사용하여 리포트 생성 중...")
            progress_bar.progress(90)
            
            # 쿼리 기반 리포트 생성
            report_data = generator.generate_report_from_query(
                query=query.strip(),
                doc_types=doc_types,
                few_shot_examples=few_shot_examples
            )
            
            progress_bar.progress(100)
            status_text.text("완료!")
            
            # 리포트 저장
            json_path = generator.save_json(report_data)
            text_path = generator.save_text(report_data)
            
            st.success(f"✅ 리포트 생성 완료!")
            st.info(f"저장 위치: {json_path.parent}")
            
            # AHP 결과 요약 표시
            input_data = report_data.get("input_data", {})
            professors = input_data.get("professors", [])
            
            if professors:
                st.markdown("---")
                st.header("📋 추천 교수 요약")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("검색 쿼리", query[:30] + "..." if len(query) > 30 else query)
                with col2:
                    st.metric("총 교수 수", len(professors))
                with col3:
                    st.metric("문서 타입", ", ".join(doc_types))
                
                # 상위 교수 목록
                st.subheader("🏆 상위 교수 순위")
                prof_data = []
                for i, prof in enumerate(professors[:10], 1):  # 상위 10명만
                    prof_data.append({
                        "순위": i,
                        "교수명": prof.get("name", ""),
                        "소속": prof.get("department", ""),
                        "종합 점수": f"{prof.get('total_score', 0):.4f}",
                        "특허": f"{prof.get('scores_by_type', {}).get('patent', 0):.4f}",
                        "논문": f"{prof.get('scores_by_type', {}).get('article', 0):.4f}",
                        "연구과제": f"{prof.get('scores_by_type', {}).get('project', 0):.4f}"
                    })
                
                df = pd.DataFrame(prof_data)
                st.dataframe(df, width='stretch')
            
            # 생성된 리포트 표시
            st.markdown("---")
            st.header("📄 생성된 리포트")
            
            # 리포트 텍스트 표시 (마크다운 형식)
            report_text = report_data.get("report_text", "")
            st.markdown(report_text)
            
            # 원본 텍스트 보기 (접을 수 있게)
            with st.expander("📋 원본 텍스트 보기"):
                st.text_area(
                    "리포트 원본 텍스트",
                    value=report_text,
                    height=400,
                    disabled=True
                )
            
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
            st.error(f"❌ 리포트 생성 중 오류 발생: {str(e)}")
            st.exception(e)
        finally:
            progress_bar.empty()
            status_text.empty()


# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>INU LLM RAG Matching Engine - 리포트 생성 시스템</div>",
    unsafe_allow_html=True
)
