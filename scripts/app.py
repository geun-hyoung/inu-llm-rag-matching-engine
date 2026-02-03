"""
Streamlit 앱 - 산학 매칭 리포트 생성 시스템
1. 쿼리 입력 → RAG → AHP → 리포트 (자동 파이프라인)
2. 기존 파일 선택 → 리포트 생성
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.reporting.report_generator import ReportGenerator
from config.settings import OPENAI_API_KEY
from src.utils.cost_tracker import get_cost_tracker


# ===== Streamlit 캐시 함수 (임베딩 모델, 벡터 저장소 등 최초 1회만 로드) =====
@st.cache_resource
def get_embedder(force_api: bool = False):
    """임베딩 모델 캐시 (최초 1회만 로드)"""
    from src.rag.embedding.embedder import Embedder
    print("임베딩 모델 로드 중... (최초 1회)")
    return Embedder(force_api=force_api)


@st.cache_resource
def get_vector_store():
    """ChromaDB 벡터 저장소 캐시 (최초 1회만 로드)"""
    from src.rag.store.vector_store import ChromaVectorStore
    print("벡터 저장소 로드 중... (최초 1회)")
    return ChromaVectorStore()


@st.cache_resource
def get_retriever(_embedder, _vector_store, doc_types_tuple: tuple):
    """
    HybridRetriever 캐시 (doc_types별로 캐시)

    Args:
        _embedder: 캐시된 Embedder (언더스코어로 시작하면 해시하지 않음)
        _vector_store: 캐시된 ChromaVectorStore
        doc_types_tuple: 문서 타입 튜플 (리스트는 해시 불가하므로 튜플로 변환)
    """
    from src.rag.query.retriever import HybridRetriever
    print(f"HybridRetriever 생성 중... (doc_types: {doc_types_tuple})")
    return HybridRetriever(
        doc_types=list(doc_types_tuple),
        embedder=_embedder,
        vector_store=_vector_store
    )


# 페이지 설정 (사이드바 미사용)
st.set_page_config(
    page_title="의미 기반 검색과 생성형 AI를 활용한 산학 매칭 추천 시스템",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 커스텀 CSS: 눈 피로 완화·아이콘과 조화 =====
st.markdown("""
<style>
    /* 배경: 부드러운 회청색 (눈부심 감소, 📋 등 아이콘과 조화) */
    .stApp { background: linear-gradient(180deg, #e2e6ec 0%, #dce0e6 50%, #d4dae2 100%) !important; }
    
    /* 메인 콘텐츠 영역(중간 배경): 읽기 편한 은은한 회색 */
    .main .block-container { background: #eaecf0 !important; border-radius: 0 0 16px 16px !important; }
    
    /* 메인 본문 텍스트 */
    .main .block-container p, .main .block-container li, .main .block-container span,
    .main [data-testid="stMarkdown"] p, .main [data-testid="stMarkdown"] li {
        color: #2d3748 !important;
    }
    
    /* 제목 계층 */
    .main h1 { font-weight: 700 !important; color: #1e3a5f !important; letter-spacing: -0.03em !important; }
    .main h2 { font-weight: 600 !important; color: #1e3a5f !important; letter-spacing: -0.02em !important; margin-top: 1.25rem !important; }
    .main h3 { font-weight: 600 !important; color: #2c5282 !important; }
    .main h4 { font-weight: 600 !important; color: #2d3748 !important; }
    
    /* 캡션 */
    .main [data-testid="stCaptionContainer"] { color: #5a6c7d !important; }
    
    /* 사이드바 숨김 */
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    
    /* 상단 헤더 배너: 보기 편한 연한 회색 (눈 부담 감소) */
    .main .block-container > div:first-child {
        background: #f2f3f6 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 12px rgba(30, 58, 95, 0.06) !important;
        border: 1px solid rgba(203, 213, 224, 0.6) !important;
    }
    
    /* 탭: 배경과 어울리는 은은한 톤 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem !important;
        background: #e8eaef !important;
        padding: 0.35rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(203, 213, 224, 0.7) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        color: #4a5568 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e5aa8 !important;
        color: #fff !important;
    }
    
    /* 버튼: 기본 반응형·터치 친화 */
    .stButton > button {
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        min-height: 48px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        cursor: pointer !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }
    .stButton > button:active { transform: translateY(0) !important; }
    /* 검색 폼 내 메인 버튼: 밝은 배경 + 진한 글자 (가독성 확보) */
    .main form .stButton > button {
        background-color: #ffffff !important;
        border: 2px solid #1e3a5f !important;
        font-weight: 600 !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main form .stButton > button *,
    .main form .stButton [data-testid="stMarkdown"],
    .main form .stButton [data-testid="stMarkdown"] *,
    .main form .stButton button p,
    .main form .stButton button span,
    .main form .stButton button div {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main form .stButton > button:hover {
        background-color: #e8eef4 !important;
        border-color: #1e3a5f !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main form .stButton > button:hover *,
    .main form .stButton > button:hover [data-testid="stMarkdown"],
    .main form .stButton > button:hover [data-testid="stMarkdown"] * {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    @media (max-width: 640px) {
        .stButton > button { min-height: 44px !important; padding: 0.6rem 1rem !important; width: 100% !important; }
    }
    
    /* 검색 쿼리 입력창: 원래 색상 유지, 반응형·실시간 입력만 */
    [data-testid="stTextInput"] input {
        font-size: 1rem !important;
        padding: 0.65rem 0.9rem !important;
        min-height: 48px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    @media (max-width: 640px) {
        [data-testid="stTextInput"] input { min-height: 44px !important; padding: 0.6rem 0.8rem !important; font-size: 16px !important; }
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    
    /* 표 */
    .main table { border-collapse: collapse !important; border-radius: 8px !important; overflow: hidden !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
    .main th { background: #e8eef4 !important; color: #1e3a5f !important; font-weight: 600 !important; padding: 0.6rem 0.75rem !important; }
    .main td { color: #2d3748 !important; background: #f4f5f7 !important; padding: 0.5rem 0.75rem !important; }
    
    /* 메트릭·알림 */
    [data-testid="stMetricValue"] { font-weight: 600 !important; color: #1e3a5f !important; }
    [data-testid="stMetricLabel"] { color: #5a6c7d !important; }
    .stSuccess { background: rgba(16, 185, 129, 0.12) !important; color: #047857 !important; border-radius: 8px !important; padding: 0.5rem 0.75rem !important; }
    .stWarning { background: rgba(245, 158, 11, 0.12) !important; color: #b45309 !important; border-radius: 8px !important; }
    .stError { background: rgba(220, 38, 38, 0.1) !important; color: #b91c1c !important; border-radius: 8px !important; }
    .stInfo { background: rgba(30, 90, 168, 0.1) !important; color: #1e5aa8 !important; border-radius: 8px !important; }
    
    /* 구분선 */
    hr { margin: 1.25rem 0 !important; border: none !important; border-top: 1px solid rgba(203, 213, 224, 0.8) !important; }
    
    /* PDF 다운로드 버튼: 배경·글자색 조화 (가독성) */
    .stDownloadButton > button,
    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important;
        border: 2px solid #1e3a5f !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button:hover,
    .stDownloadButton > button:hover * {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .stDownloadButton > button:hover {
        background-color: #e8eef4 !important;
        border-color: #1e3a5f !important;
    }
</style>
""", unsafe_allow_html=True)

# 기본값: config API 키, 전체 문서 타입, 기본 Few-shot
api_key = OPENAI_API_KEY or ""
doc_types = ["patent", "article", "project"]
default_few_shot_path = Path("data/report_few_shot_examples.json")
few_shot_examples = None
if default_few_shot_path.exists():
    try:
        with open(default_few_shot_path, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)
            few_shot_examples = few_shot_data if isinstance(few_shot_data, list) else few_shot_data.get("examples")
    except Exception:
        pass

# 헤더
st.markdown("## 의미 기반 검색과 생성형 AI를 활용한 산학 매칭 추천 시스템: AI가 만드는 산학 매칭 추천 보고서")
st.caption("인천대학교 데이터사이언스 연구실")
st.markdown("---")

# 검색 섹션: 트렌디·신뢰감 있는 문구와 뱃지
st.markdown(
    "<p style='font-size: 0.75rem; color: #5a6c7d; margin-bottom: 0.25rem;'>RAG · AHP · 생성형 AI 기반</p>",
    unsafe_allow_html=True
)
st.markdown("### 한 번의 검색으로 AI 추천 보고서까지")
st.markdown(
    "검색어만 입력하면 **의미 기반 RAG 검색**으로 특허·논문·연구과제를 찾고, "
    "**AHP 랭킹**과 **생성형 AI**가 산학 매칭 추천 보고서를 자동으로 만들어 드립니다. "
    "데이터 기반의 신뢰할 수 있는 매칭 결과를 제공합니다."
)

# 폼 사용: 입력 중에는 스크립트 재실행 없음 → 반응성 개선. 제출 시에만 실행.
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "검색 쿼리",
        placeholder="예: 딥러닝 의료영상 분석 기술 연구를 찾고 있어요",
        help="산학협력 매칭을 위한 검색 쿼리를 입력하세요. 구체적인 기술·분야 키워드를 넣으면 더 좋은 결과가 나옵니다.",
        key="query_input",
    )
    st.caption(
        "**검색 팁** · 구체적인 **기술·분야 키워드**(예: 딥러닝, 의료영상, 배터리, 감성분석)를 포함하면 매칭 정확도가 올라갑니다. "
        "· 하고 싶은 **기술 개발·연구 주제**를 문장으로 써도 됩니다(예: \"전기차 배터리 충전 시간 단축 기술\"). "
        "· 너무 짧은 단어 하나만 쓰기보다는 **2~5개 키워드** 또는 **한 문장**으로 입력하는 것을 권장합니다."
    )
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        submitted = st.form_submit_button("🚀 검색 & 리포트 생성")
# 버튼 스타일 JS 강제 (밝은 배경 + 진한 글자) — DOM에서 직접 적용
_btn_style_js = """
<script>
(function applyBtnStyle() {
  try {
    var doc = window.parent.document;
    var selectors = ['section[data-testid="stForm"] .stButton button', 'form .stButton button', '.main form .stButton button'];
    var btn = null;
    for (var i = 0; i < selectors.length; i++) {
      btn = doc.querySelector(selectors[i]);
      if (btn) break;
    }
    if (btn) {
      btn.style.setProperty('background-color', '#ffffff', 'important');
      btn.style.setProperty('color', '#1e3a5f', 'important');
      btn.style.setProperty('-webkit-text-fill-color', '#1e3a5f', 'important');
      btn.style.setProperty('border', '2px solid #1e3a5f', 'important');
      btn.style.setProperty('font-weight', '600', 'important');
      var nodes = btn.querySelectorAll('*');
      for (var j = 0; j < nodes.length; j++) {
        nodes[j].style.setProperty('color', '#1e3a5f', 'important');
        nodes[j].style.setProperty('-webkit-text-fill-color', '#1e3a5f', 'important');
      }
    }
  } catch (e) {}
}
applyBtnStyle();
setTimeout(applyBtnStyle, 200);
setTimeout(applyBtnStyle, 1000);
</script>
"""
components.html(_btn_style_js, height=0)

if submitted:
    if not api_key:
        st.error("⚠️ OpenAI API Key가 없습니다. config/settings.py에 OPENAI_API_KEY를 설정해주세요.")
    elif not query:
        st.warning("⚠️ 검색 쿼리를 입력해주세요.")
    else:
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 캐시된 리소스 가져오기 (최초 1회만 실제 로드)
            with st.spinner("임베딩 모델 및 벡터 저장소 로드 중... (최초 실행 시에만 시간이 소요됩니다)"):
                embedder = get_embedder()
                vector_store = get_vector_store()
                retriever = get_retriever(embedder, vector_store, tuple(doc_types))

            progress_bar.progress(10)

            status_text.text("ReportGenerator 초기화 중...")
            progress_bar.progress(20)

            generator = ReportGenerator(api_key=api_key)

            status_text.text("RAG 검색 수행 중...")
            progress_bar.progress(30)

            # 전체 파이프라인 실행 (캐시된 retriever 사용)
            report_data = generator.generate_report_from_query(
                query=query,
                doc_types=doc_types,
                few_shot_examples=few_shot_examples,
                retriever=retriever
            )

            progress_bar.progress(70)
            status_text.text("RAG·AHP·REPORT 결과 로그 저장 중...")

            # results/runs 하위에 RAG, AHP, REPORT 각각 로그로 저장 (동일 타임스탬프)
            ts = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            base = Path("results/runs")
            (base / "rag").mkdir(parents=True, exist_ok=True)
            (base / "ahp").mkdir(parents=True, exist_ok=True)
            (base / "report").mkdir(parents=True, exist_ok=True)
            rag_path = base / "rag" / f"rag_{ts}.json"
            ahp_path = base / "ahp" / f"ahp_results_{ts}.json"
            with open(rag_path, "w", encoding="utf-8") as f:
                json.dump(report_data.get("rag_results", {}), f, ensure_ascii=False, indent=2)
            with open(ahp_path, "w", encoding="utf-8") as f:
                json.dump(report_data.get("ahp_results", {}), f, ensure_ascii=False, indent=2)

            progress_bar.progress(90)
            status_text.text("PDF 저장 중...")

            pdf_path = generator.save_pdf(report_data)

            progress_bar.progress(100)
            status_text.text("완료!")

            cost_result = report_data.get("api_cost")
            st.success(f"✅ 리포트 생성 완료!")
            st.info(
                f"**저장된 로그** · RAG: `{rag_path}` · AHP: `{ahp_path}` · "
                + (f"PDF: `{pdf_path}`" if pdf_path and pdf_path.exists() else "PDF: 저장 실패")
            )
            if not (pdf_path and pdf_path.exists()):
                st.warning("PDF 저장 실패. pip install fpdf2 확인 후 다시 시도하세요.")
            if cost_result and cost_result.get('total_cost_usd', 0) > 0:
                st.caption(f"API 비용: ${cost_result['total_cost_usd']:.6f}")

            st.markdown("---")
            st.markdown("### 📄 생성된 보고서")

            report_text = report_data.get("report_text", "")
            st.markdown(report_text)

            with st.expander("📋 원본 텍스트 보기"):
                st.text_area("리포트 원본", value=report_text, height=320, disabled=True, label_visibility="collapsed")

            with st.expander("🔍 입력 데이터 (디버깅)"):
                st.json(report_data.get("input_data", {}))

            st.markdown("---")
            st.markdown("**PDF 다운로드**")
            if pdf_path and pdf_path.exists():
                st.download_button(
                    label="📥 PDF 다운로드",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key="query_pdf_download"
                )
            else:
                st.caption("PDF를 생성하려면: pip install markdown weasyprint")
                st.caption("또는 브라우저에서 인쇄(Ctrl+P) → PDF로 저장")

        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.exception(e)
        finally:
            progress_bar.empty()
            status_text.empty()


# 페이지 맨 끝: 검색 버튼 스타일 강제 적용 (Streamlit DOM 구조 대응)
st.markdown("""
<style>
    /* 폼 내 첫 번째 버튼 = 검색 & 리포트 생성 (여러 선택자로 확실히 적용) */
    section[data-testid="stForm"] .stButton > button,
    [data-testid="stForm"] .stButton > button,
    form .stButton > button,
    .main section .stButton > button,
    .main form .stButton > button {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 2px solid #1e3a5f !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        font-weight: 600 !important;
    }
    section[data-testid="stForm"] .stButton > button *,
    [data-testid="stForm"] .stButton > button *,
    form .stButton > button *,
    .main form .stButton > button *,
    .main form .stButton button p,
    .main form .stButton button span,
    .main form .stButton button div,
    .main form .stButton [data-testid="stMarkdown"],
    .main form .stButton [data-testid="stMarkdown"] * {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
</style>
""", unsafe_allow_html=True)

# 푸터 (인디고/RISE 스타일 참고)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #5a6c7d; font-size: 0.8rem; padding: 1rem 0; border-top: 1px solid rgba(203,213,224,0.8); margin-top: 1.5rem;'>INU 산학매칭지원 · 산학 매칭 추천 리포트 | INU LLM RAG Matching Engine</p>",
    unsafe_allow_html=True
)
