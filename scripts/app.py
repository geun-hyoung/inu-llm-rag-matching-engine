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
try:
    import markdown
except ImportError:
    markdown = None

sys.path.append(str(Path(__file__).parent.parent))
from src.reporting.report_generator import ReportGenerator
from config.settings import OPENAI_API_KEY, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from src.utils.cost_tracker import get_cost_tracker
from src.ranking.professor_aggregator import ProfessorAggregator
from src.ranking.ranker import ProfessorRanker
from config.ahp_config import DEFAULT_TYPE_WEIGHTS


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

# ===== 지정색 고정 CSS: 라이트/다크 모드에 맞춰 배경·텍스트 조화 =====
# 라이트: 앱 배경 연한 회청(#f0f4f8), 카드/폼 흰색. 다크: 앱 배경 진한 회색(#1a1d23), 카드 흰색 유지.
st.markdown("""
<style>
    /* 라이트 모드: 앱 배경만 연한 회청, 카드·폼은 흰색으로 대비 */
    .stApp { color-scheme: light !important; }
    .stApp, .main {
        background: #f0f4f8 !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main .block-container {
        background: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        border-radius: 0 0 18px 18px !important;
        box-shadow: 0 4px 24px rgba(30, 58, 95, 0.08) !important;
    }
    .main .block-container p, .main .block-container li, .main .block-container span,
    .main .block-container label, .main .block-container h1, .main .block-container h2,
    .main .block-container h3, .main .block-container h4, .main label, .main p, .main li, .main span {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main h1, .main h2, .main h3, .main h4,
    .main [data-testid="stMarkdown"] h1, .main [data-testid="stMarkdown"] h2,
    .main [data-testid="stMarkdown"] h3, .main [data-testid="stMarkdown"] h4 {
        font-weight: 600 !important; color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; background: transparent !important;
    }
    .main [data-testid="stMarkdown"] p, .main [data-testid="stMarkdown"] li,
    .main [data-testid="stMarkdown"] span, .main [data-testid="stMarkdown"] td {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    section[data-testid="stForm"] {
        background: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        padding: 1.25rem 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(30, 58, 95, 0.2) !important;
        margin: 0.5rem 0 !important;
    }
    section[data-testid="stForm"] label, section[data-testid="stForm"] p, section[data-testid="stForm"] span {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main [data-testid="stMarkdown"] {
        background: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        padding: 1rem 1.25rem !important;
        border-radius: 10px !important;
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(30, 58, 95, 0.15) !important;
    }
    .main [data-testid="stMarkdown"] strong { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    .main [data-testid="stMarkdown"] hr { border-color: rgba(30, 58, 95, 0.25) !important; }
    .main [data-testid="stMarkdown"] table, .main [data-testid="stMarkdown"] th { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    .main [data-testid="stMarkdown"] th { background: #e8eef4 !important; }
    .main [data-testid="stCaptionContainer"] { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    .main .report-content-box, div.report-content-box {
        background: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        padding: 1.5rem 1.75rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(30, 58, 95, 0.2) !important;
        margin: 0.75rem 0 !important;
        box-shadow: 0 2px 12px rgba(30, 58, 95, 0.06) !important;
        font-size: 0.95rem !important;
    }
    .main .report-content-box h1, div.report-content-box h1 { font-size: 1.15rem !important; }
    .main .report-content-box h2, div.report-content-box h2 { font-size: 1.05rem !important; }
    .main .report-content-box h3, div.report-content-box h3 { font-size: 1rem !important; }
    .main .report-content-box h4, div.report-content-box h4 { font-size: 0.98rem !important; }
    /* 관련 문서: 1단계=유형(동그라미), 2단계=실제 문서(세부 불릿) 가독성 */
    .main .report-content-box ul, div.report-content-box ul {
        list-style-type: circle !important;
        padding-left: 1.5rem !important;
        margin: 0.4rem 0 !important;
        line-height: 1.5 !important;
    }
    .main .report-content-box ul ul, div.report-content-box ul ul {
        list-style-type: disc !important;
        padding-left: 1.5rem !important;
        margin: 0.25rem 0 0.5rem 0 !important;
    }
    .main .report-content-box li, div.report-content-box li {
        margin: 0.35rem 0 !important;
        line-height: 1.5 !important;
    }
    .main .report-content-box li li, div.report-content-box li li {
        margin: 0.25rem 0 !important;
    }
    .main .report-content-box p, .main .report-content-box li, .main .report-content-box span,
    .main .report-content-box td, .main .report-content-box h1, .main .report-content-box h2,
    .main .report-content-box h3, .main .report-content-box h4, .main .report-content-box strong,
    div.report-content-box p, div.report-content-box li, div.report-content-box span,
    div.report-content-box td, div.report-content-box h1, div.report-content-box h2,
    div.report-content-box h3, div.report-content-box h4, div.report-content-box strong {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    .main .report-content-box table, div.report-content-box table { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    .main .report-content-box th, div.report-content-box th { color: #1e3a5f !important; background: #e8eef4 !important; -webkit-text-fill-color: #1e3a5f !important; }
    .main .report-content-box hr, div.report-content-box hr { border-color: rgba(30, 58, 95, 0.25) !important; }
    /* 다크 모드: 앱 배경만 진한 회색, 카드·폼은 흰색 유지해 가독성 확보 */
    [data-theme="dark"] .stApp { color-scheme: dark !important; background: #1a1d23 !important; color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    [data-theme="dark"] .main { background: #1a1d23 !important; }
    [data-theme="dark"] .stApp .main,
    [data-theme="dark"] .stApp .main .block-container,
    [data-theme="dark"] .stApp .main .block-container *,
    [data-theme="dark"] .stApp .main [data-testid="stMarkdown"],
    [data-theme="dark"] .stApp .main [data-testid="stMarkdown"] *,
    [data-theme="dark"] .stApp .main section[data-testid="stForm"],
    [data-theme="dark"] .stApp .main section[data-testid="stForm"] *,
    [data-theme="dark"] .stApp .main .report-content-box,
    [data-theme="dark"] .stApp .main .report-content-box *,
    [data-theme="dark"] .stApp div.report-content-box,
    [data-theme="dark"] .stApp div.report-content-box * {
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    [data-theme="dark"] .stApp .main .block-container,
    [data-theme="dark"] .stApp .main [data-testid="stMarkdown"],
    [data-theme="dark"] .stApp .main section[data-testid="stForm"],
    [data-theme="dark"] .stApp .main .report-content-box,
    [data-theme="dark"] .stApp div.report-content-box { background: #ffffff !important; }
    [data-theme="dark"] .stApp [data-testid="stTextInput"] input,
    [data-theme="dark"] .stApp .stTextInput input { background: #ffffff !important; color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    /* 다크 모드: 폼 버튼도 네이비 + 흰 글자 (트렌디·학술 스타일) */
    [data-theme="dark"] .stApp .stButton > button,
    [data-theme="dark"] .stApp section[data-testid="stForm"] .stButton > button,
    [data-theme="dark"] .stApp form .stButton > button {
        background: linear-gradient(165deg, #1e3a5f 0%, #2c5282 100%) !important;
        background-color: #1e3a5f !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 12px rgba(30, 58, 95, 0.35) !important;
    }
    [data-theme="dark"] .stApp .stButton > button *,
    [data-theme="dark"] .stApp section[data-testid="stForm"] .stButton > button *,
    [data-theme="dark"] .stApp form .stButton > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-theme="dark"] .stApp [data-testid="stMetricValue"], [data-theme="dark"] .stApp [data-testid="stMetricLabel"] { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    [data-theme="dark"] .stApp .stDownloadButton > button { background: #ffffff !important; color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; border-color: #1e3a5f !important; }
    [data-theme="dark"] .stApp .stDownloadButton > button * { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    
    /* 사이드바 숨김 */
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    
    /* 상단 헤더 배너: 흰 배경 + 파란 글자 */
    .main .block-container > div:first-child {
        background: #ffffff !important;
        color: #1e3a5f !important;
        border-radius: 14px !important;
        padding: 1.5rem 1.75rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 2px 16px rgba(30, 58, 95, 0.08) !important;
        border: 1px solid rgba(30, 58, 95, 0.15) !important;
    }
    .main .block-container > div:first-child p,
    .main .block-container > div:first-child h1,
    .main .block-container > div:first-child h2,
    .main .block-container > div:first-child span { color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important; }
    
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
    /* 검색 & 리포트 생성 버튼: 트렌디·학술 (흰 글자만 확실히, 폼 내 모든 버튼 + primary 타깃) */
    .main form .stButton > button,
    section[data-testid="stForm"] .stButton > button,
    section[data-testid="stForm"] button,
    [data-testid="stForm"] button,
    form[data-testid="stForm"] button,
    .main form button {
        background: linear-gradient(165deg, #1e3a5f 0%, #2c5282 100%) !important;
        background-color: #1e3a5f !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 12px rgba(30, 58, 95, 0.35) !important;
    }
    .main form .stButton > button *,
    .main form .stButton [data-testid="stMarkdown"],
    .main form .stButton [data-testid="stMarkdown"] *,
    section[data-testid="stForm"] .stButton > button *,
    section[data-testid="stForm"] button *,
    [data-testid="stForm"] button *,
    .main form button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .main form .stButton > button:hover,
    section[data-testid="stForm"] .stButton > button:hover,
    section[data-testid="stForm"] button:hover,
    .main form button:hover {
        background: linear-gradient(165deg, #2c5282 0%, #2d3748 100%) !important;
        background-color: #2c5282 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(30, 58, 95, 0.45) !important;
    }
    .main form .stButton > button:hover *,
    section[data-testid="stForm"] .stButton > button:hover *,
    section[data-testid="stForm"] button:hover *,
    .main form button:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    @media (max-width: 640px) {
        .stButton > button { min-height: 44px !important; padding: 0.6rem 1rem !important; width: 100% !important; }
    }
    
    /* 검색 쿼리 입력창: 흰 배경 + 파란 글자 (라이트/다크 공통) */
    [data-testid="stTextInput"] input,
    .stTextInput input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        border: 1px solid rgba(30, 58, 95, 0.35) !important;
        font-size: 1rem !important;
        padding: 0.65rem 0.9rem !important;
        min-height: 48px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    .stTextInput input::placeholder {
        color: #4a6fa5 !important;
        opacity: 0.85 !important;
    }
    [data-theme="dark"] [data-testid="stTextInput"] input,
    [data-theme="dark"] .stTextInput input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
        border: 1px solid rgba(30, 58, 95, 0.35) !important;
    }
    [data-theme="dark"] [data-testid="stTextInput"] input::placeholder,
    [data-theme="dark"] .stTextInput input::placeholder {
        color: #4a6fa5 !important;
        -webkit-text-fill-color: #4a6fa5 !important;
    }
    @media (max-width: 640px) {
        [data-testid="stTextInput"] input, .stTextInput input { min-height: 44px !important; padding: 0.6rem 0.8rem !important; font-size: 16px !important; }
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
    /* 인쇄 시 보고서 영역만 출력 (화면 그대로 PDF 저장용) */
    @media print {
        body * { visibility: hidden; }
        #report-for-pdf, #report-for-pdf * { visibility: visible; }
        #report-for-pdf {
            position: absolute !important;
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            padding: 1rem !important;
            box-shadow: none !important;
            border: none !important;
        }
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
st.markdown("## 의미 기반 검색과 생성형 AI를 활용한 산학 매칭 추천 시스템")
st.caption("인천대학교 데이터사이언스 연구실 · AI 기반 산학 매칭 추천 보고서")
st.markdown("---")

# 검색 섹션
st.markdown(
    "<p style='font-size: 0.8rem; color: #1e3a5f; margin-bottom: 0.35rem; letter-spacing: 0.02em;'>RAG · AHP · 생성형 AI</p>",
    unsafe_allow_html=True
)
st.markdown("### 한 번의 검색으로 AI 추천 보고서까지")
st.markdown(
    "검색어만 입력하면 **의미 기반 RAG 검색**으로 특허·논문·연구과제를 찾고, "
    "**생성형 AI**가 산학 매칭 추천 보고서를 자동으로 만들어 드립니다."
)

# 폼 사용: 입력 중에는 스크립트 재실행 없음 → 반응성 개선. 제출 시에만 실행.
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "검색",
        placeholder="예:  3D 스캐너를 활용한 기술 연구를 수행한 교수님을 찾고 있어요",
        help="산학협력 매칭을 위한 검색 쿼리를 입력하세요. 구체적인 기술·분야 키워드를 넣으면 더 좋은 결과가 나옵니다.",
        key="query_input",
        label_visibility="collapsed",
    )
    st.caption(
        "**💡 검색 팁** · 구체적인 **기술·분야 키워드**(예: 의료영상, 배터리 소재, 에이전트 개발)를 포함하면 매칭 정확도가 올라갑니다. "
        "· 하고 싶은 **기술 개발·연구 주제**를 문장으로 써도 됩니다(예: \"전기차 배터리 충전 시간 단축 기술\"). "
        "· 단어 하나만 쓰기보다는 **2~5개 키워드** 또는 **한 문장**으로 입력하는 것을 권장합니다."
    )
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        submitted = st.form_submit_button("🚀 검색 & 리포트 생성", type="primary")
def _run_pipeline(q: str, docs: list, key: str, few_shot, progress_bar, status_text):
    """파이프라인 실행 + 진행률 표시. 성공 시 session_state 설정, 실패 시 예외 발생."""
    tracker = get_cost_tracker()
    with st.spinner("준비 중..."):
        embedder = get_embedder()
        vector_store = get_vector_store()
        retriever = get_retriever(embedder, vector_store, tuple(docs))
    progress_bar.progress(10)

    generator = ReportGenerator(api_key=key)
    tracker.start_task("full_pipeline", description=q[:40])

    status_text.text("📂 문서 검색 중...")
    progress_bar.progress(20)
    raw_rag_results = retriever.retrieve(
        query=q,
        retrieval_top_k=RETRIEVAL_TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        mode="hybrid"
    )
    rag_results = generator._convert_rag_results(raw_rag_results)
    progress_bar.progress(35)

    status_text.text("👤 연구자 추천 중...")
    progress_bar.progress(45)
    aggregator = ProfessorAggregator()
    professor_data = aggregator.aggregate_by_professor(rag_results=rag_results, doc_types=docs)
    ranker = ProfessorRanker()
    ranked_professors = ranker.rank_professors(professor_data, DEFAULT_TYPE_WEIGHTS)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.reporting.report_generator import normalize_keywords_if_duplicate_query
    raw_kw = {
        "high_level": raw_rag_results.get("high_level_keywords", []),
        "low_level": raw_rag_results.get("low_level_keywords", []),
    }
    ahp_results = {
        "query": q,
        "keywords": normalize_keywords_if_duplicate_query(raw_kw, q),
        "timestamp": run_ts,
        "total_professors": len(ranked_professors),
        "type_weights": DEFAULT_TYPE_WEIGHTS,
        "ranked_professors": ranked_professors,
    }
    progress_bar.progress(60)

    status_text.text("📄 보고서 생성 중...")
    progress_bar.progress(70)
    report_data = generator.generate_report(
        ahp_results=ahp_results,
        rag_results=rag_results,
        few_shot_examples=few_shot
    )
    report_data["timestamp"] = run_ts
    report_data["rag_results"] = rag_results
    report_data["ahp_results"] = ahp_results
    cost_result = tracker.end_task()
    if cost_result:
        report_data["api_cost"] = cost_result
    progress_bar.progress(85)

    base = Path("results/runs")
    (base / "rag").mkdir(parents=True, exist_ok=True)
    (base / "ahp").mkdir(parents=True, exist_ok=True)
    (base / "report").mkdir(parents=True, exist_ok=True)
    rag_path = base / "rag" / f"rag_{run_ts}.json"
    ahp_path = base / "ahp" / f"ahp_results_{run_ts}.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_results, f, ensure_ascii=False, indent=2)
    with open(ahp_path, "w", encoding="utf-8") as f:
        json.dump(ahp_results, f, ensure_ascii=False, indent=2)

    status_text.text("보고서 마무리 중...")
    progress_bar.progress(92)
    if markdown is not None:
        report_data["report_html"] = markdown.markdown(
            report_data.get("report_text", ""),
            extensions=["extra", "nl2br"]
        )
    save_result = generator.save_pdf(report_data)
    if isinstance(save_result, tuple):
        pdf_path, pdf_via_playwright = save_result
    else:
        pdf_path, pdf_via_playwright = save_result, True

    progress_bar.progress(100)
    status_text.text("완료!")

    st.session_state["report_data"] = report_data
    st.session_state["report_pdf_path"] = str(pdf_path) if (pdf_path and pdf_path.exists()) else None
    st.session_state["report_pdf_via_playwright"] = pdf_via_playwright
    st.session_state["report_rag_path_name"] = rag_path.name
    st.session_state["report_ahp_path_name"] = ahp_path.name


def _open_pipeline_modal(q: str, docs: list, key: str, few_shot):
    """모달(팝업)로 로딩 표시. Streamlit 1.33+ 필요."""
    already_done = (
        "report_data" in st.session_state
        and st.session_state.get("report_data", {}).get("query") == q
    )
    close_key = "pipeline_modal_close"
    if already_done:
        st.success("✅ 리포트 생성이 완료되었습니다.")
        if st.button("닫기", type="primary", key=close_key):
            st.session_state.pop("_pipeline_modal_opened", None)
            st.session_state["_modal_just_closed"] = True
            st.rerun()
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        _run_pipeline(q, docs, key, few_shot, progress_bar, status_text)
        st.success("✅ 리포트 생성이 완료되었습니다.")
        if st.button("닫기", type="primary", key=close_key):
            st.session_state.pop("_pipeline_modal_opened", None)
            st.session_state["_modal_just_closed"] = True
            st.rerun()
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.exception(e)
        if st.button("닫기", key=close_key):
            st.session_state.pop("_pipeline_modal_opened", None)
            st.session_state["_modal_just_closed"] = True
            st.rerun()


if submitted:
    if not api_key:
        st.error("⚠️ OpenAI API Key가 없습니다. config/settings.py에 OPENAI_API_KEY를 설정해주세요.")
    elif not query:
        st.warning("⚠️ 검색 쿼리를 입력해주세요.")
    else:
        if hasattr(st, "dialog"):
            # 닫기/바깥 클릭 후 재검색 시 모달 다시 열리도록, 열지 않을 땐 플래그 제거.
            st.session_state.pop("_modal_just_closed", None)
            report_for_same_query = (
                "report_data" in st.session_state
                and st.session_state.get("report_data", {}).get("query") == query
            )
            modal_ok = not report_for_same_query and not st.session_state.get("_pipeline_modal_opened")
            if modal_ok:
                st.session_state["_pipeline_modal_opened"] = True
                @st.dialog("리포트 생성 중", width="small", dismissible=True)
                def run_pipeline_modal(q: str, docs: list, key: str, few_shot):
                    _open_pipeline_modal(q, docs, key, few_shot)
                run_pipeline_modal(query, doc_types, api_key, few_shot_examples)
            else:
                st.session_state.pop("_pipeline_modal_opened", None)
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                _run_pipeline(query, doc_types, api_key, few_shot_examples, progress_bar, status_text)
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()

# 보고서 표시: 방금 생성했거나, PDF 다운로드 등 버튼 클릭 후 재실행 시에도 유지
if "report_data" in st.session_state:
    report_data = st.session_state["report_data"]
    pdf_path_str = st.session_state.get("report_pdf_path")
    pdf_path = Path(pdf_path_str) if pdf_path_str and Path(pdf_path_str).exists() else None
    cost_result = report_data.get("api_cost")

    st.success("✅ 리포트 생성 완료!")
    if not pdf_path:
        st.warning("PDF 저장 실패. 터미널에서 한 번만 실행: **playwright install chromium**")

    st.markdown("---")
    st.markdown("### 📄 생성된 보고서")
    st.caption("검색 질의 기반 추천 교수 및 관련 문서 요약")

    report_text = report_data.get("report_text", "")
    if markdown is not None:
        report_html = markdown.markdown(report_text, extensions=["extra", "nl2br"])
        wrapped = f'<div id="report-for-pdf" class="report-content-box">{report_html}</div>'
        st.markdown(wrapped, unsafe_allow_html=True)
    else:
        st.markdown(report_text)

    with st.expander("📋 원본 텍스트 보기"):
        st.text_area("리포트 원본", value=report_text, height=320, disabled=True, label_visibility="collapsed")

    with st.expander("🔍 입력 데이터 (디버깅)"):
        st.json(report_data.get("input_data", {}))

    if cost_result and cost_result.get("total_cost_usd", 0) > 0:
        st.markdown("---")
        st.markdown("**💰 보고서 생성 비용**")
        st.markdown(
            f"<p style='font-size: 1.25rem; font-weight: 600; color: #1e3a5f; margin: 0.25rem 0 1rem 0;'>${cost_result['total_cost_usd']:.6f} USD</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**PDF 다운로드**")
    st.caption("위 보고서 화면(HTML)을 그대로 PDF로 변환하여 다운로드합니다.")
    if pdf_path:
        st.download_button(
            label="📥 PDF 다운로드",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            key="query_pdf_download"
        )
    else:
        st.caption("PDF 생성 실패. 터미널에서 `playwright install chromium` 실행 후 다시 검색해 주세요.")


# 페이지 맨 끝: 검색 & 리포트 생성 버튼 (최종 우선 적용)
st.markdown("""
<style>
    /* 폼 내 유일한 버튼 = 검색 & 리포트 생성 (우선순위 극대화) */
    body section[data-testid="stForm"] button,
    body .main section[data-testid="stForm"] button,
    body section[data-testid="stForm"] .stButton > button,
    section[data-testid="stForm"] .stButton > button,
    section[data-testid="stForm"] button,
    [data-testid="stForm"] .stButton > button,
    [data-testid="stForm"] button,
    .main form .stButton > button,
    .main form button {
        background: linear-gradient(165deg, #1e3a5f 0%, #2c5282 100%) !important;
        background-color: #1e3a5f !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 12px rgba(30, 58, 95, 0.35) !important;
    }
    body section[data-testid="stForm"] button *,
    section[data-testid="stForm"] .stButton > button *,
    section[data-testid="stForm"] button *,
    [data-testid="stForm"] button *,
    .main form .stButton > button *,
    .main form button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    body section[data-testid="stForm"] button:hover,
    section[data-testid="stForm"] .stButton > button:hover,
    section[data-testid="stForm"] button:hover,
    .main form .stButton > button:hover,
    .main form button:hover {
        background: linear-gradient(165deg, #2c5282 0%, #2d3748 100%) !important;
        background-color: #2c5282 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(30, 58, 95, 0.45) !important;
    }
    body section[data-testid="stForm"] button:hover *,
    section[data-testid="stForm"] .stButton > button:hover *,
    section[data-testid="stForm"] button:hover *,
    .main form button:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #1e3a5f; font-size: 0.8rem; padding: 1.25rem 0; border-top: 1px solid rgba(30,58,95,0.2); margin-top: 1.5rem; letter-spacing: 0.02em;'>Incheon National University · Data Science for Intelligent System Lab</p>",
    unsafe_allow_html=True
)
