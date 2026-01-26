"""
Context Relevance 평가 모듈
RAGAS 라이브러리 기반 Retrieval 평가
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY

# RAGAS 라이브러리 임포트
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import ContextRelevance
    from ragas.llms import llm_factory
    from openai import OpenAI
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not installed. Run: pip install ragas openai")


# 평가용 LLM 싱글톤 캐시
_evaluator_llm_cache = None
_openai_client_cache = None

def get_evaluator_llm():
    """평가용 LLM 초기화 (싱글톤) - RAGAS 0.4.x API"""
    global _evaluator_llm_cache, _openai_client_cache

    if not RAGAS_AVAILABLE:
        return None

    if _evaluator_llm_cache is None:
        _openai_client_cache = OpenAI(api_key=OPENAI_API_KEY)
        _evaluator_llm_cache = llm_factory(
            "gpt-4o-mini",
            client=_openai_client_cache
        )

    return _evaluator_llm_cache


def evaluate_context_relevance(
    query: str,
    contexts: List[str]
) -> float:
    """
    Context Relevance 평가 (Retrieval 전용)

    Ground Truth 없이 검색된 문서들이 쿼리와 관련 있는지 평가합니다.

    Args:
        query: 사용자 질문/쿼리
        contexts: 검색된 컨텍스트(문서) 리스트

    Returns:
        Context Relevance 점수 (0~1, 높을수록 관련성 높음)

    Example:
        >>> score = evaluate_context_relevance(
        ...     query="스마트팜 IoT 센서 기술",
        ...     contexts=["스마트팜 센서 네트워크 구축...", "딥러닝 이미지 분류..."]
        ... )
        >>> print(f"Context Relevance: {score:.3f}")
    """
    if not RAGAS_AVAILABLE:
        return _fallback_context_relevance(query, contexts)

    if not contexts:
        return 0.0

    try:
        evaluator_llm = get_evaluator_llm()

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts
        )

        metric = ContextRelevance(llm=evaluator_llm)
        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=evaluator_llm
        )

        df = result.to_pandas()

        # Context Relevance 컬럼명 확인
        relevance_cols = [col for col in df.columns if 'relevance' in col.lower() or 'context' in col.lower()]
        if relevance_cols:
            score = float(df[relevance_cols[0]].iloc[0])
            if score != score:  # NaN 처리
                return 0.0
            return score

        return 0.0

    except Exception as e:
        print(f"Context Relevance evaluation error: {e}")
        return _fallback_context_relevance(query, contexts)


def _fallback_context_relevance(
    query: str,
    contexts: List[str]
) -> float:
    """RAGAS 사용 불가 시 폴백 (단어 겹침 기반)"""
    if not contexts or not query:
        return 0.0

    query_words = set(query.lower().split())
    context_text = " ".join(contexts).lower()
    context_words = set(context_text.split())

    if not query_words:
        return 0.0

    overlap = len(query_words & context_words) / len(query_words)
    return min(1.0, overlap * 1.5)


def evaluate_context_relevance_batch(
    samples: List[Dict]
) -> List[float]:
    """
    배치 Context Relevance 평가

    Args:
        samples: [{"query": str, "contexts": List[str]}, ...]

    Returns:
        각 샘플의 Context Relevance 점수 리스트
    """
    if not RAGAS_AVAILABLE:
        return [_fallback_context_relevance(s["query"], s["contexts"]) for s in samples]

    try:
        evaluator_llm = get_evaluator_llm()

        ragas_samples = []
        for s in samples:
            ragas_samples.append(SingleTurnSample(
                user_input=s["query"],
                retrieved_contexts=s["contexts"]
            ))

        dataset = EvaluationDataset(samples=ragas_samples)
        metric = ContextRelevance(llm=evaluator_llm)

        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=evaluator_llm
        )

        df = result.to_pandas()
        scores = []

        # Context Relevance 컬럼명 확인
        relevance_cols = [col for col in df.columns if 'relevance' in col.lower() or 'context' in col.lower()]
        col_name = relevance_cols[0] if relevance_cols else None

        for i in range(len(df)):
            if col_name and col_name in df.columns:
                score = float(df[col_name].iloc[i])
                if score != score:  # NaN
                    score = 0.0
                scores.append(score)
            else:
                scores.append(0.0)

        return scores

    except Exception as e:
        print(f"Batch Context Relevance error: {e}")
        return [_fallback_context_relevance(s["query"], s["contexts"]) for s in samples]


if __name__ == "__main__":
    print(f"RAGAS available: {RAGAS_AVAILABLE}")

    test_query = "스마트팜 IoT 센서 기술"
    test_contexts = [
        "스마트팜에서 IoT 센서를 활용한 환경 모니터링 시스템을 구축하였다.",
        "딥러닝 이미지 분류 알고리즘의 성능을 비교 분석하였다.",
    ]

    print("\nTesting Context Relevance...")
    score = evaluate_context_relevance(test_query, test_contexts)
    print(f"  Context Relevance: {score:.3f}")
