"""
RAGAS 스타일 평가 메트릭
LLM-as-Judge 방식으로 4가지 메트릭 평가
"""

import sys
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL


# LLM 클라이언트 초기화
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=OPENAI_API_KEY)
    return _llm_client


def _call_llm(prompt: str, temperature: float = 0.0) -> str:
    """LLM 호출 헬퍼"""
    client = _get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def context_relevance(query: str, contexts: List[str]) -> float:
    """
    Context Relevance: 검색된 컨텍스트가 질문과 얼마나 관련있는지

    Args:
        query: 사용자 질문
        contexts: 검색된 컨텍스트 리스트

    Returns:
        0.0 ~ 1.0 점수
    """
    if not contexts:
        return 0.0

    context_text = "\n---\n".join(contexts[:5])  # 상위 5개만

    prompt = f"""다음 질문과 검색된 컨텍스트를 보고, 컨텍스트가 질문에 답하는데 얼마나 관련있는지 평가하세요.

질문: {query}

검색된 컨텍스트:
{context_text}

평가 기준:
- 1.0: 컨텍스트가 질문에 직접적으로 답할 수 있는 정보를 포함
- 0.7: 컨텍스트가 질문과 관련있지만 직접적인 답은 아님
- 0.4: 컨텍스트가 부분적으로 관련있음
- 0.1: 컨텍스트가 거의 관련없음
- 0.0: 컨텍스트가 전혀 관련없음

0.0에서 1.0 사이의 숫자만 출력하세요."""

    try:
        result = _call_llm(prompt)
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0


def context_precision(query: str, contexts: List[str]) -> float:
    """
    Context Precision: 상위 검색 결과가 더 관련있는지 (랭킹 품질)

    Args:
        query: 사용자 질문
        contexts: 검색된 컨텍스트 리스트 (순위대로)

    Returns:
        0.0 ~ 1.0 점수
    """
    if not contexts:
        return 0.0

    # 각 컨텍스트의 관련성을 개별 평가
    relevance_scores = []

    for i, ctx in enumerate(contexts[:5]):
        prompt = f"""다음 질문과 컨텍스트가 관련있으면 1, 관련없으면 0을 출력하세요.

질문: {query}
컨텍스트: {ctx[:500]}

1 또는 0만 출력하세요."""

        try:
            result = _call_llm(prompt)
            score = 1.0 if "1" in result else 0.0
            relevance_scores.append(score)
        except:
            relevance_scores.append(0.0)

    # Precision@K 계산 (가중 평균 - 상위 결과에 더 높은 가중치)
    if not relevance_scores:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for i, score in enumerate(relevance_scores):
        weight = 1.0 / (i + 1)  # 순위가 높을수록 가중치 높음
        weighted_sum += score * weight
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def faithfulness(query: str, contexts: List[str], answer: str) -> float:
    """
    Faithfulness: 응답이 컨텍스트에 근거했는지 (환각 여부)

    Args:
        query: 사용자 질문
        contexts: 검색된 컨텍스트 리스트
        answer: 생성된 응답

    Returns:
        0.0 ~ 1.0 점수
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n---\n".join(contexts[:5])

    prompt = f"""다음 응답이 제공된 컨텍스트에 근거했는지 평가하세요.

질문: {query}

컨텍스트:
{context_text}

응답: {answer}

평가 기준:
- 1.0: 응답의 모든 정보가 컨텍스트에 근거함
- 0.7: 응답의 대부분이 컨텍스트에 근거하지만 일부 추론 포함
- 0.4: 응답의 일부만 컨텍스트에 근거함
- 0.1: 응답이 컨텍스트와 거의 관련없음
- 0.0: 응답이 컨텍스트에 없는 정보를 포함 (환각)

0.0에서 1.0 사이의 숫자만 출력하세요."""

    try:
        result = _call_llm(prompt)
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0


def answer_relevance(query: str, answer: str) -> float:
    """
    Answer Relevance: 응답이 질문에 적절히 답했는지

    Args:
        query: 사용자 질문
        answer: 생성된 응답

    Returns:
        0.0 ~ 1.0 점수
    """
    if not answer:
        return 0.0

    prompt = f"""다음 응답이 질문에 얼마나 적절하게 답했는지 평가하세요.

질문: {query}

응답: {answer}

평가 기준:
- 1.0: 응답이 질문에 정확하고 완전하게 답함
- 0.7: 응답이 질문에 대체로 답하지만 일부 누락
- 0.4: 응답이 질문과 관련있지만 직접적인 답이 아님
- 0.1: 응답이 질문에 거의 답하지 못함
- 0.0: 응답이 질문과 무관함

0.0에서 1.0 사이의 숫자만 출력하세요."""

    try:
        result = _call_llm(prompt)
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0


def evaluate_all(
    query: str,
    contexts: List[str],
    answer: str
) -> Dict[str, float]:
    """
    모든 메트릭 한번에 평가

    Args:
        query: 사용자 질문
        contexts: 검색된 컨텍스트 리스트
        answer: 생성된 응답

    Returns:
        각 메트릭의 점수 딕셔너리
    """
    return {
        "context_relevance": context_relevance(query, contexts),
        "context_precision": context_precision(query, contexts),
        "faithfulness": faithfulness(query, contexts, answer),
        "answer_relevance": answer_relevance(query, answer)
    }


if __name__ == "__main__":
    # 테스트
    test_query = "딥러닝을 활용한 의료영상 분석 연구자는 누구인가요?"
    test_contexts = [
        "김철수 교수는 CNN 기반 의료영상 분석 연구를 수행하고 있다.",
        "이영희 교수는 자연어처리 분야를 연구한다.",
    ]
    test_answer = "김철수 교수가 딥러닝 기반 의료영상 분석 연구를 수행하고 있습니다."

    print("Testing RAGAS metrics...")
    scores = evaluate_all(test_query, test_contexts, test_answer)

    for metric, score in scores.items():
        print(f"  {metric}: {score:.2f}")
