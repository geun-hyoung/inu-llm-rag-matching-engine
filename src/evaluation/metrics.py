"""
RAGAS 라이브러리 기반 평가 메트릭
공식 라이브러리를 사용한 검증된 평가 방법론
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY

# RAGAS 라이브러리 임포트
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not installed. Run: pip install ragas langchain-openai")


def get_evaluator_llm():
    """평가용 LLM 초기화"""
    if not RAGAS_AVAILABLE:
        return None

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    return LangchainLLMWrapper(llm)


def evaluate_rag(
    query: str,
    contexts: List[str],
    answer: str,
    reference: str = None
) -> Dict[str, float]:
    """
    RAGAS를 사용한 RAG 평가

    Args:
        query: 사용자 질문
        contexts: 검색된 컨텍스트 리스트
        answer: 생성된 응답
        reference: 정답 (선택, LLMContextRecall에 필요)

    Returns:
        각 메트릭의 점수 딕셔너리
    """
    if not RAGAS_AVAILABLE:
        return _fallback_evaluate(query, contexts, answer)

    try:
        evaluator_llm = get_evaluator_llm()

        # 샘플 생성
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
            reference=reference or answer  # reference 없으면 answer 사용
        )

        # 메트릭 설정
        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]

        # reference가 있으면 Context Recall 추가
        if reference:
            metrics.append(LLMContextRecall())

        # 데이터셋 생성
        dataset = EvaluationDataset(samples=[sample])

        # 평가 실행
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )

        # 결과 추출
        scores = {}
        df = result.to_pandas()

        if 'faithfulness' in df.columns:
            scores['faithfulness'] = float(df['faithfulness'].iloc[0])
        if 'answer_relevancy' in df.columns:
            scores['answer_relevance'] = float(df['answer_relevancy'].iloc[0])
        elif 'response_relevancy' in df.columns:
            scores['answer_relevance'] = float(df['response_relevancy'].iloc[0])
        if 'llm_context_precision_without_reference' in df.columns:
            scores['context_precision'] = float(df['llm_context_precision_without_reference'].iloc[0])
        if 'context_recall' in df.columns:
            scores['context_recall'] = float(df['context_recall'].iloc[0])

        # NaN 처리
        for key in scores:
            if scores[key] != scores[key]:  # NaN check
                scores[key] = 0.0

        return scores

    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        return _fallback_evaluate(query, contexts, answer)


def _fallback_evaluate(
    query: str,
    contexts: List[str],
    answer: str
) -> Dict[str, float]:
    """
    RAGAS 사용 불가 시 폴백 평가 (간단한 휴리스틱)
    """
    scores = {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
    }

    if not answer or not contexts:
        return scores

    # 간단한 휴리스틱: 컨텍스트와 답변의 단어 겹침 비율
    context_text = " ".join(contexts).lower()
    answer_words = set(answer.lower().split())
    context_words = set(context_text.split())
    query_words = set(query.lower().split())

    # Faithfulness: 답변 단어가 컨텍스트에 얼마나 있는지
    if answer_words:
        overlap = len(answer_words & context_words) / len(answer_words)
        scores["faithfulness"] = min(1.0, overlap * 1.5)

    # Answer Relevance: 답변이 질문 키워드를 얼마나 포함하는지
    if query_words:
        overlap = len(query_words & answer_words) / len(query_words)
        scores["answer_relevance"] = min(1.0, overlap * 2)

    # Context Precision: 컨텍스트가 질문 키워드를 얼마나 포함하는지
    if query_words:
        overlap = len(query_words & context_words) / len(query_words)
        scores["context_precision"] = min(1.0, overlap * 1.5)

    return scores


def evaluate_batch(
    samples: List[Dict]
) -> List[Dict[str, float]]:
    """
    배치 평가

    Args:
        samples: [{"query": str, "contexts": List[str], "answer": str}, ...]

    Returns:
        각 샘플의 점수 리스트
    """
    if not RAGAS_AVAILABLE:
        return [_fallback_evaluate(s["query"], s["contexts"], s["answer"]) for s in samples]

    try:
        evaluator_llm = get_evaluator_llm()

        # 샘플 리스트 생성
        ragas_samples = []
        for s in samples:
            ragas_samples.append(SingleTurnSample(
                user_input=s["query"],
                retrieved_contexts=s["contexts"],
                response=s["answer"],
                reference=s.get("reference", s["answer"])
            ))

        # 데이터셋 생성
        dataset = EvaluationDataset(samples=ragas_samples)

        # 메트릭 설정
        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]

        # 평가 실행
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )

        # 결과 변환
        df = result.to_pandas()
        results = []

        for i in range(len(df)):
            scores = {}
            if 'faithfulness' in df.columns:
                scores['faithfulness'] = float(df['faithfulness'].iloc[i])
            if 'answer_relevancy' in df.columns:
                scores['answer_relevance'] = float(df['answer_relevancy'].iloc[i])
            elif 'response_relevancy' in df.columns:
                scores['answer_relevance'] = float(df['response_relevancy'].iloc[i])
            if 'llm_context_precision_without_reference' in df.columns:
                scores['context_precision'] = float(df['llm_context_precision_without_reference'].iloc[i])

            # NaN 처리
            for key in scores:
                if scores[key] != scores[key]:
                    scores[key] = 0.0

            results.append(scores)

        return results

    except Exception as e:
        print(f"RAGAS batch evaluation error: {e}")
        return [_fallback_evaluate(s["query"], s["contexts"], s["answer"]) for s in samples]


if __name__ == "__main__":
    # 테스트
    print(f"RAGAS available: {RAGAS_AVAILABLE}")

    test_query = "딥러닝을 활용한 의료영상 분석 연구자는 누구인가요?"
    test_contexts = [
        "김철수 교수는 CNN 기반 의료영상 분석 연구를 수행하고 있다.",
        "이영희 교수는 자연어처리 분야를 연구한다.",
    ]
    test_answer = "김철수 교수가 딥러닝 기반 의료영상 분석 연구를 수행하고 있습니다."

    print("\nTesting RAGAS metrics...")
    scores = evaluate_rag(test_query, test_contexts, test_answer)

    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
