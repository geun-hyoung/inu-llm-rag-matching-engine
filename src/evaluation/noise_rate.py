"""
Noise Rate@K 평가 모듈
산학협력 매칭을 위한 커스텀 진단 지표

- A1: 도메인 + 기술 동시 적합성
- A2: 적용/구현 관점 정보 존재
- Noise: A1 또는 A2 미충족 문서
"""

import sys
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed. Run: pip install openai")


# Noise 판정 프롬프트
NOISE_JUDGE_PROMPT = """당신은 산학협력 매칭을 위한 문서 관련성 판정 전문가입니다.

## 작업
기업의 기술 수요(쿼리)에 대해 검색된 문서가 실무 활용 가치가 있는지 판정하세요.

## 쿼리 (기업 기술 수요)
{query}

## 평가 대상 문서
{document}

## 판정 기준

### A1: 도메인 + 기술 동시 적합성
- 쿼리에서 요구하는 **적용 도메인**(예: 스마트팜, 제조업, 헬스케어 등)과
- **핵심 기술**(예: IoT 센서, 딥러닝, 블록체인 등)이
- 문서에서 **동시에** 다뤄지고 있는가?

### A2: 적용/구현 관점 정보 존재
문서에 아래 중 하나 이상의 **실무 적용 관련 정보**가 포함되어 있는가?
- 센서/장비 구성
- 데이터 수집/처리 방법
- 시스템 아키텍처/통신 프로토콜
- 운영/유지보수 방안
- 실제 적용 사례/실험 결과

## 출력 형식 (JSON)
{{
    "A1_domain_tech": true 또는 false,
    "A2_implementation": true 또는 false,
    "is_noise": true 또는 false,
    "reason": "판정 근거를 1-2문장으로 작성"
}}

## 판정 규칙
- is_noise = !(A1_domain_tech AND A2_implementation)
- 즉, A1과 A2 **모두 true**여야 Non-noise (is_noise: false)
- A1 또는 A2 중 하나라도 false면 Noise (is_noise: true)
"""


@dataclass
class NoiseJudgment:
    """단일 문서에 대한 Noise 판정 결과"""
    doc_id: str
    A1_domain_tech: bool
    A2_implementation: bool
    is_noise: bool
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NoiseRateResult:
    """Noise Rate@K 평가 결과"""
    noise_rate: Optional[float]  # None이면 평가 제외 (문서 0개)
    k: int
    noise_count: int
    non_noise_count: int
    judgments: List[NoiseJudgment]

    def to_dict(self) -> Dict:
        return {
            "noise_rate": self.noise_rate,
            "k": self.k,
            "noise_count": self.noise_count,
            "non_noise_count": self.non_noise_count,
            "judgments": [j.to_dict() for j in self.judgments]
        }

    def summary(self) -> str:
        """결과 요약 문자열"""
        if self.noise_rate is None:
            return f"Noise Rate@{self.k}: N/A (검색된 문서 없음)"

        lines = [
            f"Noise Rate@{self.k}: {self.noise_rate:.2%}",
            f"  - Noise: {self.noise_count}개",
            f"  - Non-noise: {self.non_noise_count}개",
            "",
            "개별 판정 결과:"
        ]
        for j in self.judgments:
            status = "Noise" if j.is_noise else "Non-noise"
            lines.append(f"  [{j.doc_id}] {status}")
            lines.append(f"    A1(도메인+기술): {j.A1_domain_tech}")
            lines.append(f"    A2(적용/구현): {j.A2_implementation}")
            lines.append(f"    사유: {j.reason}")
        return "\n".join(lines)


class NoiseRateEvaluator:
    """Noise Rate@K 평가기"""

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        """
        Args:
            api_key: OpenAI API 키 (기본값: settings.OPENAI_API_KEY)
            model: 사용할 모델 (기본값: settings.LLM_MODEL)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Run: pip install openai")

        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or LLM_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def judge_single_document(
        self,
        query: str,
        doc_id: str,
        doc_content: str
    ) -> NoiseJudgment:
        """
        단일 문서에 대한 Noise 판정

        Args:
            query: 기업 기술 수요 쿼리
            doc_id: 문서 ID
            doc_content: 문서 내용

        Returns:
            NoiseJudgment 판정 결과
        """
        prompt = NOISE_JUDGE_PROMPT.format(
            query=query,
            document=doc_content[:3000]  # 토큰 제한을 위해 앞부분만
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0  # 일관된 판정
            )

            result = json.loads(response.choices[0].message.content)

            return NoiseJudgment(
                doc_id=doc_id,
                A1_domain_tech=result.get("A1_domain_tech", False),
                A2_implementation=result.get("A2_implementation", False),
                is_noise=result.get("is_noise", True),
                reason=result.get("reason", "판정 실패")
            )

        except Exception as e:
            # 에러 시 Noise로 처리
            return NoiseJudgment(
                doc_id=doc_id,
                A1_domain_tech=False,
                A2_implementation=False,
                is_noise=True,
                reason=f"판정 오류: {str(e)}"
            )

    async def calculate_noise_rate_at_k(
        self,
        query: str,
        documents: List[Dict],
        k: int = 5
    ) -> NoiseRateResult:
        """
        Top-K 문서에 대한 Noise Rate 계산

        Args:
            query: 기업 기술 수요 쿼리
            documents: 문서 리스트 [{"doc_id": str, "content": str}, ...]
            k: 평가할 상위 문서 수

        Returns:
            NoiseRateResult 평가 결과
        """
        # K개로 제한
        top_k_docs = documents[:k]
        actual_k = len(top_k_docs)

        if actual_k == 0:
            # 문서가 없으면 평가 제외 (N/A)
            return NoiseRateResult(
                noise_rate=None,  # 평가 불가
                k=0,
                noise_count=0,
                non_noise_count=0,
                judgments=[]
            )

        # 병렬로 모든 문서 판정
        tasks = [
            self.judge_single_document(
                query=query,
                doc_id=doc.get("doc_id", f"doc_{i}"),
                doc_content=doc.get("content", "")
            )
            for i, doc in enumerate(top_k_docs)
        ]

        judgments = await asyncio.gather(*tasks)

        # 집계
        noise_count = sum(1 for j in judgments if j.is_noise)
        non_noise_count = actual_k - noise_count
        noise_rate = noise_count / actual_k

        return NoiseRateResult(
            noise_rate=noise_rate,
            k=actual_k,
            noise_count=noise_count,
            non_noise_count=non_noise_count,
            judgments=list(judgments)
        )

    def calculate_noise_rate_at_k_sync(
        self,
        query: str,
        documents: List[Dict],
        k: int = 5
    ) -> NoiseRateResult:
        """
        동기 버전의 Noise Rate 계산
        (기존 이벤트 루프 재사용)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.calculate_noise_rate_at_k(query, documents, k)
        )


# NoiseRateEvaluator 싱글톤 캐시
_noise_evaluator_cache = None

def evaluate_noise_rate(
    query: str,
    documents: List[Dict],
    k: int = 5,
    api_key: str = None,
    model: str = None
) -> NoiseRateResult:
    """
    Noise Rate@K 평가 (간편 함수)

    Args:
        query: 기업 기술 수요 쿼리
        documents: 문서 리스트 [{"doc_id": str, "content": str}, ...]
        k: 평가할 상위 문서 수
        api_key: OpenAI API 키
        model: 사용할 모델

    Returns:
        NoiseRateResult 평가 결과

    Example:
        >>> documents = [
        ...     {"doc_id": "paper_001", "content": "스마트팜 IoT 센서 네트워크..."},
        ...     {"doc_id": "paper_002", "content": "딥러닝 이미지 분류..."},
        ... ]
        >>> result = evaluate_noise_rate(
        ...     query="스마트팜 IoT 센서 기술",
        ...     documents=documents,
        ...     k=5
        ... )
        >>> print(f"Noise Rate: {result.noise_rate:.2%}")
    """
    global _noise_evaluator_cache

    # 싱글톤 패턴: 동일한 evaluator 재사용
    if _noise_evaluator_cache is None:
        _noise_evaluator_cache = NoiseRateEvaluator(api_key=api_key, model=model)

    return _noise_evaluator_cache.calculate_noise_rate_at_k_sync(query, documents, k)


if __name__ == "__main__":
    # 테스트
    print("Testing Noise Rate@K Evaluator...")
    print(f"OpenAI available: {OPENAI_AVAILABLE}")

    if not OPENAI_AVAILABLE:
        print("OpenAI not installed. Skipping test.")
        exit()

    # 테스트 데이터
    test_query = "스마트팜 환경에서 IoT 센서를 활용한 작물 생육 모니터링 시스템"

    test_documents = [
        {
            "doc_id": "paper_001",
            "content": """
            본 연구는 스마트팜에서 IoT 센서 네트워크를 구축하여 작물 생육 환경을
            실시간으로 모니터링하는 시스템을 제안한다. 온도, 습도, 토양 수분 센서를
            활용하여 데이터를 수집하고, LoRa 통신을 통해 클라우드 서버로 전송한다.
            실제 토마토 재배 농가에 적용하여 수확량 15% 증가를 확인하였다.
            """
        },
        {
            "doc_id": "paper_002",
            "content": """
            딥러닝 기반 이미지 분류 알고리즘의 성능을 비교 분석하였다.
            ResNet, VGG, EfficientNet 모델을 ImageNet 데이터셋에서 평가하였으며,
            EfficientNet이 가장 높은 정확도를 보였다.
            """
        },
        {
            "doc_id": "paper_003",
            "content": """
            스마트팜 IoT 기술의 최신 동향을 리뷰한다. 다양한 센서 기술과
            통신 프로토콜이 발전하고 있으며, 인공지능과의 결합이 주목받고 있다.
            향후 스마트팜 시장은 연평균 12% 성장할 것으로 전망된다.
            """
        },
    ]

    # 평가 실행
    result = evaluate_noise_rate(
        query=test_query,
        documents=test_documents,
        k=3
    )

    # 결과 출력
    print("\n" + "="*60)
    print(result.summary())
    print("="*60)
