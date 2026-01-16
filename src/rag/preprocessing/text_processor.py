"""
텍스트 전처리 모듈
특허/논문 데이터를 정제하여 엔티티 추출에 적합한 형태로 변환
"""

import json
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ProcessedDocument:
    """전처리된 문서 데이터 클래스"""
    doc_id: str           # 문서 고유 ID (no 필드)
    doc_type: str         # 문서 타입 (patent / article / project)
    text: str             # 전처리된 텍스트
    metadata: Dict        # 추가 메타데이터


class TextProcessor:
    """텍스트 전처리 클래스"""

    MIN_TEXT_LENGTH = 100  # 최소 텍스트 길이 (이하면 제외)

    def __init__(self):
        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped_duplicate": 0,
            "skipped_short": 0
        }

    def process_patents(self, patent_data: List[Dict]) -> List[ProcessedDocument]:
        """
        특허 데이터 전처리

        처리 로직:
        1. 기본: [특허명] + \\n + [요약] 형식
        2. 예외 1: 요약이 특허명으로 시작하는 경우 → 요약만 사용
        3. 예외 2: 요약이 특허명으로 시작 + 100자 이하 → 제외

        Args:
            patent_data: 특허 데이터 리스트

        Returns:
            ProcessedDocument 리스트
        """
        self.stats = {"total": 0, "processed": 0, "skipped_duplicate": 0, "skipped_short": 0}
        processed_docs = []

        for patent in patent_data:
            self.stats["total"] += 1

            # 필수 필드 확인
            doc_id = str(patent.get("no", ""))
            doc_type = patent.get("data_type", "patent")
            title = patent.get("kipris_application_name", "").strip()
            abstract = patent.get("kipris_abstract", "").strip()

            if not abstract:
                self.stats["skipped_short"] += 1
                continue

            # 요약이 특허명으로 시작하는지 확인
            starts_with_title = self._starts_with_title(abstract, title)

            if starts_with_title:
                # 예외 1: 요약이 특허명으로 시작 → 요약만 사용
                text = f"[요약] {abstract}"

                # 예외 2: 100자 이하면 제외
                if len(abstract) <= self.MIN_TEXT_LENGTH:
                    self.stats["skipped_short"] += 1
                    continue
            else:
                # 기본: 특허명 + 요약
                text = f"[특허명] {title}\n[요약] {abstract}"

            # 텍스트 정제
            text = self._clean_text(text)

            # 메타데이터 구성
            metadata = {
                "register_status": patent.get("kipris_register_status", ""),
                "application_date": patent.get("kipris_application_date", ""),
                "title": title
            }

            doc = ProcessedDocument(
                doc_id=doc_id,
                doc_type=doc_type,
                text=text,
                metadata=metadata
            )
            processed_docs.append(doc)
            self.stats["processed"] += 1

        return processed_docs

    def _starts_with_title(self, abstract: str, title: str) -> bool:
        """요약이 특허명으로 시작하는지 확인"""
        if not title:
            return False

        # 정규화하여 비교
        normalized_abstract = self._normalize_for_comparison(abstract)
        normalized_title = self._normalize_for_comparison(title)

        return normalized_abstract.startswith(normalized_title)

    def _normalize_for_comparison(self, text: str) -> str:
        """비교를 위한 텍스트 정규화"""
        # 공백, 특수문자 제거하고 소문자로 변환
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\w가-힣]', '', text)
        return text.lower()

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text

    def get_stats(self) -> Dict:
        """처리 통계 반환"""
        return self.stats

    def process_articles(self, article_data: List[Dict]) -> List[ProcessedDocument]:
        """
        논문 데이터 전처리

        처리 로직:
        - [논문명] + [초록] 결합
        - 100자 이하 초록은 제외

        Args:
            article_data: 논문 데이터 리스트

        Returns:
            ProcessedDocument 리스트
        """
        self.stats = {"total": 0, "processed": 0, "skipped_empty": 0, "skipped_short": 0}
        processed_docs = []

        for article in article_data:
            self.stats["total"] += 1

            # 필수 필드 추출
            doc_id = str(article.get("no", ""))
            doc_type = article.get("data_type", "article")
            title = article.get("THSS_NM", "").strip()
            abstract = article.get("abstract", "").strip()

            if not abstract:
                self.stats["skipped_empty"] += 1
                continue

            # 100자 이하면 제외
            if len(abstract) <= self.MIN_TEXT_LENGTH:
                self.stats["skipped_short"] += 1
                continue

            # 텍스트 구성: [논문명] + [초록]
            text = f"[논문명] {title}\n[초록] {abstract}"
            text = self._clean_text(text)

            # 메타데이터 구성
            metadata = {
                "title": title
            }

            doc = ProcessedDocument(
                doc_id=doc_id,
                doc_type=doc_type,
                text=text,
                metadata=metadata
            )
            processed_docs.append(doc)
            self.stats["processed"] += 1

        return processed_docs

    def process_projects(self, project_data: List[Dict]) -> List[ProcessedDocument]:
        """
        연구과제 데이터 전처리

        처리 로직:
        - [과제명] + [연구목표] + [연구내용] + [기대효과] 결합
        - 100자 이하면 제외

        Args:
            project_data: 연구과제 데이터 리스트

        Returns:
            ProcessedDocument 리스트
        """
        self.stats = {"total": 0, "processed": 0, "skipped_empty": 0, "skipped_short": 0}
        processed_docs = []

        for project in project_data:
            self.stats["total"] += 1

            # 필수 필드 추출
            doc_id = str(project.get("no", ""))
            doc_type = project.get("data_type", "project")
            title = project.get("excel_project_name_kr", "").strip()

            # 연구 내용 필드들
            objective = project.get("excel_research_objective_summary", "").strip()
            content = project.get("excel_research_content_summary", "").strip()
            effect = project.get("excel_expected_effect_summary", "").strip()

            # 내용이 모두 비어있으면 스킵
            if not objective and not content and not effect:
                self.stats["skipped_empty"] += 1
                continue

            # 텍스트 구성
            text_parts = [f"[과제명] {title}"]
            if objective:
                text_parts.append(f"[연구목표] {objective}")
            if content:
                text_parts.append(f"[연구내용] {content}")
            if effect:
                text_parts.append(f"[기대효과] {effect}")

            text = "\n".join(text_parts)
            text = self._clean_text(text)

            # 100자 이하면 제외
            if len(text) <= self.MIN_TEXT_LENGTH:
                self.stats["skipped_short"] += 1
                continue

            # 메타데이터 구성
            metadata = {
                "title": title
            }

            doc = ProcessedDocument(
                doc_id=doc_id,
                doc_type=doc_type,
                text=text,
                metadata=metadata
            )
            processed_docs.append(doc)
            self.stats["processed"] += 1

        return processed_docs


def load_patent_data(file_path: str) -> List[Dict]:
    """특허 JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 프로젝트 루트 추가
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.settings import PATENT_DATA_FILE

    # 테스트
    processor = TextProcessor()

    # 특허 데이터 로드
    patents = load_patent_data(PATENT_DATA_FILE)
    print(f"원본 특허 데이터: {len(patents)}개")

    # 전처리
    processed = processor.process_patents(patents)

    # 통계 출력
    stats = processor.get_stats()
    print(f"\n처리 통계:")
    print(f"  - 전체: {stats['total']}개")
    print(f"  - 처리됨: {stats['processed']}개")
    print(f"  - 제외 (짧음): {stats['skipped_short']}개")

    # 샘플 출력
    if processed:
        print(f"\n샘플 출력 (첫 번째 문서):")
        doc = processed[0]
        print(f"  - ID: {doc.doc_id}")
        print(f"  - 교수: {doc.professor_name} ({doc.emp_no})")
        print(f"  - 텍스트 길이: {len(doc.text)}자")
        print(f"  - 텍스트 미리보기: {doc.text[:200]}...")