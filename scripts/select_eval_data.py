"""
평가용 데이터 선별 스크립트
- 비기술 분야 제외 필터링 (단과대학, 키워드 기반)
- article, patent, project 각 100개 선별
- 임베딩 유사도 기반 문서 그룹화 후 LLM으로 자연스러운 쿼리 생성
- 1 query = N documents (그룹 기반)
"""

import json
import random
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL, OPENAI_EMBEDDING_MODEL

from openai import OpenAI

# 기술 분야 키워드 (산학협력 적합)
TECH_KEYWORDS = [
    # AI/ML
    "딥러닝", "머신러닝", "인공지능", "신경망", "CNN", "RNN", "LSTM", "트랜스포머",
    "영상처리", "이미지", "자연어처리", "NLP", "강화학습", "분류", "예측", "탐지",

    # IoT/네트워크
    "IoT", "사물인터넷", "센서", "통신", "네트워크", "임베디드", "엣지", "클라우드",
    "무선", "블루투스", "WiFi", "LoRa", "5G", "프로토콜",

    # 에너지
    "이차전지", "배터리", "리튬", "전해질", "연료전지", "태양광", "태양전지",
    "수소", "암모니아", "가스터빈", "원자력", "SMR", "발전", "에너지",

    # 바이오/의료
    "바이오", "의료", "진단", "센서", "나노", "항체", "단백질", "유전자",
    "세포", "암", "약물", "치료", "영상", "헬스케어", "웨어러블",

    # 소재/화학
    "반도체", "전극", "촉매", "고분자", "나노", "박막", "코팅", "소재",
    "합성", "화합물", "산화물", "금속", "세라믹", "복합재",

    # 제조/로봇
    "로봇", "자동화", "제조", "공정", "스마트팩토리", "CNC", "3D프린팅",
    "자율주행", "드론", "모터", "액추에이터", "제어",

    # 환경/안전
    "환경", "오염", "정화", "처리", "폐수", "대기", "안전", "재해",
    "위험", "모니터링", "예측", "시뮬레이션"
]

# 제외할 학과/분야 키워드
EXCLUDE_KEYWORDS = [
    "역사", "철학", "문학", "정치", "사회복지", "행정", "법학", "경제학",
    "심리학", "교육학", "언어학", "미술", "음악", "체육", "무용",
    "종교", "신학", "고고학", "인류학"
]

# 제외할 단과대학
EXCLUDE_COLLEGES = [
    "인문대학", "사회과학대학", "사범대학", "법과대학", "예술대학",
    "체육대학", "교육대학원"
]

# 기술 카테고리 (그룹화용)
TECH_CATEGORIES = {
    "AI_ML": ["딥러닝", "머신러닝", "인공지능", "신경망", "CNN", "RNN", "LSTM", "트랜스포머",
              "영상처리", "이미지", "자연어처리", "NLP", "강화학습", "분류", "예측", "탐지"],
    "IoT_Network": ["IoT", "사물인터넷", "센서", "통신", "네트워크", "임베디드", "엣지", "클라우드",
                   "무선", "블루투스", "WiFi", "LoRa", "5G", "프로토콜"],
    "Energy": ["이차전지", "배터리", "리튬", "전해질", "연료전지", "태양광", "태양전지",
               "수소", "암모니아", "가스터빈", "원자력", "SMR", "발전", "에너지"],
    "Bio_Medical": ["바이오", "의료", "진단", "나노", "항체", "단백질", "유전자",
                   "세포", "암", "약물", "치료", "헬스케어", "웨어러블"],
    "Materials": ["반도체", "전극", "촉매", "고분자", "박막", "코팅", "소재",
                  "합성", "화합물", "산화물", "금속", "세라믹", "복합재"],
    "Manufacturing_Robot": ["로봇", "자동화", "제조", "공정", "스마트팩토리", "CNC", "3D프린팅",
                           "자율주행", "드론", "모터", "액추에이터", "제어"],
    "Environment_Safety": ["환경", "오염", "정화", "처리", "폐수", "대기", "안전", "재해",
                          "위험", "모니터링", "시뮬레이션"]
}


def contains_tech_keyword(text: str) -> bool:
    """기술 키워드 포함 여부 확인"""
    if not text:
        return False
    text_lower = text.lower()
    for keyword in TECH_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def get_tech_category(text: str) -> str:
    """문서의 주요 기술 카테고리 반환"""
    if not text:
        return "Other"

    text_lower = text.lower()
    category_scores = {}

    for category, keywords in TECH_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            category_scores[category] = score

    if not category_scores:
        return "Other"

    return max(category_scores, key=category_scores.get)


def contains_exclude_keyword(text: str) -> bool:
    """제외 키워드 포함 여부 확인"""
    if not text:
        return False
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in text:
            return True
    return False


def is_tech_college(professor_info: dict) -> bool:
    """기술 관련 단과대학 여부 확인"""
    if not professor_info:
        return True  # 정보 없으면 일단 포함

    college = professor_info.get("COLG_NM", "")
    for exclude in EXCLUDE_COLLEGES:
        if exclude in college:
            return False
    return True


def filter_articles(data: List[Dict], target_count: int = 100) -> List[Dict]:
    """Article 데이터 필터링"""
    candidates = []

    for item in data:
        # 기본 필드 확인
        abstract = item.get("abstract", "")
        title = item.get("THSS_NM", "")
        professor_info = item.get("professor_info", {})

        # 필터링 조건
        if not abstract or len(abstract) < 100:
            continue
        if contains_exclude_keyword(title + abstract):
            continue
        if not is_tech_college(professor_info):
            continue

        candidates.append(item)

    print(f"Article candidates: {len(candidates)}")

    # 다양성을 위해 셔플 후 선택
    random.shuffle(candidates)

    return candidates[:target_count]


def filter_patents(data: List[Dict], target_count: int = 100) -> List[Dict]:
    """Patent 데이터 필터링"""
    candidates = []

    for item in data:
        abstract = item.get("kipris_abstract", "")

        if not abstract or len(abstract) < 50:
            continue

        candidates.append(item)

    print(f"Patent candidates: {len(candidates)}")

    random.shuffle(candidates)

    return candidates[:target_count]


def filter_projects(data: List[Dict], target_count: int = 100) -> List[Dict]:
    """Project 데이터 필터링"""
    candidates = []

    for item in data:
        summary = item.get("summary", "")
        title = item.get("PRJ_NM", "")
        professor_info = item.get("professor_info", {})

        if not summary or len(summary) < 100:
            continue
        if contains_exclude_keyword(title + summary):
            continue
        if not is_tech_college(professor_info):
            continue

        candidates.append(item)

    print(f"Project candidates: {len(candidates)}")

    random.shuffle(candidates)

    return candidates[:target_count]


def generate_query_with_llm(client: OpenAI, title: str, content: str, data_type: str) -> str:
    """LLM으로 자연스러운 쿼리 생성"""

    prompt = f"""당신은 기업의 기술 담당자입니다.
아래 연구 자료를 보고, 이 연구가 필요한 기업이 산학협력을 요청할 때 사용할 법한 자연스러운 질문을 1개 작성하세요.

[조건]
- 도메인(적용 분야)과 핵심 기술이 반드시 포함되어야 함
- "~찾아줘", "~있나요?", "~연구를 찾고 있어", "~기술 개발 관련 연구" 등 실제 요청 어투 사용
- 30~60자 내외로 간결하게
- 질문만 출력 (다른 설명 없이)

[연구 제목]: {title}
[연구 내용 요약]: {content[:3000]}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        query = response.choices[0].message.content.strip()
        # 따옴표 제거
        query = query.strip('"\'')
        return query
    except Exception as e:
        print(f"Query generation error: {e}")
        return f"{title} 관련 연구를 찾아줘"


def generate_group_query_with_llm(client: OpenAI, docs: List[Dict], title_field: str, content_field: str) -> str:
    """그룹 문서들에 대한 공통 쿼리 생성"""

    # 문서들의 제목과 내용 요약
    doc_summaries = []
    for i, doc in enumerate(docs[:5]):  # 최대 5개 문서만 사용
        title = doc.get(title_field, "")
        content = doc.get(content_field, "")[:500]
        doc_summaries.append(f"[문서{i+1}] {title}\n{content}")

    combined = "\n\n".join(doc_summaries)

    prompt = f"""당신은 기업의 기술 담당자입니다.
아래 여러 연구 자료들의 공통 주제를 파악하고, 이 분야의 연구가 필요한 기업이 산학협력을 요청할 때 사용할 법한 자연스러운 질문을 1개 작성하세요.

[조건]
- 여러 문서에 공통으로 해당될 수 있는 일반적인 질문
- 도메인(적용 분야)과 핵심 기술이 반드시 포함되어야 함
- "~찾아줘", "~있나요?", "~연구를 찾고 있어", "~기술 개발 관련 연구" 등 실제 요청 어투 사용
- 30~60자 내외로 간결하게
- 질문만 출력 (다른 설명 없이)

[관련 연구들]
{combined}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        query = response.choices[0].message.content.strip()
        query = query.strip('"\'')
        return query
    except Exception as e:
        print(f"Group query generation error: {e}")
        first_title = docs[0].get(title_field, "") if docs else ""
        return f"{first_title} 관련 기술 연구를 찾아줘"


def group_documents_by_category(
    docs: List[Dict],
    title_field: str,
    content_field: str,
    min_group_size: int = 2,
    max_group_size: int = 5
) -> List[List[Dict]]:
    """(deprecated) 키워드 기반 그룹화 - group_documents_by_embedding 사용 권장"""

    # 카테고리별 분류
    category_docs = defaultdict(list)
    for doc in docs:
        title = doc.get(title_field, "")
        content = doc.get(content_field, "")
        category = get_tech_category(title + " " + content)
        category_docs[category].append(doc)

    # 그룹 생성
    groups = []
    for category, cat_docs in category_docs.items():
        if category == "Other":
            continue

        # 카테고리 내 문서들을 그룹으로 분할
        random.shuffle(cat_docs)
        for i in range(0, len(cat_docs), max_group_size):
            group = cat_docs[i:i + max_group_size]
            if len(group) >= min_group_size:
                groups.append(group)

    return groups


def get_embeddings(client: OpenAI, texts: List[str], batch_size: int = 20) -> np.ndarray:
    """OpenAI API로 텍스트 임베딩 생성"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # 빈 텍스트 처리
        batch = [t if t.strip() else "empty" for t in batch]

        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def group_documents_by_embedding(
    docs: List[Dict],
    title_field: str,
    content_field: str,
    client: OpenAI,
    similarity_threshold: float = 0.75,
    min_group_size: int = 2,
    max_group_size: int = 5
) -> List[List[Dict]]:
    """
    임베딩 유사도 기반 문서 그룹화

    Args:
        docs: 문서 리스트
        title_field: 제목 필드명
        content_field: 내용 필드명
        client: OpenAI 클라이언트
        similarity_threshold: 같은 그룹으로 묶을 최소 유사도 (기본: 0.75)
        min_group_size: 최소 그룹 크기
        max_group_size: 최대 그룹 크기

    Returns:
        그룹화된 문서 리스트
    """
    if len(docs) < min_group_size:
        return []

    print(f"  Computing embeddings for {len(docs)} documents...")

    # 제목 + 내용 앞부분으로 임베딩 텍스트 생성
    texts = []
    for doc in docs:
        title = doc.get(title_field, "")
        content = doc.get(content_field, "")[:500]  # 앞 500자만
        texts.append(f"{title} {content}")

    # 임베딩 계산
    embeddings = get_embeddings(client, texts)

    # Agglomerative Clustering (유사도 기반)
    # distance_threshold로 클러스터 수 자동 결정
    distance_threshold = 1 - similarity_threshold  # cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # 클러스터별로 문서 그룹화
    cluster_docs = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_docs[label].append(docs[idx])

    # 그룹 크기 조정 및 필터링
    groups = []
    for cluster_id, cluster in cluster_docs.items():
        if len(cluster) < min_group_size:
            continue

        # max_group_size 초과 시 분할
        if len(cluster) > max_group_size:
            # 클러스터 내에서 유사도 높은 것끼리 서브그룹
            cluster_indices = [docs.index(d) for d in cluster]
            cluster_embeddings = embeddings[cluster_indices]

            # 유사도 행렬
            sim_matrix = cosine_similarity(cluster_embeddings)

            # Greedy하게 그룹 분할
            used = set()
            for i in range(len(cluster)):
                if i in used:
                    continue

                group = [cluster[i]]
                used.add(i)

                # 가장 유사한 문서들 추가
                similarities = [(j, sim_matrix[i][j]) for j in range(len(cluster)) if j not in used]
                similarities.sort(key=lambda x: x[1], reverse=True)

                for j, sim in similarities:
                    if len(group) >= max_group_size:
                        break
                    if sim >= similarity_threshold:
                        group.append(cluster[j])
                        used.add(j)

                if len(group) >= min_group_size:
                    groups.append(group)
        else:
            groups.append(cluster)

    print(f"  Created {len(groups)} groups from clustering")
    return groups


def process_data_type(
    data: List[Dict],
    data_type: str,
    filter_func,
    title_field: str,
    content_field: str,
    client: OpenAI,
    output_dir: Path,
    target_count: int = 50
) -> List[Dict]:
    """데이터 타입별 처리"""

    print(f"\n{'='*50}")
    print(f"Processing {data_type}...")
    print(f"{'='*50}")

    # 필터링
    selected = filter_func(data, target_count)
    print(f"Selected: {len(selected)}")

    # 쿼리 생성
    queries = []
    for i, item in enumerate(selected):
        title = item.get(title_field, "")
        content = item.get(content_field, "")
        doc_id = item.get("no", i + 1)

        print(f"[{i+1}/{len(selected)}] Generating query for: {title[:50]}...")

        query = generate_query_with_llm(client, title, content, data_type)

        queries.append({
            "query": query,
            "source_doc_id": doc_id,
            "data_type": data_type,
            "reference_title": title
        })

    # 선별 데이터 저장
    data_file = output_dir / f"{data_type}_50.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"Saved: {data_file}")

    return queries


def select_data_only():
    """데이터 선별만 수행 (쿼리 생성 없이)"""
    random.seed(42)

    data_dir = Path(__file__).parent.parent / "data" / "train"
    output_dir = Path(__file__).parent.parent / "data" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    with open(data_dir / "article_filtering.json", 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"Articles loaded: {len(articles)}")

    with open(data_dir / "patent_filtering.json", 'r', encoding='utf-8') as f:
        patents = json.load(f)
    print(f"Patents loaded: {len(patents)}")

    with open(data_dir / "project_filtering.json", 'r', encoding='utf-8') as f:
        projects = json.load(f)
    print(f"Projects loaded: {len(projects)}")

    # Article 선별
    print(f"\n{'='*50}")
    print("Filtering articles...")
    selected_articles = filter_articles(articles, 100)
    print(f"Selected: {len(selected_articles)}")

    with open(output_dir / "article_100.json", 'w', encoding='utf-8') as f:
        json.dump(selected_articles, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_dir / 'article_100.json'}")

    # Patent 선별
    print(f"\n{'='*50}")
    print("Filtering patents...")
    selected_patents = filter_patents(patents, 100)
    print(f"Selected: {len(selected_patents)}")

    with open(output_dir / "patent_100.json", 'w', encoding='utf-8') as f:
        json.dump(selected_patents, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_dir / 'patent_100.json'}")

    # Project 선별
    print(f"\n{'='*50}")
    print("Filtering projects...")
    selected_projects = filter_projects(projects, 100)
    print(f"Selected: {len(selected_projects)}")

    with open(output_dir / "project_100.json", 'w', encoding='utf-8') as f:
        json.dump(selected_projects, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_dir / 'project_100.json'}")

    print(f"\n{'='*50}")
    print("DATA SELECTION COMPLETE")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"  - article_100.json: {len(selected_articles)} items")
    print(f"  - patent_100.json: {len(selected_patents)} items")
    print(f"  - project_100.json: {len(selected_projects)} items")
    print("\nNext step: Run query generation with 'python select_eval_data.py --generate-queries'")


def main():
    """그룹 기반 쿼리 생성 (1 query = N documents)"""
    random.seed(42)  # 재현성

    # 경로 설정
    eval_dir = Path(__file__).parent.parent / "data" / "eval"
    output_dir = Path(__file__).parent.parent / "src" / "evaluation"

    # OpenAI 클라이언트
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 선별된 데이터 로드
    print("Loading selected evaluation data...")

    with open(eval_dir / "article_100.json", 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"Articles loaded: {len(articles)}")

    with open(eval_dir / "patent_100.json", 'r', encoding='utf-8') as f:
        patents = json.load(f)
    print(f"Patents loaded: {len(patents)}")

    with open(eval_dir / "project_100.json", 'r', encoding='utf-8') as f:
        projects = json.load(f)
    print(f"Projects loaded: {len(projects)}")

    all_queries = []

    # Article 그룹 쿼리 생성 (임베딩 기반)
    print(f"\n{'='*50}")
    print("Generating Article group queries (embedding-based)...")
    article_groups = group_documents_by_embedding(
        articles, "THSS_NM", "abstract", client,
        similarity_threshold=0.60, min_group_size=2, max_group_size=5
    )
    print(f"Article groups: {len(article_groups)}")

    for i, group in enumerate(article_groups):
        print(f"  [{i+1}/{len(article_groups)}] Group size: {len(group)}")
        query = generate_group_query_with_llm(client, group, "THSS_NM", "abstract")
        doc_ids = [doc.get("no", idx) for idx, doc in enumerate(group)]
        titles = [doc.get("THSS_NM", "")[:30] for doc in group]

        all_queries.append({
            "query": query,
            "source_doc_ids": doc_ids,
            "data_type": "article",
            "references": titles
        })
        print(f"    Query: {query}")

    # Patent 그룹 쿼리 생성 (임베딩 기반)
    print(f"\n{'='*50}")
    print("Generating Patent group queries (embedding-based)...")
    patent_groups = group_documents_by_embedding(
        patents, "kipris_application_name", "kipris_abstract", client,
        similarity_threshold=0.60, min_group_size=2, max_group_size=5
    )
    print(f"Patent groups: {len(patent_groups)}")

    for i, group in enumerate(patent_groups):
        print(f"  [{i+1}/{len(patent_groups)}] Group size: {len(group)}")
        query = generate_group_query_with_llm(client, group, "kipris_application_name", "kipris_abstract")
        doc_ids = [doc.get("no", idx) for idx, doc in enumerate(group)]
        titles = [doc.get("kipris_application_name", "")[:30] for doc in group]

        all_queries.append({
            "query": query,
            "source_doc_ids": doc_ids,
            "data_type": "patent",
            "references": titles
        })
        print(f"    Query: {query}")

    # Project 그룹 쿼리 생성 (임베딩 기반)
    print(f"\n{'='*50}")
    print("Generating Project group queries (embedding-based)...")
    project_groups = group_documents_by_embedding(
        projects, "PRJ_NM", "summary", client,
        similarity_threshold=0.60, min_group_size=2, max_group_size=5
    )
    print(f"Project groups: {len(project_groups)}")

    for i, group in enumerate(project_groups):
        print(f"  [{i+1}/{len(project_groups)}] Group size: {len(group)}")
        query = generate_group_query_with_llm(client, group, "PRJ_NM", "summary")
        doc_ids = [doc.get("no", idx) for idx, doc in enumerate(group)]
        titles = [doc.get("PRJ_NM", "")[:30] for doc in group]

        all_queries.append({
            "query": query,
            "source_doc_ids": doc_ids,
            "data_type": "project",
            "references": titles
        })
        print(f"    Query: {query}")

    # test_queries.json 생성
    test_queries = {
        "version": "4.0",
        "description": "산학협력 매칭 시스템 평가용 테스트 쿼리셋 - 그룹 기반 (1 query = N documents)",
        "queries": all_queries
    }

    queries_file = output_dir / "test_queries.json"
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved test_queries.json: {queries_file}")

    # 요약
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total queries: {len(all_queries)}")
    print(f"  - Article groups: {len(article_groups)}")
    print(f"  - Patent groups: {len(patent_groups)}")
    print(f"  - Project groups: {len(project_groups)}")
    total_docs = sum(len(q['source_doc_ids']) for q in all_queries)
    print(f"  - Total documents covered: {total_docs}")
    print(f"\nOutput: {queries_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-queries":
        main()
    else:
        select_data_only()
