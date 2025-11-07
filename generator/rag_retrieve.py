# rag_retrieve.py
import os, sys
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# == 프로젝트 설정 ==
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from config import settings

# 업서트 때 썼던 것과 "완전히 동일한" 임베딩 설정을 사용하세요.
# (예: text-embedding-3-small + dimensions=512 를 썼다면 동일하게!)
def get_vectorstore() -> PineconeVectorStore:
    emb = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model="text-embedding-3-small",
        # 업서트에서 dimensions를 지정했었다면 동일하게 지정
        # dimensions=512,
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=emb,
        pinecone_api_key=settings.pinecone_api_key,
    )

def build_filter(
    product: Optional[str] = None,
    category: Optional[str] = None,
    score_min: Optional[int] = None,
    score_max: Optional[int] = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if product:
        f["product"] = {"$ne": product}
    if category:
        f["category"] = {"$eq": category}
    if score_min is not None or score_max is not None:
        rng = {}
        if score_min is not None: rng["$gte"] = int(score_min)
        if score_max is not None: rng["$lte"] = int(score_max)
        f["score"] = rng
    return f

def make_hwc_block(docs, limit=5, max_chars=150) -> str:
    lines: List[str] = []
    for d in docs[:limit]:
        t = (d.page_content or "").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + "..."
        # meta = d.metadata or {}
        # nm = meta.get("product")
        # sc = meta.get("score")
        # lines.append(f"- ({nm}/{sc}) {t}")
        lines.append(f"{t}")
    return "\n".join(lines)

def retrieve_similar_comments(
    product_title: str,
    category: str,
    stars: int,
    k: int = 8,
):
    vs = get_vectorstore()
    # 점수 대역 매핑(원하면 조정 가능)
    score_min, score_max = {
        20: (0, 40),
        40: (0, 60),
        60: (40, 80),
        80: (60, 100),
        100:(80, 100),
    }.get(int(stars), (0, 100))

    flt = build_filter(product=product_title, category=category, score_min=score_min, score_max=score_max)

    retriever = vs.as_retriever(
        search_kwargs={"k": k, "filter": flt}
    )
    # 쿼리는 단순하게 시작 → 필요시 키워드/브랜드/속성 등 보강
    query_text = f"{product_title} {category} {stars} 후기 리뷰"
    return retriever.get_relevant_documents(query_text)
