# upsert.py
"""
all_reviews_clean.csv -> Pinecone 업서트 스크립트
- 입력 CSV 형식: Product,Category,Score,Date,Comment
- 임베딩: OpenAI text-embedding-3-small (비용↓) / 필요시 large로 변경
- 메타데이터: product, category, score(int), date
- 재실행 시 중복 방지: (product|date|comment) 기반 해시 ID
"""

import os, sys
import re
import uuid
import hashlib
import math
import pandas as pd
from typing import List, Dict

# == 프로젝트 설정 ==
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
INPUT_CSV = os.path.join(DATA_PATH, "all_reviews_clean.csv")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import settings

# == 임베딩 & 벡터 스토어 ==
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def read_csv_kr(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def score_to_int(score_val) -> int:
    """
    Score가 '80점', '100점' 같이 들어있는 경우 정수로 변환.
    숫자만 있으면 그대로 int.
    변환 실패 시 0 반환.
    """
    if pd.isna(score_val):
        return 0
    s = str(score_val)
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0


def make_id(product: str, date: str, comment: str) -> str:
    """
    재실행 시 같은 문서가 같은 id로 업서트되게 안정 ID 생성.
    """
    base = f"{product}|{date}|{comment}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return str(uuid.UUID(h[:32]))  # sha1 앞 32자리로 UUID 형식화


def build_payloads(df: pd.DataFrame):
    """
    PineconeVectorStore.add_texts 를 위한 (texts, metadatas, ids) 준비.
    """
    texts: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for row in df.itertuples(index=False):
        product = getattr(row, "Product")
        category = getattr(row, "Category")
        score_raw = getattr(row, "Score")
        date = getattr(row, "Date")
        comment = getattr(row, "Comment")

        if not isinstance(comment, str) or not comment.strip():
            continue

        score = score_to_int(score_raw)
        _id = make_id(product, str(date), comment)

        texts.append(comment)
        metadatas.append({
            "product": product,
            "category": category,
            "score": score,
            "date": str(date),
        })
        ids.append(_id)

    return texts, metadatas, ids


def batched(iterable, batch_size: int):
    total = len(iterable)
    for i in range(0, total, batch_size):
        yield i, iterable[i:i+batch_size]


def main():
    # 0) 입력 확인
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_CSV}")

    # 1) CSV 로드
    df = read_csv_kr(INPUT_CSV)
    expected_cols = {"Product", "Category", "Score", "Date", "Comment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 컬럼이 예상과 다릅니다. 누락: {missing} / 현재: {list(df.columns)}")

    # 2) 임베딩/벡터 스토어 초기화
    # - 모델: text-embedding-3-small (차원 1536). large(3072)로 바꾸면 인덱스 차원도 맞춰야 함.
    emb = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model="text-embedding-3-small"
    )

    # Pinecone 클라이언트
    # pc = Pinecone(api_key=settings.pinecone_api_key)
    # (필요하면 index 생성 코드 추가 가능)
    # 예: pc.create_index(name=settings.pinecone_index_name, dimension=1536, spec=...)

    index_name = settings.pinecone_index_name

    # LangChain VectorStore wrapper
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=emb,
        pinecone_api_key=settings.pinecone_api_key,
    )

    # 3) payloads 구성
    texts, metadatas, ids = build_payloads(df)
    if not texts:
        print("[WARN] 업서트할 레코드가 없습니다.")
        return

    # 4) 배치 업서트
    BATCH = 100  # Pinecone 권장에 맞춰 적절히 조절
    total = len(texts)
    print(f"[INFO] Upserting {total} records into Pinecone index='{index_name}' ...")
    for i, idx_batch in batched(list(range(total)), BATCH):
        batch_texts = [texts[j] for j in idx_batch]
        batch_metas = [metadatas[j] for j in idx_batch]
        batch_ids = [ids[j] for j in idx_batch]

        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids,
        )
        print(f"  - upserted {i + len(idx_batch)}/{total}")

    print("[DONE] Upsert finished.")

if __name__ == "__main__":
    main()
