# make_lgc_from_real_test.py

import os, sys
import asyncio
from typing import Dict, List
import pandas as pd
import math, hashlib, random


# == 프로젝트 설정 ==
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
INPUT_CSV = os.path.join(DATA_PATH, "real_test.csv")
OUTPUT_CSV = os.path.join(DATA_PATH, "lgc_test_generated.csv")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from generator.generate import run_once

STARS = [20, 40, 60, 80, 100]
CONCURRENCY = 4          # 동시 생성 개수 (API 한도 고려)
RETRIES = 2              # 실패 재시도 횟수
SLEEP_BETWEEN = 0.1      # 호출 사이 간격(선택)
# 1) 카테고리별 기본 길이 범위(한국어 Category 그대로 사용)
CATEGORY_LEN_CONFIG = {
    # 가전/전자 (고가 → 길게)
    "TV": (110, 150),
    "드럼세탁기": (110, 150),
    "로봇청소기": (110, 150),         # 예시: 있으면 추가
    "건조기": (110, 150),
    "노트북": (100, 140),
    "모니터": (90, 130),
    "휴대폰": (90, 130),
    "스마트워치": (80, 120),

    # 생활/스포츠/가구 (중간)
    "청소기": (90, 130),
    "침대": (90, 130),
    "소파": (90, 130),
    "매트리스/토퍼": (90, 130),
    "학생/사무용의자": (80, 120),
    "러닝화": (50, 110),
    "운동화/스니커즈": (50, 110),
    "등산화/트래킹화": (50, 110),
    "축구화": (50, 110),

    # 식품/뷰티/유아 (짧게~중간)
    "홍삼": (60, 100),
    "유산균": (60, 100),
    "비타민/미네랄": (60, 100),
    "기초세트": (50, 100),
    "스킨/토너": (50, 100),
    "로션": (50, 100),
    "크림/수딩젤": (50, 100),
    "유모차": (60, 110),
    "레고/블럭": (50, 100),
    "역할놀이/소꿉놀이": (50, 100),
    "신생아/영유아완구": (50, 100),
}

DEFAULT_LEN_RANGE = (50, 110)

# 2) 별점 보정(문자 수 +delta)
STARS_LEN_DELTA = {
    20: +5,   # 불만
    40: +0,
    60: +0,     # 보통
    80: +0,
    100: +10,  # 칭찬
}

# 3) 대상별 재현 가능한 난수 생성용 시드
def _seed_from(product: str, category: str, stars: int) -> int:
    h = hashlib.sha1(f"{product}|{category}|{stars}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def sample_target_len(category: str, stars: int, product: str = "") -> int:
    # 카테고리 기본 범위
    base_min, base_max = CATEGORY_LEN_CONFIG.get(category, DEFAULT_LEN_RANGE)

    # 별점 보정
    delta = STARS_LEN_DELTA.get(int(stars), 0)
    base_min += delta
    base_max += delta

    # 소량 지터(재현 가능)
    rng = random.Random(_seed_from(product, category, stars))
    jitter = rng.randint(-15, 15)   # ±10자

    target = int((base_min + base_max) / 2 + jitter)

    # 안전 클램프 (프롬프트 권장 30~150자 기준)
    target = max(30, min(target, 150))
    return target

def read_csv_kr(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def load_unique_products(df: pd.DataFrame) -> List[str]:
    """real_test.csv에서 중복 없이 Product만 추출."""
    if "Product" not in df.columns or "Category" not in df.columns:
        raise ValueError("real_test.csv에는 최소한 Product, Category 컬럼이 있어야 합니다.")
    return df["Product"].dropna().drop_duplicates().tolist()

def product_to_category_map(df: pd.DataFrame) -> Dict[str, str]:
    """각 Product의 대표 Category를 매핑(첫 번째 값 사용)."""
    return df.groupby("Product")["Category"].first().to_dict()

async def gen_one(product: str, category: str, score: int, sem: asyncio.Semaphore):
    """단일 (Product,Category,Score) → Comment 생성 (재시도/백오프 포함)."""
    backoff = 0.8
    attempt = 0
    async with sem:
        while True:
            try:
                target_len = sample_target_len(category=category, stars=score, product=product)
                
                out = await run_once(
                    product_title=product,
                    category=category,
                    stars=int(score),
                    target_len=int(target_len),
                )
                model_used = out.get("model") if isinstance(out, dict) else "unknown"
                inner = out.get("result", {}) if isinstance(out, dict) else {}
                comment = inner.get("answer") if isinstance(inner, dict) else str(inner)

                if not isinstance(comment, str) or not comment.strip():
                    raise ValueError("Empty comment from LLM")
                # 성공
                return {
                    "Product": product,
                    "Category": category,
                    "Score": int(score),
                    "Comment": comment.strip(),
                    "IsLGC": True,
                    "ModelUsed": model_used,
                }
            except Exception as e:
                attempt += 1
                if attempt > RETRIES:
                    # 실패시 에러 메시지라도 기록
                    print(f"[SKIP] {product}({score}) 생성 실패: {type(e).__name__} - {e}")
                    return None # 나중에 필터링에서 스킵
                
                    # return {
                    #     "Product": product,
                    #     "Category": category,
                    #     "Score": int(score),
                    #     "Comment": f"[ERROR:{type(e).__name__}] {e}",
                    #     "IsLGC": True,
                    #     "ModelUsed": model_used,
                    # }
                await asyncio.sleep(backoff)
                backoff *= 1.6

async def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"입력 파일 없음: {INPUT_CSV}")

    df = read_csv_kr(INPUT_CSV)
    products = load_unique_products(df)
    prod2cat = product_to_category_map(df)

    print(f"[INFO] Unique products in real_test.csv: {len(products)}")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []

    # 각 Product에 대해 5개 Score로 생성
    for p in products:
        category = prod2cat.get(p, "")
        if not isinstance(category, str) or not category.strip():
            # 카테고리 누락 시 스킵(또는 기본 카테고리 지정)
            print(f"[WARN] Category not found for product: {p} — skip")
            continue
        for s in STARS:
            tasks.append(gen_one(p, category, s, sem))

    results = []
    # 진행률 간단 출력
    for i in range(0, len(tasks), 20):
        chunk = tasks[i:i+20]
        chunk_results = await asyncio.gather(*chunk)
        chunk_results = [r for r in chunk_results if r is not None]
        results.extend(chunk_results)
        print(f"progress: {min(i+20, len(tasks))}/{len(tasks)}")
        await asyncio.sleep(SLEEP_BETWEEN)

    out_df = pd.DataFrame(results, columns=["Product", "Category", "Score", "Comment", "IsLGC", "ModelUsed"])

    # 완전 중복 코멘트 제거(원하면 유지 가능)
    out_df = out_df.drop_duplicates(subset=["Product", "Category", "Score", "Comment"], keep="first")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[DONE] Saved: {OUTPUT_CSV} (rows={len(out_df)})")

if __name__ == "__main__":
    asyncio.run(main())
