import os, re
import pandas as pd
import numpy as np
from typing import Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
#OUTPUT = os.path.join(DATA_PATH, "all_reviews_clean.csv")
TRAIN_OUT = os.path.join(DATA_PATH, "real_train.csv")
TEST_OUT  = os.path.join(DATA_PATH, "real_test.csv")

DANAWA_CATEGORIES: list[Tuple[str, str]] = [
    # 가전
    ("TV", "tv"),
    ("드럼세탁기", "drum_washer"),
    ("건조기", "dryer"),
    ("청소기", "vacuum_cleaner"),
    # 전자기기
    ("노트북", "laptop"),
    ("모니터", "monitor"),
    ("휴대폰", "smartphone"),
    ("스마트워치", "smartwatch"),
    # 스포츠
    ("러닝화", "running_shoes"),
    ("운동화/스니커즈", "sneakers"),
    ("등산화/트래킹화", "hiking_shoes"),
    ("축구화", "soccer_shoes"),
    # 가구
    ("침대", "bed"),
    ("매트리스/토퍼", "mattress_topper"),
    ("소파", "sofa"),
    ("학생/사무용의자", "office_chair"),
    # 식품
    ("헬스/다이어트식품", "health_diet_food"),
    ("홍삼", "red_ginseng"),
    ("유산균", "probiotics"),
    ("비타민/미네랄", "vitamin_mineral"),
    # 유아/완구
    ("유모차", "stroller"),
    ("레고/블럭", "lego_blocks"),
    ("역할놀이/소꿉놀이", "role_play_toys"),
    ("신생아/영유아완구", "infant_toys"),
    # 뷰티
    ("기초세트", "basic_skincare_set"),
    ("스킨/토너", "skin_toner"),
    ("로션", "lotion"),
    ("크림/수딩젤", "cream_soothing_gel"),
]

def read_csv_kr(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")
    
def normalize_comment(s: pd.Series) -> pd.Series:
    # 공백 정리, 앞뒤 공백 제거, None/NaN 처리
    s = s.astype(str).fillna("").str.strip()
    # 연속 공백을 하나로, 탭/개행도 공백으로 통일
    s = s.apply(lambda x: re.sub(r"\s+", " ", x))
    return s

def process_one_file(path: str, category_key: str) -> pd.DataFrame:
    df = read_csv_kr(path)

    # 컬럼 이름 양끝 공백 제거(예방)
    df.columns = [c.strip() for c in df.columns]

    # 필요한 컬럼 존재 확인
    required = {"Pid", "Product", "User_id", "Score", "Mall", "Date", "Comment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{path}] 누락 컬럼: {missing}")

    # 댓글 정규화
    df["Comment"] = normalize_comment(df["Comment"])

    # 길이 필터(> 20자)
    df = df[df["Comment"].str.len() > 20]

    # 파일 내부 중복 제거 (Comment 기준)
    df = df.drop_duplicates(subset=["Comment"], keep="first")

    # 필요시 추가 전처리 예: 결측 제거
    df = df.dropna(subset=["Comment"])

    # 카테고리 컬럼 추가 (파일 value 사용: tv, drum_washer, ...)
    df["Category"] = category_key

    # 출력 컬럼 정렬 (요청 형식)
    keep_cols = ["Product", "Category", "Score", "Date", "Comment"]
    df = df[keep_cols]

    # 원본에 불필요한 인덱스 컬럼이 있으면 제거(옵션)
    if "Pid" in df.columns:
        df = df.drop(columns=["Pid"])
    if "User_id" in df.columns:
        df = df.drop(columns=["User_id"])
    if "Mall" in df.columns:
        df = df.drop(columns=["Mall"])

    return df


def split_train_test_by_ratio(df: pd.DataFrame, ratio: float = 0.8, seed: int = 2025):
    """
    카테고리별 제품 단위로 train/test를 ratio 비율로 나눕니다.
    - ratio=0.8 → 80% train, 20% test
    - 같은 제품의 모든 댓글은 동일 split으로 유지
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for cat, g in df.groupby("Category"):
        products = g["Product"].dropna().unique().tolist()
        rng.shuffle(products)
        n = len(products)
        n_train = max(1, int(round(n * ratio)))

        tr_prods = set(products[:n_train])
        te_prods = set(products[n_train:])

        train_parts.append(g[g["Product"].isin(tr_prods)])
        test_parts.append(g[g["Product"].isin(te_prods)])

        print(f"[SPLIT] Category={cat} | products={n} → train={len(tr_prods)}, test={len(te_prods)}")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df

def main():
    frames = []
    for key, value in DANAWA_CATEGORIES:
        csv_path = os.path.join(DATA_PATH, f"{value}_reviews.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] 파일 없음: {csv_path}")
            continue

        try:
            df_clean = process_one_file(csv_path, category_key = key)
            frames.append(df_clean)
            print(f"[OK] {csv_path}: {len(df_clean)} rows")
        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    if not frames:
        print("처리할 데이터가 없습니다.")
        return

    merged = pd.concat(frames, ignore_index=True)

    # 요구사항: 중복은 “같은 파일 내에서만” 제거했으므로
    # 여기서는 교차 파일 중복 제거를 하지 않습니다.
    # (교차 파일 중복도 제거하려면 아래 주석 해제)
    # merged = merged.drop_duplicates(subset=["Comment"], keep="first")

    # 저장 
    # merged.to_csv(OUTPUT, index=False, encoding="utf-8")
    # print(f"[DONE] Saved: {OUTPUT} (rows={len(merged)})")

    # 저장
    train_df, test_df = split_train_test_by_ratio(merged, ratio=0.8, seed=2025)
    train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8")
    test_df.to_csv(TEST_OUT, index=False, encoding="utf-8")

    print(f"[DONE] Train saved: {TRAIN_OUT} (rows={len(train_df)})")
    print(f"[DONE] Test  saved: {TEST_OUT}  (rows={len(test_df)})")

if __name__ == "__main__":
    main()