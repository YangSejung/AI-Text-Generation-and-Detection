import os, sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data")
OUT_TRAIN = os.path.join(DATA_PATH, "reviews_detector_train.csv")
OUT_TEST  = os.path.join(DATA_PATH, "reviews_detector_test.csv")
OUT_HUMAN_EVAL = os.path.join(DATA_PATH, "reviews_human_eval_sample.csv")

def sample_group(g):
    return g.sample(
        n=3,
        replace=(len(g) < 3),  # 데이터가 3개 미만이면 중복 허용
        random_state=42        # 매번 동일하게 뽑기
    )


def main():
    col = ['Product', 'Score', 'Comment', 'IsLGC', 'ModelUsed']
    df_train = pd.read_csv(OUT_TRAIN)
    df_test  = pd.read_csv(OUT_TEST)

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Score 숫자만 남게 
    df["Score"] = (
        df["Score"]
        .astype(str)
        .str.replace("점", "", regex=False)
        .str.strip()
        .astype(int)
    )

    # 200자 이하
    df = df[df["Comment"].astype(str).str.len() <= 200]

    # 샘플링
    sampled = (
        df.groupby(["Score", "IsLGC"], group_keys=False)
          .apply(sample_group)
          .reset_index(drop=True)
    )
    
    # 열 선택
    sampled = sampled[col]

    # 셔플
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    # 저장
    sampled.to_csv(OUT_HUMAN_EVAL, index=False, encoding="utf-8")
    print(f"Saved human eval file to: {OUT_HUMAN_EVAL}")

if __name__ == "__main__":
    main()
