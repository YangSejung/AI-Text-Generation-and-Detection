# detector_pipeline.py
import os, sys, re, json, numpy as np, pandas as pd, torch
from typing import Dict, Any, Tuple, List
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# =========================
# 설정
# =========================
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data")
REAL_TEST = os.path.join(DATA_PATH, "real_test.csv")            # HWC (실제 리뷰)
LGC_FILE  = os.path.join(DATA_PATH, "lgc_test_generated.csv")   # LGC (생성 리뷰)

OUT_TRAIN = os.path.join(DATA_PATH, "reviews_detector_train.csv")
OUT_TEST  = os.path.join(DATA_PATH, "reviews_detector_test.csv")

SEED = 2025
TRAIN_RATIO = 0.7          # 제품 단위 7:3 분할
BALANCE_MAX_EACH = None    # (선택) 클래스 균형 맞춤 상한 (예: 1000). None이면 자동 최소치.

# 경량 한국어 특화 모델
MODEL_NAME = "klue/roberta-small"      # 대안: "beomi/KcELECTRA-base"
MAX_LEN    = 192
NUM_LABELS = 2
EPOCHS     = 4
LR         = 2e-5
BSZ_TRAIN  = 32
BSZ_EVAL   = 64

OUT_DIR = "./kor_detector_light"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 유틸
# =========================
def set_seed_all(seed: int = 2025):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def need_cols(df: pd.DataFrame, cols: List[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"누락 컬럼: {miss}")

def normalize_text(t: str) -> str:
    t = str(t)
    t = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', t)
    t = re.sub(r'\S+@\S+', ' <EMAIL> ', t)
    t = re.sub(r'\d[\d,\.]*', ' <NUM> ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def split_by_product(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, set, set]:
    rng = np.random.default_rng(seed)
    prods = df["Product"].dropna().unique().tolist()
    rng.shuffle(prods)
    n_tr = max(1, int(round(len(prods) * train_ratio)))
    prod_train = set(prods[:n_tr]); prod_test = set(prods[n_tr:])
    return (
        df[df["Product"].isin(prod_train)].copy(),
        df[df["Product"].isin(prod_test)].copy(),
        prod_train, prod_test
    )

def balance_pos_neg(pos_df: pd.DataFrame, neg_df: pd.DataFrame, max_each=None, seed=SEED):
    if max_each is None:
        n = min(len(pos_df), len(neg_df))
    else:
        n = min(max_each, len(pos_df), len(neg_df))
    pos_b = pos_df.sample(n, random_state=seed) if len(pos_df) > n else pos_df
    neg_b = neg_df.sample(n, random_state=seed) if len(neg_df) > n else neg_df
    return pos_b, neg_b

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"acc": acc, "precision": pr, "recall": rc, "f1": f1}

def slice_eval(df: pd.DataFrame, preds: np.ndarray, name: str, key: str):
    out = []
    if key not in df.columns:
        return out
    for val in sorted(df[key].dropna().unique()):
        mask = (df[key] == val).values
        if mask.sum() < 20:   # 너무 작은 슬라이스는 스킵
            continue
        y = df.loc[mask, "IsLGC"].astype(int).values
        p = preds[mask]
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        acc = accuracy_score(y, p)
        out.append({"slice": f"{name}:{key}={val}", "acc": acc, "precision": pr, "recall": rc, "f1": f1, "n": int(mask.sum())})
    return out

def to_int_score(x):
    s = str(x)
    m = re.search(r"\d+", s)
    return int(m.group()) if m else -1  # 없으면 -1 등 기본값

# =========================
# 파이프라인
# =========================
def main():
    set_seed_all(SEED)

    # ---- 1) 데이터 로드 + 기본 정리
    hwc = pd.read_csv(REAL_TEST)
    lgc = pd.read_csv(LGC_FILE)

    need_cols(hwc, ["Product","Category","Score","Comment"])
    need_cols(lgc, ["Product","Category","Score","Comment","IsLGC","ModelUsed"])

    if "IsLGC" not in hwc.columns:
        hwc["IsLGC"] = False
    if "ModelUsed" not in hwc.columns:
        hwc["ModelUsed"] = "HUMAN"
    # LGC 라벨 보정(문자열 → bool)
    if lgc["IsLGC"].dtype != bool:
        lgc["IsLGC"] = lgc["IsLGC"].astype(str).str.lower().isin(["true","1","t","y","yes"])

    # ---- 2) real_test 기준으로 제품 단위 분할 만들고 HWC/LGC 모두 동일 분할 적용
    hwc_train, hwc_test, prod_train, prod_test = split_by_product(hwc, TRAIN_RATIO, SEED)
    lgc_train = lgc[lgc["Product"].isin(prod_train)].copy()
    lgc_test  = lgc[lgc["Product"].isin(prod_test)].copy()

    print(f"[SPLIT] products total={len(prod_train|prod_test)} train={len(prod_train)} test={len(prod_test)}")
    print(f"  HWC train={len(hwc_train)}, test={len(hwc_test)} | LGC train={len(lgc_train)}, test={len(lgc_test)}")

    # ---- 3) (선택) 클래스 균형 맞추고 train/test 결합
    lgc_train_b, hwc_train_b = balance_pos_neg(lgc_train, hwc_train, BALANCE_MAX_EACH, SEED)
    lgc_test_b,  hwc_test_b  = balance_pos_neg(lgc_test,  hwc_test,  BALANCE_MAX_EACH, SEED)
    train_df = pd.concat([lgc_train_b, hwc_train_b], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df  = pd.concat([lgc_test_b,  hwc_test_b],  ignore_index=True).sample(frac=1.0,  random_state=SEED).reset_index(drop=True)

    # (원하면 저장)
    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8")
    print(f"[SAVE] train={OUT_TRAIN} ({len(train_df)} rows) | test={OUT_TEST} ({len(test_df)} rows)")

    # ---- 4) 정규화 + 토크나이즈
    for df in (train_df, test_df):
        df["Score"] = df["Score"].apply(to_int_score).astype("int64")
        df["text"] = df["Comment"].apply(normalize_text)
        df["ModelUsed"] = df.get("ModelUsed", "HUMAN")  # 없는 경우 대비
        df["label"] = df["IsLGC"].astype(int)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(b): return tok(b["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    # 누락 컬럼 대비
    for df in (train_df, test_df):
        for c in ["Product","Category","Score","ModelUsed"]:
            if c not in df.columns:
                df[c] = None

    ds_tr = Dataset.from_pandas(train_df[["text","label","Product","Category","Score","ModelUsed"]]).map(tokenize, batched=True)
    ds_te = Dataset.from_pandas(test_df[["text","label","Product","Category","Score","ModelUsed"]]).map(tokenize, batched=True)
    ds_tr = ds_tr.remove_columns(["text"]); ds_te = ds_te.remove_columns(["text"])
    ds_tr.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    ds_te.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    # ---- 5) 모델/학습 설정
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    args = TrainingArguments(
        output_dir=OUT_DIR, seed=SEED,
        per_device_train_batch_size=BSZ_TRAIN,
        per_device_eval_batch_size=BSZ_EVAL,
        learning_rate=LR, num_train_epochs=EPOCHS,
        weight_decay=0.01, warmup_ratio=0.06,
        logging_steps=50,
        save_strategy="epoch", evaluation_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_te,
        tokenizer=tok, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ---- 6) 학습/평가/저장
    trainer.train()
    metrics = trainer.evaluate()
    print("[EVAL] main:", metrics)

    trainer.save_model(os.path.join(OUT_DIR, "best"))
    tok.save_pretrained(os.path.join(OUT_DIR, "best"))

    # ---- 7) 상세 리포트 + 슬라이스 + 오분류 덤프
    pred_logits = trainer.predict(ds_te).predictions
    pred_labels = np.argmax(pred_logits, axis=1)
    true_labels = test_df["label"].values

    print("\n=== Classification Report (overall) ===")
    print(classification_report(true_labels, pred_labels, digits=4))
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:\n", cm)

    # 오분류 Top 샘플 (최대 30개)
    wrong_idx = np.where(pred_labels != true_labels)[0][:30]
    err_dump = test_df.iloc[wrong_idx][["Product","Category","Score","ModelUsed","Comment","IsLGC"]].copy()
    err_dump.to_csv(os.path.join(OUT_DIR, "misclassified_samples.csv"), index=False, encoding="utf-8-sig")
    print(f"[DUMP] misclassified -> {os.path.join(OUT_DIR, 'misclassified_samples.csv')} (rows={len(err_dump)})")

    # 슬라이스 리포트
    slices = []
    slices += slice_eval(test_df, pred_labels, "test", "Category")
    slices += slice_eval(test_df, pred_labels, "test", "Score")
    slices += slice_eval(test_df, pred_labels, "test", "ModelUsed")
    with open(os.path.join(OUT_DIR, "slice_report.json"), "w", encoding="utf-8") as f:
        json.dump(slices, f, ensure_ascii=False, indent=2)
    print(f"[SLICE] wrote {len(slices)} entries -> {os.path.join(OUT_DIR, 'slice_report.json')}")

    print("\n[DONE] Saved best model & reports in:", OUT_DIR)

if __name__ == "__main__":
    main()
