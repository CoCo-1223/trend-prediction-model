# 감성 분석 코드 

import glob
import os
import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

# ——— 설정 ———
IN_DIR   = "./data/processed"
PATTERN  = "NewsResult_*_processed.csv"
OUT_DIR  = "./data/processed"
BATCH    = 32
MODEL_ID = "snunlp/KR-FinBert-SC"

os.makedirs(OUT_DIR, exist_ok=True)

# 1) 파이프라인 로드
sent_pipe = pipeline(
    "sentiment-analysis",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,    # GPU 사용 시 0
    top_k=None    # 각 라벨별 확률 모두 반환
)

def company_of(txt: str) -> str:
    t = txt.lower()
    if "삼성전자" in t or "삼성" in t:
        return "samsung"
    if "애플" in t:
        return "apple"
    return None

def to_score(probs: dict) -> float:
    neg = probs.get("negative", 0.0)
    neu = probs.get("neutral",  0.0)
    pos = probs.get("positive", 0.0)
    # 중립=3점, 긍정·부정의 차이에 따라 ±2점 범위 매핑
    score = 3.0 + 2.0*(pos - neg)
    return max(1.0, min(5.0, score))

# 2) 연도별 처리
for path in glob.glob(os.path.join(IN_DIR, PATTERN)):
    year = os.path.basename(path).split("_")[1]
    print(f"\n▶ Processing {year}...")

    # CSV 로드
    df = pd.read_csv(path, dtype=str)
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    # 필요한 칼럼 체크
    if not {"일자","제목","키워드"}.issubset(df.columns):
        print(f"  ⚠️ {path}에 '일자','제목','키워드' 컬럼이 없습니다.")
        continue

    # 날짜 파싱 & 포맷
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 텍스트 결합 & 회사 분류
    df["text"]  = df["제목"].fillna("") + " 키워드:" + df["키워드"].fillna("")
    df["기업"]  = df["text"].apply(company_of)
    df = df.dropna(subset=["기업"]).reset_index(drop=True)

    # 감성 예측
    labels, scores = [], []
    texts = df["text"].tolist()
    for i in tqdm(range(0, len(texts), BATCH), desc=f"{year} batch"):
        batch = texts[i : i + BATCH]
        outs  = sent_pipe(batch, truncation=True, max_length=512)
        for out in outs:
            # out이 list 또는 dict 형태일 수 있음
            if isinstance(out, list):
                prob_map = {x["label"]: x["score"] for x in out}
            else:
                prob_map = out
            # 최고 확률 라벨
            best_label = max(prob_map, key=prob_map.get)
            labels.append(best_label)
            scores.append(to_score(prob_map))

    df["감정라벨"] = labels
    df["감정점수"] = scores
    df = df.sort_values("일자")

    # 회사별로 CSV 저장
    for comp in ["samsung", "apple"]:
        sub = df[df["기업"] == comp][
            ["일자","제목","키워드","감정라벨","감정점수"]
        ]
        outp = os.path.join(OUT_DIR, f"{comp}_sentiment_{year}.csv")
        sub.to_csv(outp, index=False, encoding="utf-8-sig")
        print(f"  → Saved: {os.path.basename(outp)}")
