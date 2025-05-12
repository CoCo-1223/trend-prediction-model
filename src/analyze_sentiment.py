# src/analyze_sentiment.py
# 키워드 감성 분석 코드 

import glob, os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm   # 진행바

# ——— 설정 ———
IN_DIR   = "./data/processed"
OUT_DIR  = "./data/processed"
COLUMNS  = ["일자", "제목", "키워드"]
MODEL    = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH    = 64        # 한 번에 처리할 문장 수

os.makedirs(OUT_DIR, exist_ok=True)

# ——— 감성 파이프라인 (CPU) ———
sentiment = pipeline(
    "sentiment-analysis",
    model=MODEL,
    device=-1
)

def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    df["기업"] = (
        df["제목"].str.contains("삼성") .map({True:"samsung", False:None})
        .fillna( df["키워드"].str.contains("삼성").map({True:"samsung", False:None}) )
    )
    df["기업"] = df["기업"].fillna(
        df["제목"].str.contains("애플").map({True:"apple", False:None})
    ).fillna(
        df["키워드"].str.contains("애플").map({True:"apple", False:None})
    )
    return df.dropna(subset=["기업"])

# ——— 연도별 배치 처리 ———
for path in glob.glob(os.path.join(IN_DIR, "news_*.xlsx")):
    year = os.path.basename(path).split("_")[1].split(".")[0]
    df = pd.read_excel(path, usecols=COLUMNS)
    
    # 날짜 파싱
    df["일자"] = pd.to_datetime(
        df["일자"].astype(str),
        format="%Y%m%d",
        errors="coerce"
        )
    
    # 삼성/애플 필터
    df = classify_all(df)
    
    # 문장화
    df["문장"] = df["제목"].fillna("") + " " + df["키워드"].str.replace(",", " ")
    
    # 배치로 감정점수(1~5) 계산
    scores = []
    texts  = df["문장"].tolist()
    for i in tqdm(range(0, len(texts), BATCH), desc=f"{year} batch"):
        batch = texts[i : i + BATCH]
        outs  = sentiment(batch, truncation=True, max_length=512)
        # '1 star' → 1.0, ..., '5 stars' → 5.0
        scores.extend([float(item["label"].split()[0]) for item in outs])
    
    df["감정점수"] = scores

    df["일자"] = df["일자"].dt.strftime("%Y-%m-%d")
    
    df = df.sort_values("일자")
    
    # 기업별로 저장
    for comp in ["samsung", "apple"]:
        sub = df[df["기업"] == comp][["일자","제목","키워드","감정점수"]]
        outp = os.path.join(OUT_DIR, f"{comp}_sentiment_{year}.xlsx")
        sub.to_excel(outp, index=False)
        print(f"Saved: {outp}")



# 1에 가까울수록 부정, 5에 가까울수록 긍정 
'''
1 -> 매우 부정
3 -> 중립
5 -> 매우 긍정 
'''