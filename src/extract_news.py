# src/extract_news.py
# 기존 /raw/news 에서 원하는 키워드만 추출 (일자, 제목 키워드)

import glob
import pandas as pd
import os

RAW_DIR  = "./data/raw"
OUT_DIR  = "./data/processed"
# 데이터 분석에 필요한 일자, 제목, 키워드만 추출 
COLUMNS  = ["일자", "제목", "키워드"] 

os.makedirs(OUT_DIR, exist_ok=True)

for path in glob.glob(os.path.join(RAW_DIR, "news_data_*.xlsx")):
    df = pd.read_excel(path, usecols=COLUMNS)
    # 파일명에서 연도 추출
    year = os.path.basename(path).split("_")[2][:4]
    # data/processed에 일자, 제목, 키워드 추출한 뉴스 데이터 저장 
    out_path = os.path.join(OUT_DIR, f"news_{year}.xlsx")
    df.to_excel(out_path, index=False)
    print(f"-> 추출 완료: {out_path}")
