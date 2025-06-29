# 데이터 구조 분석 리포트

## 생성일: 2025-06-08 02:34
## 팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)

---

## 📁 디렉토리 구조 분석

### 존재하는 디렉토리
- ✅ stock/
- ✅ product/
- ✅ data/
- ✅ code/
- ✅ results/

## 📈 주가 데이터 분석

총 8개 파일 발견

### Samsung_2021.csv
- 크기: 10.7 KB
- 행수: 244
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2024-01-02 ~ 2024-12-30

### Apple Stock Price History_2021.csv
- 크기: 11.5 KB
- 행수: 252
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2021-01-04 ~ 2021-12-31

### Samsung_2022.csv
- 크기: 10.8 KB
- 행수: 245
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2023-01-02 ~ 2023-12-28

### Apple Stock Price History_2022.csv
- 크기: 11.5 KB
- 행수: 251
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2022-01-03 ~ 2022-12-30

### Apple Stock Price History_2023.csv
- 크기: 11.3 KB
- 행수: 250
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2023-01-03 ~ 2023-12-29

### Samsung_2023.csv
- 크기: 10.8 KB
- 행수: 246
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2022-01-03 ~ 2022-12-29

### Samsung_2024.csv
- 크기: 10.9 KB
- 행수: 248
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2021-01-04 ~ 2021-12-30

### Apple Stock Price History_2024.csv
- 크기: 11.4 KB
- 행수: 252
- 컬럼: ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.']
- 기간: 2024-01-02 ~ 2024-12-31


## 🎭 감성 데이터 분석

총 8개 파일 발견

### apple_sentiment_2024.csv
- 크기: 4.5 MB
- 행수: 2,023
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### samsung_sentiment_2024.csv
- 크기: 34.8 MB
- 행수: 15,340
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### samsung_sentiment_2021.csv
- 크기: 25.5 MB
- 행수: 11,037
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### apple_sentiment_2023.csv
- 크기: 4.2 MB
- 행수: 1,872
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### apple_sentiment_2022.csv
- 크기: 3.3 MB
- 행수: 1,488
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### samsung_sentiment_2022.csv
- 크기: 24.8 MB
- 행수: 10,767
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### apple_sentiment_2021.csv
- 크기: 2.8 MB
- 행수: 1,139
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']

### samsung_sentiment_2023.csv
- 크기: 31.2 MB
- 행수: 13,689
- 컬럼: ['일자', '제목', '키워드', '감정라벨', '감정점수']
- 감성 컬럼: ['감정라벨', '감정점수']


## 📱 제품 출시 데이터 분석

총 2개 파일 발견

### samsung.xlsx
- 타입: OTHER

### apple.xlsx
- 타입: OTHER


---

## 📊 다음 단계

1. **7일 평균 통합 시각화**: 8개 연도별 차트 생성
2. **LSTM 모델 개선**: 주간 데이터 기반으로 전환
3. **제품 출시 임팩트 분석**: 정량적 영향 측정

## 🎯 기대 결과

- 노이즈 감소를 통한 모델 성능 개선 (R² > 0.3 목표)
- 실용적인 주간 단위 예측 시스템 구축
- 제품 출시와 감성-주가 간 상관관계 정량화
