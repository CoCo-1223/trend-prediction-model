# 뉴스 기반 기업 감성 분석 모델

이 프로젝트는 뉴스 데이터를 기반으로 삼성전자와 애플의 연도별 감성 분석을 수행하는 모델입니다.

## 기능

- 뉴스 텍스트 전처리 및 형태소 분석
- 감성 분석 모델 학습 및 평가
- 연도별 기업 감성 분석 및 시각화

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Java 설치 (KoNLPy 사용을 위해 필요):
- macOS: `brew install java`
- Windows: [Java JDK](https://www.oracle.com/java/technologies/javase-downloads.html) 설치

## 사용 방법

1. 뉴스 데이터 준비:
- `news/news_data.csv` 파일에 뉴스 데이터를 저장
- 데이터 형식: `year`, `content` 컬럼 필요

2. 감성 분석 실행:
```bash
python sentiment_analysis.py
```

## 결과

- 각 기업별 연도별 감성 분석 결과가 콘솔에 출력됩니다.
- 시각화 결과는 `삼성전자_sentiment_analysis.png`와 `애플_sentiment_analysis.png` 파일로 저장됩니다.

## 감성 분석 기준

- 긍정적 키워드: '호조', '성장', '개선', '기대'
- 부정적 키워드: '부진', '침체', '악화', '우려'

## 프로젝트 개요
소셜 미디어 데이터를 활용하여 대중의 감성 흐름을 분석하고,
이를 통해 시장 트렌드 및 비즈니스 지표(판매량등)를 예측하는 모델 개발 

## 기간 
2025.03.10 ~ 

## 팀구성 
- 현종민
- 신예원
- 김채은 

## 폴더구조
```bash
trend-prediction-model/
├── data/
│   ├── raw/                   # 원본 데이터
│   │   ├── news/              # 뉴스 데이터
│   │   ├── stock/             # 주가 데이터
│   │   └── financial/         # 재무제표 데이터
│   └── processed/             # 전처리된 데이터
│       ├── news/
│       ├── stock/
│       └── financial/
├── src/
│   ├── preprocessing/  # 데이터 전처리 관련 코드
│   │   └── news_preprocessor.py
│   ├── modeling/   # 모델링 관련 코드
│   │   └── sentiment_analysis.py
│   └── analysis/    # 분석 및 시각화 코드
│       ├── trend_analysis.py
│       ├── correlation_analysis.py
│       └── visualization.py
├── results/                 # 결과 저장
│   ├── models/              # 학습된 모델
│   ├── predictions/         # 예측 결과
│   └── visualizations/      # 시각화 결과
├── config/                  # 설정 파일
│   ├── model_config.yaml
│   └── data_config.yaml
├── tests/                   # 테스트 코드
├── requirements.txt         # 패키지 의존성
├── setup.py                 # 프로젝트 설치 파일
└── README.md                # 프로젝트 문서

