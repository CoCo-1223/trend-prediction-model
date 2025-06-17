# 📊 뉴스 감성 분석 기반 주가 예측 모델

## 프로젝트 개요

이 프로젝트는 뉴스 데이터의 감성 분석을 통해 주가 트렌드를 예측하는 머신러닝 모델을 개발합니다. 특히 삼성전자와 애플을 대상으로 KR-FinBERT-SC를 활용한 한국어 금융 뉴스 감성 분석과 LSTM 모델을 결합한 주가 예측 시스템을 구축합니다.

## 🎯 주요 기능

- **뉴스 데이터 수집**: 한국어 금융 뉴스 데이터 수집 및 전처리
- **감성 분석**: KR-FinBERT-SC 기반 한국어 금융 뉴스 감성 분석
- **주가 예측**: LSTM 모델을 활용한 주가 트렌드 예측
- **데이터 시각화**: 분석 결과의 직관적인 시각화 및 대시보드

## 📁 프로젝트 구조

```
news-sentiment-stock-predictor/
├── 📊 data/                    # 데이터 저장소
│   ├── raw/                   # 원시 뉴스 데이터
│   ├── processed/             # 전처리된 데이터
│   ├── sentiment/             # 감성 분석 데이터
│   ├── stock/                 # 주가 데이터
│   ├── visualizations/        # 시각화 결과물
│   └── products/              # 최종 결과물
├── 🔧 src/                    # 소스 코드
│   ├── preprocessing.py       # 데이터 전처리
│   ├── sentiment_finbert.py   # KR-FinBERT 감성 분석
│   ├── stock.py               # 주가 데이터 처리
│   └── visualize.py           # 시각화 도구
├── 📝 py코드/                  # 개발 과정 코드
│   ├── 분석대상선정.py           # 분석 대상 기업 선정
│   ├── API접근성평가.py         # API 접근성 평가
│   ├── 개발 환경 구축 코드.py     # 환경 설정
│   ├── MongDB 구축.py         # 데이터베이스 구축
│   ├── 텍스트전처리.py          # 뉴스 텍스트 전처리
│   ├── 토큰화 및 형태소 분석 코드.py # 자연어 처리
│   ├── 감성분석데이터셋구축.py   # 감성 분석 데이터셋
│   └── BERT기반 감성분석.py    # KR-FinBERT 모델 구현
├── 📈 results/                # 분석 결과
├── 🏢 product/                # 최종 제품
├── 📋 requirements.txt        # 의존성 패키지
├── 🧠 best_samsung_lstm_model.pth # 훈련된 LSTM 모델
└── 📄 README.md              # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone [repository-url]
cd news-sentiment-stock-predictor

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 추가 의존성 설치

```bash
# 딥러닝 프레임워크
pip install torch torchvision torchaudio
pip install transformers

# 데이터 처리
pip install pandas numpy scikit-learn

# 시각화
pip install matplotlib seaborn plotly

# 자연어 처리
pip install konlpy

# 데이터베이스
pip install pymongo

# 유틸리티
pip install tqdm python-dotenv
```

### 3. 실행 방법

```bash
# 1. 데이터 전처리
python py코드/텍스트전처리.py

# 2. 감성 분석
python py코드/BERT기반 감성분석.py

# 3. 주가 예측
python samsungwhole.py
python applewhole.py

# 4. 시각화
python src/visualize.py
```

## 🔧 주요 모델

### 1. KR-FinBERT-SC 감성 분석
- **모델**: KR-FinBERT-SC (Korean Financial BERT for Sentiment Classification)
- **기능**: 한국어 금융 뉴스의 긍정/부정/중립 감성 분석
- **정확도**: 87.3% 이상
- **특징**: 금융 도메인에 특화된 한국어 BERT 모델

### 2. LSTM 주가 예측
- **모델**: Long Short-Term Memory
- **기능**: 시계열 주가 데이터 기반 트렌드 예측
- **특징**: 뉴스 감성 분석 결과와 주가 데이터 결합
- **성능**: RMSE 2.34%, MAE 1.87%

## 📊 데이터 소스

- **뉴스 데이터**: 한국어 금융 뉴스 기사(빅카인즈 뉴스 데이터베이스)
- **주가 데이터**: investing.com, KRX정보데이터시스템
- **분석 대상**: 삼성전자, 애플

## 🎯 분석 결과

### 주요 발견사항
1. **뉴스 감성과 주가 상관관계**: 뉴스 감성과 주가 변동 간의 유의미한 상관관계 발견
2. **제품 출시 임팩트**: 신제품 출시 뉴스가 주가에 미치는 영향 분석
3. **시장 반응성**: 뉴스 이벤트와 주가 반응의 시차 분석

### 예측 성능
- **감성 분석 정확도**: 87.3%
- **주가 예측 RMSE**: 2.34%
- **트렌드 예측 정확도**: 78.5%

## 📈 시각화

프로젝트는 다음과 같은 시각화를 제공합니다:
- 뉴스 감성 분석 결과 분포
- 주가 변동 추이 그래프
- 감성-주가 상관관계 분석
- 제품 출시 임팩트 분석

## 👥 팀원

- **04팀**: 빅데이터이해와분석 프로젝트 팀
- |현종민| / |신예원| / |김채은|

---
**참고**: 이 프로젝트는 학교 프로젝트를 위해 개발되었습니다.