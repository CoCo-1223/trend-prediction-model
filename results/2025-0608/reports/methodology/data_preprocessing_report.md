# 7일 평균 LSTM 데이터 전처리 리포트

## 프로젝트 정보
- **생성일**: 2025-06-08 03:08:29
- **팀**: 현종민(팀장), 신예원(팀원), 김채은(팀원)
- **목표**: 주간 데이터 기반 LSTM 감성 예측 모델 데이터 준비

## 데이터 개요
- **전체 데이터**: 2,800건
- **Apple 데이터**: 1,339건
- **Samsung 데이터**: 1,461건
- **기간**: 2021-01-01 ~ 2024-12-31
- **총 특성 수**: 60개

## 특성 엔지니어링 결과

### 특성 그룹별 구성
- **Time_Features**: 16개
- **Sentiment_Features**: 22개
- **Stock_Features**: 16개
- **Product_Launch_Features**: 10개
- **Interaction_Features**: 12개
- **Lag_Features**: 8개
- **News_Features**: 5개
- **Momentum_Features**: 9개
- **Volatility_Features**: 4개


### 주요 특성 상관관계 (Apple vs Samsung)
- **감성-주가 상관관계 (Apple)**: -0.008
- **감성-주가 상관관계 (Samsung)**: 0.663

### LSTM 시퀀스 정보
- **시퀀스 길이**: 4주 (28일)
- **예측 타겟**: 다음 주 감성 점수
- **정규화 방법**: RobustScaler (회사별 개별 적용)

### 데이터 분할 전략
- **훈련 데이터**: 60% (시계열 순서 기준)
- **검증 데이터**: 20%
- **테스트 데이터**: 20%

## 품질 관리
- **결측값 처리**: 전진/후진 채움 + 0으로 대체
- **이상값 처리**: RobustScaler로 스케일링
- **무한값 처리**: NaN으로 변환 후 보간

## 파일 출력
1. `weekly_sentiment_features.csv` - 전체 특성 데이터
2. `apple_weekly_features.csv` - Apple 특성 데이터
3. `samsung_weekly_features.csv` - Samsung 특성 데이터
4. `lstm_training_sequences.pkl` - LSTM 훈련용 시퀀스
5. `feature_correlation_matrix.png` - 상관관계 히트맵
6. `feature_info.json` - 특성 메타데이터

## 다음 단계
1. LSTM 모델 훈련 (`10.개선된삼성LSTM.py`)
2. 특성 중요도 분석 (SHAP)
3. 예측 성능 평가 및 비교

---
*생성일: 2025-06-08 03:08:29*
