"""
뉴스 감성 분석 기반 주가 예측 모델 - 7일 평균 LSTM 데이터 전처리
생성일: 2025-06-08
팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)

목표:
- 8번 코드에서 생성된 통합 데이터를 활용한 고급 특성 엔지니어링
- 주간 데이터 기반 LSTM 시퀀스 생성
- 회사별 특성을 고려한 차별화된 특성 생성
- SHAP 분석을 위한 해석 가능한 특성 설계
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import json
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 결과물 저장 경로 설정
RESULTS_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model/results/2025-0608"
PROJECT_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model"

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_directories():
    """결과물 디렉토리 구조 생성"""
    directories = [
        f"{RESULTS_BASE}/visualizations/weekly_analysis",
        f"{RESULTS_BASE}/visualizations/impact_analysis", 
        f"{RESULTS_BASE}/visualizations/model_performance",
        f"{RESULTS_BASE}/visualizations/final_insights",
        f"{RESULTS_BASE}/data/processed",
        f"{RESULTS_BASE}/data/features",
        f"{RESULTS_BASE}/data/predictions", 
        f"{RESULTS_BASE}/data/exports",
        f"{RESULTS_BASE}/models/trained",
        f"{RESULTS_BASE}/models/evaluation",
        f"{RESULTS_BASE}/models/features_importance",
        f"{RESULTS_BASE}/models/predictions",
        f"{RESULTS_BASE}/reports/technical",
        f"{RESULTS_BASE}/reports/business",
        f"{RESULTS_BASE}/reports/methodology",
        f"{RESULTS_BASE}/reports/final"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 결과물 디렉토리 구조 생성 완료: {RESULTS_BASE}")

# 실행 시작 시 디렉토리 자동 생성
setup_directories()

class WeeklyLSTMDataProcessor:
    """7일 평균 LSTM 데이터 전처리 및 특성 엔지니어링"""
    
    def __init__(self):
        self.results_base = RESULTS_BASE
        self.project_base = PROJECT_BASE
        
        # 데이터 경로 설정
        self.weekly_data_path = f"{self.results_base}/data/processed/weekly_sentiment_stock_data.csv"
        self.product_data_path = f"{self.results_base}/data/processed/combined_product_timeline.csv"
        self.sentiment_data_path = f"{self.project_base}/data/processed"
        self.stock_data_path = f"{self.project_base}/stock"
        
        # 스케일러 초기화
        self.scalers = {}
        
        # 로그 기록
        self.processing_log = []
        
        print("📊 Weekly LSTM Data Processor 초기화 완료")
        
    def log_process(self, message):
        """처리 과정 로깅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(log_entry)
    
    def load_base_data(self):
        """8번 코드에서 생성된 통합 데이터 로딩"""
        self.log_process("기본 데이터 로딩 시작")
        
        try:
            # 통합 주간 데이터 로딩
            self.weekly_data = pd.read_csv(self.weekly_data_path)
            self.weekly_data['Date'] = pd.to_datetime(self.weekly_data['Date'])
            
            # 제품 출시 데이터 로딩
            self.product_data = pd.read_csv(self.product_data_path)
            self.product_data['Date'] = pd.to_datetime(self.product_data['Date'])
            
            self.log_process(f"통합 데이터 로딩 완료: {len(self.weekly_data):,}건")
            self.log_process(f"제품 출시 데이터 로딩 완료: {len(self.product_data):,}건")
            
            # 데이터 기본 정보 출력
            print("\n📈 통합 데이터 기본 정보:")
            print(f"- 전체 데이터: {len(self.weekly_data):,}건")
            print(f"- Apple 데이터: {len(self.weekly_data[self.weekly_data['Company'] == 'Apple']):,}건")
            print(f"- Samsung 데이터: {len(self.weekly_data[self.weekly_data['Company'] == 'Samsung']):,}건")
            print(f"- 기간: {self.weekly_data['Date'].min()} ~ {self.weekly_data['Date'].max()}")
            
            return True
            
        except Exception as e:
            self.log_process(f"데이터 로딩 실패: {str(e)}")
            return False
    
    def create_time_features(self, df):
        """시간 기반 특성 생성"""
        self.log_process("시간 기반 특성 생성 시작")
        
        # 기본 시간 특성
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # 계절성 특성
        df['is_quarter_end'] = df['Month'].isin([3, 6, 9, 12]).astype(int)
        df['is_year_end'] = (df['Month'] == 12).astype(int)
        df['is_apple_season'] = df['Month'].isin([9, 10, 11]).astype(int)  # 아이폰 출시 시즌
        
        # 주기적 특성 (사인/코사인 인코딩)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        self.log_process("시간 기반 특성 생성 완료 (12개)")
        return df
    
    def create_sentiment_features(self, df):
        """감성 관련 고급 특성 생성"""
        self.log_process("감성 특성 엔지니어링 시작")
        
        # 회사별로 처리
        for company in df['Company'].unique():
            mask = df['Company'] == company
            company_data = df[mask].copy()
            
            # 1. 감성 모멘텀 특성 (7일, 14일, 30일)
            df.loc[mask, 'sentiment_momentum_7d'] = company_data['sentiment_score_7d_avg'].pct_change(periods=1)
            df.loc[mask, 'sentiment_momentum_14d'] = company_data['sentiment_score_7d_avg'].pct_change(periods=2)
            df.loc[mask, 'sentiment_momentum_30d'] = company_data['sentiment_score_7d_avg'].pct_change(periods=4)
            
            # 2. 감성 변동성 (7일, 14일 롤링 표준편차)
            df.loc[mask, 'sentiment_volatility_7d'] = company_data['sentiment_score_7d_avg'].rolling(window=2, min_periods=1).std()
            df.loc[mask, 'sentiment_volatility_14d'] = company_data['sentiment_score_7d_avg'].rolling(window=3, min_periods=1).std()
            
            # 3. 감성 Z-스코어 (연도별 정규화)
            for year in company_data['Year'].unique():
                year_mask = (df['Company'] == company) & (df['Year'] == year)
                year_data = df[year_mask]['sentiment_score_7d_avg']
                if len(year_data) > 1:
                    df.loc[year_mask, 'sentiment_zscore'] = stats.zscore(year_data, nan_policy='omit')
            
            # 4. 감성 트렌드 (선형 회귀 기울기)
            df.loc[mask, 'sentiment_trend_7d'] = company_data['sentiment_score_7d_avg'].rolling(window=2, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            # 5. 감성 상대 포지션 (최근 4주 내 백분위)
            df.loc[mask, 'sentiment_percentile_4w'] = company_data['sentiment_score_7d_avg'].rolling(window=4, min_periods=1).rank(pct=True)
            
            # 6. 감성 안정성 (변동계수)
            rolling_mean = company_data['sentiment_score_7d_avg'].rolling(window=4, min_periods=1).mean()
            rolling_std = company_data['sentiment_score_7d_avg'].rolling(window=4, min_periods=1).std()
            df.loc[mask, 'sentiment_stability'] = rolling_std / (rolling_mean + 1e-6)  # 변동계수
        
        # 7. 뉴스 볼륨 특성
        df['news_volume_momentum'] = df.groupby('Company')['news_count'].pct_change()
        df['news_volume_spike'] = (df['news_count'] > df.groupby('Company')['news_count'].transform(lambda x: x.quantile(0.9))).astype(int)
        
        self.log_process("감성 특성 엔지니어링 완료 (13개)")
        return df
    
    def create_stock_features(self, df):
        """주가 관련 고급 특성 생성"""
        self.log_process("주가 특성 엔지니어링 시작")
        
        # 회사별로 처리
        for company in df['Company'].unique():
            mask = df['Company'] == company
            company_data = df[mask].copy()
            
            # 1. 주가 모멘텀 특성
            df.loc[mask, 'stock_momentum_7d'] = company_data['stock_price_7d_avg'].pct_change(periods=1)
            df.loc[mask, 'stock_momentum_14d'] = company_data['stock_price_7d_avg'].pct_change(periods=2)
            df.loc[mask, 'stock_momentum_30d'] = company_data['stock_price_7d_avg'].pct_change(periods=4)
            
            # 2. 주가 변동성
            df.loc[mask, 'stock_volatility_7d'] = company_data['stock_price_7d_avg'].rolling(window=2, min_periods=1).std()
            df.loc[mask, 'stock_volatility_14d'] = company_data['stock_price_7d_avg'].rolling(window=3, min_periods=1).std()
            
            # 3. 주가 Z-스코어 (연도별)
            for year in company_data['Year'].unique():
                year_mask = (df['Company'] == company) & (df['Year'] == year)
                year_data = df[year_mask]['stock_price_7d_avg']
                if len(year_data) > 1:
                    df.loc[year_mask, 'stock_zscore'] = stats.zscore(year_data, nan_policy='omit')
            
            # 4. 주가 상대 강도 (4주 내 백분위)
            df.loc[mask, 'stock_rsi_4w'] = company_data['stock_price_7d_avg'].rolling(window=4, min_periods=1).rank(pct=True)
            
            # 5. 이동평균 교차 신호
            ma_short = company_data['stock_price_7d_avg'].rolling(window=2, min_periods=1).mean()
            ma_long = company_data['stock_price_7d_avg'].rolling(window=4, min_periods=1).mean()
            df.loc[mask, 'ma_cross_signal'] = (ma_short > ma_long).astype(int)
            
            # 6. 주가 vs 기준 편차 (연도별 평균 대비)
            year_avg = company_data.groupby('Year')['stock_price_7d_avg'].transform('mean')
            df.loc[mask, 'stock_vs_year_avg'] = (company_data['stock_price_7d_avg'] - year_avg) / year_avg
        
        self.log_process("주가 특성 엔지니어링 완료 (11개)")
        return df
    
    def create_product_launch_features(self, df):
        """제품 출시 관련 특성 생성"""
        self.log_process("제품 출시 특성 엔지니어링 시작")
        
        # 회사별로 처리
        for company in df['Company'].unique():
            mask = df['Company'] == company
            company_data = df[mask].copy()
            company_products = self.product_data[self.product_data['Company'] == company]
            
            # 각 날짜에 대해 제품 출시 관련 특성 계산
            days_to_next = []
            days_since_last = []
            launch_impact_scores = []
            launch_categories = []
            launch_counts = []
            
            for date in company_data['Date']:
                # 다음 제품 출시까지의 일수
                future_launches = company_products[company_products['Date'] > date]
                if len(future_launches) > 0:
                    next_launch = future_launches['Date'].min()
                    days_to_next.append((next_launch - date).days)
                else:
                    days_to_next.append(365)  # 기본값
                
                # 마지막 제품 출시 이후 일수
                past_launches = company_products[company_products['Date'] <= date]
                if len(past_launches) > 0:
                    last_launch = past_launches['Date'].max()
                    days_since_last.append((date - last_launch).days)
                else:
                    days_since_last.append(365)  # 기본값
                
                # 임팩트 스코어 (가까운 출시일들의 가중 평균)
                nearby_launches = company_products[
                    (company_products['Date'] >= date - timedelta(days=60)) &
                    (company_products['Date'] <= date + timedelta(days=60))
                ]
                
                if len(nearby_launches) > 0:
                    # 거리 기반 가중치 계산
                    distances = abs((nearby_launches['Date'] - date).dt.days)
                    weights = np.exp(-distances / 30)  # 30일 감쇠 함수
                    impact_score = weights.sum()
                    
                    # 주요 제품 카테고리 확인
                    if company == 'Apple':
                        if any('iPhone' in prod for prod in nearby_launches['Product']):
                            category = 'iPhone'
                        elif any('iPad' in prod for prod in nearby_launches['Product']):
                            category = 'iPad'
                        else:
                            category = 'Other'
                    else:  # Samsung
                        if any('Galaxy S' in prod for prod in nearby_launches['Product']):
                            category = 'Galaxy_S'
                        elif any('Galaxy Note' in prod or 'Galaxy Z' in prod for prod in nearby_launches['Product']):
                            category = 'Galaxy_Premium'
                        else:
                            category = 'Other'
                    
                    # 해당 기간 출시 제품 수
                    launch_count = len(nearby_launches)
                else:
                    impact_score = 0
                    category = 'None'
                    launch_count = 0
                
                launch_impact_scores.append(impact_score)
                launch_categories.append(category)
                launch_counts.append(launch_count)
            
            # 특성 할당
            df.loc[mask, 'days_to_next_launch'] = days_to_next
            df.loc[mask, 'days_since_last_launch'] = days_since_last
            df.loc[mask, 'launch_impact_score'] = launch_impact_scores
            df.loc[mask, 'launch_category'] = launch_categories
            df.loc[mask, 'launch_count_nearby'] = launch_counts
            
            # 추가 파생 특성
            df.loc[mask, 'launch_proximity'] = 1 / (1 + np.minimum(df.loc[mask, 'days_to_next_launch'], 
                                                                  df.loc[mask, 'days_since_last_launch']))
            df.loc[mask, 'pre_launch_period'] = (df.loc[mask, 'days_to_next_launch'] <= 30).astype(int)
            df.loc[mask, 'post_launch_period'] = (df.loc[mask, 'days_since_last_launch'] <= 30).astype(int)
        
        # 카테고리 인코딩
        le = LabelEncoder()
        df['launch_category_encoded'] = le.fit_transform(df['launch_category'])
        
        self.log_process("제품 출시 특성 엔지니어링 완료 (9개)")
        return df
    
    def create_interaction_features(self, df):
        """상호작용 및 비율 특성 생성"""
        self.log_process("상호작용 특성 생성 시작")
        
        # 1. 감성-주가 상호작용
        df['sentiment_stock_ratio'] = df['sentiment_score_7d_avg'] / (df['stock_price_7d_avg'] + 1e-6)
        df['sentiment_stock_product'] = df['sentiment_score_7d_avg'] * df['stock_momentum_7d']
        
        # 2. 기준선 대비 편차
        for company in df['Company'].unique():
            mask = df['Company'] == company
            
            # 연도별 기준선
            year_sentiment_avg = df[mask].groupby('Year')['sentiment_score_7d_avg'].transform('mean')
            year_stock_avg = df[mask].groupby('Year')['stock_price_7d_avg'].transform('mean')
            
            df.loc[mask, 'sentiment_vs_baseline'] = (df.loc[mask, 'sentiment_score_7d_avg'] - year_sentiment_avg) / year_sentiment_avg
            df.loc[mask, 'stock_vs_baseline'] = (df.loc[mask, 'stock_price_7d_avg'] - year_stock_avg) / year_stock_avg
            
            # 전체 기준선 대비
            overall_sentiment_avg = df[mask]['sentiment_score_7d_avg'].mean()
            overall_stock_avg = df[mask]['stock_price_7d_avg'].mean()
            
            df.loc[mask, 'sentiment_vs_overall'] = (df.loc[mask, 'sentiment_score_7d_avg'] - overall_sentiment_avg) / overall_sentiment_avg
            df.loc[mask, 'stock_vs_overall'] = (df.loc[mask, 'stock_price_7d_avg'] - overall_stock_avg) / overall_stock_avg
        
        # 3. 뉴스 볼륨 상호작용
        df['news_sentiment_interaction'] = df['news_count'] * df['sentiment_score_7d_avg']
        df['news_launch_interaction'] = df['news_count'] * df['launch_impact_score']
        
        # 4. 시간-이벤트 상호작용
        df['quarter_launch_interaction'] = df['Quarter'] * df['launch_impact_score']
        df['season_sentiment_interaction'] = df['is_apple_season'] * df['sentiment_score_7d_avg']
        
        self.log_process("상호작용 특성 생성 완료 (10개)")
        return df
    
    def create_lag_features(self, df):
        """시차 특성 생성 (감성이 주가를 선행하는 패턴 반영)"""
        self.log_process("시차 특성 생성 시작")
        
        # 회사별로 처리
        for company in df['Company'].unique():
            mask = df['Company'] == company
            company_data = df[mask].copy().sort_values('Date')
            
            # 감성 선행 지표 (1-3주 전 감성)
            df.loc[mask, 'sentiment_lag_1w'] = company_data['sentiment_score_7d_avg'].shift(1)
            df.loc[mask, 'sentiment_lag_2w'] = company_data['sentiment_score_7d_avg'].shift(2)
            df.loc[mask, 'sentiment_lag_3w'] = company_data['sentiment_score_7d_avg'].shift(3)
            
            # 감성 변화 선행 지표
            df.loc[mask, 'sentiment_momentum_lag_1w'] = company_data['sentiment_momentum_7d'].shift(1)
            df.loc[mask, 'sentiment_momentum_lag_2w'] = company_data['sentiment_momentum_7d'].shift(2)
            
            # 주가 선행 지표 (다음 주 예측을 위한 현재/과거 주가)
            df.loc[mask, 'stock_lag_1w'] = company_data['stock_price_7d_avg'].shift(1)
            df.loc[mask, 'stock_lag_2w'] = company_data['stock_price_7d_avg'].shift(2)
            
            # 혼합 선행 지표
            df.loc[mask, 'sentiment_stock_lag_interaction'] = (
                company_data['sentiment_score_7d_avg'].shift(1) * 
                company_data['stock_momentum_7d'].shift(1)
            )
        
        self.log_process("시차 특성 생성 완료 (8개)")
        return df
    
    def create_all_features(self):
        """모든 특성 엔지니어링 실행"""
        self.log_process("=== 전체 특성 엔지니어링 시작 ===")
        
        # 기본 데이터 로딩
        if not self.load_base_data():
            return None
        
        # 전체 데이터 복사
        df = self.weekly_data.copy()
        
        # 각 특성 그룹 생성
        df = self.create_time_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_stock_features(df)
        df = self.create_product_launch_features(df)
        df = self.create_interaction_features(df)
        df = self.create_lag_features(df)
        
        # 무한값 및 NaN 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 결측값 처리 (회사별로)
        for company in df['Company'].unique():
            mask = df['Company'] == company
            df.loc[mask] = df.loc[mask].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_df = df
        
        # 최종 특성 목록 정리
        feature_columns = [col for col in df.columns if col not in ['Date', 'Company', 'Year']]
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        self.log_process(f"전체 특성 엔지니어링 완료: {len(numeric_features)}개 숫자형 특성")
        self.log_process(f"전체 데이터 크기: {df.shape}")
        
        return df
    
    def create_lstm_sequences(self, df, sequence_length=4, target_col='sentiment_score_7d_avg'):
        """LSTM용 시퀀스 데이터 생성"""
        self.log_process(f"LSTM 시퀀스 생성 시작 (길이: {sequence_length}주)")
        
        # 특성 컬럼 선택 (숫자형만)
        feature_columns = [col for col in df.columns if col not in ['Date', 'Company', 'Year', 'launch_category']]
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # 회사별로 시퀀스 생성
        all_sequences = []
        all_targets = []
        all_dates = []
        all_companies = []
        
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company].sort_values('Date').reset_index(drop=True)
            company_features = company_data[numeric_features].values
            company_targets = company_data[target_col].values
            
            # 데이터 정규화 (회사별 스케일러)
            scaler = RobustScaler()
            company_features_scaled = scaler.fit_transform(company_features)
            self.scalers[company] = scaler
            
            # 시퀀스 생성
            for i in range(sequence_length, len(company_data)):
                # 입력 시퀀스 (과거 4주)
                sequence = company_features_scaled[i-sequence_length:i]
                
                # 타겟 (현재 주의 감성 점수)
                target = company_targets[i]
                
                # 메타 정보
                date = company_data.iloc[i]['Date']
                
                all_sequences.append(sequence)
                all_targets.append(target)
                all_dates.append(date)
                all_companies.append(company)
        
        # numpy 배열로 변환
        X = np.array(all_sequences)
        y = np.array(all_targets)
        
        # 메타 정보 DataFrame
        meta_df = pd.DataFrame({
            'Date': all_dates,
            'Company': all_companies,
            'Target': y
        })
        
        self.log_process(f"시퀀스 생성 완료: {X.shape[0]}개 시퀀스, 입력 차원: {X.shape[1:]} → 출력: {y.shape}")
        
        # 회사별 분포 확인
        for company in meta_df['Company'].unique():
            count = (meta_df['Company'] == company).sum()
            self.log_process(f"- {company}: {count:,}개 시퀀스")
        
        return X, y, meta_df, numeric_features
    
    def create_validation_split(self, X, y, meta_df, test_size=0.2, val_size=0.2):
        """시계열 특성을 고려한 검증 데이터 분할"""
        self.log_process("검증 데이터 분할 시작")
        
        # 날짜 기준 정렬
        sort_idx = meta_df['Date'].argsort()
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        meta_sorted = meta_df.iloc[sort_idx].reset_index(drop=True)
        
        # 시간 순서 기준 분할
        n_total = len(X_sorted)
        n_train = int(n_total * (1 - test_size - val_size))
        n_val = int(n_total * val_size)
        
        # 분할 인덱스
        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, n_total)
        
        # 데이터 분할
        X_train = X_sorted[train_idx]
        X_val = X_sorted[val_idx]
        X_test = X_sorted[test_idx]
        
        y_train = y_sorted[train_idx]
        y_val = y_sorted[val_idx]
        y_test = y_sorted[test_idx]
        
        meta_train = meta_sorted.iloc[train_idx]
        meta_val = meta_sorted.iloc[val_idx]
        meta_test = meta_sorted.iloc[test_idx]
        
        self.log_process(f"데이터 분할 완료:")
        self.log_process(f"- 훈련: {len(X_train):,}개 ({train_idx.start}-{train_idx.stop})")
        self.log_process(f"- 검증: {len(X_val):,}개 ({val_idx.start}-{val_idx.stop})")
        self.log_process(f"- 테스트: {len(X_test):,}개 ({test_idx.start}-{test_idx.stop})")
        self.log_process(f"- 훈련 기간: {meta_train['Date'].min()} ~ {meta_train['Date'].max()}")
        self.log_process(f"- 검증 기간: {meta_val['Date'].min()} ~ {meta_val['Date'].max()}")
        self.log_process(f"- 테스트 기간: {meta_test['Date'].min()} ~ {meta_test['Date'].max()}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'meta_train': meta_train, 'meta_val': meta_val, 'meta_test': meta_test
        }
    
    def analyze_feature_importance_preparation(self, df, numeric_features):
        """특성 중요도 분석 준비"""
        self.log_process("특성 중요도 분석 준비 시작")
        
        # 특성 그룹 정의
        feature_groups = {
            'Time_Features': [col for col in numeric_features if any(x in col.lower() for x in ['month', 'quarter', 'year', 'day', 'season'])],
            'Sentiment_Features': [col for col in numeric_features if 'sentiment' in col.lower()],
            'Stock_Features': [col for col in numeric_features if 'stock' in col.lower()],
            'Product_Launch_Features': [col for col in numeric_features if any(x in col.lower() for x in ['launch', 'days_to', 'days_since'])],
            'Interaction_Features': [col for col in numeric_features if any(x in col.lower() for x in ['ratio', 'product', 'interaction', 'vs_'])],
            'Lag_Features': [col for col in numeric_features if 'lag' in col.lower()],
            'News_Features': [col for col in numeric_features if 'news' in col.lower()],
            'Momentum_Features': [col for col in numeric_features if 'momentum' in col.lower()],
            'Volatility_Features': [col for col in numeric_features if 'volatility' in col.lower()]
        }
        
        # 그룹별 특성 개수 로깅
        for group, features in feature_groups.items():
            self.log_process(f"- {group}: {len(features)}개")
        
        return feature_groups
    
    def create_correlation_analysis(self, df, numeric_features):
        """상관관계 분석 및 시각화"""
        self.log_process("상관관계 분석 시작")
        
        # 핵심 특성들만 선별 (너무 많으면 시각화가 어려움)
        key_features = [
            'sentiment_score_7d_avg', 'stock_price_7d_avg', 
            'sentiment_momentum_7d', 'stock_momentum_7d',
            'sentiment_volatility_7d', 'stock_volatility_7d',
            'launch_impact_score', 'days_to_next_launch',
            'sentiment_stock_ratio', 'news_count',
            'sentiment_vs_baseline', 'stock_vs_baseline',
            'sentiment_lag_1w', 'sentiment_lag_2w'
        ]
        
        # 실제 존재하는 특성만 선택
        available_features = [f for f in key_features if f in numeric_features]
        
        # 회사별 상관관계 계산
        correlation_results = {}
        
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company][available_features]
            corr_matrix = company_data.corr()
            correlation_results[company] = corr_matrix
        
        # 전체 상관관계 (회사 구분 없이)
        overall_corr = df[available_features].corr()
        correlation_results['Overall'] = overall_corr
        
        # 상관관계 히트맵 생성
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, y=0.98)
        
        # Apple 상관관계
        sns.heatmap(correlation_results['Apple'], annot=True, cmap='RdBu_r', center=0,
                   ax=axes[0,0], cbar_kws={'shrink': .8})
        axes[0,0].set_title('Apple - Feature Correlations')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].tick_params(axis='y', rotation=0)
        
        # Samsung 상관관계
        sns.heatmap(correlation_results['Samsung'], annot=True, cmap='RdBu_r', center=0,
                   ax=axes[0,1], cbar_kws={'shrink': .8})
        axes[0,1].set_title('Samsung - Feature Correlations')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].tick_params(axis='y', rotation=0)
        
        # 전체 상관관계
        sns.heatmap(correlation_results['Overall'], annot=True, cmap='RdBu_r', center=0,
                   ax=axes[1,0], cbar_kws={'shrink': .8})
        axes[1,0].set_title('Overall - Feature Correlations')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].tick_params(axis='y', rotation=0)
        
        # 감성-주가 시차 상관관계 (별도 분석)
        lag_features = ['sentiment_score_7d_avg', 'sentiment_lag_1w', 'sentiment_lag_2w', 'sentiment_lag_3w']
        stock_feature = 'stock_price_7d_avg'
        
        lag_corr_data = []
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company]
            for lag_feat in lag_features:
                if lag_feat in company_data.columns:
                    corr = company_data[lag_feat].corr(company_data[stock_feature])
                    lag_name = lag_feat.replace('sentiment_', '').replace('_7d_avg', '_current')
                    lag_corr_data.append({'Company': company, 'Lag_Feature': lag_name, 'Correlation': corr})
        
        lag_corr_df = pd.DataFrame(lag_corr_data)
        if not lag_corr_df.empty:
            lag_pivot = lag_corr_df.pivot(index='Lag_Feature', columns='Company', values='Correlation')
            sns.heatmap(lag_pivot, annot=True, cmap='RdBu_r', center=0,
                       ax=axes[1,1], cbar_kws={'shrink': .8})
            axes[1,1].set_title('Sentiment-Stock Lag Correlations')
            axes[1,1].tick_params(axis='x', rotation=0)
            axes[1,1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # 저장
        correlation_path = f"{self.results_base}/visualizations/model_performance/feature_correlation_matrix.png"
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.log_process(f"상관관계 분석 완료 및 저장: {correlation_path}")
        
        return correlation_results
    
    def create_feature_statistics(self, df, numeric_features):
        """특성 통계 정보 생성"""
        self.log_process("특성 통계 정보 생성 시작")
        
        # 전체 통계
        overall_stats = df[numeric_features].describe()
        
        # 회사별 통계
        company_stats = {}
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company][numeric_features]
            company_stats[company] = company_data.describe()
        
        # 특성별 변동계수 (CV) 계산
        cv_stats = {}
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company][numeric_features]
            cv = company_data.std() / (company_data.mean() + 1e-6)
            cv_stats[company] = cv
        
        # 결과 저장
        stats_results = {
            'overall_statistics': overall_stats,
            'company_statistics': company_stats,
            'coefficient_of_variation': cv_stats
        }
        
        # CSV로 저장
        overall_stats.to_csv(f"{self.results_base}/data/features/overall_feature_statistics.csv")
        
        for company, stats in company_stats.items():
            stats.to_csv(f"{self.results_base}/data/features/{company.lower()}_feature_statistics.csv")
        
        self.log_process("특성 통계 정보 생성 완료")
        return stats_results
    
    def save_processed_data(self, df, sequences_data, numeric_features, feature_groups):
        """처리된 데이터 저장"""
        self.log_process("처리된 데이터 저장 시작")
        
        # 1. 특성 엔지니어링된 데이터 저장
        df.to_csv(f"{self.results_base}/data/features/weekly_sentiment_features.csv", index=False)
        
        # 2. 회사별 데이터 저장
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company]
            company_data.to_csv(f"{self.results_base}/data/features/{company.lower()}_weekly_features.csv", index=False)
        
        # 3. LSTM 시퀀스 데이터 저장
        X, y, meta_df, split_data = sequences_data
        
        # 시퀀스 데이터 pickle로 저장
        lstm_data = {
            'X': X,
            'y': y,
            'meta_df': meta_df,
            'split_data': split_data,
            'feature_names': numeric_features,
            'scalers': self.scalers
        }
        
        with open(f"{self.results_base}/data/features/lstm_training_sequences.pkl", 'wb') as f:
            pickle.dump(lstm_data, f)
        
        # 4. 특성 정보 저장
        feature_info = {
            'total_features': len(numeric_features),
            'feature_names': numeric_features,
            'feature_groups': feature_groups,
            'sequence_length': 4,
            'target_column': 'sentiment_score_7d_avg',
            'companies': df['Company'].unique().tolist(),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            }
        }
        
        with open(f"{self.results_base}/data/features/feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)
        
        # 5. 처리 로그 저장
        log_content = '\n'.join(self.processing_log)
        with open(f"{self.results_base}/data/processed/feature_engineering_log.txt", 'w') as f:
            f.write(log_content)
        
        self.log_process("모든 데이터 저장 완료")
    
    def generate_preprocessing_report(self, df, numeric_features, feature_groups, correlation_results, stats_results):
        """전처리 리포트 생성"""
        self.log_process("전처리 리포트 생성 시작")
        
        report_content = f"""# 7일 평균 LSTM 데이터 전처리 리포트

## 프로젝트 정보
- **생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **팀**: 현종민(팀장), 신예원(팀원), 김채은(팀원)
- **목표**: 주간 데이터 기반 LSTM 감성 예측 모델 데이터 준비

## 데이터 개요
- **전체 데이터**: {len(df):,}건
- **Apple 데이터**: {len(df[df['Company'] == 'Apple']):,}건
- **Samsung 데이터**: {len(df[df['Company'] == 'Samsung']):,}건
- **기간**: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}
- **총 특성 수**: {len(numeric_features)}개

## 특성 엔지니어링 결과

### 특성 그룹별 구성
"""
        
        for group, features in feature_groups.items():
            report_content += f"- **{group}**: {len(features)}개\n"
        
        report_content += f"""

### 주요 특성 상관관계 (Apple vs Samsung)
- **감성-주가 상관관계 (Apple)**: {correlation_results['Apple'].loc['sentiment_score_7d_avg', 'stock_price_7d_avg']:.3f}
- **감성-주가 상관관계 (Samsung)**: {correlation_results['Samsung'].loc['sentiment_score_7d_avg', 'stock_price_7d_avg']:.3f}

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
*생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 리포트 저장
        with open(f"{self.results_base}/reports/methodology/data_preprocessing_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.log_process("전처리 리포트 생성 완료")

def main():
    """메인 실행 함수"""
    print("🚀 7일 평균 LSTM 데이터 전처리 시작")
    print("=" * 80)
    
    # 데이터 프로세서 초기화
    processor = WeeklyLSTMDataProcessor()
    
    try:
        # 1. 특성 엔지니어링 실행
        print("\n📊 Step 1: 특성 엔지니어링 실행")
        df = processor.create_all_features()
        
        if df is None:
            print("❌ 특성 엔지니어링 실패")
            return
        
        # 2. LSTM 시퀀스 생성
        print("\n🔄 Step 2: LSTM 시퀀스 생성")
        feature_columns = [col for col in df.columns if col not in ['Date', 'Company', 'Year', 'launch_category']]
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        X, y, meta_df, _ = processor.create_lstm_sequences(df)
        
        # 3. 검증 데이터 분할
        print("\n📋 Step 3: 검증 데이터 분할")
        split_data = processor.create_validation_split(X, y, meta_df)
        
        # 4. 특성 중요도 분석 준비
        print("\n🔍 Step 4: 특성 분석")
        feature_groups = processor.analyze_feature_importance_preparation(df, numeric_features)
        correlation_results = processor.create_correlation_analysis(df, numeric_features)
        stats_results = processor.create_feature_statistics(df, numeric_features)
        
        # 5. 데이터 저장
        print("\n💾 Step 5: 데이터 저장")
        sequences_data = (X, y, meta_df, split_data)
        processor.save_processed_data(df, sequences_data, numeric_features, feature_groups)
        
        # 6. 리포트 생성
        print("\n📄 Step 6: 리포트 생성")
        processor.generate_preprocessing_report(df, numeric_features, feature_groups, 
                                              correlation_results, stats_results)
        
        print("\n✅ 7일 평균 LSTM 데이터 전처리 완료!")
        print(f"📁 결과물 저장 위치: {RESULTS_BASE}")
        print("\n📈 최종 결과 요약:")
        print(f"- 전체 특성 수: {len(numeric_features)}개")
        print(f"- LSTM 시퀀스: {X.shape[0]:,}개 (입력: {X.shape[1:]} → 출력: {y.shape})")
        print(f"- 훈련 데이터: {len(split_data['X_train']):,}개")
        print(f"- 검증 데이터: {len(split_data['X_val']):,}개")
        print(f"- 테스트 데이터: {len(split_data['X_test']):,}개")
        
        print("\n🎯 다음 단계: 10.개선된삼성LSTM.py 실행")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()