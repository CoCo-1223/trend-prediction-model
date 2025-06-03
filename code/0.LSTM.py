"""
빅카인즈 뉴스 데이터 기반 PyTorch LSTM 감성 트렌드 예측 모델
프로젝트: 뉴스 데이터 기반 감성 트렌드 예측 모델

주요 기능:
1. 빅카인즈 뉴스 데이터 전처리
2. FinBERT 기반 감성 분석  
3. PyTorch LSTM 모델을 활용한 시계열 감성 트렌드 예측
4. 삼성전자, 애플 감성 트렌드 시각화 및 비교 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyTorch 및 NLP 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 텍스트 전처리
import re
from tqdm.auto import tqdm

# 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋 클래스"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """PyTorch LSTM 모델 클래스"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 완전연결층
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # bidirectional이므로 *2
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM 순전파
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시간 스텝의 출력 사용
        lstm_out = lstm_out[:, -1, :]
        
        # 완전연결층 순전파
        out = self.dropout(lstm_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

class BigKindsPyTorchLSTMAnalyzer:
    def __init__(self, data_path):
        """
        빅카인즈 데이터 기반 PyTorch LSTM 감성 분석기 초기화
        
        Args:
            data_path (str): 빅카인즈 데이터 파일들이 있는 경로
        """
        self.data_path = data_path
        self.file_paths = [
            f"{data_path}/NewsResult_20210101-20211231.xlsx",
            f"{data_path}/NewsResult_20220101-20221231.xlsx", 
            f"{data_path}/NewsResult_20230101-20231231.xlsx",
            f"{data_path}/NewsResult_20240101-20241231.xlsx"
        ]
        
        # 감성 분석 모델 초기화 (KR-FinBERT-SC)
        print("🤖 FinBERT 모델 로딩 중...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="snunlp/KR-FinBert-SC",
            tokenizer="snunlp/KR-FinBert-SC",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None
        )
        
        # 데이터 저장용
        self.raw_data = None
        self.processed_data = None
        self.sentiment_data = None
        self.lstm_models = {}
        self.scalers = {}
        
    def load_data(self):
        """빅카인즈 엑셀 파일들을 불러와서 통합"""
        print("📊 빅카인즈 뉴스 데이터 로딩 중...")
        
        all_data = []
        
        for file_path in self.file_paths:
            try:
                print(f"   - {file_path.split('/')[-1]} 로딩 중...")
                df = pd.read_excel(file_path)
                
                # 컬럼명 확인 및 출력
                print(f"     컬럼: {list(df.columns)}")
                
                # 필요한 컬럼 찾기 (유연한 컬럼명 매칭)
                date_col = None
                title_col = None
                keyword_col = None
                content_col = None
                media_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['일자', '날짜', 'date']):
                        date_col = col
                    elif any(keyword in col_lower for keyword in ['제목', 'title']):
                        title_col = col
                    elif any(keyword in col_lower for keyword in ['키워드', 'keyword']):
                        keyword_col = col
                    elif any(keyword in col_lower for keyword in ['본문', 'content', '내용']):
                        content_col = col
                    elif any(keyword in col_lower for keyword in ['언론사', '매체', 'media']):
                        media_col = col
                
                # 필수 컬럼 확인
                if not date_col or not title_col:
                    print(f"     ⚠️ 필수 컬럼 누락: 일자({date_col}), 제목({title_col})")
                    continue
                
                # 필요한 컬럼만 선택
                selected_cols = [date_col, title_col]
                if keyword_col:
                    selected_cols.append(keyword_col)
                if content_col:
                    selected_cols.append(content_col)
                if media_col:
                    selected_cols.append(media_col)
                
                df_selected = df[selected_cols].copy()
                
                # 컬럼명 표준화
                df_selected = df_selected.rename(columns={
                    date_col: '일자',
                    title_col: '제목',
                    keyword_col: '키워드' if keyword_col else None,
                    content_col: '본문' if content_col else None,
                    media_col: '언론사' if media_col else None
                })
                
                # None 컬럼 제거
                df_selected = df_selected.loc[:, df_selected.columns.notna()]
                
                all_data.append(df_selected)
                print(f"     ✓ {len(df_selected)}건 로딩 완료")
                
            except Exception as e:
                print(f"     ✗ 파일 로딩 실패: {e}")
                continue
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            print(f"📈 총 {len(self.raw_data)}건의 뉴스 데이터 로딩 완료\n")
            
            # 데이터 기본 정보 출력
            print("📋 데이터 기본 정보:")
            print(f"   - 컬럼: {list(self.raw_data.columns)}")
            if '일자' in self.raw_data.columns:
                print(f"   - 기간: {self.raw_data['일자'].min()} ~ {self.raw_data['일자'].max()}")
            
            return True
        else:
            print("❌ 데이터 로딩 실패")
            return False
    
    def preprocess_data(self):
        """데이터 전처리"""
        print("🔧 데이터 전처리 시작...")
        
        if self.raw_data is None:
            print("❌ 데이터가 로딩되지 않았습니다.")
            return False
        
        df = self.raw_data.copy()
        
        # 1. 날짜 처리
        print("   - 날짜 데이터 처리 중...")
        try:
            if df['일자'].dtype == 'object':
                # 다양한 날짜 형식 처리
                df['일자'] = pd.to_datetime(df['일자'], errors='coerce', infer_datetime_format=True)
            elif df['일자'].dtype in ['int64', 'float64']:
                # 숫자 형태인 경우 (예: 20210101)
                df['일자'] = pd.to_datetime(df['일자'].astype(str), format='%Y%m%d', errors='coerce')
        except Exception as e:
            print(f"     날짜 처리 오류: {e}")
            return False
        
        # 2. 결측값 처리
        print("   - 결측값 처리 중...")
        original_count = len(df)
        df = df.dropna(subset=['일자', '제목'])  # 필수 컬럼의 결측값 제거
        print(f"     결측값 제거: {original_count} → {len(df)}건")
        
        # 빈 컬럼 채우기
        for col in ['키워드', '본문', '언론사']:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # 3. 텍스트 결합 (제목 + 키워드 + 본문)
        print("   - 텍스트 데이터 결합 중...")
        def combine_text(row):
            text_parts = []
            if pd.notna(row.get('제목', '')) and str(row['제목']).strip():
                text_parts.append(str(row['제목']).strip())
            if '키워드' in row and pd.notna(row.get('키워드', '')) and str(row['키워드']).strip():
                text_parts.append(f"키워드: {str(row['키워드']).strip()}")
            if '본문' in row and pd.notna(row.get('본문', '')) and str(row['본문']).strip():
                # 본문이 너무 길면 앞부분만 사용
                content = str(row['본문']).strip()[:500]
                text_parts.append(content)
            return ' '.join(text_parts)
        
        df['combined_text'] = df.apply(combine_text, axis=1)
        
        # 4. 기업 분류 (삼성/애플)
        print("   - 기업 분류 중...")
        def classify_company(text):
            text_lower = str(text).lower()
            samsung_keywords = ['삼성전자', '삼성', 'samsung', '갤럭시', 'galaxy', 'sdi', '반도체', 'dram', 'nand']
            apple_keywords = ['애플', 'apple', '아이폰', 'iphone', '아이패드', 'ipad', '맥북', 'macbook', 'ios', 'macos']
            
            samsung_count = sum(1 for keyword in samsung_keywords if keyword in text_lower)
            apple_count = sum(1 for keyword in apple_keywords if keyword in text_lower)
            
            if samsung_count > apple_count and samsung_count > 0:
                return 'samsung'
            elif apple_count > samsung_count and apple_count > 0:
                return 'apple'
            elif samsung_count > 0 and apple_count > 0:
                return 'both'
            else:
                return 'none'
        
        df['company'] = df['combined_text'].apply(classify_company)
        
        # 5. 텍스트 정제
        print("   - 텍스트 정제 중...")
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            text = str(text)
            # HTML 태그 제거
            text = re.sub(r'<[^>]+>', '', text)
            # 특수문자 정리 (한글, 영문, 숫자, 공백, 일부 특수문자만 유지)
            text = re.sub(r'[^\w\s가-힣.,!?:%-]', ' ', text)
            # 연속된 공백 제거
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
        
        # 6. 빈 텍스트 및 관련 없는 기사 제거
        original_count = len(df)
        df = df[df['cleaned_text'].str.len() > 10]  # 10자 이상의 텍스트만 유지
        df = df[df['company'].isin(['samsung', 'apple'])]  # 관련 기업만 유지
        print(f"     필터링: {original_count} → {len(df)}건")
        
        self.processed_data = df.reset_index(drop=True)
        
        samsung_count = len(df[df['company'] == 'samsung'])
        apple_count = len(df[df['company'] == 'apple'])
        
        print(f"✅ 전처리 완료: {len(self.processed_data)}건")
        print(f"   - 삼성: {samsung_count}건")
        print(f"   - 애플: {apple_count}건\n")
        
        return True
    
    def analyze_sentiment(self, batch_size=16):
        """FinBERT 기반 감성 분석"""
        print("🎯 FinBERT 감성 분석 시작...")
        
        if self.processed_data is None:
            print("❌ 전처리된 데이터가 없습니다.")
            return False
        
        df = self.processed_data.copy()
        
        def sentiment_to_score(sentiment_result):
            """감성 분석 결과를 1-5 점수로 변환"""
            try:
                if isinstance(sentiment_result, list):
                    # 확률 딕셔너리 생성
                    prob_dict = {item['label']: item['score'] for item in sentiment_result}
                    
                    # 최고 확률의 감성 선택
                    best_label = max(prob_dict, key=prob_dict.get)
                    best_score = prob_dict[best_label]
                    
                    # 3가지 감성에 대한 확률
                    pos_prob = prob_dict.get('positive', 0)
                    neg_prob = prob_dict.get('negative', 0)
                    neu_prob = prob_dict.get('neutral', 0)
                    
                    # 1-5 스케일로 변환 (가중평균 방식)
                    score = 1.0 * neg_prob + 3.0 * neu_prob + 5.0 * pos_prob
                    
                else:
                    # 단일 결과인 경우
                    label = sentiment_result['label']
                    confidence = sentiment_result['score']
                    
                    if label == 'positive':
                        score = 3.0 + 2.0 * confidence
                    elif label == 'negative':
                        score = 3.0 - 2.0 * confidence
                    else:  # neutral
                        score = 3.0
                    
                    best_label = label
                
                return max(1.0, min(5.0, score)), best_label
                
            except Exception as e:
                print(f"감성 점수 변환 오류: {e}")
                return 3.0, 'neutral'
        
        # 배치별로 감성 분석 수행
        sentiment_scores = []
        sentiment_labels = []
        
        texts = df['cleaned_text'].tolist()
        
        print(f"   - 총 {len(texts)}개 텍스트 분석 중 (배치 크기: {batch_size})...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="감성 분석"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 각 텍스트를 512자로 제한
                batch_texts = [text[:512] for text in batch_texts]
                
                # 감성 분석 수행
                results = self.sentiment_analyzer(batch_texts, truncation=True, max_length=512)
                
                for result in results:
                    score, label = sentiment_to_score(result)
                    sentiment_scores.append(score)
                    sentiment_labels.append(label)
                    
            except Exception as e:
                print(f"     배치 {i//batch_size + 1} 분석 실패: {e}")
                # 실패한 배치는 중립으로 처리
                for _ in range(len(batch_texts)):
                    sentiment_scores.append(3.0)
                    sentiment_labels.append('neutral')
        
        df['sentiment_score'] = sentiment_scores
        df['sentiment_label'] = sentiment_labels
        
        self.sentiment_data = df
        
        print(f"✅ 감성 분석 완료")
        print(f"   - 긍정: {len(df[df['sentiment_label']=='positive'])}건")
        print(f"   - 중립: {len(df[df['sentiment_label']=='neutral'])}건")
        print(f"   - 부정: {len(df[df['sentiment_label']=='negative'])}건")
        print(f"   - 평균 감성 점수: {df['sentiment_score'].mean():.2f}\n")
        
        return True
    
    def prepare_time_series_data(self, company='samsung'):
        """시계열 데이터 준비"""
        print(f"📈 {company.upper()} 시계열 데이터 준비 중...")
        
        if self.sentiment_data is None:
            print("❌ 감성 분석 데이터가 없습니다.")
            return None
        
        # 해당 기업 데이터 필터링
        company_data = self.sentiment_data[self.sentiment_data['company'] == company].copy()
        
        if len(company_data) == 0:
            print(f"❌ {company} 관련 데이터가 없습니다.")
            return None
        
        # 일별 집계
        daily_data = company_data.groupby('일자').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x)  # 긍정 비율
        }).reset_index()
        
        # 컬럼명 정리
        daily_data.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count', 'positive_ratio']
        daily_data['sentiment_std'] = daily_data['sentiment_std'].fillna(0)
        
        # 날짜순 정렬
        daily_data = daily_data.sort_values('date').reset_index(drop=True)
        
        # 이동평균 계산
        daily_data['sentiment_ma7'] = daily_data['sentiment_mean'].rolling(window=7, min_periods=1).mean()
        daily_data['sentiment_ma30'] = daily_data['sentiment_mean'].rolling(window=30, min_periods=1).mean()
        daily_data['news_count_ma7'] = daily_data['news_count'].rolling(window=7, min_periods=1).mean()
        
        # 결측값 처리
        daily_data = daily_data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ 시계열 데이터 준비 완료: {len(daily_data)}일간의 데이터")
        print(f"   - 기간: {daily_data['date'].min()} ~ {daily_data['date'].max()}")
        print(f"   - 일평균 기사 수: {daily_data['news_count'].mean():.1f}건\n")
        
        return daily_data
    
    def create_sequences(self, data, features, target, sequence_length=30):
        """LSTM용 시퀀스 데이터 생성"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[features].iloc[i-sequence_length:i].values)
            y.append(data[target].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_pytorch_lstm(self, company='samsung', sequence_length=30, epochs=100, learning_rate=0.001):
        """PyTorch LSTM 모델 훈련"""
        print(f"🤖 {company.upper()} PyTorch LSTM 모델 훈련 시작...")
        
        # 시계열 데이터 준비
        time_series_data = self.prepare_time_series_data(company)
        if time_series_data is None:
            return None
        
        # 특성 선택
        features = ['sentiment_mean', 'sentiment_std', 'news_count', 'positive_ratio', 
                   'sentiment_ma7', 'news_count_ma7']
        target = 'sentiment_mean'
        
        # 데이터 정규화
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(time_series_data[features])
        y_scaled = scaler_y.fit_transform(time_series_data[[target]])
        
        # 정규화된 데이터로 DataFrame 생성
        scaled_data = pd.DataFrame(X_scaled, columns=features)
        scaled_data[target] = y_scaled.flatten()
        
        # LSTM 시퀀스 생성
        X, y = self.create_sequences(scaled_data, features, target, sequence_length)
        
        if len(X) == 0:
            print("❌ 충분한 시퀀스 데이터가 없습니다.")
            return None
        
        # 훈련/테스트 데이터 분할
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"   - 훈련 데이터: {len(X_train)}개 시퀀스")
        print(f"   - 테스트 데이터: {len(X_test)}개 시퀀스")
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 모델 초기화
        model = LSTMModel(input_size=len(features)).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # 훈련 기록
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("   - 훈련 시작...")
        
        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 검증 단계
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs.squeeze(), y_batch).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 학습률 스케줄러
            scheduler.step(val_loss)
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최상의 모델 저장
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"     조기 종료 (에포크 {epoch+1})")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"     에포크 [{epoch+1}/{epochs}] - 훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
        
        # 최상의 모델 로드
        model.load_state_dict(best_model_state)
        
        # 테스트 데이터 예측
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        # 정규화 해제
        predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 성능 평가
        mse = mean_squared_error(actuals_original, predictions_original)
        mae = mean_absolute_error(actuals_original, predictions_original)
        r2 = r2_score(actuals_original, predictions_original)
        
        print(f"✅ 모델 훈련 완료")
        print(f"   - MSE: {mse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - R²: {r2:.4f}\n")
        
        # 모델 정보 저장
        self.lstm_models[company] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'sequence_length': sequence_length,
            'features': features,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'time_series_data': time_series_data,
            'test_results': {
                'predictions': predictions_original,
                'actuals': actuals_original,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        }
        
        return self.lstm_models[company]
    
    def predict_future_sentiment(self, company='samsung', days=30):
        """미래 감성 점수 예측"""
        print(f"🔮 {company.upper()} 향후 {days}일 감성 예측...")
        
        if company not in self.lstm_models:
            print(f"❌ {company} 모델이 훈련되지 않았습니다.")
            return None
        
        model_info = self.lstm_models[company]
        model = model_info['model']
        scaler_X = model_info['scaler_X']
        scaler_y = model_info['scaler_y']
        sequence_length = model_info['sequence_length']
        features = model_info['features']
        time_series_data = model_info['time_series_data']
        
        # 마지막 시퀀스 가져오기
        last_sequence = scaler_X.transform(time_series_data[features].tail(sequence_length))
        
        model.eval()
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(days):
                # 예측 수행
                pred_scaled = model(current_sequence)
                pred_value = pred_scaled.cpu().numpy()[0, 0]
                
                # 정규화 해제하여 실제 값으로 변환
                pred_original = scaler_y.inverse_transform([[pred_value]])[0, 0]
                predictions.append(pred_original)
                
                # 시퀀스 업데이트 (간단한 방법: 예측값을 첫 번째 특성으로 사용)
                new_features = current_sequence[0, -1].clone()
                new_features[0] = pred_scaled[0, 0]  # sentiment_mean 업데이트
                
                # 시퀀스 롤링
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_features.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        # 미래 날짜 생성
        last_date = time_series_data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sentiment': predictions
        })
        
        print(f"✅ 미래 예측 완료")
        print(f"   - 예측 기간: {future_dates[0]} ~ {future_dates[-1]}")
        print(f"   - 평균 예측 감성 점수: {np.mean(predictions):.2f}\n")
        
        return future_df
    
    def visualize_results(self, company='samsung'):
        """결과 시각화"""
        print(f"📊 {company.upper()} 분석 결과 시각화...")
        
        if company not in self.lstm_models:
            print(f"❌ {company} 모델이 훈련되지 않았습니다.")
            return
        
        model_info = self.lstm_models[company]
        time_series_data = model_info['time_series_data']
        test_results = model_info['test_results']
        train_losses = model_info['train_losses']
        val_losses = model_info['val_losses']
        
        # 1. 전체 감성 트렌드 시각화
        plt.figure(figsize=(20, 15))
        
        # 서브플롯 1: 일별 감성 점수 및 이동평균
        plt.subplot(3, 3, 1)
        plt.plot(time_series_data['date'], time_series_data['sentiment_mean'], 
                alpha=0.4, color='blue', label='일별 감성 점수', linewidth=1)
        plt.plot(time_series_data['date'], time_series_data['sentiment_ma7'], 
                color='orange', label='7일 이동평균', linewidth=2)
        plt.plot(time_series_data['date'], time_series_data['sentiment_ma30'], 
                color='red', label='30일 이동평균', linewidth=2)
        plt.title(f'{company.upper()} 일별 감성 점수 트렌드', fontsize=14)
        plt.ylabel('감성 점수')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 서브플롯 2: 일별 기사 수
        plt.subplot(3, 3, 2)
        plt.bar(time_series_data['date'], time_series_data['news_count'], 
               alpha=0.6, color='green', label='일별 기사 수')
        plt.plot(time_series_data['date'], time_series_data['news_count_ma7'], 
                color='darkgreen', label='7일 이동평균', linewidth=2)
        plt.title(f'{company.upper()} 일별 뉴스 기사 수', fontsize=14)
        plt.ylabel('기사 수')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 서브플롯 3: 긍정 비율
        plt.subplot(3, 3, 3)
        plt.plot(time_series_data['date'], time_series_data['positive_ratio'], 
                color='purple', marker='o', markersize=3, label='긍정 비율', linewidth=1.5)
        plt.title(f'{company.upper()} 일별 긍정 뉴스 비율', fontsize=14)
        plt.ylabel('긍정 비율')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 서브플롯 4: LSTM 모델 학습 과정
        plt.subplot(3, 3, 4)
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.title('PyTorch LSTM 모델 학습 과정', fontsize=14)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 5: 예측 vs 실제 (테스트 데이터)
        plt.subplot(3, 3, 5)
        test_dates = time_series_data['date'].iloc[-len(test_results['actuals']):]
        plt.plot(test_dates, test_results['actuals'], 'b-', label='실제 값', alpha=0.8, linewidth=2)
        plt.plot(test_dates, test_results['predictions'], 'r--', label='예측 값', alpha=0.8, linewidth=2)
        plt.title('LSTM 예측 성능 (테스트 데이터)', fontsize=14)
        plt.ylabel('감성 점수')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 서브플롯 6: 산점도 (예측 vs 실제)
        plt.subplot(3, 3, 6)
        plt.scatter(test_results['actuals'], test_results['predictions'], alpha=0.7, s=50)
        min_val = min(test_results['actuals'].min(), test_results['predictions'].min())
        max_val = max(test_results['actuals'].max(), test_results['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        plt.xlabel('실제 값')
        plt.ylabel('예측 값')
        plt.title(f'예측 정확도 (R² = {test_results["r2"]:.3f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 7: 감성 점수 분포
        plt.subplot(3, 3, 7)
        company_sentiment = self.sentiment_data[self.sentiment_data['company'] == company]['sentiment_score']
        plt.hist(company_sentiment, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(company_sentiment.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {company_sentiment.mean():.2f}')
        plt.title(f'{company.upper()} 감성 점수 분포', fontsize=14)
        plt.xlabel('감성 점수')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 8: 월별 감성 트렌드
        plt.subplot(3, 3, 8)
        monthly_data = time_series_data.copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        monthly_avg = monthly_data.groupby('year_month')['sentiment_mean'].mean()
        
        plt.plot(range(len(monthly_avg)), monthly_avg.values, marker='o', linewidth=2, markersize=6)
        plt.title(f'{company.upper()} 월별 평균 감성 점수', fontsize=14)
        plt.xlabel('월')
        plt.ylabel('평균 감성 점수')
        plt.xticks(range(0, len(monthly_avg), max(1, len(monthly_avg)//6)), 
                  [str(month) for i, month in enumerate(monthly_avg.index) if i % max(1, len(monthly_avg)//6) == 0], 
                  rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 9: 감성 변동성 (표준편차)
        plt.subplot(3, 3, 9)
        plt.plot(time_series_data['date'], time_series_data['sentiment_std'], 
                color='orange', marker='o', markersize=3, label='감성 변동성', linewidth=1.5)
        plt.title(f'{company.upper()} 일별 감성 변동성', fontsize=14)
        plt.ylabel('표준편차')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{company}_pytorch_lstm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 미래 예측 시각화
        future_pred = self.predict_future_sentiment(company, days=30)
        
        if future_pred is not None:
            plt.figure(figsize=(16, 10))
            
            # 상단: 과거 + 미래 예측
            plt.subplot(2, 1, 1)
            recent_data = time_series_data.tail(90)  # 최근 90일
            plt.plot(recent_data['date'], recent_data['sentiment_mean'], 
                    'b-', label='과거 실제 감성 점수', linewidth=2.5, alpha=0.8)
            plt.plot(recent_data['date'], recent_data['sentiment_ma7'], 
                    'orange', label='과거 7일 이동평균', linewidth=2, alpha=0.8)
            
            # 미래 예측
            plt.plot(future_pred['date'], future_pred['predicted_sentiment'], 
                    'r--', label='미래 예측 감성 점수', linewidth=3, marker='o', markersize=4)
            
            # 현재 시점 표시
            plt.axvline(x=time_series_data['date'].max(), color='green', 
                       linestyle='-', alpha=0.8, linewidth=2, label='현재')
            
            # 신뢰구간 (단순히 ±0.2 범위로 표시)
            plt.fill_between(future_pred['date'], 
                           future_pred['predicted_sentiment'] - 0.2,
                           future_pred['predicted_sentiment'] + 0.2,
                           alpha=0.2, color='red', label='예측 신뢰구간')
            
            plt.title(f'{company.upper()} 감성 점수 예측 (과거 90일 + 미래 30일)', fontsize=16)
            plt.xlabel('날짜')
            plt.ylabel('감성 점수')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 하단: 예측 트렌드 분석
            plt.subplot(2, 1, 2)
            
            # 예측값의 변화율 계산
            pred_values = future_pred['predicted_sentiment'].values
            trend_changes = np.diff(pred_values)
            
            colors = ['red' if x < 0 else 'green' for x in trend_changes]
            plt.bar(range(len(trend_changes)), trend_changes, color=colors, alpha=0.7)
            plt.title('일별 감성 점수 변화 예측', fontsize=14)
            plt.xlabel('예측 일수')
            plt.ylabel('감성 점수 변화량')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{company}_future_prediction.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✅ {company.upper()} 시각화 완료\n")
    
    def generate_comprehensive_report(self):
        """종합 분석 리포트 생성"""
        print("📝 종합 분석 리포트 생성 중...")
        
        report = []
        report.append("=" * 80)
        report.append("📊 빅카인즈 뉴스 데이터 기반 PyTorch LSTM 감성 트렌드 분석 리포트")
        report.append("=" * 80)
        report.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"사용 디바이스: {device}")
        report.append("")
        
        # 1. 데이터 개요
        if self.sentiment_data is not None:
            report.append("📈 데이터 개요")
            report.append("-" * 50)
            report.append(f"총 뉴스 기사 수: {len(self.sentiment_data):,}건")
            report.append(f"분석 기간: {self.sentiment_data['일자'].min().strftime('%Y-%m-%d')} ~ {self.sentiment_data['일자'].max().strftime('%Y-%m-%d')}")
            
            samsung_count = len(self.sentiment_data[self.sentiment_data['company'] == 'samsung'])
            apple_count = len(self.sentiment_data[self.sentiment_data['company'] == 'apple'])
            
            report.append(f"삼성 관련 뉴스: {samsung_count:,}건 ({samsung_count/len(self.sentiment_data)*100:.1f}%)")
            report.append(f"애플 관련 뉴스: {apple_count:,}건 ({apple_count/len(self.sentiment_data)*100:.1f}%)")
            report.append("")
            
            # 2. 감성 분석 결과
            sentiment_dist = self.sentiment_data['sentiment_label'].value_counts()
            report.append("🎯 전체 감성 분석 결과")
            report.append("-" * 50)
            for sentiment, count in sentiment_dist.items():
                percentage = (count / len(self.sentiment_data)) * 100
                report.append(f"{sentiment.upper()}: {count:,}건 ({percentage:.1f}%)")
            report.append(f"평균 감성 점수: {self.sentiment_data['sentiment_score'].mean():.3f}/5.0")
            report.append(f"감성 점수 표준편차: {self.sentiment_data['sentiment_score'].std():.3f}")
            report.append("")
            
            # 3. 기업별 감성 분석
            report.append("🏢 기업별 감성 분석 비교")
            report.append("-" * 50)
            
            for company in ['samsung', 'apple']:
                company_data = self.sentiment_data[self.sentiment_data['company'] == company]
                if len(company_data) > 0:
                    avg_sentiment = company_data['sentiment_score'].mean()
                    std_sentiment = company_data['sentiment_score'].std()
                    positive_ratio = (company_data['sentiment_label'] == 'positive').mean()
                    negative_ratio = (company_data['sentiment_label'] == 'negative').mean()
                    
                    report.append(f"{company.upper()}:")
                    report.append(f"  - 평균 감성 점수: {avg_sentiment:.3f}/5.0")
                    report.append(f"  - 감성 점수 표준편차: {std_sentiment:.3f}")
                    report.append(f"  - 긍정 뉴스 비율: {positive_ratio:.1%}")
                    report.append(f"  - 부정 뉴스 비율: {negative_ratio:.1%}")
                    report.append("")
        
        # 4. LSTM 모델 성능
        report.append("🤖 PyTorch LSTM 모델 성능 분석")
        report.append("-" * 50)
        
        for company in ['samsung', 'apple']:
            if company in self.lstm_models:
                model_info = self.lstm_models[company]
                test_results = model_info['test_results']
                
                report.append(f"{company.upper()} LSTM 모델:")
                report.append(f"  - MSE (평균제곱오차): {test_results['mse']:.4f}")
                report.append(f"  - MAE (평균절대오차): {test_results['mae']:.4f}")
                report.append(f"  - R² (결정계수): {test_results['r2']:.4f}")
                
                # 성능 해석
                if test_results['r2'] > 0.8:
                    performance = "매우 우수"
                elif test_results['r2'] > 0.6:
                    performance = "우수"
                elif test_results['r2'] > 0.4:
                    performance = "양호"
                elif test_results['r2'] > 0.2:
                    performance = "보통"
                else:
                    performance = "개선 필요"
                
                report.append(f"  - 모델 성능 평가: {performance}")
                
                # 시계열 데이터 정보
                time_series_data = model_info['time_series_data']
                report.append(f"  - 시계열 데이터 포인트: {len(time_series_data)}일")
                report.append(f"  - 일평균 기사 수: {time_series_data['news_count'].mean():.1f}건")
                report.append("")
        
        # 5. 시간적 트렌드 분석
        if self.sentiment_data is not None:
            report.append("📅 시간적 트렌드 분석")
            report.append("-" * 50)
            
            # 연도별 트렌드
            yearly_sentiment = self.sentiment_data.groupby(self.sentiment_data['일자'].dt.year)['sentiment_score'].mean()
            report.append("연도별 평균 감성 점수:")
            for year, score in yearly_sentiment.items():
                report.append(f"  - {year}년: {score:.3f}")
            
            # 전체 트렌드 방향
            if len(yearly_sentiment) > 1:
                trend_direction = "상승" if yearly_sentiment.iloc[-1] > yearly_sentiment.iloc[0] else "하락"
                trend_magnitude = abs(yearly_sentiment.iloc[-1] - yearly_sentiment.iloc[0])
                report.append(f"전체 트렌드: {trend_direction} ({trend_magnitude:.3f}점 변화)")
            report.append("")
            
            # 월별 변동성 분석
            monthly_sentiment = self.sentiment_data.groupby(self.sentiment_data['일자'].dt.to_period('M'))['sentiment_score'].agg(['mean', 'std'])
            volatility = monthly_sentiment['std'].mean()
            report.append(f"월별 감성 변동성 (평균): {volatility:.3f}")
            report.append("")
        
        # 6. 미래 예측 결과
        report.append("🔮 미래 감성 트렌드 예측")
        report.append("-" * 50)
        
        for company in self.lstm_models.keys():
            future_pred = self.predict_future_sentiment(company, days=30)
            if future_pred is not None:
                current_sentiment = self.lstm_models[company]['time_series_data']['sentiment_mean'].iloc[-1]
                future_avg = future_pred['predicted_sentiment'].mean()
                trend_change = future_avg - current_sentiment
                
                report.append(f"{company.upper()} 향후 30일 예측:")
                report.append(f"  - 현재 감성 점수: {current_sentiment:.3f}")
                report.append(f"  - 예측 평균 감성 점수: {future_avg:.3f}")
                report.append(f"  - 예상 변화: {'+' if trend_change > 0 else ''}{trend_change:.3f}")
                
                if abs(trend_change) > 0.1:
                    direction = "개선" if trend_change > 0 else "악화"
                    report.append(f"  - 트렌드 전망: {direction} 예상")
                else:
                    report.append(f"  - 트렌드 전망: 안정적 유지 예상")
                report.append("")
        
        # 7. 주요 인사이트 및 권고사항
        report.append("💡 주요 인사이트 및 권고사항")
        report.append("-" * 50)
        
        insights = []
        
        # 기업별 감성 비교
        if len(self.lstm_models) >= 2:
            samsung_sentiment = self.sentiment_data[self.sentiment_data['company'] == 'samsung']['sentiment_score'].mean()
            apple_sentiment = self.sentiment_data[self.sentiment_data['company'] == 'apple']['sentiment_score'].mean()
            
            if samsung_sentiment > apple_sentiment:
                insights.append(f"삼성전자의 뉴스 감성이 애플보다 {samsung_sentiment - apple_sentiment:.3f}점 높음")
            else:
                insights.append(f"애플의 뉴스 감성이 삼성전자보다 {apple_sentiment - samsung_sentiment:.3f}점 높음")
        
        # 모델 성능 기반 인사이트
        reliable_models = []
        for company, model_info in self.lstm_models.items():
            if model_info['test_results']['r2'] > 0.5:
                reliable_models.append(company)
        
        if reliable_models:
            insights.append(f"{', '.join([c.upper() for c in reliable_models])} 모델의 예측 신뢰도가 높음 (R² > 0.5)")
        
        # 변동성 분석
        if self.sentiment_data is not None:
            high_volatility_threshold = 0.5
            for company in ['samsung', 'apple']:
                company_data = self.sentiment_data[self.sentiment_data['company'] == company]
                if len(company_data) > 0:
                    volatility = company_data['sentiment_score'].std()
                    if volatility > high_volatility_threshold:
                        insights.append(f"{company.upper()}의 감성 변동성이 높음 (σ={volatility:.3f})")
        
        for i, insight in enumerate(insights, 1):
            report.append(f"{i}. {insight}")
        
        report.append("")
        report.append("📋 활용 방안")
        report.append("-" * 50)
        applications = [
            "실시간 기업 이미지 모니터링 시스템 구축",
            "제품 출시 타이밍 최적화를 위한 감성 분석",
            "경쟁사 대비 브랜드 포지셔닝 분석",
            "위기 관리를 위한 부정 감성 조기 감지",
            "마케팅 캠페인 효과 측정 및 최적화",
            "투자 의사결정 보조 지표로 활용",
            "언론 대응 전략 수립을 위한 감성 트렌드 분석"
        ]
        
        for i, app in enumerate(applications, 1):
            report.append(f"{i}. {app}")
        
        report.append("")
        report.append("🔧 모델 개선 방안")
        report.append("-" * 50)
        improvements = [
            "외부 데이터 통합 (주가, 경제지표, 소셜미디어)",
            "어텐션 메커니즘 적용으로 중요 시점 가중치 부여",
            "앙상블 모델 구성 (LSTM + GRU + Transformer)",
            "하이퍼파라미터 최적화 (베이지안 최적화)",
            "실시간 데이터 파이프라인 구축",
            "설명 가능한 AI 기법 적용 (SHAP, LIME)",
            "다중 시간 해상도 예측 (일/주/월별)"
        ]
        
        for i, improvement in enumerate(improvements, 1):
            report.append(f"{i}. {improvement}")
        
        report.append("")
        report.append("=" * 80)
        report.append("리포트 생성 완료")
        report.append("=" * 80)
        
        # 리포트 저장
        report_text = "\n".join(report)
        with open("PyTorch_LSTM_Sentiment_Analysis_Report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print("✅ 종합 분석 리포트 생성 완료: PyTorch_LSTM_Sentiment_Analysis_Report.txt")
        return report_text
    
    def run_complete_analysis(self):
        """전체 분석 파이프라인 실행"""
        print("🚀 빅카인즈 뉴스 데이터 PyTorch LSTM 감성 분석 시작!")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. 데이터 로딩
            print("1️⃣ 데이터 로딩 단계")
            if not self.load_data():
                print("❌ 데이터 로딩 실패로 분석을 중단합니다.")
                return False
            
            # 2. 데이터 전처리
            print("2️⃣ 데이터 전처리 단계")
            if not self.preprocess_data():
                print("❌ 데이터 전처리 실패로 분석을 중단합니다.")
                return False
            
            # 3. 감성 분석
            print("3️⃣ 감성 분석 단계")
            if not self.analyze_sentiment():
                print("❌ 감성 분석 실패로 분석을 중단합니다.")
                return False
            
            # 4. LSTM 모델 훈련
            print("4️⃣ LSTM 모델 훈련 단계")
            trained_models = []
            
            for company in ['samsung', 'apple']:
                company_data = self.sentiment_data[self.sentiment_data['company'] == company]
                if len(company_data) > 100:  # 충분한 데이터가 있는 경우만 훈련
                    print(f"   {company.upper()} 모델 훈련 중...")
                    model_info = self.train_pytorch_lstm(company)
                    if model_info:
                        trained_models.append(company)
                else:
                    print(f"   ⚠️ {company.upper()} 데이터 부족 (필요: 100개, 현재: {len(company_data)}개)")
            
            if not trained_models:
                print("❌ 훈련된 모델이 없습니다.")
                return False
            
            # 5. 결과 시각화
            print("5️⃣ 결과 시각화 단계")
            for company in trained_models:
                print(f"   {company.upper()} 시각화 생성 중...")
                self.visualize_results(company)
            
            # 6. 종합 리포트 생성
            print("6️⃣ 종합 리포트 생성 단계")
            self.generate_comprehensive_report()
            
            # 실행 시간 계산
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            print("\n🎉 전체 분석 완료!")
            print("=" * 80)
            print(f"⏱️  총 실행 시간: {execution_time}")
            print(f"📊 훈련된 모델: {', '.join([c.upper() for c in trained_models])}")
            print(f"📈 생성된 파일:")
            print(f"   - 시각화: {', '.join([f'{c}_pytorch_lstm_analysis.png' for c in trained_models])}")
            print(f"   - 예측 차트: {', '.join([f'{c}_future_prediction.png' for c in trained_models])}")
            print(f"   - 종합 리포트: PyTorch_LSTM_Sentiment_Analysis_Report.txt")
            
            return True
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def interactive_analysis(self):
        """대화형 분석 모드"""
        print("🔍 대화형 분석 모드")
        print("=" * 50)
        
        while True:
            print("\n사용 가능한 명령어:")
            print("1. 감성 데이터 요약 보기")
            print("2. 특정 기업 시각화")
            print("3. 미래 예측 (사용자 정의 기간)")
            print("4. 모델 성능 비교")
            print("5. 종료")
            
            choice = input("\n선택하세요 (1-5): ").strip()
            
            if choice == '1':
                if self.sentiment_data is not None:
                    print("\n📊 감성 데이터 요약:")
                    print(f"총 기사 수: {len(self.sentiment_data):,}건")
                    print(f"평균 감성 점수: {self.sentiment_data['sentiment_score'].mean():.3f}")
                    print("\n기업별 통계:")
                    for company in ['samsung', 'apple']:
                        data = self.sentiment_data[self.sentiment_data['company'] == company]
                        if len(data) > 0:
                            print(f"{company.upper()}: {len(data)}건, 평균 {data['sentiment_score'].mean():.3f}")
                else:
                    print("❌ 감성 분석 데이터가 없습니다.")
            
            elif choice == '2':
                company = input("기업명 입력 (samsung/apple): ").strip().lower()
                if company in self.lstm_models:
                    self.visualize_results(company)
                else:
                    print(f"❌ {company} 모델을 찾을 수 없습니다.")
            
            elif choice == '3':
                company = input("기업명 입력 (samsung/apple): ").strip().lower()
                try:
                    days = int(input("예측 기간 입력 (일): ").strip())
                    if company in self.lstm_models and days > 0:
                        future_pred = self.predict_future_sentiment(company, days)
                        if future_pred is not None:
                            print(f"\n📈 {company.upper()} {days}일 예측 결과:")
                            print(f"평균 예측 감성: {future_pred['predicted_sentiment'].mean():.3f}")
                            print(f"예측 범위: {future_pred['predicted_sentiment'].min():.3f} ~ {future_pred['predicted_sentiment'].max():.3f}")
                    else:
                        print("❌ 유효하지 않은 입력입니다.")
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
            
            elif choice == '4':
                if self.lstm_models:
                    print("\n🏆 모델 성능 비교:")
                    for company, model_info in self.lstm_models.items():
                        results = model_info['test_results']
                        print(f"{company.upper()}:")
                        print(f"  R²: {results['r2']:.4f}")
                        print(f"  MAE: {results['mae']:.4f}")
                else:
                    print("❌ 훈련된 모델이 없습니다.")
            
            elif choice == '5':
                print("👋 대화형 분석 모드를 종료합니다.")
                break
            
            else:
                print("❌ 유효하지 않은 선택입니다.")


# 메인 실행 부분
if __name__ == "__main__":
    try:
        # 분석기 초기화
        data_path = r"C:\Users\jmzxc\OneDrive\바탕 화면\빅카인즈"
        print(f"📁 데이터 경로: {data_path}")
        
        analyzer = BigKindsPyTorchLSTMAnalyzer(data_path)
        
        # 전체 분석 실행
        print("\n🚀 자동 분석 시작...")
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\n" + "="*60)
            print("🎊 분석 성공적으로 완료!")
            print("="*60)
            
            # 결과 요약
            print("\n📋 생성된 결과물:")
            print("🖼️  시각화 파일:")
            for company in analyzer.lstm_models.keys():
                print(f"   - {company}_pytorch_lstm_analysis.png (종합 분석)")
                print(f"   - {company}_future_prediction.png (미래 예측)")
            
            print("\n📄 리포트 파일:")
            print("   - PyTorch_LSTM_Sentiment_Analysis_Report.txt")
            
            print("\n📊 모델 성능 요약:")
            for company, model_info in analyzer.lstm_models.items():
                r2 = model_info['test_results']['r2']
                print(f"   - {company.upper()}: R² = {r2:.3f}")
            
            # 대화형 모드 제안
            interactive = input("\n🔍 대화형 분석 모드를 실행하시겠습니까? (y/n): ").strip().lower()
            if interactive == 'y':
                analyzer.interactive_analysis()
            
            print("\n📚 추가 사용 가능한 함수들:")
            print("analyzer.predict_future_sentiment('samsung', days=60)")
            print("analyzer.visualize_results('apple')")
            print("analyzer.sentiment_data.head()")
            print("analyzer.generate_comprehensive_report()")
            
        else:
            print("\n❌ 분석 실패")
            print("다음 사항을 확인해주세요:")
            print("1. 데이터 파일 경로가 올바른지 확인")
            print("2. 엑셀 파일에 필요한 컬럼이 있는지 확인")
            print("3. 인터넷 연결 상태 확인 (FinBERT 모델 다운로드)")
            print("4. 메모리 부족 시 배치 사이즈 줄이기")
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n💥 예기치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 문제 해결 방법:")
        print("1. 필요한 라이브러리가 모두 설치되었는지 확인")
        print("2. Python 버전 호환성 확인")
        print("3. 데이터 파일 형식 확인")


"""
🔧 설치 필요 라이브러리:
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn tqdm openpyxl

📊 프로젝트 주요 특징:
1. PyTorch 기반 Bidirectional LSTM 모델
2. KR-FinBERT-SC를 활용한 한국어 금융 뉴스 감성 분석
3. 자동화된 데이터 전처리 및 기업 분류
4. 종합적인 시각화 (9개 서브플롯)
5. 미래 감성 트렌드 예측
6. 상세한 성능 분석 리포트
7. 대화형 분석 모드

🎯 프로젝트 목표 달성:
✅ 뉴스 데이터 기반 감성 분석
✅ LSTM 모델을 활용한 시계열 예측
✅ 삼성전자/애플 비교 분석
✅ 감성 트렌드 시각화
✅ 자동화된 리포트 생성

💡 개선 및 확장 가능 사항:
- 실시간 뉴스 API 연동
- 주가 데이터와의 상관관계 분석
- 웹 대시보드 구축 (Streamlit/Dash)
- 모델 앙상블 및 하이퍼파라미터 최적화
- 설명 가능한 AI 기법 적용
"""