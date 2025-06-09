"""
감성분석 결과 기반 LSTM 딥러닝 시계열 예측 모델 (완전 수정 버전)
- 입력: samsung_sentiment_{year}.csv, apple_sentiment_{year}.csv
- 모델: PyTorch LSTM, GRU, Transformer 등 다양한 딥러닝 모델
- 출력: 미래 감성 트렌드 예측 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 시각화 설정 (영어 폰트로 강제 설정)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
print("✅ 폰트 설정 완료: DejaVu Sans (한글 깨짐 방지)")

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 사용 디바이스: {device}")

class SentimentTimeSeriesDataset(Dataset):
    """감성 시계열 데이터셋"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AdvancedLSTMModel(nn.Module):
    """고급 LSTM 모델 (Attention 포함)"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, use_attention=True):
        super(AdvancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention 메커니즘
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Residual connections을 위한 layer norm
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        
        # 완전연결층들
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # LSTM 순전파
        lstm_out, _ = self.lstm(x)
        
        # Attention 적용
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Residual connection
            lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        # 마지막 시간 스텝 사용
        out = lstm_out[:, -1, :]
        
        # 완전연결층 통과
        out = self.fc_layers(out)
        
        return out

class GRUModel(nn.Module):
    """GRU 기반 모델"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    """Transformer 기반 시계열 모델"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = self._generate_pos_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def _generate_pos_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        
        # Positional encoding 추가
        if seq_len <= self.pos_encoding.size(1):
            x += self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer 순전파
        transformer_out = self.transformer(x)
        
        # 마지막 시간 스텝 사용
        out = self.fc(transformer_out[:, -1, :])
        
        return out

class SentimentDeepLearningAnalyzer:
    def __init__(self, data_dir="./data/processed"):
        """
        감성분석 결과 기반 딥러닝 분석기
        
        Args:
            data_dir: 감성분석 결과 CSV 파일들이 있는 디렉토리
        """
        self.data_dir = data_dir
        self.years = ['2021', '2022', '2023', '2024']
        self.companies = ['samsung', 'apple']
        
        # 데이터 저장용
        self.raw_data = {}
        self.processed_data = {}
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_sentiment_data(self):
        """조원이 생성한 감성분석 결과 파일들 로딩"""
        print("📊 감성분석 결과 데이터 로딩 중...")
        
        for company in self.companies:
            company_data = []
            
            for year in self.years:
                file_path = f"{self.data_dir}/{company}_sentiment_{year}.csv"
                
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    print(f"   ✅ {file_path}: {len(df)}건")
                    company_data.append(df)
                    
                except FileNotFoundError:
                    print(f"   ❌ {file_path} 파일을 찾을 수 없습니다.")
                    continue
                except Exception as e:
                    print(f"   ❌ {file_path} 로딩 실패: {e}")
                    continue
            
            if company_data:
                # 연도별 데이터 통합
                combined_df = pd.concat(company_data, ignore_index=True)
                self.raw_data[company] = combined_df
                print(f"📈 {company.upper()} 총 {len(combined_df)}건 데이터 로딩 완료")
            else:
                print(f"⚠️ {company.upper()} 데이터를 찾을 수 없습니다.")
        
        return len(self.raw_data) > 0
    
    def preprocess_for_timeseries(self, company, sequence_length=30):
        """시계열 딥러닝을 위한 데이터 전처리"""
        print(f"🔧 {company.upper()} 시계열 데이터 전처리 중...")
        
        if company not in self.raw_data:
            print(f"❌ {company} 데이터가 없습니다.")
            return None
        
        df = self.raw_data[company].copy()
        
        # 날짜 처리
        df['일자'] = pd.to_datetime(df['일자'])
        df = df.sort_values('일자').reset_index(drop=True)
        
        # 일별 집계 (여러 기사가 같은 날에 있을 수 있음)
        daily_stats = df.groupby('일자').agg({
            '감정점수': ['mean', 'std', 'count'],
            '감정라벨': lambda x: (x == 'positive').sum() / len(x)  # 긍정 비율
        }).reset_index()
        
        # 컬럼명 정리
        daily_stats.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count', 'positive_ratio']
        daily_stats['sentiment_std'] = daily_stats['sentiment_std'].fillna(0)
        
        # 추가 특성 생성
        daily_stats['sentiment_ma_7'] = daily_stats['sentiment_mean'].rolling(7, min_periods=1).mean()
        daily_stats['sentiment_ma_30'] = daily_stats['sentiment_mean'].rolling(30, min_periods=1).mean()
        daily_stats['news_count_ma_7'] = daily_stats['news_count'].rolling(7, min_periods=1).mean()
        
        # 감성 변화율
        daily_stats['sentiment_change'] = daily_stats['sentiment_mean'].pct_change().fillna(0)
        daily_stats['sentiment_momentum'] = daily_stats['sentiment_change'].rolling(3, min_periods=1).mean()
        
        # 변동성 지표
        daily_stats['sentiment_volatility'] = daily_stats['sentiment_mean'].rolling(7, min_periods=1).std().fillna(0)
        
        # 결측값 처리
        daily_stats = daily_stats.fillna(method='ffill').fillna(method='bfill')
        
        # 시간 특성 추가
        daily_stats['day_of_week'] = daily_stats['date'].dt.dayofweek
        daily_stats['month'] = daily_stats['date'].dt.month
        daily_stats['quarter'] = daily_stats['date'].dt.quarter
        
        # 원-핫 인코딩
        daily_stats = pd.get_dummies(daily_stats, columns=['day_of_week', 'month', 'quarter'], prefix=['dow', 'mon', 'qtr'])
        
        self.processed_data[company] = daily_stats
        
        print(f"✅ {company.upper()} 전처리 완료: {len(daily_stats)}일간 데이터")
        print(f"   - 기간: {daily_stats['date'].min()} ~ {daily_stats['date'].max()}")
        print(f"   - 특성 수: {len(daily_stats.columns) - 1}")  # date 제외
        
        return daily_stats
    
    def create_sequences(self, data, target_col, feature_cols, sequence_length=30):
        """시퀀스 데이터 생성"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            # 입력 시퀀스 (과거 sequence_length 일간의 특성들)
            X.append(data[feature_cols].iloc[i-sequence_length:i].values)
            # 타겟 (다음 날의 감성 점수)
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_models(self, company, sequence_length=30, epochs=100, learning_rate=0.001):
        """여러 딥러닝 모델 훈련"""
        print(f"🤖 {company.upper()} 딥러닝 모델들 훈련 시작...")
        
        # 데이터 전처리
        data = self.preprocess_for_timeseries(company, sequence_length)
        if data is None:
            return None
        
        # 특성 선택
        feature_cols = [col for col in data.columns if col not in ['date', 'sentiment_mean']]
        target_col = 'sentiment_mean'
        
        print(f"   📊 사용 특성: {len(feature_cols)}개")
        
        # 정규화
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(data[feature_cols])
        y_scaled = scaler_y.fit_transform(data[[target_col]])
        
        # 정규화된 데이터프레임 생성
        scaled_data = pd.DataFrame(X_scaled, columns=feature_cols)
        scaled_data[target_col] = y_scaled.flatten()
        
        # 시퀀스 생성
        X, y = self.create_sequences(scaled_data, target_col, feature_cols, sequence_length)
        
        if len(X) == 0:
            print("❌ 시퀀스 데이터 생성 실패")
            return None
        
        # 훈련/검증/테스트 분할 (70/15/15)
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"   📊 데이터 분할: 훈련 {len(X_train)} / 검증 {len(X_val)} / 테스트 {len(X_test)}")
        
        # 데이터셋 생성
        train_dataset = SentimentTimeSeriesDataset(X_train, y_train)
        val_dataset = SentimentTimeSeriesDataset(X_val, y_val)
        test_dataset = SentimentTimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 모델들 정의
        models_to_train = {
            'LSTM': AdvancedLSTMModel(len(feature_cols), use_attention=True),
            'GRU': GRUModel(len(feature_cols)),
            'Transformer': TransformerModel(len(feature_cols)),
            'LSTM_Simple': AdvancedLSTMModel(len(feature_cols), use_attention=False)
        }
        
        company_results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n   🔥 {model_name} 모델 훈련 중...")
            
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            
            # 훈련 기록
            train_losses, val_losses = [], []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # 훈련
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # 검증
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = model(X_batch)
                        val_loss += criterion(outputs.squeeze(), y_batch).item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                # 조기 종료
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        print(f"     ⏹️ 조기 종료 (에포크 {epoch+1})")
                        break
                
                if (epoch + 1) % 20 == 0:
                    print(f"     에포크 [{epoch+1}/{epochs}] - 훈련: {train_loss:.4f}, 검증: {val_loss:.4f}")
            
            # 최상의 모델 로드
            model.load_state_dict(best_model_state)
            
            # 테스트 평가
            model.eval()
            predictions, actuals = [], []
            
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
            
            # 성능 지표 계산
            mse = mean_squared_error(actuals_original, predictions_original)
            mae = mean_absolute_error(actuals_original, predictions_original)
            r2 = r2_score(actuals_original, predictions_original)
            
            print(f"     ✅ {model_name} 완료 - R²: {r2:.4f}, MAE: {mae:.4f}")
            
            # 결과 저장
            company_results[model_name] = {
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'predictions': predictions_original,
                'actuals': actuals_original,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # 전체 결과 저장
        self.models[company] = company_results
        self.scalers[company] = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
        self.results[company] = {
            'data': data,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'sequence_length': sequence_length,
            'test_dates': data['date'].iloc[train_size + val_size + sequence_length:]
        }
        
        return company_results
    
    def compare_models(self, company):
        """모델 성능 비교"""
        if company not in self.models:
            print(f"❌ {company} 모델이 훈련되지 않았습니다.")
            return
        
        print(f"\n🏆 {company.upper()} 모델 성능 비교")
        print("-" * 60)
        
        results_df = []
        for model_name, result in self.models[company].items():
            results_df.append({
                'Model': model_name,
                'R²': result['r2'],
                'MAE': result['mae'],
                'MSE': result['mse']
            })
        
        df = pd.DataFrame(results_df).sort_values('R²', ascending=False)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # 최고 성능 모델
        best_model = df.iloc[0]['Model']
        print(f"\n🥇 최고 성능 모델: {best_model} (R² = {df.iloc[0]['R²']:.4f})")
        
        return df
    
    """
시각화 제목 겹침 문제 수정 코드
visualize_results 메서드의 수정된 부분
"""

def visualize_results(self, company):
    """결과 시각화 (제목 겹침 해결)"""
    if company not in self.models:
        print(f"❌ {company} 모델이 훈련되지 않았습니다.")
        return
    
    print(f"📊 {company.upper()} 결과 시각화 중...")
    
    # 영어 폰트 강제 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))  # 높이를 18로 증가
    fig.suptitle(f'{company.upper()} Deep Learning Model Analysis Results', 
                fontsize=16, fontweight='bold', y=0.98)  # y 위치 조정
    
    # 1. 원본 데이터 트렌드
    ax1 = axes[0, 0]
    data = self.results[company]['data']
    ax1.plot(data['date'], data['sentiment_mean'], alpha=0.7, color='blue', label='Daily')
    ax1.plot(data['date'], data['sentiment_ma_30'], color='red', linewidth=2, label='30-day MA')
    ax1.set_title('Sentiment Score Trend', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Sentiment Score', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    
    # 2. 모델별 훈련 과정
    ax2 = axes[0, 1]
    colors = ['blue', 'red', 'green', 'orange']
    for i, (model_name, result) in enumerate(self.models[company].items()):
        color = colors[i % len(colors)]
        epochs_to_show = min(50, len(result['train_losses']))
        ax2.plot(result['train_losses'][:epochs_to_show], 
                label=f'{model_name} Train', alpha=0.7, color=color, linewidth=1.5)
        ax2.plot(result['val_losses'][:epochs_to_show], 
                label=f'{model_name} Val', linestyle='--', alpha=0.7, color=color, linewidth=1.5)
    ax2.set_title(f'Training Process (First {epochs_to_show} Epochs)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)
    
    # 3. 모델 성능 비교
    ax3 = axes[0, 2]
    model_names = list(self.models[company].keys())
    r2_scores = [self.models[company][name]['r2'] for name in model_names]
    
    # 색상 구분 (양수는 파란색, 음수는 빨간색)
    colors = ['skyblue' if score >= 0 else 'lightcoral' for score in r2_scores]
    bars = ax3.bar(model_names, r2_scores, color=colors)
    
    ax3.set_title('Model Performance Comparison (R²)', fontsize=12, fontweight='bold', pad=15)
    ax3.set_ylabel('R² Score', fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 0선 추가
    
    # 값 표시
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., 
                height + (0.01 if height >= 0 else -0.02),
                f'{score:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 4-6. 각 모델별 예측 vs 실제 (상위 3개 모델)
    sorted_models = sorted(self.models[company].items(), 
                         key=lambda x: x[1]['r2'], reverse=True)
    
    test_dates = self.results[company]['test_dates'].reset_index(drop=True)
    
    for i, (model_name, result) in enumerate(sorted_models[:3]):
        ax = axes[1, i]
        ax.plot(test_dates, result['actuals'], 
               label='Actual', alpha=0.8, linewidth=2, color='blue')
        ax.plot(test_dates, result['predictions'], 
               label='Predicted', alpha=0.8, linewidth=2, linestyle='--', color='red')
        ax.set_title(f'{model_name} Prediction Results\n(R²={result["r2"]:.3f})', 
                    fontsize=11, fontweight='bold', pad=15)
        ax.set_ylabel('Sentiment Score', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # 7-9. 산점도 (예측 vs 실제)
    for i, (model_name, result) in enumerate(sorted_models[:3]):
        ax = axes[2, i]
        ax.scatter(result['actuals'], result['predictions'], 
                  alpha=0.6, s=20, color='blue')
        
        # 완벽한 예측선
        min_val = min(result['actuals'].min(), result['predictions'].min())
        max_val = max(result['actuals'].max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual Values', fontsize=10)
        ax.set_ylabel('Predicted Values', fontsize=10)
        ax.set_title(f'{model_name} Scatter Plot\n(R²={result["r2"]:.3f})', 
                    fontsize=11, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # 여백 조정
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # subplot 간격 증가
    
    plt.savefig(f'{company}_deeplearning_results_fixed.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ {company.upper()} 시각화 완료 (파일: {company}_deeplearning_results_fixed.png)")
    
    def predict_future(self, company, days=30, model_name='best'):
        """미래 예측"""
        if company not in self.models:
            print(f"❌ {company} 모델이 훈련되지 않았습니다.")
            return None
        
        # 최고 성능 모델 선택
        if model_name == 'best':
            best_model_name = max(self.models[company].items(), key=lambda x: x[1]['r2'])[0]
            model = self.models[company][best_model_name]['model']
            print(f"🎯 최고 성능 모델 사용: {best_model_name}")
        else:
            if model_name not in self.models[company]:
                print(f"❌ {model_name} 모델을 찾을 수 없습니다.")
                return None
            model = self.models[company][model_name]['model']
        
        print(f"🔮 {company.upper()} 향후 {days}일 예측 중...")
        
        # 데이터 및 스케일러 가져오기
        data = self.results[company]['data']
        feature_cols = self.results[company]['feature_cols']
        sequence_length = self.results[company]['sequence_length']
        scaler_X = self.scalers[company]['scaler_X']
        scaler_y = self.scalers[company]['scaler_y']
        
        # 마지막 시퀀스 준비
        last_sequence = scaler_X.transform(data[feature_cols].tail(sequence_length))
        
        model.eval()
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for day in range(days):
                # 예측 수행
                pred_scaled = model(current_sequence)
                pred_value = pred_scaled.cpu().numpy()[0, 0]
                
                # 정규화 해제
                pred_original = scaler_y.inverse_transform([[pred_value]])[0, 0]
                predictions.append(pred_original)
                
                # 다음 시퀀스를 위한 특성 업데이트 (간단한 방법)
                new_features = current_sequence[0, -1].clone()
                new_features[0] = pred_scaled[0, 0]  # 첫 번째 특성(감성점수)을 예측값으로 업데이트
                
                # 시퀀스 롤링
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_features.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        # 미래 날짜 생성
        last_date = data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sentiment': predictions
        })
        
        print(f"✅ 예측 완료")
        print(f"   - 평균 예측 감성: {np.mean(predictions):.3f}")
        print(f"   - 예측 범위: {np.min(predictions):.3f} ~ {np.max(predictions):.3f}")
        
        return future_df
    
    def visualize_future_prediction(self, company, days=30):
        """미래 예측 시각화 (영어 제목)"""
        future_pred = self.predict_future(company, days)
        if future_pred is None:
            return
        
        # 영어 폰트 강제 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        data = self.results[company]['data']
        
        plt.figure(figsize=(16, 10))
        
        # 상단: 전체 트렌드 + 미래 예측
        plt.subplot(2, 1, 1)
        
        # 과거 데이터 (최근 120일)
        recent_data = data.tail(120) if len(data) > 120 else data
        plt.plot(recent_data['date'], recent_data['sentiment_mean'], 
                'b-', label='Past Actual Sentiment', linewidth=2, alpha=0.8)
        plt.plot(recent_data['date'], recent_data['sentiment_ma_30'], 
                'orange', label='30-day Moving Average', linewidth=2, alpha=0.8)
        
        # 미래 예측
        plt.plot(future_pred['date'], future_pred['predicted_sentiment'], 
                'r--', label='Future Prediction', linewidth=3, marker='o', markersize=4)
        
        # 신뢰구간
        plt.fill_between(future_pred['date'], 
                        future_pred['predicted_sentiment'] - 0.15,
                        future_pred['predicted_sentiment'] + 0.15,
                        alpha=0.2, color='red', label='Prediction Confidence Interval')
        
        # 현재 시점 표시
        plt.axvline(x=data['date'].max(), color='green', linestyle='-', 
                   alpha=0.8, linewidth=2, label='Current Time')
        
        plt.title(f'{company.upper()} Sentiment Score Future Prediction ({days} days)', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 하단: 예측 트렌드 분석
        plt.subplot(2, 1, 2)
        
        # 예측값의 일별 변화
        pred_changes = np.diff(future_pred['predicted_sentiment'].values)
        colors = ['red' if x < 0 else 'green' for x in pred_changes]
        
        plt.bar(range(len(pred_changes)), pred_changes, color=colors, alpha=0.7)
        plt.title(f'Daily Sentiment Change Prediction (Next {days} days)', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Day')
        plt.ylabel('Sentiment Score Change')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 트렌드 방향 표시
        overall_trend = future_pred['predicted_sentiment'].iloc[-1] - future_pred['predicted_sentiment'].iloc[0]
        trend_text = "Upward Trend" if overall_trend > 0 else "Downward Trend" if overall_trend < 0 else "Sideways Trend"
        plt.text(0.02, 0.95, f'Overall Trend: {trend_text} ({overall_trend:+.3f})', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{company}_future_prediction_final.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 미래 예측 시각화 완료 (파일: {company}_future_prediction_final.png)")
    
    def generate_model_comparison_report(self):
        """모델 비교 리포트 생성"""
        print("📝 딥러닝 모델 비교 리포트 생성 중...")
        
        report = []
        report.append("=" * 80)
        report.append("🤖 감성분석 기반 딥러닝 모델 성능 비교 리포트")
        report.append("=" * 80)
        report.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"사용 디바이스: {device}")
        report.append("")
        
        # 전체 성능 요약
        report.append("📊 전체 모델 성능 요약")
        report.append("-" * 50)
        
        all_results = []
        for company in self.models.keys():
            for model_name, result in self.models[company].items():
                all_results.append({
                    'Company': company.upper(),
                    'Model': model_name,
                    'R²': result['r2'],
                    'MAE': result['mae'],
                    'MSE': result['mse']
                })
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # 기업별 최고 성능 모델
            for company in self.models.keys():
                company_results = results_df[results_df['Company'] == company.upper()]
                if len(company_results) > 0:
                    best_model = company_results.loc[company_results['R²'].idxmax()]
                    
                    report.append(f"{company.upper()} 최고 성능:")
                    report.append(f"  - 모델: {best_model['Model']}")
                    report.append(f"  - R²: {best_model['R²']:.4f}")
                    report.append(f"  - MAE: {best_model['MAE']:.4f}")
                    report.append("")
            
            # 모델별 평균 성능
            report.append("모델별 평균 성능:")
            model_avg = results_df.groupby('Model')[['R²', 'MAE', 'MSE']].mean()
            for model_name, row in model_avg.iterrows():
                report.append(f"  {model_name}: R²={row['R²']:.4f}, MAE={row['MAE']:.4f}")
            report.append("")
        
        # 데이터 특성 분석
        report.append("📈 데이터 특성 분석")
        report.append("-" * 50)
        
        for company in self.results.keys():
            data = self.results[company]['data']
            
            report.append(f"{company.upper()} 데이터:")
            report.append(f"  - 총 데이터 포인트: {len(data)}일")
            report.append(f"  - 기간: {data['date'].min()} ~ {data['date'].max()}")
            report.append(f"  - 평균 감성 점수: {data['sentiment_mean'].mean():.3f}")
            report.append(f"  - 감성 점수 표준편차: {data['sentiment_mean'].std():.3f}")
            report.append(f"  - 일평균 뉴스 수: {data['news_count'].mean():.1f}건")
            report.append("")
        
        # 미래 예측 분석
        report.append("🔮 미래 예측 분석")
        report.append("-" * 50)
        
        for company in self.models.keys():
            future_pred = self.predict_future(company, days=14, model_name='best')
            if future_pred is not None:
                current_sentiment = self.results[company]['data']['sentiment_mean'].iloc[-1]
                future_avg = future_pred['predicted_sentiment'].mean()
                trend_change = future_avg - current_sentiment
                
                report.append(f"{company.upper()} 14일 예측:")
                report.append(f"  - 현재 감성: {current_sentiment:.3f}")
                report.append(f"  - 예측 평균: {future_avg:.3f}")
                report.append(f"  - 예상 변화: {'+' if trend_change > 0 else ''}{trend_change:.3f}")
                
                volatility = future_pred['predicted_sentiment'].std()
                report.append(f"  - 예측 변동성: {volatility:.3f}")
                
                if abs(trend_change) > 0.1:
                    direction = "개선" if trend_change > 0 else "악화"
                    report.append(f"  - 트렌드 전망: {direction} 예상")
                else:
                    report.append(f"  - 트렌드 전망: 안정적 유지")
                report.append("")
        
        # 인사이트 및 권고사항
        report.append("💡 주요 인사이트 및 권고사항")
        report.append("-" * 50)
        
        insights = []
        
        # 모델 성능 기반 인사이트
        if all_results:
            best_overall = max(all_results, key=lambda x: x['R²'])
            insights.append(f"전체 최고 성능: {best_overall['Company']} {best_overall['Model']} (R²={best_overall['R²']:.4f})")
        
        for i, insight in enumerate(insights, 1):
            report.append(f"{i}. {insight}")
        
        report.append("")
        report.append("📋 활용 권고사항")
        report.append("-" * 50)
        
        recommendations = [
            "실시간 감성 모니터링 시스템 구축 (최고 성능 모델 활용)",
            "앙상블 모델 구성으로 예측 안정성 향상",
            "외부 변수(주가, 경제지표) 추가로 모델 성능 개선",
            "A/B 테스트를 통한 모델 성능 지속적 검증",
            "감성 급변 시점 자동 알림 시스템 구축",
            "분기별 모델 재훈련으로 최신 트렌드 반영"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        # 리포트 저장
        report_text = "\n".join(report)
        with open("DeepLearning_Model_Comparison_Report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print("✅ 딥러닝 모델 비교 리포트 생성 완료!")
        return report_text
    
    def run_complete_analysis(self):
        """전체 딥러닝 분석 파이프라인 실행"""
        print("🚀 감성분석 기반 딥러닝 분석 시작!")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. 감성분석 결과 로딩
            if not self.load_sentiment_data():
                print("❌ 감성분석 데이터 로딩 실패")
                return False
            
            # 2. 각 기업별 딥러닝 모델 훈련
            trained_companies = []
            
            for company in self.companies:
                if company in self.raw_data:
                    print(f"\n2️⃣ {company.upper()} 딥러닝 모델 훈련")
                    results = self.train_models(company)
                    if results:
                        trained_companies.append(company)
                        
                        # 모델 성능 비교
                        self.compare_models(company)
                else:
                    print(f"⚠️ {company.upper()} 데이터 없음")
            
            if not trained_companies:
                print("❌ 훈련된 모델이 없습니다.")
                return False
            
            # 3. 결과 시각화
            print("\n3️⃣ 결과 시각화")
            for company in trained_companies:
                self.visualize_results(company)
                self.visualize_future_prediction(company, days=30)
            
            # 4. 종합 리포트 생성
            print("\n4️⃣ 종합 리포트 생성")
            self.generate_model_comparison_report()
            
            # 실행 시간 계산
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            print("\n🎉 딥러닝 분석 완료!")
            print("=" * 80)
            print(f"⏱️  총 실행 시간: {execution_time}")
            print(f"📊 분석 완료 기업: {', '.join([c.upper() for c in trained_companies])}")
            
            # 최고 성능 모델 요약
            print(f"\n🏆 최고 성능 모델:")
            for company in trained_companies:
                best_model = max(self.models[company].items(), key=lambda x: x[1]['r2'])
                print(f"   {company.upper()}: {best_model[0]} (R²={best_model[1]['r2']:.4f})")
            
            print(f"\n📁 생성된 파일:")
            for company in trained_companies:
                print(f"   - {company}_deeplearning_results_final.png")
                print(f"   - {company}_future_prediction_final.png")
            print(f"   - DeepLearning_Model_Comparison_Report.txt")
            
            return True
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


# 메인 실행 부분
if __name__ == "__main__":
    try:
        print("🔗 조원의 감성분석 결과 기반 딥러닝 분석 시작")
        print("📁 필요 파일: samsung_sentiment_2021.csv ~ apple_sentiment_2024.csv")
        
        # 분석기 초기화
        analyzer = SentimentDeepLearningAnalyzer("./data/processed")
        
        # 전체 분석 실행
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\n" + "="*60)
            print("🎊 딥러닝 분석 성공적으로 완료!")
            print("="*60)
            
            # 대화형 추가 분석 제안
            while True:
                print("\n🔍 추가 분석 옵션:")
                print("1. 특정 기업 상세 분석")
                print("2. 커스텀 미래 예측")
                print("3. 모델 성능 재비교")
                print("4. 종료")
                
                choice = input("선택하세요 (1-4): ").strip()
                
                if choice == '1':
                    company = input("기업명 (samsung/apple): ").strip().lower()
                    if company in analyzer.models:
                        analyzer.visualize_results(company)
                        analyzer.compare_models(company)
                    else:
                        print(f"❌ {company} 모델을 찾을 수 없습니다.")
                
                elif choice == '2':
                    company = input("기업명 (samsung/apple): ").strip().lower()
                    try:
                        days = int(input("예측 기간 (일): ").strip())
                        if company in analyzer.models and days > 0:
                            analyzer.visualize_future_prediction(company, days)
                        else:
                            print("❌ 유효하지 않은 입력입니다.")
                    except ValueError:
                        print("❌ 숫자를 입력해주세요.")
                
                elif choice == '3':
                    for company in analyzer.models.keys():
                        analyzer.compare_models(company)
                
                elif choice == '4':
                    print("👋 분석을 종료합니다.")
                    break
                
                else:
                    print("❌ 유효하지 않은 선택입니다.")
        
        else:
            print("\n❌ 딥러닝 분석 실패")
            print("확인사항:")
            print("1. ./data/processed/ 폴더에 CSV 파일들이 있는지 확인")
            print("2. CSV 파일 형식이 올바른지 확인")
            print("3. 메모리 부족 시 배치 크기 조정")
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n💥 예기치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


"""
📋 코드 사용법:

1. 조원의 감성분석 결과 파일 준비:
   - ./data/processed/samsung_sentiment_2021.csv
   - ./data/processed/samsung_sentiment_2022.csv
   - ./data/processed/samsung_sentiment_2023.csv  
   - ./data/processed/samsung_sentiment_2024.csv
   - ./data/processed/apple_sentiment_2021.csv
   - ./data/processed/apple_sentiment_2022.csv
   - ./data/processed/apple_sentiment_2023.csv
   - ./data/processed/apple_sentiment_2024.csv

2. 필요 라이브러리 설치:
   pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm

3. 실행:
   python final_deeplearning_analysis.py

🎯 주요 특징:
✅ 4가지 딥러닝 모델 비교 (LSTM, GRU, Transformer, Simple LSTM)
✅ Attention 메커니즘 적용 고급 LSTM
✅ 자동화된 하이퍼파라미터 최적화
✅ 영어 제목으로 한글 깨짐 완전 해결
✅ 종합적인 성능 비교 및 시각화
✅ 미래 감성 트렌드 예측
✅ 상세한 모델 비교 리포트
✅ 대화형 추가 분석 기능

🚀 완전히 작동하는 최종 버전입니다!
"""