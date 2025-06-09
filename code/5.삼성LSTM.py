import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("🚀 삼성전자 감성 분석 고도화 LSTM 모델")
print("=" * 60)

class SamsungDataLoader:
    """삼성 데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.data = None
        
    def load_all_years(self):
        """2021-2024년 모든 데이터 로딩 (UTF-8 인코딩 사용)"""
        all_data = []
        
        for year in range(2021, 2025):
            try:
                file_path = f"{self.base_path}/samsung_sentiment_{year}.csv"
                df = pd.read_csv(file_path, encoding='utf-8')  # UTF-8로 고정
                df['year'] = year
                all_data.append(df)
                print(f"✅ {year}년 데이터 로딩 완료: {len(df)}개 기사")
            except FileNotFoundError:
                print(f"❌ {year}년 데이터 파일을 찾을 수 없습니다: {file_path}")
            except Exception as e:
                print(f"❌ {year}년 데이터 로딩 오류: {e}")
                
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 전체 데이터 통합 완료: {len(self.data)}개 기사")
            print(f"📋 첫 5행 데이터 확인:")
            print(self.data.head())
            return self.data
        else:
            raise Exception("❌ 로딩된 데이터가 없습니다.")
            
    def preprocess_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        if self.data is None:
            raise Exception("먼저 데이터를 로딩해주세요.")
        
        # 데이터 컬럼 확인
        print(f"📋 데이터 컬럼 목록: {list(self.data.columns)}")
        print(f"📊 데이터 형태: {self.data.shape}")
        
        # 실제 데이터 구조에 맞춘 컬럼 매핑
        date_col = '일자'
        sentiment_col = '감정점수'
        title_col = '제목'
        keyword_col = '키워드'
        label_col = '감정라벨'
        
        print(f"✅ 날짜 컬럼: '{date_col}'")
        print(f"✅ 감성 점수 컬럼: '{sentiment_col}'")
        print(f"✅ 제목 컬럼: '{title_col}'")
        print(f"✅ 키워드 컬럼: '{keyword_col}'")
        print(f"✅ 감정라벨 컬럼: '{label_col}'")
            
        # 날짜 파싱
        self.data['date'] = pd.to_datetime(self.data[date_col], errors='coerce')
        self.data = self.data.dropna(subset=['date'])
        
        # 감성 점수 처리
        self.data['sentiment_score'] = pd.to_numeric(self.data[sentiment_col], errors='coerce')
        self.data = self.data.dropna(subset=['sentiment_score'])
        
        # 감정라벨을 숫자로 변환 (추가 특성으로 활용)
        label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.data['sentiment_label_numeric'] = self.data[label_col].map(label_mapping)
        
        # 제목 길이 특성 추가
        self.data['title_length'] = self.data[title_col].str.len()
        
        # 키워드 개수 특성 추가
        self.data['keyword_count'] = self.data[keyword_col].str.count(',') + 1
            
        self.data['sentiment_score'] = pd.to_numeric(self.data[sentiment_col], errors='coerce')
        self.data = self.data.dropna(subset=['sentiment_score'])
        
        # 감정라벨을 숫자로 변환 (추가 특성으로 활용)
        label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.data['sentiment_label_numeric'] = self.data[label_col].map(label_mapping)
        
        # 제목 길이 특성 추가
        self.data['title_length'] = self.data[title_col].str.len()
        
        # 키워드 개수 특성 추가
        self.data['keyword_count'] = self.data[keyword_col].str.count(',') + 1
        
        # 일별 집계 및 특성 엔지니어링
        daily_data = self.data.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
            'sentiment_label_numeric': ['mean', 'sum'],  # 긍정/부정 비율
            'title_length': 'mean',
            'keyword_count': 'mean'
        }).round(4)
        
        daily_data.columns = [
            'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_min', 'sentiment_max',
            'label_ratio', 'positive_news_count', 'avg_title_length', 'avg_keyword_count'
        ]
        daily_data = daily_data.reset_index()
        
        # 결측값 처리
        daily_data['sentiment_std'] = daily_data['sentiment_std'].fillna(0)
        
        # 추가 특성 생성
        daily_data['sentiment_range'] = daily_data['sentiment_max'] - daily_data['sentiment_min']
        daily_data['sentiment_momentum_3'] = daily_data['sentiment_mean'].rolling(3).mean()
        daily_data['sentiment_momentum_7'] = daily_data['sentiment_mean'].rolling(7).mean()
        daily_data['volatility_7'] = daily_data['sentiment_mean'].rolling(7).std()
        daily_data['news_volume_ma_7'] = daily_data['news_count'].rolling(7).mean()
        
        # 새로운 특성들
        daily_data['positive_ratio'] = daily_data['positive_news_count'] / daily_data['news_count']
        daily_data['sentiment_velocity'] = daily_data['sentiment_mean'].diff()  # 감성 변화율
        daily_data['news_surge'] = (daily_data['news_count'] > daily_data['news_volume_ma_7'] * 1.5).astype(int)
        
        # 요일 및 월 특성
        daily_data['weekday'] = daily_data['date'].dt.dayofweek
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['is_weekend'] = (daily_data['weekday'] >= 5).astype(int)
        
        # 정렬 및 결측값 제거
        daily_data = daily_data.sort_values('date')
        daily_data = daily_data.dropna()
        
        print(f"📈 전처리 완료: {len(daily_data)}일 데이터")
        print(f"📅 기간: {daily_data['date'].min()} ~ {daily_data['date'].max()}")
        print(f"📊 감성 점수 범위: {daily_data['sentiment_mean'].min():.3f} ~ {daily_data['sentiment_mean'].max():.3f}")
        print(f"📰 일평균 뉴스 수: {daily_data['news_count'].mean():.1f}개")
        
        return daily_data

class AdvancedLSTMDataset(Dataset):
    """고도화된 LSTM 데이터셋"""
    
    def __init__(self, data, sequence_length=30, prediction_length=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # 특성 선택 (새로운 특성들 포함)
        self.feature_columns = [
            'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_range',
            'sentiment_momentum_3', 'sentiment_momentum_7', 'volatility_7',
            'news_volume_ma_7', 'positive_ratio', 'sentiment_velocity', 'news_surge',
            'avg_title_length', 'avg_keyword_count', 'weekday', 'month', 'is_weekend'
        ]
        
        self.target_column = 'sentiment_mean'
        
        # 스케일링
        self.scaler_features = RobustScaler()
        self.scaler_target = RobustScaler()
        
        # 특성과 타겟 분리
        self.features = self.scaler_features.fit_transform(data[self.feature_columns])
        self.targets = self.scaler_target.fit_transform(data[[self.target_column]])
        
        self.sequences, self.labels = self._create_sequences()
        
    def _create_sequences(self):
        sequences = []
        labels = []
        
        for i in range(len(self.features) - self.sequence_length - self.prediction_length + 1):
            seq = self.features[i:i + self.sequence_length]
            label = self.targets[i + self.sequence_length:i + self.sequence_length + self.prediction_length]
            sequences.append(seq)
            labels.append(label)
            
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.labels[idx])

class AdvancedLSTMModel(nn.Module):
    """고도화된 LSTM 모델"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2, 
                 prediction_length=1, use_attention=True):
        super(AdvancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 어텐션 메커니즘
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,  # bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # 완전연결층
        fc_input_size = hidden_size * 2 if not use_attention else hidden_size * 2
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_length)
        )
        
        # 배치 정규화
        self.batch_norm = nn.BatchNorm1d(fc_input_size)
        
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # 셀프 어텐션
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # 마지막 타임스텝
            output = attn_output[:, -1, :]
        else:
            # 마지막 타임스텝
            output = lstm_out[:, -1, :]
        
        # 배치 정규화
        output = self.batch_norm(output)
        
        # 완전연결층
        output = self.fc_layers(output)
        
        return output

class ModelTrainer:
    """모델 훈련 및 평가 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001, 
                   patience=15, min_delta=1e-6):
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🎯 모델 훈련 시작 - Device: {self.device}")
        
        for epoch in range(epochs):
            # 훈련
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.squeeze(-1))
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 검증
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y.squeeze(-1))
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 학습률 스케줄링
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # 학습률 변화 출력
            if old_lr != new_lr:
                print(f"   📉 학습률 감소: {old_lr:.6f} → {new_lr:.6f}")
            
            # 조기 종료
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), 'best_samsung_lstm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"⏹️ 조기 종료 - Epoch {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {new_lr:.6f}")
        
        print(f"✅ 훈련 완료 - 최고 검증 손실: {best_val_loss:.6f}")
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_samsung_lstm_model.pth'))
        
    def predict(self, test_loader, scaler_target):
        """예측 수행"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                predictions.extend(output.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        # 스케일 복원
        predictions = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        return predictions, actuals

def comprehensive_evaluation(y_true, y_pred, model_name="LSTM"):
    """종합적인 모델 평가"""
    
    # 기본 메트릭
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 방향성 정확도
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    direction_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 임계값 기반 정확도 (±0.1 내에서 정확)
    threshold_accuracy = np.mean(np.abs(y_true - y_pred) <= 0.1)
    
    results = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Direction_Accuracy': direction_accuracy,
        'MAPE': mape,
        'Threshold_Accuracy': threshold_accuracy
    }
    
    return results

def create_advanced_visualizations(data, predictions, actuals, test_dates, trainer):
    """고도화된 시각화"""
    
    # 전체 시각화 설정
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            '삼성전자 감성 점수 시계열 (전체 기간)',
            '예측 vs 실제 (테스트 기간)',
            '훈련/검증 손실 곡선',
            '예측 오차 분포',
            '월별 감성 트렌드',
            '뉴스 볼륨 vs 감성 점수'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # 1. 전체 시계열
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['sentiment_mean'],
                  mode='lines', name='전체 감성 점수', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 2. 예측 vs 실제
    fig.add_trace(
        go.Scatter(x=test_dates, y=actuals,
                  mode='lines+markers', name='실제값', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=test_dates, y=predictions,
                  mode='lines+markers', name='예측값', line=dict(color='blue')),
        row=1, col=2
    )
    
    # 3. 훈련 곡선
    epochs = range(1, len(trainer.train_losses) + 1)
    fig.add_trace(
        go.Scatter(x=list(epochs), y=trainer.train_losses,
                  mode='lines', name='훈련 손실', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=trainer.val_losses,
                  mode='lines', name='검증 손실', line=dict(color='red')),
        row=2, col=1
    )
    
    # 4. 오차 분포
    errors = actuals - predictions
    fig.add_trace(
        go.Histogram(x=errors, name='예측 오차', nbinsx=20),
        row=2, col=2
    )
    
    # 5. 월별 트렌드
    monthly_sentiment = data.groupby(data['date'].dt.month)['sentiment_mean'].mean()
    fig.add_trace(
        go.Bar(x=list(monthly_sentiment.index), y=monthly_sentiment.values,
               name='월별 평균 감성', marker_color='lightblue'),
        row=3, col=1
    )
    
    # 6. 뉴스 볼륨 vs 감성
    fig.add_trace(
        go.Scatter(x=data['news_count'], y=data['sentiment_mean'],
                  mode='markers', name='뉴스볼륨 vs 감성',
                  marker=dict(color='green', size=4)),
        row=3, col=2
    )
    
    # 레이아웃 업데이트
    fig.update_layout(
        height=1200,
        title_text="삼성전자 감성 분석 고도화 LSTM 모델 - 종합 분석 결과",
        title_x=0.5,
        showlegend=True
    )
    
    return fig

def predict_future(model, dataset, scaler_target, days=30):
    """미래 예측"""
    model.eval()
    
    # 마지막 시퀀스 가져오기
    last_sequence = dataset.features[-dataset.sequence_length:]
    last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(next(model.parameters()).device)
    
    predictions = []
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(days):
            # 예측
            pred = model(current_sequence)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            # 다음 입력을 위한 시퀀스 업데이트 (단순화된 버전)
            # 실제로는 더 정교한 방법이 필요할 수 있음
            new_features = current_sequence[0, -1, :].clone()
            new_features[0] = pred[0, 0]  # 감성 점수 업데이트
            
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                new_features.unsqueeze(0).unsqueeze(0)
            ], dim=1)
    
    # 스케일 복원
    predictions = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    return predictions

def main():
    """메인 실행 함수"""
    
    # 1. 데이터 로딩
    print("1️⃣ 데이터 로딩 중...")
    base_path = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model/data/processed"
    
    loader = SamsungDataLoader(base_path)
    raw_data = loader.load_all_years()
    processed_data = loader.preprocess_data()
    
    print(f"📊 처리된 데이터 정보:")
    print(f"   - 총 {len(processed_data)}일")
    print(f"   - 감성 점수 범위: {processed_data['sentiment_mean'].min():.3f} ~ {processed_data['sentiment_mean'].max():.3f}")
    print(f"   - 평균 일일 뉴스 수: {processed_data['news_count'].mean():.1f}개")
    
    # 2. 데이터셋 생성
    print("\n2️⃣ 데이터셋 생성 중...")
    sequence_length = 30  # 30일 시퀀스
    dataset = AdvancedLSTMDataset(processed_data, sequence_length=sequence_length)
    
    # 시계열 분할
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    # 데이터 로더
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   - 훈련 데이터: {len(train_dataset)}개")
    print(f"   - 검증 데이터: {len(val_dataset)}개")
    print(f"   - 테스트 데이터: {len(test_dataset)}개")
    
    # 3. 모델 생성
    print("\n3️⃣ 모델 생성 중...")
    input_size = len(dataset.feature_columns)
    model = AdvancedLSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        use_attention=True
    )
    
    print(f"   - 입력 특성 수: {input_size}")
    print(f"   - 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 모델 훈련
    print("\n4️⃣ 모델 훈련 중...")
    trainer = ModelTrainer(model)
    trainer.train_model(train_loader, val_loader, epochs=100, lr=0.001)
    
    # 5. 예측 및 평가
    print("\n5️⃣ 모델 평가 중...")
    predictions, actuals = trainer.predict(test_loader, dataset.scaler_target)
    
    # 테스트 날짜 계산
    test_start_idx = train_size + val_size + sequence_length
    test_dates = processed_data['date'].iloc[test_start_idx:test_start_idx + len(predictions)]
    
    # 평가 결과
    evaluation_results = comprehensive_evaluation(actuals, predictions, "Advanced LSTM")
    
    print("\n📊 모델 성능 평가 결과:")
    print("=" * 50)
    for key, value in evaluation_results.items():
        if key != 'Model':
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # 6. 시각화
    print("\n6️⃣ 결과 시각화 중...")
    
    # Plotly 상호작용 차트
    fig = create_advanced_visualizations(processed_data, predictions, actuals, test_dates, trainer)
    fig.write_html("samsung_advanced_lstm_analysis.html")
    print("   - 상호작용 차트 저장: samsung_advanced_lstm_analysis.html")
    
    # 정적 차트 (Matplotlib)
    plt.figure(figsize=(20, 12))
    
    # 서브플롯 1: 전체 시계열
    plt.subplot(3, 3, 1)
    plt.plot(processed_data['date'], processed_data['sentiment_mean'], alpha=0.7, linewidth=1)
    plt.title('Samsung Sentiment Score Time Series (2021-2024)')
    plt.xticks(rotation=45)
    
    # 서브플롯 2: 예측 vs 실제
    plt.subplot(3, 3, 2)
    plt.plot(test_dates, actuals, 'r-', label='Actual', linewidth=2)
    plt.plot(test_dates, predictions, 'b--', label='Predicted', linewidth=2)
    plt.title('Prediction vs Actual (Test Period)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 서브플롯 3: 훈련 곡선
    plt.subplot(3, 3, 3)
    plt.plot(trainer.train_losses, label='Training Loss', color='blue')
    plt.plot(trainer.val_losses, label='Validation Loss', color='red')
    plt.title('Training/Validation Loss Curve')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 서브플롯 4: 오차 분포
    plt.subplot(3, 3, 4)
    errors = actuals - predictions
    plt.hist(errors, bins=20, alpha=0.7, color='lightblue')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    # 서브플롯 5: 산점도 (실제 vs 예측)
    plt.subplot(3, 3, 5)
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Scatter Plot')
    
    # 서브플롯 6: 월별 평균 감성
    plt.subplot(3, 3, 6)
    monthly_sentiment = processed_data.groupby(processed_data['date'].dt.month)['sentiment_mean'].mean()
    plt.bar(monthly_sentiment.index, monthly_sentiment.values, color='lightgreen')
    plt.title('Monthly Average Sentiment Score')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    
    # 서브플롯 7: 뉴스 볼륨 vs 감성
    plt.subplot(3, 3, 7)
    plt.scatter(processed_data['news_count'], processed_data['sentiment_mean'], alpha=0.5)
    plt.xlabel('Daily News Count')
    plt.ylabel('Sentiment Score')
    plt.title('News Volume vs Sentiment Score')
    
    # 서브플롯 8: 연도별 감성 트렌드
    plt.subplot(3, 3, 8)
    yearly_sentiment = processed_data.groupby(processed_data['date'].dt.year)['sentiment_mean'].mean()
    plt.bar(yearly_sentiment.index, yearly_sentiment.values, color='orange')
    plt.title('Yearly Average Sentiment Score')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    
    # 서브플롯 9: 변동성 분석
    plt.subplot(3, 3, 9)
    plt.plot(processed_data['date'], processed_data['volatility_7'], color='purple', alpha=0.7)
    plt.title('7-Day Moving Volatility')
    plt.xticks(rotation=45)
    plt.ylabel('Volatility')
    
    plt.tight_layout()
    plt.savefig('samsung_advanced_lstm_static_analysis.png', dpi=300, bbox_inches='tight')
    print("   - 정적 차트 저장: samsung_advanced_lstm_static_analysis.png")
    
    # 7. 미래 예측
    print("\n7️⃣ 미래 30일 예측 중...")
    future_predictions = predict_future(model, dataset, dataset.scaler_target, days=30)
    
    # 미래 날짜 생성
    last_date = processed_data['date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
    
    # 미래 예측 시각화
    plt.figure(figsize=(15, 8))
    
    # 최근 60일과 미래 30일 시각화
    recent_data = processed_data.tail(60)
    
    plt.plot(recent_data['date'], recent_data['sentiment_mean'], 
             'b-', label='Actual Sentiment Score', linewidth=2)
    plt.plot(future_dates, future_predictions, 
             'r--', label='Future Prediction', linewidth=2, marker='o', markersize=4)
    
    # 신뢰구간 추가 (단순화된 버전)
    std_error = np.std(predictions - actuals)
    confidence_interval = 1.96 * std_error
    plt.fill_between(future_dates, 
                     future_predictions - confidence_interval,
                     future_predictions + confidence_interval,
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
    plt.title('Samsung Sentiment Score 30-Day Future Prediction', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('samsung_future_prediction_30days.png', dpi=300, bbox_inches='tight')
    print("   - 미래 예측 차트 저장: samsung_future_prediction_30days.png")
    
    # 8. 상세 분석 리포트 생성
    print("\n8️⃣ 상세 분석 리포트 생성 중...")
    
    # 특성 중요도 분석 (단순화된 버전)
    feature_importance = {}
    for i, feature in enumerate(dataset.feature_columns):
        # 특성별 표준편차를 중요도로 사용 (단순화)
        importance = np.std(dataset.features[:, i])
        feature_importance[feature] = importance
    
    # 정렬
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 성능 비교 (이전 결과와 비교)
    performance_comparison = {
        '이전 기본 LSTM': {'R²': 0.096, 'MAE': 0.15},
        '현재 고도화 LSTM': {'R²': evaluation_results['R²'], 'MAE': evaluation_results['MAE']}
    }
    
    # 리포트 작성
    report = f"""
# 삼성전자 감성 분석 고도화 LSTM 모델 - 상세 분석 리포트

## 📊 프로젝트 개요
- **분석 대상**: 삼성전자 관련 뉴스 기사 (2021-2024)
- **데이터 포인트**: {len(processed_data):,}일
- **모델**: Advanced LSTM with Attention
- **분석 일자**: {datetime.now().strftime('%Y년 %m월 %d일')}

## 🎯 모델 성능
### 주요 지표
- **R² Score**: {evaluation_results['R²']:.4f}
- **RMSE**: {evaluation_results['RMSE']:.4f}
- **MAE**: {evaluation_results['MAE']:.4f}
- **방향성 정확도**: {evaluation_results['Direction_Accuracy']:.1%}
- **MAPE**: {evaluation_results['MAPE']:.2f}%
- **임계값 정확도**: {evaluation_results['Threshold_Accuracy']:.1%}

### 성능 개선도
"""

    for model_name, metrics in performance_comparison.items():
        report += f"- **{model_name}**: R² = {metrics['R²']:.4f}, MAE = {metrics['MAE']:.4f}\n"
    
    improvement_r2 = evaluation_results['R²'] - 0.096
    improvement_pct = (improvement_r2 / 0.096) * 100 if 0.096 != 0 else 0
    
    report += f"\n**개선도**: R² {improvement_r2:+.4f} ({improvement_pct:+.1f}%)\n"
    
    report += f"""

## 📈 데이터 특성 분석
### 기본 통계
- **감성 점수 범위**: {processed_data['sentiment_mean'].min():.3f} ~ {processed_data['sentiment_mean'].max():.3f}
- **평균 감성 점수**: {processed_data['sentiment_mean'].mean():.3f}
- **표준편차**: {processed_data['sentiment_mean'].std():.3f}
- **평균 일일 뉴스 수**: {processed_data['news_count'].mean():.1f}개

### 특성 중요도 (상위 5개)
"""

    for i, (feature, importance) in enumerate(sorted_features[:5]):
        report += f"{i+1}. **{feature}**: {importance:.4f}\n"
    
    report += f"""

## 🔮 미래 예측 결과
### 30일 예측 요약
- **예측 시작일**: {last_date.strftime('%Y-%m-%d')}
- **예측 종료일**: {future_dates[-1].strftime('%Y-%m-%d')}
- **예측 평균값**: {np.mean(future_predictions):.3f}
- **예측 변동성**: {np.std(future_predictions):.3f}
- **트렌드**: {"상승" if future_predictions[-1] > future_predictions[0] else "하락" if future_predictions[-1] < future_predictions[0] else "횡보"}

### 주요 인사이트
"""

    # 트렌드 분석
    if future_predictions[-1] > future_predictions[0]:
        trend_msg = "향후 30일 동안 삼성전자에 대한 감성이 점진적으로 개선될 것으로 예상됩니다."
    elif future_predictions[-1] < future_predictions[0]:
        trend_msg = "향후 30일 동안 삼성전자에 대한 감성이 다소 하락할 것으로 예상됩니다."
    else:
        trend_msg = "향후 30일 동안 삼성전자에 대한 감성이 현재 수준을 유지할 것으로 예상됩니다."
    
    report += f"- {trend_msg}\n"
    
    # 변동성 분석
    volatility_level = "높음" if np.std(future_predictions) > processed_data['sentiment_mean'].std() else "보통" if np.std(future_predictions) > processed_data['sentiment_mean'].std() * 0.5 else "낮음"
    report += f"- 예상 변동성은 **{volatility_level}** 수준입니다.\n"
    
    # 연도별 비교
    yearly_avg = processed_data.groupby(processed_data['date'].dt.year)['sentiment_mean'].mean()
    current_year_avg = yearly_avg.iloc[-1] if len(yearly_avg) > 0 else processed_data['sentiment_mean'].mean()
    
    if np.mean(future_predictions) > current_year_avg:
        year_comparison = "작년 대비 개선된"
    elif np.mean(future_predictions) < current_year_avg:
        year_comparison = "작년 대비 하락한"
    else:
        year_comparison = "작년과 유사한"
    
    report += f"- 예측된 감성 수준은 {year_comparison} 수준입니다.\n"
    
    report += f"""

## 🔍 모델 분석
### 모델 아키텍처
- **입력 특성**: {len(dataset.feature_columns)}개
- **시퀀스 길이**: {sequence_length}일
- **LSTM 레이어**: 3층 (양방향)
- **어텐션 메커니즘**: 멀티헤드 어텐션 (8헤드)
- **정규화**: Dropout (0.2), 배치 정규화

### 훈련 과정
- **총 에포크**: {len(trainer.train_losses)}
- **최종 훈련 손실**: {trainer.train_losses[-1]:.6f}
- **최종 검증 손실**: {trainer.val_losses[-1]:.6f}
- **조기 종료**: {"적용됨" if len(trainer.train_losses) < 100 else "미적용"}

## 📋 결론 및 제언
### 주요 성과
1. **모델 성능 개선**: 기존 대비 R² 점수 향상
2. **다차원 분석**: 11개 특성을 활용한 종합적 분석
3. **실용적 예측**: 방향성 정확도 {evaluation_results['Direction_Accuracy']:.1%} 달성

### 한계점
1. **예측 정확도**: 여전히 R² < 0.5 수준
2. **노이즈 영향**: 일별 데이터의 높은 변동성
3. **외부 요인**: 주가, 경제지표 등 미반영

### 개선 방안
1. **데이터 확장**: 소셜미디어, 재무데이터 통합
2. **시간 단위 조정**: 주간/월간 단위 분석 고려
3. **앙상블 모델**: 여러 모델 조합으로 성능 향상
4. **실시간 업데이트**: 새로운 뉴스 데이터 자동 반영

## 📊 활용 방안
### 비즈니스 적용
- **마케팅 타이밍**: 감성 개선 시점에 마케팅 집중
- **위기 관리**: 감성 급락 예상 시 선제적 대응
- **투자 참고**: 감성 트렌드를 투자 의사결정에 참고

### 기술적 발전
- **모델 고도화**: Transformer, BERT 등 최신 모델 적용
- **실시간 시스템**: 뉴스 수집부터 예측까지 자동화
- **다기업 확장**: 다른 기업으로 모델 확장 적용

---
*본 리포트는 AI 기반 감성 분석 모델의 결과이며, 투자 조언이 아닌 참고 자료로 활용하시기 바랍니다.*
"""

    # 리포트 저장
    with open('samsung_advanced_lstm_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   - 상세 리포트 저장: samsung_advanced_lstm_report.md")
    
    # 9. 요약 출력
    print("\n" + "="*60)
    print("🎉 삼성전자 감성 분석 고도화 LSTM 모델 분석 완료!")
    print("="*60)
    print(f"📈 최종 성능: R² = {evaluation_results['R²']:.4f}")
    print(f"🎯 방향성 정확도: {evaluation_results['Direction_Accuracy']:.1%}")
    print(f"📊 생성된 파일:")
    print("   - samsung_advanced_lstm_analysis.html (상호작용 차트)")
    print("   - samsung_advanced_lstm_static_analysis.png (정적 차트)")
    print("   - samsung_future_prediction_30days.png (미래 예측)")
    print("   - samsung_advanced_lstm_report.md (상세 리포트)")
    print("   - best_samsung_lstm_model.pth (훈련된 모델)")
    
    # 개선 제안
    if evaluation_results['R²'] < 0.3:
        print("\n💡 성능 개선 제안:")
        print("   1. 주간 단위로 데이터 집계하여 노이즈 감소")
        print("   2. 주가, 거래량 등 외부 데이터 추가")
        print("   3. 감성 분석 품질 개선 (KoBERT 재학습)")
        print("   4. 앙상블 모델 적용")
    
    return {
        'model': model,
        'dataset': dataset,
        'evaluation_results': evaluation_results,
        'predictions': predictions,
        'actuals': actuals,
        'future_predictions': future_predictions,
        'processed_data': processed_data
    }

if __name__ == "__main__":
    # 실행
    results = main()
    print("\n✅ 모든 분석이 완료되었습니다!")