"""
뉴스 감성 분석 기반 주가 예측 모델 - 10.개선된삼성LSTM.py
생성일: 2025-06-08
팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)

9번 코드에서 생성된 7일 평균 데이터를 활용하여
Samsung 중심의 고도화된 LSTM 모델을 구축합니다.

목표: R² > 0.3 (기존 -0.19에서 대폭 개선)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 결과물 저장 경로 설정
RESULTS_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model/results/2025-0608"
PROJECT_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model"

# 9번 코드에서 생성된 데이터 경로
LSTM_SEQUENCES_PATH = f"{RESULTS_BASE}/data/features/lstm_training_sequences.pkl"
FEATURE_INFO_PATH = f"{RESULTS_BASE}/data/features/feature_info.json"
WEEKLY_FEATURES_PATH = f"{RESULTS_BASE}/data/features/weekly_sentiment_features.csv"

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_directories():
    """결과물 디렉토리 구조 생성"""
    directories = [
        f"{RESULTS_BASE}/models/trained",
        f"{RESULTS_BASE}/models/evaluation",
        f"{RESULTS_BASE}/models/features_importance",
        f"{RESULTS_BASE}/models/predictions",
        f"{RESULTS_BASE}/visualizations/model_performance",
        f"{RESULTS_BASE}/reports/technical"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ 모델 관련 디렉토리 구조 생성 완료")

# 디렉토리 자동 생성
setup_directories()

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 사용 디바이스: {device}")

class SamsungLSTMDataset(Dataset):
    """Samsung LSTM 학습용 데이터셋"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AdvancedSamsungLSTM(nn.Module):
    """고도화된 Samsung LSTM 모델"""
    def __init__(self, input_size=58, hidden_size=128, num_layers=3, dropout=0.3):
        super(AdvancedSamsungLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # 멀티헤드 어텐션
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # LSTM 통과
        lstm_out, _ = self.lstm(x)
        
        # Self-Attention 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Layer Normalization
        normalized = self.layer_norm(attn_out)
        
        # 마지막 시간 스텝의 출력 사용
        final_output = normalized[:, -1, :]
        
        # 최종 예측
        prediction = self.fc(final_output)
        
        return prediction

def load_preprocessed_data():
    """9번에서 생성된 전처리 데이터 로딩"""
    print("📁 9번 코드에서 생성된 전처리 데이터 로딩 중...")
    
    try:
        with open(LSTM_SEQUENCES_PATH, 'rb') as f:
            lstm_data = pickle.load(f)
        
        print("🔍 실제 데이터 구조 확인 중...")
        print(f"   - 로딩된 키들: {list(lstm_data.keys())}")
        
        X = lstm_data['X']
        y = lstm_data['y']
        split_data = lstm_data.get('split_data', None)
        scalers = lstm_data.get('scalers', None)
        
        print(f"✅ 데이터 로딩 완료")
        print(f"   - 전체 시퀀스 수: {X.shape[0]}")
        print(f"   - 시퀀스 길이: {X.shape[1]}")
        print(f"   - 특성 개수: {X.shape[2]}")
        
        if 'company_info' in lstm_data:
            company_info = lstm_data['company_info']
            print(f"   - Samsung 샘플 수: {company_info['Samsung']['count']}")
            print(f"   - Apple 샘플 수: {company_info['Apple']['count']}")
        else:
            print("   - company_info 없음. 메타데이터에서 회사 정보 추출 예정")
            company_info = None
        
        return X, y, split_data, scalers, company_info
        
    except FileNotFoundError:
        print("❌ 9번 코드 결과물을 찾을 수 없습니다. 9번 코드를 먼저 실행해주세요.")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류 발생: {e}")
        return None, None, None, None, None

def filter_samsung_data(X, y, company_info=None):
    """Samsung 데이터만 추출"""
    print("🔍 Samsung 데이터 필터링 중...")
    
    if company_info is not None and 'Samsung' in company_info:
        samsung_indices = company_info['Samsung']['indices']
        X_samsung = X[samsung_indices]
        y_samsung = y[samsung_indices]
    else:
        try:
            weekly_features_path = f"{RESULTS_BASE}/data/features/weekly_sentiment_features.csv"
            if os.path.exists(weekly_features_path):
                weekly_df = pd.read_csv(weekly_features_path)
                samsung_mask = weekly_df['Company'] == 'Samsung'
                samsung_indices = samsung_mask[samsung_mask].index.tolist()
                X_samsung = X[samsung_indices]
                y_samsung = y[samsung_indices]
                print(f"   - 메타데이터에서 Samsung 인덱스 추출: {len(samsung_indices)}개")
            else:
                print("   - 메타데이터 없음. 전체 데이터의 후반부를 Samsung으로 가정")
                mid_point = len(X) // 2
                X_samsung = X[mid_point:]
                y_samsung = y[mid_point:]
        except Exception as e:
            print(f"   - 메타데이터 로딩 실패: {e}")
            mid_point = len(X) // 2
            X_samsung = X[mid_point:]
            y_samsung = y[mid_point:]
    
    print(f"✅ Samsung 데이터 추출 완료: {len(X_samsung)}개 시퀀스")
    return X_samsung, y_samsung

def create_samsung_train_val_test_split(X_samsung, y_samsung, train_ratio=0.6, val_ratio=0.2):
    """Samsung 데이터 시간 순서 유지하며 분할"""
    print("📊 Samsung 데이터 분할 중...")
    
    n_samples = len(X_samsung)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X_samsung[:train_size]
    y_train = y_samsung[:train_size]
    
    X_val = X_samsung[train_size:train_size + val_size]
    y_val = y_samsung[train_size:train_size + val_size]
    
    X_test = X_samsung[train_size + val_size:]
    y_test = y_samsung[train_size + val_size:]
    
    print(f"✅ 데이터 분할 완료")
    print(f"   - 훈련: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    print(f"   - 검증: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    print(f"   - 테스트: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def comprehensive_evaluation(y_true, y_pred):
    """다차원 평가 지표"""
    
    def direction_accuracy(y_true, y_pred):
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction) * 100
    
    def threshold_accuracy(y_true, y_pred, threshold=0.1):
        return np.mean(np.abs(y_true - y_pred) < threshold) * 100
    
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'direction_accuracy': direction_accuracy(y_true, y_pred),
        'threshold_accuracy': threshold_accuracy(y_true, y_pred, threshold=0.1)
    }
    
    return metrics

class ModelTrainer:
    """LSTM 모델 훈련 및 평가 클래스"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=200, learning_rate=0.001, patience=20):
        print("🚀 Samsung LSTM 모델 훈련 시작...")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate_epoch(val_loader, criterion)
            scheduler.step(val_loss)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          f"{RESULTS_BASE}/models/trained/best_samsung_lstm_weekly.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            if early_stopping_counter >= patience:
                print(f"⏰ 조기 종료: {patience}회 연속 개선 없음 (Epoch {epoch+1})")
                break
        
        self.model.load_state_dict(torch.load(f"{RESULTS_BASE}/models/trained/best_samsung_lstm_weekly.pth"))
        print("✅ 훈련 완료 및 최고 성능 모델 로드")
        
        return self.training_history
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(actuals), np.array(predictions)

def create_comprehensive_visualizations(y_true, y_pred, training_history):
    """종합 시각화 생성 (9개 서브플롯)"""
    print("📊 종합 시각화 생성 중...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Samsung LSTM Model - Comprehensive Analysis', fontsize=20, y=0.98)
    
    # 1. 시계열 예측 결과
    ax1 = axes[0, 0]
    ax1.plot(y_true, label='Actual', color='blue', alpha=0.7)
    ax1.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    ax1.set_title('Time Series Prediction Results')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sentiment Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 예측 vs 실제 산점도
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.6, color='purple')
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax2.set_title('Predicted vs Actual')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.grid(True, alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 3. 훈련 곡선
    ax3 = axes[0, 2]
    ax3.plot(training_history['train_loss'], label='Train Loss', color='blue')
    ax3.plot(training_history['val_loss'], label='Validation Loss', color='orange')
    ax3.set_title('Training Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 오차 분포
    ax4 = axes[1, 0]
    errors = y_pred - y_true
    ax4.hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Frequency')
    ax4.axvline(0, color='red', linestyle='--', alpha=0.8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 잔차 플롯
    ax5 = axes[1, 1]
    ax5.scatter(y_pred, errors, alpha=0.6, color='orange')
    ax5.axhline(0, color='red', linestyle='--', alpha=0.8)
    ax5.set_title('Residual Plot')
    ax5.set_xlabel('Predicted Values')
    ax5.set_ylabel('Residuals')
    ax5.grid(True, alpha=0.3)
    
    # 6. 방향성 정확도
    ax6 = axes[1, 2]
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    direction_acc = np.mean(true_direction == pred_direction) * 100
    
    categories = ['Correct', 'Incorrect']
    correct_count = np.sum(true_direction == pred_direction)
    incorrect_count = len(true_direction) - correct_count
    values = [correct_count, incorrect_count]
    
    ax6.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    ax6.set_title(f'Direction Accuracy: {direction_acc:.1f}%')
    
    # 7. 성능 지표 요약
    ax7 = axes[2, 0]
    ax7.axis('off')
    
    metrics = comprehensive_evaluation(y_true, y_pred)
    metrics_text = f"""
    Performance Metrics:
    
    R² Score: {metrics['r2_score']:.4f}
    RMSE: {metrics['rmse']:.4f}
    MAE: {metrics['mae']:.4f}
    MAPE: {metrics['mape']:.2f}%
    Direction Accuracy: {metrics['direction_accuracy']:.1f}%
    Threshold Accuracy: {metrics['threshold_accuracy']:.1f}%
    """
    
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    # 8. 최근 30일 예측 vs 실제
    ax8 = axes[2, 1]
    last_30_true = y_true[-30:]
    last_30_pred = y_pred[-30:]
    
    ax8.plot(range(30), last_30_true, 'o-', label='Actual', color='blue', markersize=4)
    ax8.plot(range(30), last_30_pred, 's-', label='Predicted', color='red', markersize=4)
    ax8.set_title('Last 30 Days - Detailed View')
    ax8.set_xlabel('Days')
    ax8.set_ylabel('Sentiment Score')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 예측 신뢰구간
    ax9 = axes[2, 2]
    
    error_std = np.std(errors)
    upper_bound = y_pred + 1.96 * error_std
    lower_bound = y_pred - 1.96 * error_std
    
    recent_range = range(len(y_true) - 50, len(y_true))
    ax9.plot(recent_range, y_true[-50:], 'o-', label='Actual', color='blue', markersize=3)
    ax9.plot(recent_range, y_pred[-50:], 's-', label='Predicted', color='red', markersize=3)
    ax9.fill_between(recent_range, upper_bound[-50:], lower_bound[-50:], 
                     alpha=0.3, color='gray', label='95% Confidence')
    ax9.set_title('Prediction with Confidence Interval')
    ax9.set_xlabel('Time Index')
    ax9.set_ylabel('Sentiment Score')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    viz_path = f"{RESULTS_BASE}/visualizations/model_performance/samsung_lstm_comprehensive_analysis.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 종합 시각화 저장: {viz_path}")
    
    plt.show()
    
    return metrics

def predict_future_30_days(model, last_sequence, device):
    """30일 미래 예측"""
    print("🔮 30일 미래 예측 생성 중...")
    
    model.eval()
    predictions = []
    current_sequence = last_sequence.copy()
    
    with torch.no_grad():
        for day in range(30):
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            pred = model(input_tensor)
            pred_value = pred.cpu().numpy()[0, 0]
            
            predictions.append(pred_value)
            
            new_features = current_sequence[-1].copy()
            new_features[0] = pred_value
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_features
    
    return np.array(predictions)

def create_future_prediction_chart(future_predictions, y_true, y_pred):
    """미래 예측 차트 생성"""
    print("📈 미래 예측 차트 생성 중...")
    
    plt.figure(figsize=(15, 8))
    
    past_range = range(-60, 0)
    plt.plot(past_range, y_true[-60:], 'o-', label='Actual (Past)', color='blue', markersize=3)
    plt.plot(past_range, y_pred[-60:], 's-', label='Predicted (Past)', color='red', markersize=3)
    
    future_range = range(0, 30)
    plt.plot(future_range, future_predictions, '^-', label='Future Prediction', 
             color='green', markersize=4, linewidth=2)
    
    past_error_std = np.std(y_pred - y_true)
    upper_bound = future_predictions + 1.96 * past_error_std
    lower_bound = future_predictions - 1.96 * past_error_std
    
    plt.fill_between(future_range, upper_bound, lower_bound, 
                     alpha=0.3, color='green', label='95% Confidence Interval')
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Current Time')
    
    plt.title('Samsung Sentiment Score - 30 Days Future Prediction', fontsize=16)
    plt.xlabel('Days (Relative to Current Time)')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    stats_text = f"""
    Future Prediction Summary:
    Mean: {np.mean(future_predictions):.3f}
    Trend: {(future_predictions[-1] - future_predictions[0]):.3f}
    Volatility: {np.std(future_predictions):.3f}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    future_path = f"{RESULTS_BASE}/visualizations/model_performance/samsung_future_prediction_30days.png"
    plt.savefig(future_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 미래 예측 차트 저장: {future_path}")
    
    plt.show()

def save_model_results(model, metrics, training_history, future_predictions, y_true, y_pred, input_size):
    """모델 결과 저장"""
    print("💾 모델 결과 저장 중...")
    
    model_config = {
        'model_type': 'AdvancedSamsungLSTM',
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'attention_heads': 8,
        'training_epochs': len(training_history['train_loss']),
        'best_val_loss': min(training_history['val_loss']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{RESULTS_BASE}/models/evaluation/samsung_model_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    with open(f"{RESULTS_BASE}/models/evaluation/model_performance_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"{RESULTS_BASE}/models/evaluation/training_history.csv", index=False)
    
    predictions_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true)
    })
    predictions_df.to_csv(f"{RESULTS_BASE}/models/predictions/test_predictions.csv", index=False)
    
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sentiment': future_predictions,
        'prediction_day': range(1, 31)
    })
    future_df.to_csv(f"{RESULTS_BASE}/models/predictions/30day_future_predictions.csv", index=False)
    
    report = f"""# Samsung LSTM Model Performance Report

## Model Configuration
- Model Type: Advanced Samsung LSTM with Attention
- Input Features: {input_size} dimensions
- Sequence Length: 4 weeks
- Hidden Size: 128
- LSTM Layers: 3 (Bidirectional)
- Attention Heads: 8
- Training Epochs: {len(training_history['train_loss'])}

## Performance Metrics
- R² Score: {metrics['r2_score']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- MAPE: {metrics['mape']:.2f}%
- Direction Accuracy: {metrics['direction_accuracy']:.1f}%
- Threshold Accuracy (±0.1): {metrics['threshold_accuracy']:.1f}%

## Model Analysis
- Best Validation Loss: {min(training_history['val_loss']):.6f}
- Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Future Prediction Trend: {(future_predictions[-1] - future_predictions[0]):.3f}

## Business Insights
- Model Performance Level: {'Excellent' if metrics['r2_score'] > 0.5 else 'Good' if metrics['r2_score'] > 0.3 else 'Moderate' if metrics['r2_score'] > 0.1 else 'Needs Improvement'}
- Practical Usage: {'Highly Recommended' if metrics['direction_accuracy'] > 70 else 'Recommended' if metrics['direction_accuracy'] > 60 else 'Conditional Use'}
- Risk Level: {'Low' if metrics['rmse'] < 0.3 else 'Medium' if metrics['rmse'] < 0.5 else 'High'}
"""
    
    with open(f"{RESULTS_BASE}/reports/technical/samsung_lstm_performance_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 모델 결과 저장 완료")
    print(f"   - 모델 설정: models/evaluation/samsung_model_config.json")
    print(f"   - 성능 지표: models/evaluation/model_performance_metrics.json")
    print(f"   - 훈련 이력: models/evaluation/training_history.csv")
    print(f"   - 예측 결과: models/predictions/test_predictions.csv")
    print(f"   - 미래 예측: models/predictions/30day_future_predictions.csv")
    print(f"   - 성능 리포트: reports/technical/samsung_lstm_performance_report.md")

def main():
    """메인 실행 함수"""
    print("🚀 Samsung LSTM 모델 개발 시작!")
    print("=" * 60)
    
    # 1. 데이터 로딩
    X, y, split_data, scalers, company_info = load_preprocessed_data()
    
    if X is None:
        print("❌ 데이터 로딩 실패. 프로그램을 종료합니다.")
        return
    
    # 2. Samsung 데이터 필터링
    X_samsung, y_samsung = filter_samsung_data(X, y, company_info)
    
    # 3. 데이터 분할
    X_train, y_train, X_val, y_val, X_test, y_test = create_samsung_train_val_test_split(
        X_samsung, y_samsung, train_ratio=0.6, val_ratio=0.2
    )
    
    # 4. 데이터셋 및 데이터로더 생성
    print("📦 데이터로더 생성 중...")
    train_dataset = SamsungLSTMDataset(X_train, y_train)
    val_dataset = SamsungLSTMDataset(X_val, y_val)
    test_dataset = SamsungLSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. 모델 생성
    print("🧠 고도화된 Samsung LSTM 모델 생성 중...")
    input_size = X_train.shape[2]
    model = AdvancedSamsungLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    )
    
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 실제 입력 특성 개수: {input_size}")
    
    # 6. 모델 훈련
    trainer = ModelTrainer(model, device)
    training_history = trainer.train(
        train_loader, val_loader, 
        num_epochs=200, 
        learning_rate=0.001, 
        patience=20
    )
    
    # 7. 모델 평가
    print("📊 모델 성능 평가 중...")
    y_true, y_pred = trainer.predict(test_loader)
    
    # 8. 종합 시각화
    metrics = create_comprehensive_visualizations(y_true, y_pred, training_history)
    
    # 9. 성능 결과 출력
    print("\n" + "=" * 60)
    print("🎯 최종 모델 성능 결과")
    print("=" * 60)
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
    print(f"Threshold Accuracy: {metrics['threshold_accuracy']:.1f}%")
    
    # 성능 평가
    if metrics['r2_score'] > 0.5:
        print("🏆 우수한 성능! 실무 활용 강력 추천")
    elif metrics['r2_score'] > 0.3:
        print("✅ 좋은 성능! 실무 활용 가능")
    elif metrics['r2_score'] > 0.1:
        print("⚠️ 보통 성능. 추가 개선 필요")
    else:
        print("❌ 성능 부족. 모델 재설계 필요")
    
    # 10. 미래 예측
    print("\n🔮 30일 미래 예측 생성 중...")
    last_sequence = X_test[-1]
    future_predictions = predict_future_30_days(model, last_sequence, device)
    
    # 11. 미래 예측 시각화
    create_future_prediction_chart(future_predictions, y_true, y_pred)
    
    # 12. 모든 결과 저장
    save_model_results(model, metrics, training_history, future_predictions, y_true, y_pred, input_size)
    
    # 13. 최종 요약
    print("\n" + "=" * 60)
    print("🎊 Samsung LSTM 모델 개발 완료!")
    print("=" * 60)
    print("📁 생성된 결과물:")
    print(f"   - 훈련된 모델: models/trained/best_samsung_lstm_weekly.pth")
    print(f"   - 종합 분석 차트: visualizations/model_performance/samsung_lstm_comprehensive_analysis.png")
    print(f"   - 미래 예측 차트: visualizations/model_performance/samsung_future_prediction_30days.png")
    print(f"   - 성능 리포트: reports/technical/samsung_lstm_performance_report.md")
    print(f"   - 예측 결과: models/predictions/test_predictions.csv")
    print(f"   - 미래 예측: models/predictions/30day_future_predictions.csv")
    
    print("\n✨ 다음 단계: 11번 제품출시임팩트분석.py 실행")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'future': future_predictions
        },
        'training_history': training_history
    }

if __name__ == "__main__":
    results = main()