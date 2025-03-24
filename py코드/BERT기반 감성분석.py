# src/models/bert_sentiment_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 토크나이징
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    def __init__(self, model_name='klue/bert-base', num_labels=3, config_path='config.json'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # 훈련 기록
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def prepare_dataloader(self, texts, labels, batch_size=16, max_len=128, shuffle=True):
        """데이터 로더 준비"""
        dataset = SentimentDataset(texts, labels, self.tokenizer, max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_dataloader, val_dataloader=None, epochs=4, lr=2e-5, warmup_steps=0, output_dir='models/sentiment'):
        """모델 훈련"""
        # 옵티마이저 및 스케줄러 설정
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        # 손실 함수
        loss_fn = nn.CrossEntropyLoss()
        
        # 훈련 시작
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            
            for batch in progress_bar:
                # 배치 데이터 준비
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # 역전파
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # 통계 업데이트
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 진행 상황 업데이트
                progress_bar.set_postfix({
                    'loss': running_loss / (progress_bar.n + 1),
                    'acc': 100. * correct / total
                })
            
            # 에폭 평균 손실 및 정확도
            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = 100. * correct / total
            
            # 훈련 기록 업데이트
            self.train_history['loss'].append(epoch_loss)
            self.train_history['accuracy'].append(epoch_acc)
            
            logger.info(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
            # 검증 데이터로 평가
            if val_dataloader:
                val_loss, val_acc = self.evaluate(val_dataloader)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_accuracy'].append(val_acc)
                
                logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 모델 저장
        model_path = f"{output_dir}/bert_sentiment_{timestamp}"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # 훈련 기록 저장
        history_path = f"{output_dir}/training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def evaluate(self, dataloader):
        """모델 평가"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 배치 데이터 준비
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 순전파
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # 통계 업데이트
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 평균 손실 및 정확도
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, texts, batch_size=16, max_len=128):
        """텍스트에 대한 감성 예측"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 데이터셋 및 데이터로더 생성
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(texts, dummy_labels, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # 예측 수행
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 배치 데이터 준비
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                # 순전파
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # 예측 저장
                _, preds = torch.max(logits, 1)
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities
    
    def compute_metrics(self, true_labels, predicted_labels):
        """모델 성능 평가 지표 계산"""
        accuracy =
