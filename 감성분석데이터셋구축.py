# src/features/sentiment_dataset_builder.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import random
from text_preprocessing import TextPreprocessor

class SentimentDatasetBuilder:
    def __init__(self, config_path='config.json'):
        self.preprocessor = TextPreprocessor(config_path)
        
        # 감성 레이블 정의
        self.sentiment_labels = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
        # 샘플링 옵션
        self.sampling_options = {
            'random': self._random_sampling,
            'stratified_by_brand': self._stratified_by_brand_sampling,
            'stratified_by_time': self._stratified_by_time_sampling
        }
    
    def _random_sampling(self, df, n_samples):
        """무작위 샘플링"""
        if len(df) <= n_samples:
            return df
        
        return df.sample(n=n_samples, random_state=42)
    
    def _stratified_by_brand_sampling(self, df, n_samples):
        """브랜드별 층화 샘플링"""
        if 'brand' not in df.columns:
            return self._random_sampling(df, n_samples)
        
        # 브랜드별 비율 계산
        brand_counts = df['brand'].value_counts(normalize=True)
        
        # 브랜드별 샘플 수 계산
        brand_samples = {brand: int(np.ceil(n_samples * ratio)) for brand, ratio in brand_counts.items()}
        
        # 각 브랜드에서 샘플링
        samples = []
        for brand, count in brand_samples.items():
            brand_df = df[df['brand'] == brand]
            if len(brand_df) <= count:
                samples.append(brand_df)
            else:
                samples.append(brand_df.sample(n=count, random_state=42))
        
        # 샘플 합치기
        sampled_df = pd.concat(samples)
        
        # 원하는 샘플 수에 맞게 조정
        if len(sampled_df) > n_samples:
            sampled_df = sampled_df.sample(n=n_samples, random_state=42)
        
        return sampled_df
    
    def _stratified_by_time_sampling(self, df, n_samples):
        """시간별 층화 샘플링"""
        if 'created_at' not in df.columns and 'date' not in df.columns:
            return self._random_sampling(df, n_samples)
        
        # 날짜 필드 선택
        date_field = 'created_at' if 'created_at' in df.columns else 'date'
        
        # 날짜 형식 변환
        if df[date_field].dtype != 'datetime64[ns]':
            df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
        
        # 월별 분류
        df['month'] = df[date_field].dt.strftime('%Y-%m')
        
        # 월별 비율 계산
        month_counts = df['month'].value_counts(normalize=True)
        
        # 월별 샘플 수 계산
        month_samples = {month: int(np.ceil(n_samples * ratio)) for month, ratio in month_counts.items()}
        
        # 각 월에서 샘플링
        samples = []
        for month, count in month_samples.items():
            month_df = df[df['month'] == month]
            if len(month_df) <= count:
                samples.append(month_df)
            else:
                samples.append(month_df.sample(n=count, random_state=42))
        
        # 샘플 합치기
        sampled_df = pd.concat(samples)
        
        # 원하는 샘플 수에 맞게 조정
        if len(sampled_df) > n_samples:
            sampled_df = sampled_df.sample(n=n_samples, random_state=42)
        
        # 불필요한 열 제거
        sampled_df = sampled_df.drop(columns=['month'])
        
        return sampled_df
    
    def sample_for_labeling(self, df, n_samples=1000, sampling_method='stratified_by_brand'):
        """레이블링을 위한 샘플 추출"""
        if sampling_method not in self.sampling_options:
            raise ValueError(f"지원하지 않는 샘플링 방법: {sampling_method}")
        
        # 샘플링 함수 선택
        sampling_func = self.sampling_options[sampling_method]
        
        # 텍스트 전처리
        text_column = next((col for col in df.columns if col in ['text', 'body', 'content']), None)
        if text_column is None:
            raise ValueError("텍스트 열을 찾을 수 없습니다")
        
        # 전처리된 열이 없으면 전처리 수행
        if f'cleaned_{text_column}' not in df.columns:
            df = self.preprocessor.preprocess_dataframe(df, text_column)
        
        # 샘플링
        sampled_df = sampling_func(df, n_samples)
        
        # 필수 열만 선택
        required_columns = ['id', text_column, f'cleaned_{text_column}', 'brand'] if 'brand' in df.columns else ['id', text_column, f'cleaned_{text_column}']
        sampled_df = sampled_df[required_columns].copy()
        
        # 레이블링 열 추가
        sampled_df['sentiment'] = None
        
        return sampled_df
    
    def generate_labeling_guide(self, output_path='docs/labeling_guide.md'):
        """레이블링 가이드라인 생성"""
        guide_content = """
# 소셜 미디어 감성 분석 레이블링 가이드라인

## 개요
이 문서는 소셜 미디어 텍스트에 대한 감성 레이블링 방법을 안내합니다. 각 텍스트는 다음 세 가지 감성 중 하나로 분류됩니다.

## 감성 레이블 정의

1. **부정적(Negative) - 0**
   - 제품/브랜드에 대한 불만, 실망, 불평, 비판
   - 화난 감정, 좌절감이 드러난 표현
   - 부정적인 단어(문제, 안좋음, 실망, 불만 등) 포함

2. **중립적(Neutral) - 1**
   - 감정 없이 사실만 전달하는 내용
   - 질문이나 정보 요청
   - 긍정도 부정도 아닌 중간적인 평가
   - 뉴스 기사나 객관적 정보 공유

3. **긍정적(Positive) - 2**
   - 제품/브랜드에 대한 칭찬, 감사, 만족감 표현
   - 기쁨, 흥분, 놀라움 등 긍정적 감정
   - 긍정적인 단어(좋음, 훌륭함, 추천, 만족 등) 포함

## 레이블링 지침

1. **전체 문맥 고려하기**
   - 개별 단어나 구절보다 전체 문맥을 고려하여 판단
   - 아이러니나 반어법 사용 여부 확인

2. **복합적 감정 처리**
   - 긍정과 부정이 동시에 나타나면, 주된 감정 또는 결론에 따라 분류
   - 명확하지 않은 경우 중립으로 분류

3. **브랜드/제품 관련성**
   - 브랜드나 제품과 직접 관련된 감정만 고려
   - 관련 없는 내용의 감성은 무시

4. **특별 사례**
   - 이모티콘만 있는 경우: 이모티콘의 의미에 따라 분류
   - 광고/홍보 글: 중립으로 분류
   - 짧은 단어나 불완전한 문장: 가능한 해석하되, 판단 불가 시 중립

## 예시

### 부정적 (0)
- "이 스마트폰 배터리 너무 빨리 닳아서 실망했어요."
- "AS 서비스가 형편없습니다. 다시는 이 브랜드 안 삽니다."
- "화면이 자꾸 멈춰서 짜증나요. 쓸모없어요."

### 중립적 (1)
- "이 제품 언제 출시되나요?"
- "흰색과 검정색 중 어떤 색상이 더 인기있나요?"
- "오늘 새 모델이 출시되었다고 합니다."

### 긍정적 (2)
- "카메라 성능이 정말 좋아요! 사진이 너무 잘 나옵니다."
- "빠른 배송과 친절한 서비스에 감사드립니다."
- "디자인이 예쁘고 성능도 훌륭해서 대만족입니다!"

## 레이블링 방법

제공된 엑셀 파일에서 'sentiment' 열에 다음과 같이 입력:
- 부정적: 0
- 중립적: 1
- 긍정적: 2
        """
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 가이드라인 파일 작성
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"레이블링 가이드라인이 {output_path}에 생성되었습니다.")
    
    def create_labeling_files(self, sampled_df, output_dir='data/interim/labeling'):
        """레이블링을 위한 파일 생성"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 필요한 열만 선택
        text_column = next((col for col in sampled_df.columns if col in ['text', 'body', 'content']), None)
        cleaned_column = f'cleaned_{text_column}'
        
        # 레이블링용 데이터프레임 생성
        labeling_df = sampled_df[['id', text_column, cleaned_column, 'sentiment']].copy()
        labeling_df.columns = ['id', 'original_text', 'cleaned_text', 'sentiment']
        
        # 파일 저장
        excel_path = f"{output_dir}/sentiment_labeling_{timestamp}.xlsx"
        labeling_df.to_excel(excel_path, index=False)
        
        print(f"레이블링 파일이 {excel_path}에 생성되었습니다.")
        return excel_path
    
    def load_labeled_data(self, labeled_file_path):
        """레이블링된 데이터 로드"""
        if not os.path.exists(labeled_file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {labeled_file_path}")
        
        # 파일 확장자 확인
        ext = os.path.splitext(labeled_file_path)[1].lower()
        if ext == '.xlsx' or ext == '.xls':
            labeled_df = pd.read_excel(labeled_file_path)
        elif ext == '.csv':
            labeled_df = pd.read_csv(labeled_file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        
        # 필요한 열 확인
        required_columns = ['id', 'original_text', 'cleaned_text', 'sentiment']
        for col in required_columns:
            if col not in labeled_df.columns:
                raise ValueError(f"필수 열이 누락되었습니다: {col}")
        
        # 레이블 확인
        labeled_df = labeled_df.dropna(subset=['sentiment'])
        labeled_df['sentiment'] = labeled_df['sentiment'].astype(int)
        
        # 레이블 분포 확인
        sentiment_counts = labeled_df['sentiment'].value_counts()
        print("감성 레이블 분포:")
        for sentiment, count in sentiment_counts.items():
            label_name = {0: '부정적', 1: '중립적', 2: '긍정적'}.get(sentiment, '알 수 없음')
            print(f"  {label_name} ({sentiment}): {count} ({count/len(labeled_df)*100:.1f}%)")
        
        return labeled_df
    
    def balance_dataset(self, labeled_df, method='undersample'):
        """데이터셋 균형 조정"""
        sentiment_counts = labeled_df['sentiment'].value_counts()
        min_count = sentiment_counts.min()
        balanced_df = pd.DataFrame()
        
        if method == 'undersample':
            # 언더샘플링: 가장 적은 클래스 수에 맞춤
            for sentiment in sentiment_counts.index:
                sentiment_df = labeled_df[labeled_df['sentiment'] == sentiment]
                balanced_df = pd.concat([balanced_df, sentiment_df.sample(n=min_count, random_state=42)])
        
        elif method == 'oversample':
            # 오버샘플링: 가장 많은 클래스 수에 맞춤
            max_count = sentiment_counts.max()
            for sentiment in sentiment_counts.index:
                sentiment_df = labeled_df[labeled_df['sentiment'] == sentiment]
                if len(sentiment_df) < max_count:
                    # 복원 추출로 오버샘플링
                    sampled = sentiment_df.sample(n=max_count-len(sentiment_df), replace=True, random_state=42)
                    sentiment_df = pd.concat([sentiment_df, sampled])
                balanced_df = pd.concat([balanced_df, sentiment_df])
        
        else:
            raise ValueError(f"지원하지 않는 균형 조정 방법: {method}")
        
        # 결과 확인
        balanced_counts = balanced_df['sentiment'].value_counts()
        print(f"균형 조정 후 감성 레이블 분포 ({method}):")
        for sentiment, count in balanced_counts.items():
            label_name = {0: '부정적', 1: '중립적', 2: '긍정적'}.get(sentiment, '알 수 없음')
            print(f"  {label_name} ({sentiment}): {count} ({count/len(balanced_df)*100:.1f}%)")
        
        return balanced_df
    
    def split_dataset(self, labeled_df, test_size=0.2, val_size=0.1):
        """데이터셋 분할 (훈련/검증/테스트)"""
        # 훈련 및 테스트 분할
        train_df, test_df = train_test_split(
            labeled_df,
            test_size=test_size,
            stratify=labeled_df['sentiment'],
            random_state=42
        )
        
        if val_size > 0:
            # 훈련 및 검증 분할
            val_size_adjusted = val_size / (1 - test_size)  # 남은 데이터에서의 비율로 조정
            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size_adjusted,
                stratify=train_df['sentiment'],
                random_state=42
            )
            
            print(f"데이터셋 분할 완료: 훈련={len(train_df)}, 검증={len(val_df)}, 테스트={len(test_df)}")
            return train_df, val_df, test_df
        else:
            print(f"데이터셋 분할 완료: 훈련={len(train_df)}, 테스트={len(test_df)}")
            return train_df, test_df
    
    def save_datasets(self, datasets, output_dir='data/processed/sentiment'):
        """분할된 데이터셋 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 데이터셋 저장
        if len(datasets) == 3:
            train_df, val_df, test_df = datasets
            train_df.to_csv(f"{output_dir}/train_{timestamp}.csv", index=False, encoding='utf-8')
            val_df.to_csv(f"{output_dir}/val_{timestamp}.csv", index=False, encoding='utf-8')
            test_df.to_csv(f"{output_dir}/test_{timestamp}.csv", index=False, encoding='utf-8')
            print(f"3개 데이터셋이 {output_dir}에 저장되었습니다.")
        elif len(datasets) == 2:
            train_df, test_df = datasets
            train_df.to_csv(f"{output_dir}/train_{timestamp}.csv", index=False, encoding='utf-8')
            test_df.to_csv(f"{output_dir}/test_{timestamp}.csv", index=False, encoding='utf-8')
            print(f"2개 데이터셋이 {output_dir}에 저장되었습니다.")
        else:
            raise ValueError(f"예상치 못한 데이터셋 수: {len(datasets)}")
        
        return timestamp

# 실행 예시
if __name__ == "__main__":
    # 데이터셋 빌더 생성
    dataset_builder = SentimentDatasetBuilder()
    
    # 레이블링 가이드라인 생성
    dataset_builder.generate_labeling_guide()
    
    # 샘플 데이터 로드 (실제로는 데이터베이스에서 로드)
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'text': [f"샘플 텍스트 {i}" for i in range(1, 101)],
        'brand': np.random.choice(['삼성전자', 'LG전자', '애플'], 100)
    })
    
    # 레이블링을 위한 샘플 추출
    sampled_df = dataset_builder.sample_for_labeling(sample_data, n_samples=50)
    
    # 레이블링 파일 생성
    labeling_file = dataset_builder.create_labeling_files(sampled_df)
    
    # 실제 프로젝트에서는 이 파일을 사람이 레이블링한 후 아래 코드 실행
    
    # 가상의 레이블링된 데이터 생성 (실습용)
    sampled_df['sentiment'] = np.random.choice([0, 1, 2], len(sampled_df))
    
    # 레이블링된 데이터셋 균형 조정
    balanced_df = dataset_builder.balance_dataset(sampled_df, method='undersample')
    
    # 데이터셋 분할
    train_df, val_df, test_df = dataset_builder.split_dataset(balanced_df)
    
    # 데이터셋 저장
    timestamp = dataset_builder.save_datasets([train_df, val_df, test_df])
