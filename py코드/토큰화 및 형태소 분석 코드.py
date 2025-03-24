# src/features/vectorizer.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os
from gensim.models import Word2Vec
from datetime import datetime

class TextVectorizer:
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.count_vectorizer = CountVectorizer(max_features=10000)
        self.word2vec_params = {
            'vector_size': vector_size,
            'window': window,
            'min_count': min_count,
            'workers': 4
        }
        self.word2vec_model = None
    
    def fit_tfidf(self, tokens_list):
        """TF-IDF 벡터라이저 학습"""
        # 토큰 리스트를 문자열로 변환
        corpus = [' '.join(tokens) for tokens in tokens_list]
        self.tfidf_vectorizer.fit(corpus)
        return self
    
    def transform_tfidf(self, tokens_list):
        """TF-IDF 변환"""
        corpus = [' '.join(tokens) for tokens in tokens_list]
        return self.tfidf_vectorizer.transform(corpus)
    
    def fit_transform_tfidf(self, tokens_list):
        """TF-IDF 학습 및 변환"""
        corpus = [' '.join(tokens) for tokens in tokens_list]
        return self.tfidf_vectorizer.fit_transform(corpus)
    
    def fit_count(self, tokens_list):
        """Count 벡터라이저 학습"""
        corpus = [' '.join(tokens) for tokens in tokens_list]
        self.count_vectorizer.fit(corpus)
        return self
    
    def transform_count(self, tokens_list):
        """Count 변환"""
        corpus = [' '.join(tokens) for tokens in tokens_list]
        return self.count_vectorizer.transform(corpus)
    
    def fit_transform_count(self, tokens_list):
        """Count 학습 및 변환"""
        corpus = [' '.join(tokens) for tokens in tokens_list]
        return self.count_vectorizer.fit_transform(corpus)
    
    def train_word2vec(self, tokens_list):
        """Word2Vec 모델 학습"""
        # 빈 토큰 리스트 제거
        filtered_tokens = [tokens for tokens in tokens_list if tokens]
        
        if not filtered_tokens:
            raise ValueError("유효한 토큰 리스트가 없습니다.")
        
        # Word2Vec 모델 학습
        self.word2vec_model = Word2Vec(
            sentences=filtered_tokens,
            **self.word2vec_params
        )
        
        return self
    
    def get_word2vec_vector(self, tokens, method='mean'):
        """토큰 리스트를 Word2Vec 벡터로 변환"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec 모델이 학습되지 않았습니다.")
        
        if not tokens:
            return np.zeros(self.word2vec_params['vector_size'])
        
        # 모델에 있는 토큰만 필터링
        vectors = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
        
        if not vectors:
            return np.zeros(self.word2vec_params['vector_size'])
        
        # 벡터 통합 방법
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
    
    def transform_texts_to_word2vec(self, tokens_list, method='mean'):
        """토큰 리스트 배열을 Word2Vec 벡터 배열로 변환"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec 모델이 학습되지 않았습니다.")
        
        vectors = []
        for tokens in tokens_list:
            vector = self.get_word2vec_vector(tokens, method=method)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def get_most_similar_words(self, word, topn=10):
        """Word2Vec 모델에서 유사 단어 조회"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec 모델이 학습되지 않았습니다.")
        
        if word not in self.word2vec_model.wv:
            return []
        
        return self.word2vec_model.wv.most_similar(word, topn=topn)
    
    def save_models(self, dir_path='models/vectorizers'):
        """벡터라이저 모델 저장"""
        os.makedirs(dir_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # TF-IDF 벡터라이저 저장
        with open(f"{dir_path}/tfidf_vectorizer_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Count 벡터라이저 저장
        with open(f"{dir_path}/count_vectorizer_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.count_vectorizer, f)
        
        # Word2Vec 모델 저장
        if self.word2vec_model is not None:
            self.word2vec_model.save(f"{dir_path}/word2vec_model_{timestamp}.model")
        
        print(f"모델 저장 완료: {dir_path}")
    
    def load_models(self, tfidf_path, count_path, word2vec_path=None):
        """저장된 벡터라이저 모델 로드"""
        # TF-IDF 벡터라이저 로드
        with open(tfidf_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Count 벡터라이저 로드
        with open(count_path, 'rb') as f:
            self.count_vectorizer = pickle.load(f)
        
        # Word2Vec 모델 로드 (선택적)
        if word2vec_path:
            self.word2vec_model = Word2Vec.load(word2vec_path)
        
        print("모델 로드 완료")
        return self

# 실행 예시
if __name__ == "__main__":
    from text_preprocessing import TextPreprocessor
    
    # 텍스트 전처리
    preprocessor = TextPreprocessor()
    
    # 테스트 데이터
    test_df = pd.DataFrame({
        'id': range(1, 6),
        'text': [
            "삼성전자 갤럭시는 훌륭한 스마트폰이다.",
            "애플 아이폰의 카메라 성능이 뛰어나다.",
            "LG전자 제품의 디자인이 세련되었다.",
            "삼성전자의 최신 TV는 화질이 좋다.",
            "애플 맥북의 성능과 디자인이 인상적이다."
        ]
    })
    
    # 텍스트 전처리 및 토큰화
    processed_df = preprocessor.preprocess_dataframe(test_df, 'text', extract_nouns=True)
    
    # 벡터라이저 생성 및 학습
    vectorizer = TextVectorizer(vector_size=50, window=3, min_count=1)
    
    # TF-IDF 학습 및 변환
    tfidf_matrix = vectorizer.fit_transform_tfidf(processed_df['tokens'].tolist())
    print(f"TF-IDF 행렬 형태: {tfidf_matrix.shape}")
    
    # Word2Vec 학습
    vectorizer.train_word2vec(processed_df['tokens'].tolist())
    
    # Word2Vec 변환
    word2vec_vectors = vectorizer.transform_texts_to_word2vec(processed_df['tokens'].tolist())
    print(f"Word2Vec 벡터 형태: {word2vec_vectors.shape}")
    
    # 유사 단어 확인
    if '삼성전자' in vectorizer.word2vec_model.wv:
        similar_words = vectorizer.get_most_similar_words('삼성전자')
        print(f"'삼성전자'와 유사한 단어: {similar_words}")
    
    # 모델 저장
    vectorizer.save_models()
