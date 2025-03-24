# src/features/text_preprocessing.py
import re
import pandas as pd
import numpy as np
import json
from konlpy.tag import Okt, Mecab
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, config_path='config.json'):
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 한국어 형태소 분석기 초기화
        try:
            self.mecab = Mecab()
            self.use_mecab = True
        except:
            print("Mecab 초기화 실패, Okt를 사용합니다.")
            self.okt = Okt()
            self.use_mecab = False
        
        # 불용어 사전 로드
        self.stopwords_ko = self._load_korean_stopwords()
        self.stopwords_en = set(stopwords.words('english'))
        
        # 이모지와 특수문자 패턴
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
    def _load_korean_stopwords(self):
        """한국어 불용어 사전 로드"""
        # 기본 한국어 불용어 목록
        default_stopwords = {
            '이', '그', '저', '것', '이것', '저것', '그것', '이번', '저번', '그번',
            '이거', '저거', '그거', '여기', '저기', '거기', '이런', '저런', '그런',
            '하다', '되다', '있다', '없다', '같다', '이다', '아니다', '때문', '말',
            '통해', '지금', '오늘', '내일', '어제', '매우', '정말', '아주', '너무'
        }
        
        try:
            # 외부 불용어 파일이 있으면 로드
            with open('data/external/korean_stopwords.txt', 'r', encoding='utf-8') as f:
                custom_stopwords = set(line.strip() for line in f.readlines())
            return default_stopwords.union(custom_stopwords)
        except:
            return default_stopwords
    
    def clean_text(self, text):
        """텍스트 기본 정제 함수"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # URL 제거
        text = self.url_pattern.sub('', text)
        
        # HTML 태그 제거
        text = self.html_pattern.sub('', text)
        
        # 이모지 제거
        text = emoji.replace_emoji(text, replace='')
        
        # 멘션 제거
        text = self.mention_pattern.sub('', text)
        
        # 해시태그 제외한 텍스트만 유지 (해시태그 기호 제거)
        text = self.hashtag_pattern.sub(lambda x: x.group(0)[1:], text)
        
        # 특수문자 제거 및 공백 정리
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text):
        """텍스트 언어 감지"""
        # 간단한 언어 감지 로직 (한글/영어)
        if not text:
            return 'unknown'
        
        # 한글 문자 비율 계산
        korean_chars = sum(1 for char in text if ord('가') <= ord(char) <= ord('힣'))
        korean_ratio = korean_chars / len(text)
        
        if korean_ratio > 0.3:
            return 'ko'
        else:
            # 영어 문자 비율 계산
            english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
            english_ratio = english_chars / len(text)
            
            if english_ratio > 0.3:
                return 'en'
            else:
                return 'unknown'
    
    def tokenize_text(self, text, language=None):
        """텍스트 토큰화"""
        if not text:
            return []
        
        # 언어 감지
        if language is None:
            language = self.detect_language(text)
        
        # 언어별 토큰화
        if language == 'ko':
            # 한국어 토큰화
            if self.use_mecab:
                tokens = self.mecab.morphs(text)
            else:
                tokens = self.okt.morphs(text, stem=True)
            
            # 불용어 제거
            tokens = [token for token in tokens if token not in self.stopwords_ko and len(token) > 1]
        
        elif language == 'en':
            # 영어 토큰화
            tokens = word_tokenize(text.lower())
            
            # 불용어 제거
            tokens = [token for token in tokens if token not in self.stopwords_en and len(token) > 1]
        
        else:
            # 기본 공백 기준 토큰화
            tokens = text.split()
        
        return tokens
    
    def pos_tagging(self, text, language=None):
        """품사 태깅"""
        if not text:
            return []
        
        # 언어 감지
        if language is None:
            language = self.detect_language(text)
        
        # 언어별 품사 태깅
        if language == 'ko':
            # 한국어 품사 태깅
            if self.use_mecab:
                tagged = self.mecab.pos(text)
            else:
                tagged = self.okt.pos(text)
        
        elif language == 'en':
            # 영어 품사 태깅
            nltk.download('averaged_perceptron_tagger')
            tokens = word_tokenize(text.lower())
            tagged = nltk.pos_tag(tokens)
        
        else:
            tagged = []
        
        return tagged
    
    def extract_nouns(self, text, language=None):
        """명사 추출"""
        if not text:
            return []
        
        # 언어 감지
        if language is None:
            language = self.detect_language(text)
        
        # 언어별 명사 추출
        if language == 'ko':
            # 한국어 명사 추출
            if self.use_mecab:
                nouns = self.mecab.nouns(text)
            else:
                nouns = self.okt.nouns(text)
            
            # 불용어 제거 및 길이 필터링
            nouns = [noun for noun in nouns if noun not in self.stopwords_ko and len(noun) > 1]
        
        elif language == 'en':
            # 영어 명사 추출 (품사 태깅 후 명사만 필터링)
            nltk.download('averaged_perceptron_tagger')
            tokens = word_tokenize(text.lower())
            tagged = nltk.pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stopwords_en and len(word) > 1]
        
        else:
            nouns = []
        
        return nouns
    
    def preprocess_dataframe(self, df, text_column, new_column=None, extract_nouns=False):
        """데이터프레임 전처리"""
        if text_column not in df.columns:
            raise ValueError(f"열 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
        
        if new_column is None:
            new_column = f"cleaned_{text_column}"
        
        # 텍스트 정제
        df[new_column] = df[text_column].apply(self.clean_text)
        
        # 언어 감지
        df['detected_language'] = df[new_column].apply(self.detect_language)
        
        # 토큰화 또는 명사 추출
        if extract_nouns:
            df['tokens'] = df.apply(lambda row: self.extract_nouns(row[new_column], row['detected_language']), axis=1)
        else:
            df['tokens'] = df.apply(lambda row: self.tokenize_text(row[new_column], row['detected_language']), axis=1)
        
        return df

# 실행 예시
if __name__ == "__main__":
    # 텍스트 전처리기 생성
    preprocessor = TextPreprocessor()
    
    # 테스트 데이터
    test_df = pd.DataFrame({
        'id': [1, 2, 3],
        'text': [
            '안녕하세요! #인공지능 기술에 대한 이야기를 해보겠습니다. https://example.com',
            'Hello world! This is an #AI example. @user https://ai.com',
            '삼성전자 갤럭시 S22는 정말 좋은 스마트폰입니다! 카메라 성능이 뛰어나요. #갤럭시 #삼성'
        ]
    })
    
    # 데이터프레임 전처리
    processed_df = preprocessor.preprocess_dataframe(test_df, 'text', extract_nouns=True)
    
    # 결과 확인
    print(processed_df[['text', 'cleaned_text', 'detected_language', 'tokens']])
