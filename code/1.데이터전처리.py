import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_news_data(file_path):
    """
    빅카인즈에서 다운로드한 뉴스 데이터를 전처리하는 함수
    
    Args:
        file_path (str): 뉴스 데이터 Excel 파일 경로
    
    Returns:
        pd.DataFrame: 전처리된 뉴스 데이터
    """
    print(f"파일 로딩 중: {file_path}")
    
    # Excel 파일 로드
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return pd.DataFrame()
    
    print(f"원본 데이터 크기: {df.shape}")
    
    # 필요한 컬럼만 선택
    # 필요한 컬럼: '뉴스 식별자', '일자', '언론사', '제목', '본문', 'URL', '키워드'
    necessary_columns = ['뉴스 식별자', '일자', '언론사', '제목', '본문', 'URL', '키워드']
    
    # 컬럼명이 다를 수 있으므로 매핑
    columns_mapping = {
        '뉴스 식별자': ['뉴스 식별자', '뉴스식별자', 'news_id', 'ID'],
        '일자': ['일자', '날짜', 'date', '일시'],
        '언론사': ['언론사', '신문사', 'media', '출처'],
        '제목': ['제목', 'title'],
        '본문': ['본문', 'content'],
        'URL': ['URL', 'url'],
        '키워드': ['키워드', 'keyword']
    }
    
    # 컬럼명 확인 출력
    print(f"파일의 컬럼: {df.columns.tolist()}")
    
    selected_columns = []
    renamed_columns = {}
    
    for target_col, possible_names in columns_mapping.items():
        for col_name in possible_names:
            if col_name in df.columns:
                selected_columns.append(col_name)
                renamed_columns[col_name] = target_col
                break
    
    # 필요한 컬럼이 모두 있는지 확인
    if len(selected_columns) < len(necessary_columns):
        print(f"경고: 일부 필요한 컬럼을 찾을 수 없습니다. 찾은 컬럼: {selected_columns}")
    
    # 필요한 컬럼만 선택하고 컬럼명 통일
    if selected_columns:
        df = df[selected_columns].rename(columns=renamed_columns)
    
    # 1. 중복 기사 제거
    # 1-1. 완전히 동일한 제목을 가진 기사 제거
    duplicate_titles = df[df.duplicated(subset=['제목'], keep='first')]['제목'].unique()
    print(f"동일한 제목을 가진 중복 기사 수: {len(duplicate_titles)}")
    df = df.drop_duplicates(subset=['제목'], keep='first')
    
    # 1-2. 유사한 제목을 가진 기사 확인 (코사인 유사도 사용)
    titles = df['제목'].tolist()
    
    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 유사도가 높은 기사 쌍 찾기 (임계값 0.8)
    similar_indices = []
    for i in range(len(cosine_sim)):
        for j in range(i+1, len(cosine_sim)):
            if cosine_sim[i][j] > 0.8:  # 유사도 임계값 (조정 가능)
                # 두 기사의 날짜를 비교해서 더 최신 기사를 유지
                if df.iloc[i]['일자'] >= df.iloc[j]['일자']:
                    similar_indices.append(j)
                else:
                    similar_indices.append(i)
    
    similar_indices = list(set(similar_indices))  # 중복 제거
    print(f"유사한 제목을 가진 중복 기사 수: {len(similar_indices)}")
    
    # 유사한 기사 제거
    df = df.drop(similar_indices).reset_index(drop=True)
    
    # 2. 본문이 너무 짧은 기사 제거 (의미 있는 내용이 없는 경우)
    min_content_length = 200  # 최소 본문 길이 (조정 가능)
    
    # None 값 처리
    df['본문'] = df['본문'].fillna('')
    
    short_articles = df[df['본문'].str.len() < min_content_length]
    print(f"너무 짧은 본문을 가진 기사 수: {len(short_articles)}")
    df = df[df['본문'].str.len() >= min_content_length]
    
    # 3. 삼성전자나 애플과 관련 없는 기사 제거 (프로젝트 목적에 맞게)
    # 키워드 및 본문 내용에서 관련 키워드 검색
    relevant_keywords = ['삼성전자', '삼성', '갤럭시', '스마트폰', '반도체', 'DRAM', 'NAND', '디스플레이',
                        '애플', '아이폰', '아이패드', '맥북', '애플워치', '에어팟']
    
    # None 값 처리
    df['키워드'] = df['키워드'].fillna('')
    
    # 키워드 컬럼과 본문에서 관련 키워드 체크
    has_relevant_keyword = df.apply(
        lambda row: any(keyword in str(row['키워드']) or keyword in str(row['본문']) or keyword in str(row['제목']) 
                       for keyword in relevant_keywords),
        axis=1
    )
    
    irrelevant_articles = df[~has_relevant_keyword]
    print(f"관련 없는 기사 수: {len(irrelevant_articles)}")
    df = df[has_relevant_keyword]
    
    # 4. 날짜 형식 통일 (YYYYMMDD -> YYYY-MM-DD)
    def standardize_date(date_val):
        if pd.isna(date_val):
            return None
            
        date_str = str(date_val).split()[0]  # 시간 부분 제거하고 날짜만 사용
        
        # 이미 표준 형식이면 그대로 반환
        if '-' in date_str:
            return date_str
        
        date_str = date_str.strip()
        if len(date_str) == 8:  # YYYYMMDD 형식인 경우
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str
    
    df['일자'] = df['일자'].apply(standardize_date)
    
    # 5. 필요 없는 특수문자 및 HTML 태그 제거
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        # 특수문자 정리 (일부 특수문자만 제거)
        text = re.sub(r'[\n\r\t]', ' ', text)
        # 여러 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    df['본문'] = df['본문'].apply(clean_text)
    df['제목'] = df['제목'].apply(clean_text)
    
    # 6. 결과 저장
    # 원본 파일명 추출
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.join(os.path.dirname(file_path), "processed_data")
    
    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{file_name_without_ext}_processed.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 한글 깨짐 방지
    
    print(f"전처리 완료. 결과 파일: {output_path}")
    print(f"전처리 후 데이터 크기: {df.shape}")
    
    return df

def batch_preprocess_files(file_paths):
    """
    여러 파일을 일괄 처리하는 함수
    
    Args:
        file_paths (list): 처리할 파일 경로 리스트
    
    Returns:
        pd.DataFrame: 모든 파일을 합친 전처리된 데이터프레임
    """
    all_data = []
    
    for file_path in file_paths:
        processed_df = preprocess_news_data(file_path)
        if not processed_df.empty:
            all_data.append(processed_df)
    
    if not all_data:
        print("처리된 데이터가 없습니다.")
        return pd.DataFrame()
        
    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 중복 제거 한번 더 (파일 간 중복 제거)
    combined_df = combined_df.drop_duplicates(subset=['제목'], keep='first')
    
    # 날짜순 정렬
    combined_df['일자'] = pd.to_datetime(combined_df['일자'], errors='coerce')
    combined_df = combined_df.sort_values('일자')
    
    # 결과 저장 디렉토리 지정
    base_dir = os.path.dirname(file_paths[0])
    output_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 최종 결과 저장
    output_path = os.path.join(output_dir, 'all_news_processed.csv')
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 한글 깨짐 방지
    print(f"모든 파일 병합 완료. 최종 데이터 크기: {combined_df.shape}")
    
    return combined_df

# 메모리 사용량 모니터링 함수 추가
def print_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        print(f"현재 메모리 사용량: {memory_usage_mb:.2f} MB")
    except ImportError:
        print("psutil 라이브러리가 설치되어 있지 않습니다. pip install psutil로 설치하세요.")

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정 (Windows 경로)
    base_dir = r"C:\Users\jmzxc\OneDrive\바탕 화면\빅카인즈"
    
    file_paths = [
        os.path.join(base_dir, "NewsResult_20210101-20211231.xlsx"),
        os.path.join(base_dir, "NewsResult_20220101-20221231.xlsx"),
        os.path.join(base_dir, "NewsResult_20230101-20231231.xlsx"),
        os.path.join(base_dir, "NewsResult_20240101-20241231.xlsx")
    ]
    
    # 각 파일의 존재 여부 확인
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"파일이 존재합니다: {file_path}")
        else:
            print(f"경고: 파일이 존재하지 않습니다: {file_path}")
    
    # 메모리 사용량 출력
    print_memory_usage()
    
    # 여러 파일 일괄 처리
    batch_preprocess_files(file_paths)
    
    # 처리 후 메모리 사용량 출력
    print_memory_usage()