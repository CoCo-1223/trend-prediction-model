import pandas as pd
import numpy as np

def check_data_structure():
    """데이터 구조 확인 함수"""
    
    base_path = r"C:\Users\jmzxc\OneDrive\바탕 화면\빅카인즈\data\processed"
    
    # 2021년 삼성 데이터 확인
    file_path = f"{base_path}/samsung_sentiment_2021.csv"
    
    print("🔍 데이터 구조 분석 시작...")
    print("=" * 60)
    
    # 여러 인코딩으로 시도
    encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            print(f"\n📂 {encoding} 인코딩으로 시도 중...")
            
            # 처음 5행만 읽기
            df_sample = pd.read_csv(file_path, nrows=5, encoding=encoding)
            
            print(f"✅ {encoding} 인코딩 성공!")
            print(f"📊 데이터 형태: {df_sample.shape}")
            print(f"📋 컬럼 목록 ({len(df_sample.columns)}개):")
            
            for i, col in enumerate(df_sample.columns):
                print(f"   {i:2d}: '{col}'")
            
            print(f"\n🔍 샘플 데이터:")
            print(df_sample.head())
            
            print(f"\n📈 데이터 타입:")
            print(df_sample.dtypes)
            
            # 숫자형 컬럼 찾기
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            print(f"\n🔢 숫자형 컬럼들:")
            for col in numeric_cols:
                sample_values = df_sample[col].dropna().head(3).tolist()
                print(f"   - {col}: {sample_values}")
            
            # 텍스트 컬럼 찾기
            text_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
            print(f"\n📝 텍스트 컬럼들:")
            for col in text_cols:
                sample_value = str(df_sample[col].iloc[0])[:50] if len(df_sample) > 0 else "N/A"
                print(f"   - {col}: {sample_value}...")
            
            # 성공하면 루프 종료
            return df_sample, encoding
            
        except UnicodeDecodeError as e:
            print(f"❌ {encoding} 인코딩 실패: {e}")
            continue
        except Exception as e:
            print(f"❌ {encoding} 인코딩으로 파일 읽기 실패: {e}")
            continue
    
    # 모든 인코딩 실패
    print("❌ 모든 인코딩 방식 실패")
    return None, None

def suggest_column_mapping(df):
    """컬럼 매핑 제안"""
    if df is None:
        return
    
    print("\n💡 컬럼 매핑 제안:")
    print("=" * 40)
    
    # 날짜 컬럼 추정
    date_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', '날짜', '일자', '시간', 'time']):
            date_candidates.append(col)
    
    print(f"📅 날짜 컬럼 후보: {date_candidates}")
    
    # 감성/점수 컬럼 추정
    sentiment_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['sentiment', '감성', '감정', 'score', '점수', 'polarity']):
            sentiment_candidates.append(col)
    
    print(f"😊 감성 점수 컬럼 후보: {sentiment_candidates}")
    
    # 제목/내용 컬럼 추정
    content_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['title', '제목', 'content', '내용', 'text', '본문']):
            content_candidates.append(col)
    
    print(f"📰 뉴스 내용 컬럼 후보: {content_candidates}")
    
    # 숫자형 컬럼 중 감성 점수 가능성
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"🔢 숫자형 컬럼들 (감성 점수 가능성): {numeric_cols}")
    
    if numeric_cols:
        print("\n🎯 숫자형 컬럼별 값 범위:")
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   - {col}: {values.min():.3f} ~ {values.max():.3f} (평균: {values.mean():.3f})")

if __name__ == "__main__":
    # 데이터 구조 확인
    df_sample, successful_encoding = check_data_structure()
    
    # 컬럼 매핑 제안
    suggest_column_mapping(df_sample)
    
    if successful_encoding:
        print(f"\n✅ 권장 인코딩: {successful_encoding}")
        print("💻 다음 단계: 위 정보를 바탕으로 LSTM 코드의 컬럼명을 수정하세요.")
    else:
        print("\n❌ 데이터 파일을 읽을 수 없습니다. 파일 경로와 형식을 확인해주세요.")