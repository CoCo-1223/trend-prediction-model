"""
빅카인즈 뉴스 데이터 감성분석 전처리 파이프라인
- 조원의 감성분석 코드와 동일한 결과 생성
- 입력: 빅카인즈 엑셀 파일 (NewsResult_20XX0101-20XX1231.xlsx)
- 출력: samsung_sentiment_{year}.csv, apple_sentiment_{year}.csv
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from transformers import pipeline
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class BigKindsSentimentProcessor:
    def __init__(self, data_dir, output_dir="./data/processed"):
        """
        빅카인즈 감성분석 처리기 초기화
        
        Args:
            data_dir (str): 빅카인즈 엑셀 파일들이 있는 디렉토리
            output_dir (str): 결과 CSV 파일을 저장할 디렉토리
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = 32
        self.model_id = "snunlp/KR-FinBert-SC"
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # FinBERT 감성분석 파이프라인 초기화
        print("🤖 FinBERT 모델 로딩 중...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1,  # GPU 사용 시 0으로 변경
            top_k=None  # 모든 라벨 확률 반환
        )
        print("✅ FinBERT 모델 로딩 완료")
        
    def find_excel_files(self):
        """빅카인즈 엑셀 파일들 찾기"""
        pattern = os.path.join(self.data_dir, "NewsResult_*-*.xlsx")
        files = glob.glob(pattern)
        
        if not files:
            print(f"❌ {self.data_dir}에서 빅카인즈 파일을 찾을 수 없습니다.")
            print(f"찾는 패턴: NewsResult_*-*.xlsx")
            return []
        
        # 연도별로 정렬
        files.sort()
        print(f"📁 발견된 파일들:")
        for file in files:
            print(f"   - {os.path.basename(file)}")
        
        return files
    
    def extract_year_from_filename(self, filename):
        """파일명에서 연도 추출"""
        basename = os.path.basename(filename)
        # NewsResult_20230101-20231231.xlsx -> 2023 추출
        try:
            year = basename.split('_')[1][:4]
            return year
        except:
            # 다른 형식의 파일명 처리
            import re
            year_match = re.search(r'20\d{2}', basename)
            if year_match:
                return year_match.group()
            return "unknown"
    
    def load_and_clean_excel(self, file_path):
        """엑셀 파일 로드 및 정제"""
        print(f"\n📊 {os.path.basename(file_path)} 처리 중...")
        
        try:
            # 엑셀 파일 로드
            df = pd.read_excel(file_path)
            print(f"   - 원본 데이터: {len(df)}건")
            
            # 컬럼명 공백 제거
            df.columns = df.columns.str.strip()
            
            print(f"   - 컬럼: {list(df.columns)}")
            
            # 필요한 컬럼 확인 및 매핑
            required_columns = ['일자', '제목', '키워드']
            column_mapping = {}
            
            for col in df.columns:
                col_lower = col.lower()
                if any(date_word in col_lower for date_word in ['일자', '날짜', 'date', '기사일자']):
                    column_mapping[col] = '일자'
                elif any(title_word in col_lower for title_word in ['제목', 'title', '기사제목']):
                    column_mapping[col] = '제목'
                elif any(keyword_word in col_lower for keyword_word in ['키워드', 'keyword', '주요키워드']):
                    column_mapping[col] = '키워드'
                elif any(content_word in col_lower for content_word in ['본문', 'content', '내용']):
                    column_mapping[col] = '본문'
                elif any(media_word in col_lower for media_word in ['언론사', '매체', 'media']):
                    column_mapping[col] = '언론사'
            
            # 컬럼명 변경
            df = df.rename(columns=column_mapping)
            
            # 필수 컬럼 확인
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"   ⚠️ 누락된 필수 컬럼: {missing_columns}")
                return None
            
            # 데이터 타입을 문자열로 변환 (조원 코드와 동일)
            for col in df.columns:
                df[col] = df[col].astype(str)
            
            # 결측값 처리
            df = df.dropna(subset=['일자', '제목'])
            df['키워드'] = df['키워드'].fillna('')
            if '본문' in df.columns:
                df['본문'] = df['본문'].fillna('')
            
            print(f"   - 정제 후 데이터: {len(df)}건")
            
            return df
            
        except Exception as e:
            print(f"   ❌ 파일 로드 실패: {e}")
            return None
    
    def parse_and_format_date(self, df):
        """날짜 파싱 및 포맷팅 (조원 코드와 동일)"""
        print("   - 날짜 처리 중...")
        
        try:
            # 다양한 날짜 형식 처리
            df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
            
            # NaT 값 제거
            before_count = len(df)
            df = df.dropna(subset=["일자"])
            after_count = len(df)
            
            if before_count != after_count:
                print(f"     날짜 오류로 제거된 데이터: {before_count - after_count}건")
            
            # YYYY-MM-DD 형식으로 변환
            df["일자"] = df["일자"].dt.strftime("%Y-%m-%d")
            
            print(f"     날짜 범위: {df['일자'].min()} ~ {df['일자'].max()}")
            
        except Exception as e:
            print(f"     날짜 처리 오류: {e}")
            # 날짜 처리 실패 시 원본 유지
            pass
        
        return df
    
    def create_text_and_classify_company(self, df):
        """텍스트 결합 및 기업 분류 (조원 코드와 동일)"""
        print("   - 텍스트 결합 및 기업 분류 중...")
        
        # 텍스트 결합
        df["text"] = df["제목"].fillna("") + " 키워드:" + df["키워드"].fillna("")
        
        # 본문이 있는 경우 추가
        if "본문" in df.columns:
            df["text"] = df["text"] + " " + df["본문"].fillna("")
        
        # 기업 분류 함수 (조원 코드와 동일)
        def company_of(txt: str) -> str:
            if pd.isna(txt):
                return None
            
            t = str(txt).lower()
            
            # 삼성 키워드
            samsung_keywords = ['삼성전자', '삼성', 'samsung']
            # 애플 키워드  
            apple_keywords = ['애플', 'apple']
            
            samsung_found = any(keyword in t for keyword in samsung_keywords)
            apple_found = any(keyword in t for keyword in apple_keywords)
            
            if samsung_found and not apple_found:
                return "samsung"
            elif apple_found and not samsung_found:
                return "apple"
            elif samsung_found and apple_found:
                # 둘 다 있는 경우 더 많이 언급된 쪽으로
                samsung_count = sum(t.count(keyword) for keyword in samsung_keywords)
                apple_count = sum(t.count(keyword) for keyword in apple_keywords)
                return "samsung" if samsung_count >= apple_count else "apple"
            else:
                return None
        
        # 기업 분류 적용
        df["기업"] = df["text"].apply(company_of)
        
        # 기업이 분류된 데이터만 유지
        before_count = len(df)
        df = df.dropna(subset=["기업"]).reset_index(drop=True)
        after_count = len(df)
        
        samsung_count = len(df[df["기업"] == "samsung"])
        apple_count = len(df[df["기업"] == "apple"])
        
        print(f"     기업 분류 결과: 삼성 {samsung_count}건, 애플 {apple_count}건")
        print(f"     분류되지 않은 데이터 제거: {before_count - after_count}건")
        
        return df
    
    def perform_sentiment_analysis(self, df):
        """감성분석 수행 (조원 코드와 동일)"""
        print("   - 감성분석 수행 중...")
        
        # 감성 점수 변환 함수 (조원 코드와 동일)
        def to_score(probs: dict) -> float:
            neg = probs.get("negative", 0.0)
            neu = probs.get("neutral", 0.0) 
            pos = probs.get("positive", 0.0)
            
            # 중립=3점, 긍정·부정의 차이에 따라 ±2점 범위 매핑
            score = 3.0 + 2.0 * (pos - neg)
            return max(1.0, min(5.0, score))
        
        labels, scores = [], []
        texts = df["text"].tolist()
        
        # 배치별 감성분석
        for i in tqdm(range(0, len(texts), self.batch_size), desc="감성분석"):
            batch = texts[i : i + self.batch_size]
            
            try:
                # 텍스트 길이 제한
                batch = [str(text)[:512] for text in batch]
                
                # 감성분석 수행
                outputs = self.sentiment_pipeline(batch, truncation=True, max_length=512)
                
                for out in outputs:
                    # 결과 처리 (list 또는 dict 형태 모두 처리)
                    if isinstance(out, list):
                        prob_map = {x["label"]: x["score"] for x in out}
                    else:
                        prob_map = out
                    
                    # 최고 확률 라벨
                    best_label = max(prob_map, key=prob_map.get)
                    labels.append(best_label)
                    scores.append(to_score(prob_map))
                    
            except Exception as e:
                print(f"     배치 {i//self.batch_size + 1} 처리 실패: {e}")
                # 실패한 배치는 중립으로 처리
                for _ in range(len(batch)):
                    labels.append("neutral")
                    scores.append(3.0)
        
        df["감정라벨"] = labels
        df["감정점수"] = scores
        
        # 감성분석 결과 요약
        sentiment_dist = df["감정라벨"].value_counts()
        print(f"     감성분석 결과:")
        for sentiment, count in sentiment_dist.items():
            print(f"       {sentiment}: {count}건")
        print(f"     평균 감정점수: {df['감정점수'].mean():.2f}")
        
        return df
    
    def save_company_files(self, df, year):
        """기업별 CSV 파일 저장 (조원 코드와 동일)"""
        print("   - 기업별 파일 저장 중...")
        
        # 날짜순 정렬
        df = df.sort_values("일자")
        
        saved_files = []
        
        # 기업별로 저장
        for company in ["samsung", "apple"]:
            company_data = df[df["기업"] == company][
                ["일자", "제목", "키워드", "감정라벨", "감정점수"]
            ]
            
            if len(company_data) > 0:
                output_path = os.path.join(self.output_dir, f"{company}_sentiment_{year}.csv")
                company_data.to_csv(output_path, index=False, encoding="utf-8-sig")
                
                print(f"     ✅ {os.path.basename(output_path)}: {len(company_data)}건 저장")
                saved_files.append(output_path)
            else:
                print(f"     ⚠️ {company} 데이터 없음")
        
        return saved_files
    
    def process_single_file(self, file_path):
        """단일 파일 처리"""
        year = self.extract_year_from_filename(file_path)
        
        # 1. 엑셀 파일 로드 및 정제
        df = self.load_and_clean_excel(file_path)
        if df is None:
            return []
        
        # 2. 날짜 처리
        df = self.parse_and_format_date(df)
        
        # 3. 텍스트 결합 및 기업 분류
        df = self.create_text_and_classify_company(df)
        
        if len(df) == 0:
            print("   ⚠️ 분류된 기업 데이터가 없습니다.")
            return []
        
        # 4. 감성분석
        df = self.perform_sentiment_analysis(df)
        
        # 5. 결과 저장
        saved_files = self.save_company_files(df, year)
        
        return saved_files
    
    def process_all_files(self):
        """모든 빅카인즈 파일 처리"""
        print("🚀 빅카인즈 뉴스 감성분석 처리 시작")
        print("=" * 60)
        
        # 파일 찾기
        excel_files = self.find_excel_files()
        if not excel_files:
            return []
        
        all_saved_files = []
        
        # 각 파일 처리
        for file_path in excel_files:
            try:
                saved_files = self.process_single_file(file_path)
                all_saved_files.extend(saved_files)
                
            except Exception as e:
                print(f"❌ {os.path.basename(file_path)} 처리 실패: {e}")
                continue
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("🎉 전체 처리 완료!")
        print("=" * 60)
        
        if all_saved_files:
            print(f"📁 생성된 파일 ({len(all_saved_files)}개):")
            for file_path in all_saved_files:
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   - {os.path.basename(file_path)} ({file_size:.1f} KB)")
            
            print(f"\n📂 저장 위치: {os.path.abspath(self.output_dir)}")
            
            # 데이터 통계
            print(f"\n📊 처리 결과 요약:")
            samsung_files = [f for f in all_saved_files if 'samsung' in f]
            apple_files = [f for f in all_saved_files if 'apple' in f]
            
            print(f"   - 삼성 데이터: {len(samsung_files)}개 연도")
            print(f"   - 애플 데이터: {len(apple_files)}개 연도")
            
            # 각 파일의 데이터 수 확인
            total_records = 0
            for file_path in all_saved_files:
                try:
                    df = pd.read_csv(file_path)
                    records = len(df)
                    total_records += records
                    company = "삼성" if "samsung" in file_path else "애플"
                    year = file_path.split("_")[-1].replace(".csv", "")
                    print(f"     {company} {year}: {records:,}건")
                except:
                    pass
            
            print(f"   - 총 감성분석 레코드: {total_records:,}건")
            
        else:
            print("❌ 생성된 파일이 없습니다.")
        
        return all_saved_files
    
    def verify_output_format(self):
        """출력 파일 형식 검증"""
        print("\n🔍 출력 파일 형식 검증 중...")
        
        output_files = glob.glob(os.path.join(self.output_dir, "*_sentiment_*.csv"))
        
        if not output_files:
            print("❌ 검증할 파일이 없습니다.")
            return False
        
        all_valid = True
        required_columns = ['일자', '제목', '키워드', '감정라벨', '감정점수']
        
        for file_path in output_files:
            try:
                df = pd.read_csv(file_path)
                
                # 컬럼 확인
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"❌ {os.path.basename(file_path)}: 누락 컬럼 {missing_cols}")
                    all_valid = False
                    continue
                
                # 데이터 타입 확인
                if not pd.api.types.is_numeric_dtype(df['감정점수']):
                    print(f"❌ {os.path.basename(file_path)}: 감정점수가 숫자가 아님")
                    all_valid = False
                    continue
                
                # 감정점수 범위 확인
                if df['감정점수'].min() < 1 or df['감정점수'].max() > 5:
                    print(f"❌ {os.path.basename(file_path)}: 감정점수 범위 오류")
                    all_valid = False
                    continue
                
                print(f"✅ {os.path.basename(file_path)}: 형식 올바름 ({len(df)}건)")
                
            except Exception as e:
                print(f"❌ {os.path.basename(file_path)}: 검증 실패 - {e}")
                all_valid = False
        
        if all_valid:
            print("🎉 모든 파일의 형식이 올바릅니다!")
        else:
            print("⚠️ 일부 파일에 문제가 있습니다.")
        
        return all_valid


def main():
    """메인 실행 함수"""
    # 데이터 경로 설정 (유니코드 오류 방지)
    data_dir = "C:/Users/jmzxc/OneDrive/바탕 화면/빅카인즈"
    output_dir = "./data/processed"
    
    try:
        print("📋 빅카인즈 뉴스 감성분석 전처리 시작")
        print(f"📁 입력 경로: {data_dir}")
        print(f"📁 출력 경로: {output_dir}")
        
        # 처리기 초기화
        processor = BigKindsSentimentProcessor(data_dir, output_dir)
        
        # 모든 파일 처리
        saved_files = processor.process_all_files()
        
        if saved_files:
            # 출력 형식 검증
            processor.verify_output_format()
            
            print(f"\n🔗 다음 단계:")
            print(f"1. 생성된 CSV 파일들이 ./data/processed/ 폴더에 있는지 확인")
            print(f"2. LSTM 딥러닝 코드 실행:")
            print(f"   python sentiment_deeplearning_analysis.py")
            
            return True
        else:
            print("❌ 처리 실패")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        return False
        
    except Exception as e:
        print(f"❌ 예기치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*50)
        print("🎊 감성분석 전처리 완료!")
        print("🚀 이제 LSTM 딥러닝 분석을 시작할 수 있습니다!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("💥 처리 중 문제가 발생했습니다.")
        print("📝 확인사항:")
        print("1. 빅카인즈 엑셀 파일 경로가 올바른지 확인")
        print("2. 필요한 라이브러리가 설치되었는지 확인")
        print("3. 인터넷 연결 상태 확인 (FinBERT 모델 다운로드)")
        print("="*50)


"""
📋 사용법:

1. 필요한 라이브러리 설치:
   pip install pandas numpy transformers torch tqdm openpyxl

2. 빅카인즈 파일 준비:
   - C:/Users/jmzxc/OneDrive/바탕 화면/빅카인즈/NewsResult_20210101-20211231.xlsx
   - C:/Users/jmzxc/OneDrive/바탕 화면/빅카인즈/NewsResult_20220101-20221231.xlsx  
   - C:/Users/jmzxc/OneDrive/바탕 화면/빅카인즈/NewsResult_20230101-20231231.xlsx
   - C:/Users/jmzxc/OneDrive/바탕 화면/빅카인즈/NewsResult_20240101-20241231.xlsx

3. 코드 실행:
   python bigkinds_sentiment_processor.py

4. 결과 확인:
   ./data/processed/ 폴더에 다음 파일들이 생성됩니다:
   - samsung_sentiment_2021.csv
   - samsung_sentiment_2022.csv
   - samsung_sentiment_2023.csv  
   - samsung_sentiment_2024.csv
   - apple_sentiment_2021.csv
   - apple_sentiment_2022.csv
   - apple_sentiment_2023.csv
   - apple_sentiment_2024.csv

🎯 조원 코드와의 완벽한 호환성:
✅ 동일한 FinBERT 모델 (snunlp/KR-FinBert-SC)
✅ 동일한 감성점수 변환 공식
✅ 동일한 기업 분류 로직
✅ 동일한 CSV 출력 형식
✅ 동일한 컬럼명 및 데이터 구조

🚀 다음 단계:
이 코드로 감성분석을 완료한 후, 
sentiment_deeplearning_analysis.py 코드로 LSTM 시계열 분석을 진행하세요!
"""