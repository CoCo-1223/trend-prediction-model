"""
뉴스 감성 분석 기반 주가 예측 모델 - 제품 출시 일정 정리
생성일: 2025-06-08
팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)
목적: Excel 형태의 제품 출시 데이터를 체계적으로 정리하고 표준화
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import openpyxl
warnings.filterwarnings('ignore')

# 결과물 저장 경로 설정
RESULTS_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model/results/2025-0608"
PROJECT_BASE = "/Users/jm/Desktop/충북대학교/충대 4학년 1학기/2. 빅데이터이해와분석/팀프로젝트/trend-prediction-model"

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_directories():
    """결과물 디렉토리 구조 생성"""
    directories = [
        f"{RESULTS_BASE}/visualizations/weekly_analysis",
        f"{RESULTS_BASE}/visualizations/impact_analysis", 
        f"{RESULTS_BASE}/visualizations/model_performance",
        f"{RESULTS_BASE}/visualizations/final_insights",
        f"{RESULTS_BASE}/data/processed",
        f"{RESULTS_BASE}/data/features",
        f"{RESULTS_BASE}/data/predictions", 
        f"{RESULTS_BASE}/data/exports",
        f"{RESULTS_BASE}/models/trained",
        f"{RESULTS_BASE}/models/evaluation",
        f"{RESULTS_BASE}/models/features_importance",
        f"{RESULTS_BASE}/models/predictions",
        f"{RESULTS_BASE}/reports/technical",
        f"{RESULTS_BASE}/reports/business",
        f"{RESULTS_BASE}/reports/methodology",
        f"{RESULTS_BASE}/reports/final"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 결과물 디렉토리 구조 생성 완료: {RESULTS_BASE}")

# 실행 시작 시 디렉토리 자동 생성
setup_directories()

class ProductLaunchDataProcessor:
    def __init__(self):
        self.project_base = PROJECT_BASE
        self.results_base = RESULTS_BASE
        self.apple_data = None
        self.samsung_data = None
        self.combined_data = None
        
        print("🍎📱 제품 출시 일정 정리 시스템 초기화 완료")
        
    def safe_read_excel(self, file_path, sheet_name=None):
        """안전한 Excel 파일 읽기"""
        try:
            print(f"📂 Excel 파일 읽는 중: {file_path}")
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                # 첫 번째 시트 읽기
                df = pd.read_excel(file_path)
            print(f"✅ 성공적으로 로딩: {len(df)}개 행")
            return df
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Excel 파일 읽기 오류: {str(e)}")
            return pd.DataFrame()
    
    def parse_dates_safely(self, date_series):
        """다양한 날짜 형식을 안전하게 파싱"""
        print("📅 날짜 데이터 파싱 중...")
        
        # 이미 datetime인 경우
        if pd.api.types.is_datetime64_any_dtype(date_series):
            return date_series
            
        try:
            # 기본 파싱 시도
            return pd.to_datetime(date_series, infer_datetime_format=True)
        except:
            # 여러 형식 시도
            for fmt in ['%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d', '%Y년 %m월 %d일']:
                try:
                    return pd.to_datetime(date_series, format=fmt)
                except:
                    continue
            
            # 문자열 전처리 후 재시도
            if date_series.dtype == 'object':
                cleaned_dates = date_series.astype(str).str.replace('년|월|일', '-', regex=True)
                try:
                    return pd.to_datetime(cleaned_dates, errors='coerce')
                except:
                    pass
            
            # 최후의 수단: 강제 변환
            return pd.to_datetime(date_series, errors='coerce')
    
    def load_excel_data(self):
        """Excel 제품 출시 데이터 로딩"""
        print("\n🔍 제품 출시 Excel 데이터 로딩 시작...")
        
        # 애플 데이터 로딩
        apple_path = f"{self.project_base}/product/apple.xlsx"
        apple_raw = self.safe_read_excel(apple_path)
        
        if not apple_raw.empty:
            print(f"🍎 Apple 데이터 로딩 완료: {len(apple_raw)}개 제품")
            print(f"   컬럼: {list(apple_raw.columns)}")
            self.apple_data = apple_raw
        else:
            print("❌ Apple 데이터 로딩 실패")
        
        # 삼성 데이터 로딩
        samsung_path = f"{self.project_base}/product/samsung.xlsx"
        samsung_raw = self.safe_read_excel(samsung_path)
        
        if not samsung_raw.empty:
            print(f"📱 Samsung 데이터 로딩 완료: {len(samsung_raw)}개 제품")
            print(f"   컬럼: {list(samsung_raw.columns)}")
            self.samsung_data = samsung_raw
        else:
            print("❌ Samsung 데이터 로딩 실패")
    
    def standardize_apple_data(self):
        """Apple 데이터 표준화"""
        if self.apple_data is None or self.apple_data.empty:
            print("❌ Apple 데이터가 없어 표준화를 건너뜁니다.")
            return pd.DataFrame()
        
        print("\n🍎 Apple 데이터 표준화 시작...")
        df = self.apple_data.copy()
        
        # 컬럼명 표준화 시도
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if '날짜' in col_lower or 'date' in col_lower or '출시' in col_lower:
                column_mapping[col] = 'Date'
            elif '제품' in col_lower or 'product' in col_lower or '모델' in col_lower:
                column_mapping[col] = 'Product'
            elif '카테고리' in col_lower or 'category' in col_lower or '분류' in col_lower:
                column_mapping[col] = 'Category'
            elif '타입' in col_lower or 'type' in col_lower or '종류' in col_lower:
                column_mapping[col] = 'Type'
        
        df = df.rename(columns=column_mapping)
        
        # 필수 컬럼이 없는 경우 생성
        if 'Date' not in df.columns:
            if len(df.columns) > 0:
                df['Date'] = df.iloc[:, 0]  # 첫 번째 컬럼을 날짜로 가정
        
        if 'Product' not in df.columns:
            if len(df.columns) > 1:
                df['Product'] = df.iloc[:, 1]  # 두 번째 컬럼을 제품명으로 가정
        
        # 표준 형식으로 변환
        standardized = pd.DataFrame({
            'Date': self.parse_dates_safely(df.get('Date', pd.Series())),
            'Product': df.get('Product', 'Unknown Product'),
            'Company': 'Apple',
            'Category': df.get('Category', 'Unknown'),
            'Type': df.get('Type', 'Unknown')
        })
        
        # 2021-2024년 데이터만 필터링
        standardized = standardized[
            (standardized['Date'].dt.year >= 2021) & 
            (standardized['Date'].dt.year <= 2024)
        ]
        
        # 결측값 제거
        standardized = standardized.dropna(subset=['Date'])
        
        print(f"✅ Apple 데이터 표준화 완료: {len(standardized)}개 제품 (2021-2024)")
        return standardized.sort_values('Date').reset_index(drop=True)
    
    def standardize_samsung_data(self):
        """Samsung 데이터 표준화"""
        if self.samsung_data is None or self.samsung_data.empty:
            print("❌ Samsung 데이터가 없어 표준화를 건너뜁니다.")
            return pd.DataFrame()
        
        print("\n📱 Samsung 데이터 표준화 시작...")
        df = self.samsung_data.copy()
        
        # 컬럼명 표준화 시도
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if '날짜' in col_lower or 'date' in col_lower or '출시' in col_lower:
                column_mapping[col] = 'Date'
            elif '제품' in col_lower or 'product' in col_lower or '모델' in col_lower:
                column_mapping[col] = 'Product'
            elif '카테고리' in col_lower or 'category' in col_lower or '분류' in col_lower:
                column_mapping[col] = 'Category'
            elif '타입' in col_lower or 'type' in col_lower or '종류' in col_lower:
                column_mapping[col] = 'Type'
        
        df = df.rename(columns=column_mapping)
        
        # 필수 컬럼이 없는 경우 생성
        if 'Date' not in df.columns:
            if len(df.columns) > 0:
                df['Date'] = df.iloc[:, 0]  # 첫 번째 컬럼을 날짜로 가정
        
        if 'Product' not in df.columns:
            if len(df.columns) > 1:
                df['Product'] = df.iloc[:, 1]  # 두 번째 컬럼을 제품명으로 가정
        
        # 표준 형식으로 변환
        standardized = pd.DataFrame({
            'Date': self.parse_dates_safely(df.get('Date', pd.Series())),
            'Product': df.get('Product', 'Unknown Product'),
            'Company': 'Samsung',
            'Category': df.get('Category', 'Unknown'),
            'Type': df.get('Type', 'Unknown')
        })
        
        # 2021-2024년 데이터만 필터링
        standardized = standardized[
            (standardized['Date'].dt.year >= 2021) & 
            (standardized['Date'].dt.year <= 2024)
        ]
        
        # 결측값 제거
        standardized = standardized.dropna(subset=['Date'])
        
        print(f"✅ Samsung 데이터 표준화 완료: {len(standardized)}개 제품 (2021-2024)")
        return standardized.sort_values('Date').reset_index(drop=True)
    
    def validate_data_quality(self, df, company_name):
        """데이터 품질 검증"""
        print(f"\n🔍 {company_name} 데이터 품질 검증...")
        
        issues = []
        
        # 연도 분포 확인
        if 'Date' in df.columns and not df.empty:
            year_counts = df['Date'].dt.year.value_counts().sort_index()
            print(f"   연도별 제품 수: {dict(year_counts)}")
            
            # 2021-2024년 범위 확인
            valid_years = set(range(2021, 2025))
            actual_years = set(year_counts.index)
            missing_years = valid_years - actual_years
            if missing_years:
                issues.append(f"Missing years: {missing_years}")
        
        # 결측값 확인
        missing_ratio = df.isnull().sum() / len(df) if not df.empty else pd.Series()
        high_missing = missing_ratio[missing_ratio > 0.1]
        if not high_missing.empty:
            issues.append(f"High missing values in: {list(high_missing.index)}")
        
        # 중복 확인
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        if issues:
            print(f"⚠️  발견된 문제점:")
            for issue in issues:
                print(f"     - {issue}")
        else:
            print(f"✅ 데이터 품질 양호")
        
        return issues
    
    def create_fallback_data(self):
        """Excel 파일을 읽을 수 없는 경우 기본 제품 출시 데이터 생성"""
        print("\n🔧 기본 제품 출시 데이터 생성 중...")
        
        # Apple 기본 데이터
        apple_fallback = [
            {"Date": "2021-04-20", "Product": "iPad Pro 5th Gen", "Category": "Tablet", "Type": "Professional"},
            {"Date": "2021-09-14", "Product": "iPhone 13 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2021-10-18", "Product": "MacBook Pro M1 Pro/Max", "Category": "Laptop", "Type": "Professional"},
            {"Date": "2022-03-08", "Product": "Mac Studio", "Category": "Desktop", "Type": "Professional"},
            {"Date": "2022-09-16", "Product": "iPhone 14 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2022-10-24", "Product": "iPad Pro 6th Gen", "Category": "Tablet", "Type": "Professional"},
            {"Date": "2023-06-05", "Product": "Mac Pro M2 Ultra", "Category": "Desktop", "Type": "Professional"},
            {"Date": "2023-09-22", "Product": "iPhone 15 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2024-05-07", "Product": "iPad Pro M4", "Category": "Tablet", "Type": "Professional"},
            {"Date": "2024-09-20", "Product": "iPhone 16 Series", "Category": "Smartphone", "Type": "Flagship"}
        ]
        
        # Samsung 기본 데이터
        samsung_fallback = [
            {"Date": "2021-01-14", "Product": "Galaxy S21 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2021-08-11", "Product": "Galaxy Z Fold3/Flip3", "Category": "Smartphone", "Type": "Foldable"},
            {"Date": "2021-10-20", "Product": "Galaxy Tab S8 Series", "Category": "Tablet", "Type": "Professional"},
            {"Date": "2022-02-09", "Product": "Galaxy S22 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2022-08-10", "Product": "Galaxy Z Fold4/Flip4", "Category": "Smartphone", "Type": "Foldable"},
            {"Date": "2022-10-21", "Product": "Galaxy Tab S8 FE", "Category": "Tablet", "Type": "Mid-range"},
            {"Date": "2023-02-01", "Product": "Galaxy S23 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2023-07-26", "Product": "Galaxy Z Fold5/Flip5", "Category": "Smartphone", "Type": "Foldable"},
            {"Date": "2024-01-17", "Product": "Galaxy S24 Series", "Category": "Smartphone", "Type": "Flagship"},
            {"Date": "2024-07-10", "Product": "Galaxy Z Fold6/Flip6", "Category": "Smartphone", "Type": "Foldable"}
        ]
        
        # DataFrame으로 변환
        apple_df = pd.DataFrame(apple_fallback)
        apple_df['Date'] = pd.to_datetime(apple_df['Date'])
        apple_df['Company'] = 'Apple'
        
        samsung_df = pd.DataFrame(samsung_fallback)
        samsung_df['Date'] = pd.to_datetime(samsung_df['Date'])
        samsung_df['Company'] = 'Samsung'
        
        print(f"🍎 Apple 기본 데이터: {len(apple_df)}개")
        print(f"📱 Samsung 기본 데이터: {len(samsung_df)}개")
        
        return apple_df, samsung_df
    
    def combine_data(self, apple_df, samsung_df):
        """Apple과 Samsung 데이터 결합"""
        print("\n🔗 데이터 결합 중...")
        
        combined = pd.concat([apple_df, samsung_df], ignore_index=True)
        combined = combined.sort_values('Date').reset_index(drop=True)
        
        # 연도 및 월 정보 추가
        combined['Year'] = combined['Date'].dt.year
        combined['Month'] = combined['Date'].dt.month
        combined['Quarter'] = combined['Date'].dt.quarter
        combined['DayOfYear'] = combined['Date'].dt.dayofyear
        
        print(f"✅ 통합 데이터: {len(combined)}개 제품")
        print(f"   기간: {combined['Date'].min().strftime('%Y-%m-%d')} ~ {combined['Date'].max().strftime('%Y-%m-%d')}")
        
        return combined
    
    def create_timeline_visualization(self, combined_df):
        """제품 출시 타임라인 시각화"""
        print("\n📊 제품 출시 타임라인 시각화 생성 중...")
        
        # 기본 설정
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Product Launch Timeline Analysis (2021-2024)', fontsize=20, fontweight='bold')
        
        # 1. 연도별 제품 출시 수 (상단 좌측)
        ax1 = axes[0, 0]
        yearly_counts = combined_df.groupby(['Year', 'Company']).size().unstack(fill_value=0)
        yearly_counts.plot(kind='bar', ax=ax1, color=['#007AFF', '#FF6B35'], width=0.7)
        ax1.set_title('Annual Product Launches by Company', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Products')
        ax1.legend(title='Company')
        ax1.grid(True, alpha=0.3)
        
        # 2. 카테고리별 분포 (상단 우측)
        ax2 = axes[0, 1]
        category_counts = combined_df.groupby(['Category', 'Company']).size().unstack(fill_value=0)
        category_counts.plot(kind='bar', ax=ax2, color=['#007AFF', '#FF6B35'], width=0.7)
        ax2.set_title('Product Categories Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Number of Products')
        ax2.legend(title='Company')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 월별 출시 패턴 (하단 좌측)
        ax3 = axes[1, 0]
        monthly_counts = combined_df.groupby(['Month', 'Company']).size().unstack(fill_value=0)
        monthly_counts.plot(kind='line', ax=ax3, color=['#007AFF', '#FF6B35'], 
                          marker='o', linewidth=2, markersize=6)
        ax3.set_title('Monthly Launch Patterns', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Products')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.legend(title='Company')
        ax3.grid(True, alpha=0.3)
        
        # 4. 타임라인 스캐터 플롯 (하단 우측)
        ax4 = axes[1, 1]
        
        # Apple 데이터
        apple_data = combined_df[combined_df['Company'] == 'Apple']
        ax4.scatter(apple_data['Date'], [1] * len(apple_data), 
                   c='#007AFF', s=100, alpha=0.7, label='Apple')
        
        # Samsung 데이터
        samsung_data = combined_df[combined_df['Company'] == 'Samsung']
        ax4.scatter(samsung_data['Date'], [2] * len(samsung_data), 
                   c='#FF6B35', s=100, alpha=0.7, label='Samsung')
        
        ax4.set_title('Product Launch Timeline', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Company')
        ax4.set_yticks([1, 2])
        ax4.set_yticklabels(['Apple', 'Samsung'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        save_path = f"{self.results_base}/visualizations/weekly_analysis/product_timeline_overview.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 타임라인 시각화 저장: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_processed_data(self, apple_df, samsung_df, combined_df):
        """처리된 데이터 저장"""
        print("\n💾 처리된 데이터 저장 중...")
        
        # Apple 데이터 저장
        apple_path = f"{self.results_base}/data/processed/apple_product_launches.csv"
        apple_df.to_csv(apple_path, index=False, encoding='utf-8')
        print(f"🍎 Apple 데이터 저장: {apple_path}")
        
        # Samsung 데이터 저장  
        samsung_path = f"{self.results_base}/data/processed/samsung_product_launches.csv"
        samsung_df.to_csv(samsung_path, index=False, encoding='utf-8')
        print(f"📱 Samsung 데이터 저장: {samsung_path}")
        
        # 통합 타임라인 저장
        combined_path = f"{self.results_base}/data/processed/combined_product_timeline.csv"
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        print(f"🔗 통합 데이터 저장: {combined_path}")
        
        return apple_path, samsung_path, combined_path
    
    def generate_summary_report(self, apple_df, samsung_df, combined_df):
        """제품 출시 데이터 요약 리포트 생성"""
        print("\n📋 요약 리포트 생성 중...")
        
        # Year 컬럼이 없는 경우 생성
        if 'Year' not in apple_df.columns:
            apple_df['Year'] = apple_df['Date'].dt.year
        if 'Year' not in samsung_df.columns:
            samsung_df['Year'] = samsung_df['Date'].dt.year
        if 'Month' not in apple_df.columns:
            apple_df['Month'] = apple_df['Date'].dt.month
        if 'Month' not in samsung_df.columns:
            samsung_df['Month'] = samsung_df['Date'].dt.month
        
        # 안전한 통계 계산
        try:
            apple_year_stats = apple_df['Year'].value_counts().sort_index().to_string()
        except:
            apple_year_stats = "연도별 데이터 계산 불가"
        
        try:
            samsung_year_stats = samsung_df['Year'].value_counts().sort_index().to_string()
        except:
            samsung_year_stats = "연도별 데이터 계산 불가"
        
        try:
            apple_category_stats = apple_df['Category'].value_counts().to_string()
        except:
            apple_category_stats = "카테고리 데이터 없음"
        
        try:
            samsung_category_stats = samsung_df['Category'].value_counts().to_string()
        except:
            samsung_category_stats = "카테고리 데이터 없음"
        
        try:
            apple_peak_month = apple_df['Month'].mode().iloc[0] if not apple_df['Month'].mode().empty else "N/A"
        except:
            apple_peak_month = "N/A"
        
        try:
            samsung_peak_month = samsung_df['Month'].mode().iloc[0] if not samsung_df['Month'].mode().empty else "N/A"
        except:
            samsung_peak_month = "N/A"
        
        try:
            apple_categories = apple_df['Category'].nunique()
        except:
            apple_categories = 0
        
        try:
            samsung_categories = samsung_df['Category'].nunique()
        except:
            samsung_categories = 0
        
        report = f"""# 제품 출시 일정 정리 보고서
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 데이터 개요
- **전체 제품 수**: {len(combined_df)}개
- **분석 기간**: {combined_df['Date'].min().strftime('%Y-%m-%d')} ~ {combined_df['Date'].max().strftime('%Y-%m-%d')}
- **Apple 제품**: {len(apple_df)}개
- **Samsung 제품**: {len(samsung_df)}개

## 🍎 Apple 제품 분석
### 연도별 출시 수
{apple_year_stats}

### 카테고리별 분포
{apple_category_stats}

## 📱 Samsung 제품 분석
### 연도별 출시 수
{samsung_year_stats}

### 카테고리별 분포
{samsung_category_stats}

## 📈 주요 인사이트
1. **출시 패턴**: 
   - Apple: {apple_peak_month}월에 가장 많이 출시
   - Samsung: {samsung_peak_month}월에 가장 많이 출시

2. **제품 다양성**:
   - Apple 카테고리 수: {apple_categories}개
   - Samsung 카테고리 수: {samsung_categories}개

3. **출시 빈도**:
   - 연평균 Apple 제품: {len(apple_df) / 4:.1f}개
   - 연평균 Samsung 제품: {len(samsung_df) / 4:.1f}개

## 📁 생성된 파일
- apple_product_launches.csv: Apple 제품 출시 데이터
- samsung_product_launches.csv: Samsung 제품 출시 데이터  
- combined_product_timeline.csv: 통합 제품 출시 타임라인
- product_timeline_overview.png: 시각화 차트
"""
        
        # 리포트 저장
        report_path = f"{self.results_base}/reports/technical/product_launch_data_summary.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 요약 리포트 저장: {report_path}")
        return report_path
    
    def process_all(self):
        """전체 처리 과정 실행"""
        print("🚀 제품 출시 일정 정리 시작!")
        print("=" * 60)
        
        # 1. Excel 데이터 로딩
        self.load_excel_data()
        
        # 2. 데이터 표준화
        if self.apple_data is not None and not self.apple_data.empty:
            apple_standardized = self.standardize_apple_data()
            self.validate_data_quality(apple_standardized, "Apple")
        else:
            apple_standardized = pd.DataFrame()
        
        if self.samsung_data is not None and not self.samsung_data.empty:
            samsung_standardized = self.standardize_samsung_data()
            self.validate_data_quality(samsung_standardized, "Samsung")
        else:
            samsung_standardized = pd.DataFrame()
        
        # 3. 데이터가 없거나 부족한 경우 기본 데이터 사용
        if apple_standardized.empty or samsung_standardized.empty or len(apple_standardized) < 5 or len(samsung_standardized) < 5:
            print("\n⚠️  Excel 데이터가 부족하여 기본 제품 출시 데이터를 사용합니다.")
            apple_standardized, samsung_standardized = self.create_fallback_data()
        
        # 4. 데이터 결합
        combined_data = self.combine_data(apple_standardized, samsung_standardized)
        
        # 5. 시각화 생성
        self.create_timeline_visualization(combined_data)
        
        # 6. 데이터 저장
        file_paths = self.save_processed_data(apple_standardized, samsung_standardized, combined_data)
        
        # 7. 요약 리포트 생성
        report_path = self.generate_summary_report(apple_standardized, samsung_standardized, combined_data)
        
        print("\n" + "=" * 60)
        print("🎉 제품 출시 일정 정리 완료!")
        print(f"📊 총 {len(combined_data)}개 제품 데이터 정리 완료")
        print(f"🍎 Apple: {len(apple_standardized)}개")
        print(f"📱 Samsung: {len(samsung_standardized)}개")
        print("\n💾 생성된 파일:")
        print(f"   - {file_paths[0]}")
        print(f"   - {file_paths[1]}")
        print(f"   - {file_paths[2]}")
        print(f"   - {report_path}")
        print(f"   - product_timeline_overview.png")
        
        # 8번 코드를 위한 데이터 검증
        self.validate_for_next_step(combined_data)
        
        return apple_standardized, samsung_standardized, combined_data
    
    def validate_for_next_step(self, combined_df):
        """8번 코드(7일 평균 통합 시각화)를 위한 데이터 검증"""
        print("\n🔍 다음 단계(8번 코드) 준비 상태 검증...")
        
        # 기본 검증
        if combined_df.empty:
            print("❌ 제품 출시 데이터가 없습니다.")
            return False
        
        # 연도별 데이터 확인
        year_coverage = set(combined_df['Year'].unique())
        required_years = {2021, 2022, 2023, 2024}
        missing_years = required_years - year_coverage
        
        if missing_years:
            print(f"⚠️  일부 연도 데이터 부족: {missing_years}")
        else:
            print("✅ 모든 연도(2021-2024) 데이터 확보")
        
        # 회사별 데이터 확인
        companies = set(combined_df['Company'].unique())
        if 'Apple' not in companies:
            print("❌ Apple 데이터 없음")
        if 'Samsung' not in companies:
            print("❌ Samsung 데이터 없음")
        
        if 'Apple' in companies and 'Samsung' in companies:
            print("✅ Apple, Samsung 모든 회사 데이터 확보")
        
        # 날짜 형식 검증
        if combined_df['Date'].dtype == 'datetime64[ns]':
            print("✅ 날짜 형식 정상")
        else:
            print("⚠️  날짜 형식 확인 필요")
        
        # 8번 코드 실행 가이드
        print("\n🎯 다음 단계 실행 가이드:")
        print("1. 8.7일평균통합시각화.py 실행")
        print("2. 생성된 CSV 파일들이 올바르게 로딩되는지 확인")
        print("3. 제품 출시일이 차트에 올바르게 표시되는지 확인")
        
        return True


def main():
    """메인 실행 함수"""
    try:
        # 제품 출시 데이터 처리 시스템 초기화
        processor = ProductLaunchDataProcessor()
        
        # 전체 처리 과정 실행
        apple_data, samsung_data, combined_data = processor.process_all()
        
        print("\n" + "🎯" * 20)
        print("7번 코드 실행 완료!")
        print("이제 8번 코드(7일 평균 통합 시각화)를 실행할 준비가 되었습니다.")
        print("🎯" * 20)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("📞 디버깅을 위해 오류 세부 정보를 확인하세요.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()