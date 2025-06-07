"""
뉴스 감성 분석 기반 주가 예측 모델 - 데이터 구조 확인 및 결과물 폴더 설정
생성일: 2025-06-08
팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)
목적: 프로젝트 데이터 구조 분석 및 체계적인 결과물 관리 시스템 구축
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import warnings
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
    return directories

class DataStructureAnalyzer:
    """프로젝트 데이터 구조 분석 클래스"""
    
    def __init__(self, project_base_path):
        self.project_base = project_base_path
        self.stock_path = f"{project_base_path}/stock"
        self.product_path = f"{project_base_path}/product"
        self.data_path = f"{project_base_path}/data"
        self.code_path = f"{project_base_path}/code"
        
        self.analysis_results = {}
        
    def analyze_directory_structure(self):
        """프로젝트 디렉토리 구조 분석"""
        print("🗂️ 프로젝트 디렉토리 구조 분석")
        print("=" * 50)
        
        # 기본 디렉토리들 확인
        base_dirs = ['stock', 'product', 'data', 'code', 'results']
        existing_dirs = []
        missing_dirs = []
        
        for dir_name in base_dirs:
            dir_path = f"{self.project_base}/{dir_name}"
            if os.path.exists(dir_path):
                existing_dirs.append(dir_name)
                print(f"✅ {dir_name}/ - 존재")
            else:
                missing_dirs.append(dir_name)
                print(f"❌ {dir_name}/ - 누락")
        
        self.analysis_results['directory_structure'] = {
            'existing': existing_dirs,
            'missing': missing_dirs
        }
        
        return existing_dirs, missing_dirs
    
    def analyze_stock_data(self):
        """주가 데이터 구조 분석"""
        print("\n📈 주가 데이터 구조 분석")
        print("=" * 50)
        
        stock_files = glob.glob(f"{self.stock_path}/*.csv")
        print(f"주가 데이터 파일 수: {len(stock_files)}")
        
        stock_analysis = {}
        
        for file_path in stock_files:
            file_name = os.path.basename(file_path)
            print(f"\n📁 {file_name}")
            
            try:
                df = pd.read_csv(file_path)
                
                # 기본 정보
                analysis = {
                    'file_size': f"{os.path.getsize(file_path) / 1024:.1f} KB",
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': None,
                    'sample_data': df.head(3).to_dict('records')
                }
                
                # 날짜 컬럼 찾기 및 분석
                date_cols = [col for col in df.columns if 'date' in col.lower() or '일자' in col]
                if date_cols:
                    date_col = date_cols[0]
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        analysis['date_range'] = {
                            'start': str(df[date_col].min().date()),
                            'end': str(df[date_col].max().date()),
                            'days': (df[date_col].max() - df[date_col].min()).days
                        }
                    except:
                        analysis['date_range'] = "날짜 파싱 실패"
                
                # 출력
                print(f"  - 크기: {analysis['file_size']}")
                print(f"  - 행수: {analysis['rows']:,}")
                print(f"  - 컬럼: {analysis['columns']}")
                if analysis['date_range'] and isinstance(analysis['date_range'], dict):
                    print(f"  - 기간: {analysis['date_range']['start']} ~ {analysis['date_range']['end']}")
                    print(f"  - 일수: {analysis['date_range']['days']}일")
                
                stock_analysis[file_name] = analysis
                
            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                stock_analysis[file_name] = {'error': str(e)}
        
        self.analysis_results['stock_data'] = stock_analysis
        return stock_analysis
    
    def analyze_sentiment_data(self):
        """감성 데이터 구조 분석"""
        print("\n🎭 감성 데이터 구조 분석")
        print("=" * 50)
        
        # 기존 감성 데이터 파일들 찾기
        sentiment_files = []
        
        # data 폴더에서 찾기
        if os.path.exists(self.data_path):
            sentiment_files.extend(glob.glob(f"{self.data_path}/**/*.csv", recursive=True))
        
        # 프로젝트 루트에서 찾기
        sentiment_files.extend(glob.glob(f"{self.project_base}/*.csv"))
        
        # 감성 관련 파일 필터링
        sentiment_files = [f for f in sentiment_files if any(keyword in os.path.basename(f).lower() 
                          for keyword in ['samsung', 'apple', '삼성', '감성', 'sentiment'])]
        
        print(f"감성 데이터 파일 수: {len(sentiment_files)}")
        
        sentiment_analysis = {}
        
        for file_path in sentiment_files:
            file_name = os.path.basename(file_path)
            print(f"\n📁 {file_name}")
            
            try:
                # 파일 크기 확인 (큰 파일은 샘플만 읽기)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if file_size_mb > 100:  # 100MB 이상
                    df = pd.read_csv(file_path, nrows=10000)  # 샘플만 읽기
                    print(f"  ⚠️ 큰 파일 ({file_size_mb:.1f}MB) - 샘플 10,000행만 분석")
                else:
                    df = pd.read_csv(file_path)
                
                # 기본 정보
                analysis = {
                    'file_size': f"{file_size_mb:.1f} MB",
                    'rows': len(df),
                    'columns': list(df.columns),
                    'sample_data': df.head(3).to_dict('records')
                }
                
                # 감성 관련 컬럼 찾기
                sentiment_cols = [col for col in df.columns if any(keyword in col.lower() 
                                for keyword in ['감정', '감성', 'sentiment', '점수', 'score'])]
                if sentiment_cols:
                    analysis['sentiment_columns'] = sentiment_cols
                    
                    # 감성 점수 분포 분석
                    for col in sentiment_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            analysis[f'{col}_stats'] = {
                                'min': float(df[col].min()),
                                'max': float(df[col].max()),
                                'mean': float(df[col].mean()),
                                'unique_values': int(df[col].nunique())
                            }
                
                # 날짜 컬럼 분석
                date_cols = [col for col in df.columns if 'date' in col.lower() or '일자' in col]
                if date_cols:
                    analysis['date_columns'] = date_cols
                
                # 출력
                print(f"  - 크기: {analysis['file_size']}")
                print(f"  - 행수: {analysis['rows']:,}")
                print(f"  - 컬럼: {analysis['columns']}")
                if 'sentiment_columns' in analysis:
                    print(f"  - 감성 컬럼: {analysis['sentiment_columns']}")
                
                sentiment_analysis[file_name] = analysis
                
            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                sentiment_analysis[file_name] = {'error': str(e)}
        
        self.analysis_results['sentiment_data'] = sentiment_analysis
        return sentiment_analysis
    
    def analyze_product_data(self):
        """제품 출시 데이터 구조 분석"""
        print("\n📱 제품 출시 데이터 구조 분석")
        print("=" * 50)
        
        if not os.path.exists(self.product_path):
            print("❌ product 폴더가 존재하지 않습니다.")
            return {}
        
        product_files = glob.glob(f"{self.product_path}/*")
        print(f"제품 출시 관련 파일 수: {len(product_files)}")
        
        product_analysis = {}
        
        for file_path in product_files:
            file_name = os.path.basename(file_path)
            print(f"\n📁 {file_name}")
            
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    analysis = {
                        'type': 'CSV',
                        'rows': len(df),
                        'columns': list(df.columns),
                        'sample_data': df.head(3).to_dict('records')
                    }
                elif file_path.endswith(('.txt', '.md')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    analysis = {
                        'type': 'TEXT',
                        'lines': len(content.split('\n')),
                        'characters': len(content),
                        'preview': content[:200] + "..." if len(content) > 200 else content
                    }
                else:
                    analysis = {
                        'type': 'OTHER',
                        'file_size': f"{os.path.getsize(file_path) / 1024:.1f} KB"
                    }
                
                print(f"  - 타입: {analysis['type']}")
                if 'rows' in analysis:
                    print(f"  - 행수: {analysis['rows']:,}")
                if 'columns' in analysis:
                    print(f"  - 컬럼: {analysis['columns']}")
                
                product_analysis[file_name] = analysis
                
            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                product_analysis[file_name] = {'error': str(e)}
        
        self.analysis_results['product_data'] = product_analysis
        return product_analysis
    
    def create_data_overview_visualization(self):
        """데이터 개요 시각화 생성"""
        print("\n📊 데이터 개요 시각화 생성")
        print("=" * 50)
        
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Project Data Structure Overview', fontsize=16, fontweight='bold')
        
        # 1. 디렉토리 구조 (파이 차트)
        ax1 = axes[0, 0]
        if 'directory_structure' in self.analysis_results:
            existing = len(self.analysis_results['directory_structure']['existing'])
            missing = len(self.analysis_results['directory_structure']['missing'])
            
            labels = ['Existing Directories', 'Missing Directories']
            sizes = [existing, missing]
            colors = ['#2ecc71', '#e74c3c']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Directory Structure Status')
        
        # 2. 파일 타입별 분포 (막대 차트)
        ax2 = axes[0, 1]
        file_types = {'Stock Files': 0, 'Sentiment Files': 0, 'Product Files': 0}
        
        if 'stock_data' in self.analysis_results:
            file_types['Stock Files'] = len(self.analysis_results['stock_data'])
        if 'sentiment_data' in self.analysis_results:
            file_types['Sentiment Files'] = len(self.analysis_results['sentiment_data'])
        if 'product_data' in self.analysis_results:
            file_types['Product Files'] = len(self.analysis_results['product_data'])
        
        bars = ax2.bar(file_types.keys(), file_types.values(), 
                      color=['#3498db', '#9b59b6', '#f39c12'])
        ax2.set_title('File Type Distribution')
        ax2.set_ylabel('Number of Files')
        
        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. 데이터 크기 분석 (가로 막대 차트)
        ax3 = axes[1, 0]
        data_sizes = []
        labels = []
        
        # 주가 데이터 크기
        if 'stock_data' in self.analysis_results:
            for file_name, info in self.analysis_results['stock_data'].items():
                if 'rows' in info:
                    data_sizes.append(info['rows'])
                    labels.append(f"Stock: {file_name[:15]}...")
        
        # 감성 데이터 크기
        if 'sentiment_data' in self.analysis_results:
            for file_name, info in self.analysis_results['sentiment_data'].items():
                if 'rows' in info:
                    data_sizes.append(info['rows'])
                    labels.append(f"Sentiment: {file_name[:15]}...")
        
        if data_sizes:
            y_pos = np.arange(len(labels))
            ax3.barh(y_pos, data_sizes, color='#1abc9c')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels, fontsize=8)
            ax3.set_xlabel('Number of Rows')
            ax3.set_title('Data Size Comparison')
        
        # 4. 프로젝트 진행 상황 (텍스트 요약)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
PROJECT ANALYSIS SUMMARY

📁 Directories: {len(self.analysis_results.get('directory_structure', {}).get('existing', []))} found
📈 Stock Files: {len(self.analysis_results.get('stock_data', {}))} files
🎭 Sentiment Files: {len(self.analysis_results.get('sentiment_data', {}))} files  
📱 Product Files: {len(self.analysis_results.get('product_data', {}))} files

STATUS: Data structure analysis completed
NEXT: Proceed with 7-day average visualization
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # 시각화 저장
        viz_path = f"{RESULTS_BASE}/visualizations/weekly_analysis/data_overview.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 데이터 개요 시각화 저장: {viz_path}")
        
        plt.show()
        
    def save_analysis_report(self):
        """분석 결과 리포트 저장"""
        print("\n📄 분석 결과 리포트 저장")
        print("=" * 50)
        
        # 마크다운 리포트 생성
        report_content = f"""# 데이터 구조 분석 리포트

## 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}
## 팀: 현종민(팀장), 신예원(팀원), 김채은(팀원)

---

## 📁 디렉토리 구조 분석

### 존재하는 디렉토리
"""
        
        if 'directory_structure' in self.analysis_results:
            for dir_name in self.analysis_results['directory_structure']['existing']:
                report_content += f"- ✅ {dir_name}/\n"
            
            if self.analysis_results['directory_structure']['missing']:
                report_content += "\n### 누락된 디렉토리\n"
                for dir_name in self.analysis_results['directory_structure']['missing']:
                    report_content += f"- ❌ {dir_name}/\n"
        
        # 주가 데이터 분석 결과
        if 'stock_data' in self.analysis_results:
            report_content += f"\n## 📈 주가 데이터 분석\n\n"
            report_content += f"총 {len(self.analysis_results['stock_data'])}개 파일 발견\n\n"
            
            for file_name, info in self.analysis_results['stock_data'].items():
                if 'error' not in info:
                    report_content += f"### {file_name}\n"
                    report_content += f"- 크기: {info.get('file_size', 'N/A')}\n"
                    report_content += f"- 행수: {info.get('rows', 'N/A'):,}\n"
                    report_content += f"- 컬럼: {info.get('columns', [])}\n"
                    if info.get('date_range') and isinstance(info['date_range'], dict):
                        report_content += f"- 기간: {info['date_range']['start']} ~ {info['date_range']['end']}\n"
                    report_content += "\n"
        
        # 감성 데이터 분석 결과
        if 'sentiment_data' in self.analysis_results:
            report_content += f"\n## 🎭 감성 데이터 분석\n\n"
            report_content += f"총 {len(self.analysis_results['sentiment_data'])}개 파일 발견\n\n"
            
            for file_name, info in self.analysis_results['sentiment_data'].items():
                if 'error' not in info:
                    report_content += f"### {file_name}\n"
                    report_content += f"- 크기: {info.get('file_size', 'N/A')}\n"
                    report_content += f"- 행수: {info.get('rows', 'N/A'):,}\n"
                    report_content += f"- 컬럼: {info.get('columns', [])}\n"
                    if 'sentiment_columns' in info:
                        report_content += f"- 감성 컬럼: {info['sentiment_columns']}\n"
                    report_content += "\n"
        
        # 제품 데이터 분석 결과
        if 'product_data' in self.analysis_results:
            report_content += f"\n## 📱 제품 출시 데이터 분석\n\n"
            report_content += f"총 {len(self.analysis_results['product_data'])}개 파일 발견\n\n"
            
            for file_name, info in self.analysis_results['product_data'].items():
                if 'error' not in info:
                    report_content += f"### {file_name}\n"
                    report_content += f"- 타입: {info.get('type', 'N/A')}\n"
                    if 'rows' in info:
                        report_content += f"- 행수: {info['rows']:,}\n"
                    if 'columns' in info:
                        report_content += f"- 컬럼: {info['columns']}\n"
                    report_content += "\n"
        
        report_content += f"""
---

## 📊 다음 단계

1. **7일 평균 통합 시각화**: 8개 연도별 차트 생성
2. **LSTM 모델 개선**: 주간 데이터 기반으로 전환
3. **제품 출시 임팩트 분석**: 정량적 영향 측정

## 🎯 기대 결과

- 노이즈 감소를 통한 모델 성능 개선 (R² > 0.3 목표)
- 실용적인 주간 단위 예측 시스템 구축
- 제품 출시와 감성-주가 간 상관관계 정량화
"""
        
        # 리포트 저장
        report_path = f"{RESULTS_BASE}/reports/technical/data_structure_analysis.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 마크다운 리포트 저장: {report_path}")
        
        # 간단한 텍스트 요약도 저장
        summary_path = f"{RESULTS_BASE}/reports/technical/data_quality_report.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"데이터 구조 분석 요약 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"주가 파일: {len(self.analysis_results.get('stock_data', {}))}개\n")
            f.write(f"감성 파일: {len(self.analysis_results.get('sentiment_data', {}))}개\n")
            f.write(f"제품 파일: {len(self.analysis_results.get('product_data', {}))}개\n")
            f.write(f"\n분석 완료: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        print(f"✅ 요약 리포트 저장: {summary_path}")

def main():
    """메인 실행 함수"""
    print("🚀 뉴스 감성 분석 기반 주가 예측 모델 - 데이터 구조 확인")
    print("=" * 60)
    print(f"프로젝트 기본 경로: {PROJECT_BASE}")
    print(f"결과물 저장 경로: {RESULTS_BASE}")
    print("=" * 60)
    
    # 1. 결과물 디렉토리 구조 생성
    print("\n1️⃣ 결과물 디렉토리 구조 생성")
    setup_directories()
    
    # 2. 데이터 구조 분석기 초기화
    print("\n2️⃣ 데이터 구조 분석 시작")
    analyzer = DataStructureAnalyzer(PROJECT_BASE)
    
    # 3. 각 데이터 타입별 분석 수행
    analyzer.analyze_directory_structure()
    analyzer.analyze_stock_data()
    analyzer.analyze_sentiment_data()
    analyzer.analyze_product_data()
    
    # 4. 시각화 생성
    print("\n3️⃣ 데이터 개요 시각화 생성")
    analyzer.create_data_overview_visualization()
    
    # 5. 분석 리포트 저장
    print("\n4️⃣ 분석 결과 리포트 저장")
    analyzer.save_analysis_report()
    
    print("\n" + "=" * 60)
    print("✅ 데이터 구조 확인 및 결과물 폴더 설정 완료!")
    print(f"📁 결과물 확인: {RESULTS_BASE}")
    print("🎯 다음 단계: 7.제품출시일정정리.py 실행")
    print("=" * 60)

if __name__ == "__main__":
    main()