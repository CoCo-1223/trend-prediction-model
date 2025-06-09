import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def quick_korean_font_setup():
    """빠른 한글 폰트 설정"""
    
    # 1. Windows 기본 폰트들 직접 시도
    windows_fonts = [
        'Malgun Gothic',  # 맑은 고딕
        'Microsoft YaHei',
        'SimHei', 
        'Arial Unicode MS',
        'Gulim',  # 굴림
        'Batang',  # 바탕
        'Dotum'   # 돋움
    ]
    
    print("🔍 한글 폰트 찾는 중...")
    
    # 2. 시스템에 설치된 모든 폰트 확인
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"총 {len(available_fonts)}개 폰트 발견")
    
    # 3. 한글 폰트 후보 찾기
    korean_candidates = []
    for font in available_fonts:
        if any(target.lower() in font.lower() for target in ['malgun', 'gulim', 'batang', 'dotum', 'nanum']):
            korean_candidates.append(font)
    
    print(f"한글 폰트 후보: {korean_candidates}")
    
    # 4. 폰트 설정 시도
    for font_name in windows_fonts + korean_candidates:
        try:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            
            # 간단 테스트
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, '한글 테스트: 삼성전자 감성분석', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title(f'폰트 테스트: {font_name}')
            
            # 파일로 저장해서 확인
            plt.savefig(f'font_test_{font_name.replace(" ", "_")}.png', 
                       dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 성공: {font_name} - font_test_{font_name.replace(' ', '_')}.png 파일 확인")
            return font_name
            
        except Exception as e:
            print(f"❌ 실패: {font_name} - {e}")
            continue
    
    # 5. 모든 폰트 실패 시 영어 폰트 사용
    print("🚨 한글 폰트 설정 실패 - 영어 폰트 사용")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

def create_test_chart():
    """테스트 차트 생성"""
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    sentiment_scores = 3 + 0.5 * np.sin(np.linspace(0, 4*np.pi, 30)) + 0.2 * np.random.randn(30)
    
    # 차트 생성
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(dates, sentiment_scores, 'b-o', linewidth=2, markersize=4)
    plt.title('삼성전자 일별 감성 점수', fontsize=14, fontweight='bold')
    plt.ylabel('감성 점수')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.hist(sentiment_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('감성 점수 분포', fontsize=14, fontweight='bold')
    plt.xlabel('감성 점수')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    categories = ['긍정', '중립', '부정']
    values = [45, 35, 20]
    colors = ['green', 'yellow', 'red']
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('감성 분류 비율', fontsize=14, fontweight='bold')
    
    plt.subplot(2, 2, 4)
    months = ['1월', '2월', '3월', '4월']
    scores = [3.2, 3.5, 3.1, 3.8]
    plt.bar(months, scores, color='lightcoral', alpha=0.7)
    plt.title('월별 평균 감성 점수', fontsize=14, fontweight='bold')
    plt.ylabel('평균 점수')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('한글 폰트 테스트 - 뉴스 감성 분석 대시보드', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('korean_font_complete_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 완전한 테스트 차트 생성 완료!")
    print("📁 korean_font_complete_test.png 파일을 확인해보세요!")

if __name__ == "__main__":
    import pandas as pd
    
    print("🚀 빠른 한글 폰트 설정 시작")
    print("="*50)
    
    # 폰트 설정
    selected_font = quick_korean_font_setup()
    
    print(f"\n🎯 선택된 폰트: {selected_font}")
    
    # 테스트 차트 생성
    print(f"\n📊 테스트 차트 생성 중...")
    create_test_chart()
    
    print(f"\n✅ 폰트 설정 완료!")
    print(f"이제 딥러닝 코드에서 다음과 같이 설정하세요:")
    print(f"plt.rcParams['font.family'] = '{selected_font}'")
    print(f"plt.rcParams['axes.unicode_minus'] = False")