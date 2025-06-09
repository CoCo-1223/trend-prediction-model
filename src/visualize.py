# 주요 제품 출시 + 주가 시각화 + 감정 분석 점수 시각화 코드
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 기본 경로 설정
base_path = "./data"
output_dir = "./data/visualizations"
os.makedirs(output_dir, exist_ok=True)

# 연도별 파일 경로
stock_files = {
    "apple": [f"{base_path}/stock/apple_stock_{year}.csv" for year in range(2021, 2025)],
    "samsung": [f"{base_path}/stock/samsung_stock_{year}.csv" for year in range(2021, 2025)],
}
sentiment_files = {
    "apple": [f"{base_path}/sentiment/apple_sentiment_{year}.csv" for year in range(2021, 2025)],
    "samsung": [f"{base_path}/sentiment/samsung_sentiment_{year}.csv" for year in range(2021, 2025)],
}
product_files = {
    "apple": f"{base_path}/products/apple_products.xlsx",
    "samsung": f"{base_path}/products/samsung_products.xlsx",
}

# 연도 필터링 함수
def filter_year(df, year_col='Date', year=2021):
    df[year_col] = pd.to_datetime(df[year_col])
    return df[(df[year_col].dt.year == year)].copy()

# 시각화 함수
def visualize_by_year(company):
    stock_df = pd.concat([pd.read_csv(f) for f in stock_files[company]])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    sentiment_df = pd.concat([pd.read_csv(f) for f in sentiment_files[company]])
    sentiment_df['일자'] = pd.to_datetime(sentiment_df['일자'])
    sentiment_df = sentiment_df.groupby('일자', as_index=False)['감정점수'].mean()

    product_df = pd.read_excel(product_files[company])
    product_df['Date'] = pd.to_datetime(product_df['Date'], errors='coerce')

    for year in range(2021, 2025):
        stock_y = filter_year(stock_df, 'Date', year)
        sentiment_y = filter_year(sentiment_df, '일자', year)
        product_y = product_df[product_df['Date'].dt.year == year]

        fig, ax1 = plt.subplots(figsize=(50, 30))

        # 주가 선
        line1, = ax1.plot(stock_y['Date'], stock_y['Close'], label='Stock Price', color='blue')
        ax1.set_ylabel('Stock Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

        # 제품명 겹침 방지 표시
        y_max = ax1.get_ylim()[1]
        y_min = ax1.get_ylim()[0]
        step_height = (y_max - y_min) * 0.08
        label_y_base = y_max * 0.98
        used_positions = {}
        previous_date = None

        for idx, (_, row) in enumerate(product_y.iterrows()):
            current_date = row['Date']
            if pd.isnull(current_date):
                continue

            # 날짜 간격 기반 겹침 방지 로직
            if previous_date and abs((current_date - previous_date).days) < 15:
                offset = used_positions.get(current_date.date(), 0) + 1
            else:
                offset = 0
            used_positions[current_date.date()] = offset
            y_position = label_y_base - offset * step_height

            # 제품 출시 영역과 텍스트 표시
            ax1.axvspan(current_date, current_date + pd.Timedelta(days=1), color='gray', alpha=0.1)
            ax1.text(current_date, y_position, row['Product'], rotation=90,
                     verticalalignment='bottom', fontsize=10, color='black')
            previous_date = current_date

        # 감정점수 보조 y축
        ax2 = ax1.twinx()
        sentiment_y['MA7'] = sentiment_y['감정점수'].rolling(window=7).mean()
        line2, = ax2.plot(sentiment_y['일자'], sentiment_y['MA7'], label='Sentiment MA(7)', color='orange', linestyle='--')
        ax2.set_ylabel('Sentiment Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 범례 통합 및 개선
        lines, labels = [line1, line2], ['Stock Price', 'Sentiment MA(7)']
        ax1.legend(lines, labels, loc='upper left', fontsize=18, fancybox=True, framealpha=0.6)

        # 기타 설정
        plt.title(f"{company.capitalize()} - {year}", fontsize=24)
        fig.tight_layout()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{company}_{year}_visualize.png")
        plt.close()

# 실행
visualize_by_year("apple")
visualize_by_year("samsung")
