# 주가 시각화 코드 

import pandas as pd
import matplotlib.pyplot as plt

# 애플과 삼성 데이터 파일 리스트
apple_file_paths = ['/stock/apple_stock_2021.csv',
                    '/stock/apple_stock_2022.csv',
                    '/stock/apple_stock_2023.csv',
                    '/stock/apple_stock_2024.csv']

samsung_files = ['/stock/samsung_stock_2021.csv',
                 '/stock/samsung_stock_2022.csv',
                 '/stock/samsung_stock_2023.csv',
                 '/stock/samsung_stock_2024.csv']

# 애플 데이터 읽기
df_apple_list = [pd.read_csv(file, encoding='ISO-8859-1') for file in apple_file_paths]
df_apple = pd.concat(df_apple_list, ignore_index=True)

# 삼성 데이터 읽기
# 인코딩 주의해서 불러오기
df_samsung_list = [pd.read_csv(file, encoding='cp949') for file in samsung_files]
df_samsung = pd.concat(df_samsung_list, ignore_index=True)

# 날짜 처리
df_apple['Date'] = pd.to_datetime(df_apple['Date'])
df_samsung['Date'] = pd.to_datetime(df_samsung['Date'])

# 날짜를 인덱스로 설정
df_apple.set_index('Date', inplace=True)
df_samsung.set_index('Date', inplace=True)

# 연도별 시각화
for year in range(2021, 2025):
    # 애플 데이터 필터링
    df_apple_year = df_apple[df_apple.index.year == year]

    # 삼성 데이터 필터링
    df_samsung_year = df_samsung[df_samsung.index.year == year]

    # 애플 주식 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(df_apple_year['Close'], label=f'Apple {year} Closing Price', color='blue')
    plt.title(f"Apple Stock Closing Price - {year}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 삼성 주식 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(df_samsung_year['Close'], label=f'Samsung {year} Closing Price', color='green')
    plt.title(f"Samsung Stock Closing Price - {year}")
    plt.xlabel("Date")
    plt.ylabel("Price (KRW)")
    plt.legend()
    plt.grid(True)
    plt.show()
    