import pandas as pd
import numpy as np

def select_target_brands(candidates, weights=None):
    # 기본 가중치 설정
    if weights is None:
        weights = {
            'social_mentions': 0.3,
            'data_accessibility': 0.3,
            'business_metrics': 0.2,
            'market_relevance': 0.1,
            'sentiment_variance': 0.1
        }
    
    # 평가 데이터프레임 생성
    eval_df = pd.DataFrame(candidates)
    
    # 가중 점수 계산
    for criterion, weight in weights.items():
        eval_df[f'{criterion}_weighted'] = eval_df[criterion] * weight
    
    # 종합 점수 계산
    eval_df['total_score'] = eval_df[[col for col in eval_df.columns if '_weighted' in col]].sum(axis=1)
    
    # 상위 브랜드 선정
    selected_brands = eval_df.sort_values('total_score', ascending=False).head(5)
    
    return selected_brands

# 사용 예시
candidates = [
    {'brand': '삼성전자', 'social_mentions': 85, 'data_accessibility': 90, 'business_metrics': 95, 'market_relevance': 80, 'sentiment_variance': 70},
    {'brand': 'LG전자', 'social_mentions': 75, 'data_accessibility': 85, 'business_metrics': 90, 'market_relevance': 75, 'sentiment_variance': 65},
    # 추가 브랜드...
]

selected = select_target_brands(candidates)
print("선정된 브랜드:")
print(selected[['brand', 'total_score']])
