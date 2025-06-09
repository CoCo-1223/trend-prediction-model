def evaluate_api_accessibility(platform_name):
    # API 평가 기준 정의
    criteria = {
        'rate_limits': 0,  # API 호출 제한
        'data_access': 0,  # 데이터 접근 범위
        'authentication': 0,  # 인증 복잡성
        'cost': 0,  # 비용
        'documentation': 0  # 문서화 품질
    }
    
    # 각 플랫폼별 평가 로직
    if platform_name.lower() == 'twitter':
        criteria['rate_limits'] = 3  # 1-5 척도 (5가 가장 좋음)
        criteria['data_access'] = 4
        criteria['authentication'] = 3
        criteria['cost'] = 2
        criteria['documentation'] = 5
    elif platform_name.lower() == 'instagram':
        # Instagram 평가 로직
        pass
    # 다른 플랫폼 평가 로직 추가
    
    # 종합 점수 계산
    total_score = sum(criteria.values()) / len(criteria)
    return criteria, total_score

# 평가 실행 예시
twitter_eval, twitter_score = evaluate_api_accessibility('twitter')
print(f"Twitter API 종합 평가 점수: {twitter_score}/5")
