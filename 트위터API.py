# src/data/twitter_collector.py
import tweepy
import pandas as pd
import json
import time
from datetime import datetime, timedelta

class TwitterDataCollector:
    def __init__(self, config_path='config.json'):
        # API 키 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        twitter_keys = config['api_keys']['twitter']
        
        # API 인증
        self.client = tweepy.Client(
            bearer_token=twitter_keys['bearer_token'],
            consumer_key=twitter_keys['api_key'],
            consumer_secret=twitter_keys['api_secret'],
            access_token=twitter_keys['access_token'],
            access_token_secret=twitter_keys['access_secret']
        )
        
        self.selected_brands = config['selected_brands']
    
    def collect_tweets(self, query, max_results=100, start_time=None, end_time=None):
        """특정 쿼리에 대한 트윗 수집"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        
        tweets = []
        try:
            # 트윗 검색 및 수집
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'lang', 'public_metrics', 'source', 'geo']
            )
            
            if response.data:
                for tweet in response.data:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'lang': tweet.lang,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'source': tweet.source,
                        'query': query
                    }
                    tweets.append(tweet_data)
            
            return pd.DataFrame(tweets)
        
        except Exception as e:
            print(f"에러 발생: {e}")
            return pd.DataFrame(tweets)
    
    def collect_brand_tweets(self, days_back=7, max_results=100):
        """선정된 브랜드에 대한 트윗 수집"""
        all_tweets = pd.DataFrame()
        
        for brand in self.selected_brands:
            print(f"{brand} 관련 트윗 수집 중...")
            
            # 브랜드 이름과 관련 키워드로 검색
            query = f"{brand} lang:ko -is:retweet"
            brand_tweets = self.collect_tweets(
                query=query,
                max_results=max_results,
                start_time=datetime.now() - timedelta(days=days_back)
            )
            
            if not brand_tweets.empty:
                brand_tweets['brand'] = brand
                all_tweets = pd.concat([all_tweets, brand_tweets])
            
            # API 속도 제한 고려
            time.sleep(2)
        
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        all_tweets.to_csv(f'data/raw/twitter_{timestamp}.csv', index=False, encoding='utf-8')
        print(f"수집 완료: {len(all_tweets)} 트윗 저장됨")
        
        return all_tweets

# 실행 예시
if __name__ == "__main__":
    collector = TwitterDataCollector()
    tweets_df = collector.collect_brand_tweets(days_back=30, max_results=500)
