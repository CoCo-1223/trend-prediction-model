# src/data/reddit_collector.py
import praw
import pandas as pd
import json
from datetime import datetime
import time

class RedditDataCollector:
    def __init__(self, config_path='config.json'):
        # 설정 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        reddit_keys = config['api_keys']['reddit']
        
        # PRAW 클라이언트 설정
        self.reddit = praw.Reddit(
            client_id=reddit_keys['client_id'],
            client_secret=reddit_keys['client_secret'],
            user_agent=reddit_keys['user_agent']
        )
        
        self.selected_brands = config['selected_brands']
        
    def collect_subreddit_posts(self, subreddit_name, limit=100):
        """특정 서브레딧의 게시물 수집"""
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=limit):
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'body': post.selftext,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'author': str(post.author),
                    'subreddit': subreddit_name
                }
                posts.append(post_data)
                
                # 댓글 수집 (상위 10개)
                post.comments.replace_more(limit=0)
                for comment in list(post.comments)[:10]:
                    comment_data = {
                        'id': comment.id,
                        'post_id': post.id,
                        'body': comment.body,
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'author': str(comment.author),
                        'subreddit': subreddit_name
                    }
                    posts.append(comment_data)
            
            return pd.DataFrame(posts)
        
        except Exception as e:
            print(f"에러 발생: {e}")
            return pd.DataFrame(posts)
    
    def search_reddit_for_brands(self, limit=100):
        """브랜드 관련 게시물 검색"""
        all_posts = pd.DataFrame()
        
        for brand in self.selected_brands:
            print(f"{brand} 관련 Reddit 게시물 검색 중...")
            
            # 브랜드 이름으로 전체 Reddit 검색
            for submission in self.reddit.subreddit('all').search(brand, limit=limit):
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'body': submission.selftext,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'url': submission.url,
                    'author': str(submission.author),
                    'subreddit': submission.subreddit.display_name,
                    'brand': brand
                }
                all_posts = pd.concat([all_posts, pd.DataFrame([post_data])])
            
            # API 속도 제한 고려
            time.sleep(2)
        
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        all_posts.to_csv(f'data/raw/reddit_{timestamp}.csv', index=False, encoding='utf-8')
        print(f"수집 완료: {len(all_posts)} 게시물 저장됨")
        
        return all_posts

# 실행 예시
if __name__ == "__main__":
    collector = RedditDataCollector()
    # 서브레딧 수집
    korea_posts = collector.collect_subreddit_posts('korea', limit=200)
    # 브랜드 검색
    brand_posts = collector.search_reddit_for_brands(limit=200)
