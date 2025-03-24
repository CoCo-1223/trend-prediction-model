# src/data/database_manager.py
from pymongo import MongoClient
import json
import pandas as pd
from datetime import datetime

class SocialMediaDatabase:
    def __init__(self, config_path='config.json'):
        # 설정 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        db_config = config['database']
        
        # MongoDB 연결
        self.client = MongoClient(db_config['connection_string'])
        self.db = self.client[db_config['db_name']]
        
        # 컬렉션 정의
        self.collections = {
            'twitter': self.db['twitter_data'],
            'reddit': self.db['reddit_data'],
            'instagram': self.db['instagram_data'],
            'business_metrics': self.db['business_metrics']
        }
        
        # 인덱스 생성
        self._create_indexes()
        
    def _create_indexes(self):
        """데이터베이스 인덱스 생성"""
        # Twitter 컬렉션 인덱스
        self.collections['twitter'].create_index('id', unique=True)
        self.collections['twitter'].create_index('created_at')
        self.collections['twitter'].create_index('brand')
        
        # Reddit 컬렉션 인덱스
        self.collections['reddit'].create_index('id', unique=True)
        self.collections['reddit'].create_index('created_utc')
        self.collections['reddit'].create_index('brand')
        
        # Instagram 컬렉션 인덱스
        self.collections['instagram'].create_index('post_url', unique=True)
        self.collections['instagram'].create_index('date')
        self.collections['instagram'].create_index('brand')
        
        # 비즈니스 지표 컬렉션 인덱스
        self.collections['business_metrics'].create_index([('brand', 1), ('date', 1)], unique=True)
    
    def insert_twitter_data(self, dataframe):
        """Twitter 데이터 삽입"""
        if dataframe.empty:
            print("삽입할 Twitter 데이터가 없습니다.")
            return 0
        
        # 데이터프레임을 사전 리스트로 변환
        records = dataframe.to_dict('records')
        
        # 중복 방지를 위한 upsert 작업
        inserted_count = 0
        for record in records:
            try:
                result = self.collections['twitter'].update_one(
                    {'id': record['id']},
                    {'$set': record},
                    upsert=True
                )
                if result.upserted_id or result.modified_count:
                    inserted_count += 1
            except Exception as e:
                print(f"Twitter 데이터 삽입 오류: {e}")
        
        print(f"{inserted_count}개의 Twitter 데이터가 데이터베이스에 저장되었습니다.")
        return inserted_count
    
    def insert_reddit_data(self, dataframe):
        """Reddit 데이터 삽입"""
        if dataframe.empty:
            print("삽입할 Reddit 데이터가 없습니다.")
            return 0
        
        records = dataframe.to_dict('records')
        
        inserted_count = 0
        for record in records:
            try:
                result = self.collections['reddit'].update_one(
                    {'id': record['id']},
                    {'$set': record},
                    upsert=True
                )
                if result.upserted_id or result.modified_count:
                    inserted_count += 1
            except Exception as e:
                print(f"Reddit 데이터 삽입 오류: {e}")
        
        print(f"{inserted_count}개의 Reddit 데이터가 데이터베이스에 저장되었습니다.")
        return inserted_count
    
    def insert_instagram_data(self, dataframe):
        """Instagram 데이터 삽입"""
        if dataframe.empty:
            print("삽입할 Instagram 데이터가 없습니다.")
            return 0
        
        records = dataframe.to_dict('records')
        
        inserted_count = 0
        for record in records:
            try:
                result = self.collections['instagram'].update_one(
                    {'post_url': record['post_url']},
                    {'$set': record},
                    upsert=True
                )
                if result.upserted_id or result.modified_count:
                    inserted_count += 1
            except Exception as e:
                print(f"Instagram 데이터 삽입 오류: {e}")
        
        print(f"{inserted_count}개의 Instagram 데이터가 데이터베이스에 저장되었습니다.")
        return inserted_count
    
    def insert_business_metrics(self, brand, date, metrics):
        """비즈니스 지표 데이터 삽입"""
        try:
            record = {
                'brand': brand,
                'date': date,
                **metrics,
                'updated_at': datetime.now()
            }
            
            result = self.collections['business_metrics'].update_one(
                {'brand': brand, 'date': date},
                {'$set': record},
                upsert=True
            )
            
            return bool(result.upserted_id or result.modified_count)
        
        except Exception as e:
            print(f"비즈니스 지표 삽입 오류: {e}")
            return False
    
    def get_data_by_brand(self, platform, brand, start_date=None, end_date=None):
        """특정 브랜드의 플랫폼 데이터 조회"""
        query = {'brand': brand}
        
        # 날짜 필터 추가
        if start_date or end_date:
            date_query = {}
            date_field = 'created_at' if platform == 'twitter' else 'created_utc' if platform == 'reddit' else 'date'
            
            if start_date:
                date_query['$gte'] = start_date
            if end_date:
                date_query['$lte'] = end_date
            
            if date_query:
                query[date_field] = date_query
        
        # 데이터 조회
        cursor = self.collections[platform].find(query)
        
        # Pandas DataFrame으로 변환
        data = pd.DataFrame(list(cursor))
        
        return data
    
    def close(self):
        """데이터베이스 연결 종료"""
        self.client.close()

# 실행 예시
if __name__ == "__main__":
    # 데이터베이스 매니저 생성
    db_manager = SocialMediaDatabase()
    
    # Twitter 데이터 로드 및 삽입
    twitter_df = pd.read_csv('data/raw/twitter_20250325_120000.csv')
    db_manager.insert_twitter_data(twitter_df)
    
    # 특정 브랜드 데이터 조회
    samsung_data = db_manager.get_data_by_brand('twitter', '삼성전자')
    print(f"삼성전자 관련 트윗 수: {len(samsung_data)}")
    
    # 데이터베이스 연결 종료
    db_manager.close()
