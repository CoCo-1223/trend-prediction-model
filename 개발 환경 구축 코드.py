# 프로젝트 초기화 스크립트 (init_project.py)
import os
import subprocess
import json

def initialize_project_environment():
    # 디렉토리 구조 생성
    directories = [
        'data/raw', 'data/processed', 'data/interim', 'data/external',
        'notebooks', 'src/data', 'src/features', 'src/models', 'src/visualization',
        'reports/figures', 'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # requirements.txt 파일 생성
    requirements = [
        "numpy==1.21.0",
        "pandas==1.3.0",
        "scikit-learn==1.0.2",
        "matplotlib==3.4.2",
        "seaborn==0.11.1",
        "plotly==5.3.1",
        "dash==2.0.0",
        "streamlit==1.2.0",
        "tweepy==4.4.0",
        "praw==7.5.0",
        "beautifulsoup4==4.10.0",
        "selenium==4.1.0",
        "konlpy==0.5.2",
        "nltk==3.6.2",
        "torch==1.10.0",
        "tensorflow==2.7.0",
        "transformers==4.12.3",
        "pymongo==4.0.1",
        "sqlalchemy==1.4.27",
        "statsmodels==0.13.1",
        "prophet==1.0.1"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    # 가상환경 생성 및 패키지 설치
    subprocess.run(['python', '-m', 'venv', 'venv'])
    print("가상환경 생성 완료")
    
    # config.json 생성
    config = {
        "project_name": "소셜미디어_감성분석_트렌드예측",
        "api_keys": {
            "twitter": {"api_key": "", "api_secret": "", "access_token": "", "access_secret": ""},
            "reddit": {"client_id": "", "client_secret": "", "user_agent": ""},
            "instagram": {"username": "", "password": ""}
        },
        "database": {
            "type": "mongodb",
            "connection_string": "mongodb://localhost:27017/",
            "db_name": "social_sentiment_db"
        },
        "selected_brands": []
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("프로젝트 환경 초기화 완료")

if __name__ == "__main__":
    initialize_project_environment()
