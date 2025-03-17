# 키워드 검색의 응답을 캐싱
import sqlite3
from utils.recommendation import makeRecommendPrompt
from utils.vector_db import search_vec
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")
cache_path = os.path.join(base_dir, "cache", "keyword_cache.db")
load_dotenv(dotenv_path=env_path)

model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY") 
)

# 캐싱을 위한 모든 키워드 조합 생성 함수
def listArray(keywords, n, result, start=0, current=[]):
    if len(current) == n:
        result.append(current[:])
        return
    if start >= len(keywords):
        return
    for i in range(start, len(keywords)):
        current.append(keywords[i])
        listArray(keywords, n, result, i+1, current)
        current.pop()

# 키워드로 캐싱된 응답 불러오는 함수
def get_cached_response(query):
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    sql = "SELECT response from keyword_cache WHERE text = ?"
    cursor.execute(sql, (query,))
    cached_response = cursor.fetchone()
    cursor.close()
    conn.close()
    return cached_response

# 캐시 생성하는 함수
def cache_keywords():
    cache_path = cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keyword_cache (
                    text TEXT PRIMARY KEY,
                    response TEXT)
    """)
    conn.commit()
    # {"한식": ["칼국수", "국밥", "비빔밥", "찌개", "고기", "감자탕"]},
    ctgs = [
        {"구내식당": ["한식뷔페"]},
        {"중식": ["짜장면", "마라탕", "동파육"]},
        {"양식": ["파스타", "샐러드", "스테이크", "피자"]},
        {"일식": ["라멘", "소바", "초밥/사시미", "카츠", "텐동", "규동", "오코노미야끼", "타코야끼", "우동"]},
        {"아시아식": ["베트남 음식", "태국 음식", "인도 음식", "퓨전 음식"]},
        {"패스트푸드": ["햄버거", "샌드위치", "치킨"]},
        {"분식": ["김밥", "떡볶이", "핫도그"]}
    ]
    for ctg in ctgs:
        ctg1 = list(ctg.keys())[0]
        keywords = list(ctg.values())[0]

        # 리스트에서 1~3개 선택해서 연결하도록 하는 함수
        for i in range(3):
            result = []
            listArray(keywords, i+1, result)
            for ctg2_comb in result:
                ctg2 = ', '.join(k for k in ctg2_comb)
                text = ctg1 + ', ' + ctg2
                # 유사도 검색
                matched_ids = search_vec(text)
                # 식당 데이터 조회 후, 프롬프트 구성
                recommend = makeRecommendPrompt(matched_ids, text)
                # 모델 통해 응답 생성
                messages = []
                messages.append({"role": "user", "content": text})
                messages.append({"role": "system", "content": recommend})
                response = model.invoke(messages)
                # 캐시에 저장
                cursor.execute("SELECT response FROM keyword_cache WHERE text = ?", (text,))
                cached_response = cursor.fetchone()
                if cached_response:
                    print(f"이미 캐싱된 키워드: {text}")
                    continue
                cursor.execute("INSERT INTO keyword_cache (text, response) VALUES (?, ?)", (text, response.content))
                conn.commit()
                print(f"새로운 키워드 캐싱: {text}")
    cursor.close()
    conn.close()
    print("키워드 캐싱 완료")