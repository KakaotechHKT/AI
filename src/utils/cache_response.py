# 키워드 검색의 응답을 캐싱
import sqlite3
from utils.recommendation import makeRecommendPrompt
from utils.vector_db import search_vec
import os
from typing import List
from queue import Queue
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")
cache_path = os.path.join(base_dir, "cache", "keyword_cache.db")
load_dotenv(dotenv_path=env_path)

category_map = {
    "한식": ["매콤", "구수한", "뜨끈한", "칼칼한", "밥", "면", "국물", "고기"],
    "중식": ["짜짱/짬뽕", "마라류", "양꼬치", "훠궈", "딤섬"],
    "분식": ["김밥", "떡볶이", "핫도그"],
    "일식": ["덮밥류", "초밥/사시미", "카츠", "나베", "야키토리", "튀김류", "라멘", "소바", "오마카세"],
    "아시아식": ["쌀국수/팟타이", "카레", "케밥"],
    "구내식당": ["구내식당"],
    "패스트푸드": ["타코", "피자", "치킨", "햄버거", "편의점", "샌드위치/토스트"],
    "양식": ["파스타", "스테이크", "화덕 피자", "수제 버거", "리조또", "샐러드"],
}

CACHE_POOL_SIZE = 5
cache_pool = Queue()
for _ in range(CACHE_POOL_SIZE):
    cache_pool.put(sqlite3.connect(cache_path, check_same_thread=False))

def refresh_cache():
    global cache_pool
    new_pool = Queue()
    for _ in range(CACHE_POOL_SIZE):
        new_pool.put(sqlite3.connect(cache_path))
    cache_pool = new_pool

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
def get_cached_response(ctg1, ctg2):
    ctg2_bit = generate_bitmask(ctg1, ctg2)

    conn = cache_pool.get()
    try:
        cursor = conn.cursor()
        sql = "SELECT response FROM keyword_cache WHERE category=? AND bitmask=?"
        cursor.execute(sql, (ctg1, ctg2_bit))
        cached_response = cursor.fetchone()
        cursor.close()
    finally:
        cache_pool.put(conn)
    return cached_response


def generate_bitmask(category: str, subcategories: List[str]) -> str:
    """소분류 리스트를 10자리 이진 비트마스크로 변환"""
    subcategory_list = category_map[category]
    bitmask = ["0"] * 10

    for sub in subcategories:
        if sub in subcategory_list:
            bitmask[subcategory_list.index(sub)] = "1"

    return "".join(bitmask)