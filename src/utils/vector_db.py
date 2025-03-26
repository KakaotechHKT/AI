# 벡터 DB 생성
from utils.embedding import get_openai_embedding
import faiss, os, logging
import numpy as np
import pandas as pd
from queue import Queue

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(base_dir, "vec_db", "restaurants_def.csv") # NOTE: vector DB에 저장할 파일명 하드코딩
index_path = os.path.join(base_dir, "vec_db", "faiss_index.bin")
faiss_pool = Queue()

VEC_POOL_SIZE = 5
logger = logging.getLogger(__name__)

for _ in range(VEC_POOL_SIZE):
    faiss_pool.put(faiss.read_index(index_path))

def refresh_vecDB():
    global faiss_pool
    new_pool = Queue()
    for _ in range(VEC_POOL_SIZE):
        new_pool.put(faiss.read_index(index_path))
    faiss_pool = new_pool

def make_vecDB():
    df = pd.read_csv(csv_path)
    index_file = index_path

    # description 컬럼에서 텍스트 데이터 추출
    text_column = "description"
    texts = df[text_column].dropna().astype(str).tolist()  # NaN 제거 및 문자열 변환

    # 모든 식당 설명(description) 임베딩 변환
    embeddings = np.array([get_openai_embedding(text) for text in texts])

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 인덱스 생성
    index.add(embeddings)  # FAISS에 임베딩 추가
    faiss.write_index(index, index_file)

    print("vector DB 저장 완료")

def search_vec(user_query):
    query = get_openai_embedding(user_query)
    try:
        index = faiss_pool.get()
    except Exception as e:
        logger.exception("벡터DB 커넥션 얻지 못함")
        raise

    try:
        distances, indices = index.search(np.array([query]), 5) # XXX: 추천 식당 개수 최대 5개로 하드코딩
    except Exception as e:
        logger.exception(f"벡터DB 검색 실패 - user query: {user_query}")
        raise
    finally:
        faiss_pool.put(index)

    matched_ids = []
    for i in range(5):  # XXX: 추천 식당 개수 최대 5개로 하드코딩
        idx = indices[0][i]
        similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

        # 유사도 임계값을 넘는 경우만 저장
        if similarity >= 0.3: # XXX: 유사도 0.7로 하드코딩
            matched_ids.append(idx)
            
    if not matched_ids:
        logger.info(f"벡터DB 추천 결과 없음 - user query: {user_query}")

    return matched_ids   