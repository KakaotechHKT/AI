# 벡터 DB 생성
from embedding import get_openai_embedding
import faiss
import numpy as np
import pandas as pd

def make_vecDB():
    df = pd.read_csv("restaurants_def.csv") # NOTE: vector DB에 저장할 파일명 하드코딩
    index_file = "faiss_index.bin"

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
    index_file = "faiss_index.bin"
    index = faiss.read_index(index_file)

    query = get_openai_embedding(user_query)
    distances, indices = index.search(np.array([query]), 5) # XXX: 추천 식당 개수 최대 5개로 하드코딩

    matched_ids = []
    for i in range(5):  # XXX: 추천 식당 개수 최대 5개로 하드코딩
        idx = indices[0][i]
        similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

        # 유사도 임계값을 넘는 경우만 저장
        if similarity >= 0.3: # XXX: 유사도 0.7로 하드코딩
            matched_ids.append(idx+1)
    
    return matched_ids   