import faiss, logging
import numpy as np
import pandas as pd
from queue import Queue
from typing import List

from app.core.config import settings
from app.core.chatbot.embedding import get_openai_embedding

logger = logging.getLogger(__name__)

INDEX_PATH = settings.INDEX_PATH
CSV_PATH = settings.CSV_PATH

TOP_K = 5
VEC_POOL_SIZE = 5
SIMILARITY_THRESHOLD = 0.3

class FaissVectorStore:
    def __init__(self, index_path: str = INDEX_PATH, top_k: int = TOP_K, sim_threshold: float = SIMILARITY_THRESHOLD):
        self.index_path = index_path
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.pool = Queue()
        self._init_pool()

    def _init_pool(self):
        for _ in range(VEC_POOL_SIZE):
            self.pool.put(faiss.read_index(str(INDEX_PATH)))

    def refresh(self):
        self.pool = Queue()
        self._init_pool()
    
    def search(self, query: str) -> List[int]:
        query_vec = get_openai_embedding(query)
        try:
            index = self.pool.get()
        except Exception as e:
            logger.exception("벡터DB 커넥션 얻지 못함")
            raise

        try:
            distances, indices = index.search(np.array([query_vec]), self.top_k)
        except Exception as e:
            logger.exception(f"벡터DB 검색 실패 - user query: {query}")
            raise
        finally:
            self.pool.put(index)

        matched_ids = []
        for i in range(self.top_k):
            idx = indices[0][i]
            similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

            # 유사도 임계값을 넘는 경우만 저장
            if similarity >= self.sim_threshold: 
                matched_ids.append(idx)
                
        if not matched_ids:
            logger.info(f"벡터DB 추천 결과 없음 - user query: {query}")

        return matched_ids   

def make_vecDB(csv_path: str = CSV_PATH, index_path: str = INDEX_PATH):
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

    logger.info("vector DB 저장 완료")

faiss_store = FaissVectorStore()
