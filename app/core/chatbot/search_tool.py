import numpy as np
from langchain_core.tools import tool
from app.repositories.vector_db import faiss_store
from app.core.chatbot.recommendation import makeRecommendPrompt

@tool
def search(query: str)->str:
    """
    유저가 식당 추천을 원하는 경우, 가지고 있는 벡터 데이터베이스에서 식당을 찾아서 반환합니다.
    모델은 search 함수에서 반환된 식당들은 유저에게 추천해야 합니다.
    """
    matched_ids = faiss_store.search(query)

    recommend_text = makeRecommendPrompt(matched_ids, query)
    return recommend_text 