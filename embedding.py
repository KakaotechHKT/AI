import openai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=[text],  # 리스트 형태로 입력
        model="text-embedding-3-small"  # 최신 임베딩 모델 사용 가능
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
