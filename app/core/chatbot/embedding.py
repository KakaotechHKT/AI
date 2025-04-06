import openai
import numpy as np
from app.core.config import settings

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

def get_openai_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small" 
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"OpenAI 임베딩 생성 실패: {e}")
