import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("restaurant_db")

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    # JSON 데이터를 벡터 DB에 삽입
    def ingest_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file) 

        ids, embeddings, metadatas = [], [], []


        for data in dataset:
            # 메뉴 정보를 문자열로 변환
            menu_items = data.get('menu', []) or []
            menu_text = ', '.join(
                [f"{menu.get('name', 'Unknown')} ({menu.get('price', '0')}원)" for menu in menu_items]
            )

            # 키워드 정보 처리
            keywords = data.get('keywords', []) or []
            ctg1 = keywords[0] if len(keywords) > 0 else "Unknown"
            ctg2 = keywords[1] if len(keywords) > 1 else ""

            # 벡터 변환용 텍스트 생성
            text_data = f"{data.get('name', 'Unknown')} - {menu_text} - {', '.join(keywords)}"

            # 텍스트를 벡터로 변환
            vector = self.embedding_function([text_data])[0]

            # metadata 구성
            metadata = {
                "name": data.get('name', 'Unknown'),
                "menu": menu_text or "",
                "keywords": ', '.join(keywords) or "",
                "thumbnail": data.get('thumbnail', '') or "",
                "ctg1": ctg1,
                "ctg2": ctg2,
                "isDiet": bool(data.get('isDiet', False)),
                "isCheap": bool(data.get('isCheap', False))
            }


            ids.append(str(data['id']))
            embeddings.append(vector)
            metadatas.append(metadata)

        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

        print(f"벡터 DB 구축 완료! 현재 저장된 벡터 개수: {self.collection.count()}")

    # 벡터 DB 내 데이터 개수 확인
    def check_vector_count(self):
        return self.collection.count()
    



if __name__ == "__main__":
    vector_db = VectorDB()
    vector_db.ingest_json("../store_data.json")