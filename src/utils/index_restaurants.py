from sqlalchemy import create_engine, text
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

engine = create_engine(f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")

es = Elasticsearch(hosts=["http://localhost:9200"])

# 검색 엔진 삭제 후, 재생성 필요할 경우
# if es.indices.exists(index="restaurant"):
#     es.indices.delete(index="restaurant")


es.indices.create(
    index="restaurant",
    body={
        "settings": {
            "index": {
                "max_ngram_diff": 10
            },
            "analysis": {
                "tokenizer": {
                    "ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 5,
                        "token_chars": ["letter", "digit"]
                    },
                    "nori_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "mixed"
                    }
                },
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ngram_tokenizer"
                    },
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer"
                    },
                }
            }
        },
        "mappings": {
            "properties": {
                "name": {
                    "type": "text",
                    "fields": {
                        "nori": {
                            "type": "text",
                            "analyzer": "nori_analyzer"
                        },
                        "ngram": {
                            "type": "text",
                            "analyzer": "ngram_analyzer",
                            "search_analyzer": "ngram_analyzer"
                        }
                    }
                }
            }
        }
    }
)

with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, name FROM restaurant")).fetchall()
    for row in rows:
        es.index(index="restaurant", id=row.id, document={"name": row.name})