import time
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from sqlalchemy import create_engine, text
from elasticsearch import Elasticsearch
from app.core.config import settings

engine = create_engine(f"mysql+mysqlconnector://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_NAME}")

es = Elasticsearch(hosts=["http://elasticsearch:9200"])

# 검색 엔진 삭제 후, 재생성 필요할 경우
# if es.indices.exists(index="restaurant"):
#     es.indices.delete(index="restaurant")

def init_elasticsearch():

    max_retries = 10
    retry_delay = 2

    for i in range(max_retries):
        try:
            if es.ping():
                break
            else:
                raise ESConnectionError("Elasticsearch ping 실패")
        except ESConnectionError:
            print(f"Elasticsearch 연결 재시도 {i+1}")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("Elasticsearch 연결 실패")

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