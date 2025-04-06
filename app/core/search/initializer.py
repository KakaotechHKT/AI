import time

from app.repositories.restaurant_db import fetchall_for_es
from app.repositories.elasticsearch import get_es_client
from app.models.elasticsearch import es_index_body
from elasticsearch.exceptions import ConnectionError as ESConnectionError

def wait_for_es_ready(es, max_retries = 10, retry_delay = 2):
    for i in range(max_retries):
        try:
            if es.ping():
                return
            else:
                raise ESConnectionError("Elasticsearch ping 실패")
        except ESConnectionError:
            print(f"Elasticsearch 연결 재시도 {i+1}")
            time.sleep(retry_delay)
    raise RuntimeError("Elasticsearch 연결 실패")

def init_es_tool():
    es = get_es_client()
    wait_for_es_ready(es)

    if es.indices.exists(index="restaurant"):
        es.indices.delete(index="restaurant")

    es.indices.create(index="restaurant", body=es_index_body)


    rows = fetchall_for_es()
    for row in rows:
        es.index(index="restaurant", id=row["id"], document={"name": row["name"]})
