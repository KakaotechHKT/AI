from elasticsearch import Elasticsearch

def get_es_client():
    return Elasticsearch(
        hosts=["http://elasticsearch:9200"],
        max_retries=3,
        retry_on_timeout=True,
        timeout=10
    )