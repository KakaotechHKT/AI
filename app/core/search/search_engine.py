from app.repositories.elasticsearch import get_es_client

# 검색 엔진 삭제 후, 재생성 필요할 경우
# if es.indices.exists(index="restaurant"):
#     es.indices.delete(index="restaurant")

def search_es_tool(q: str):
    es = get_es_client()
    response = es.search(index="restaurant", query={
        "bool": {
            "should": [
                {
                    "match": {
                        "name.ngram": q 
                    }
                },
                {
                    "match": {
                        "name.nori": {
                            "query": q,
                            "fuzziness": "AUTO"
                        }
                    }
                }
            ]
        }
    })
    results = [hit["_source"]["name"] for hit in response["hits"]["hits"]]
    return results


