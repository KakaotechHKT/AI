from fastapi import APIRouter, Query
from elasticsearch import Elasticsearch

router = APIRouter(tags=["search"])
es = Elasticsearch(
    hosts=["http://elasticsearch:9200"],
    max_retries=3,
    retry_on_timeout=True,
    timeout=10
)

@router.get("/search")
async def search_restaunrant(q: str = Query(..., min_length=1)):
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
    return {"results": results}