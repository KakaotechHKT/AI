from fastapi import APIRouter, Query

from app.services.search.search_service import search_elasticsearch

router = APIRouter(tags=["search"])

@router.get("/search")
async def search_restaurant(q: str = Query(..., min_length=1)):
    results = search_elasticsearch(q)
    return {"results": results}