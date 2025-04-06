from app.core.search.search_engine import search_es_tool
from app.core.search.initializer import init_es_tool

def init_elasticsearch():
    init_es_tool()

def search_elasticsearch(q: str):
    results = search_es_tool(q)
    return results