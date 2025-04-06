from fastapi import APIRouter
from app.api.routers import ping, chat, search

api_router = APIRouter()
api_router.include_router(ping.router)
api_router.include_router(chat.router)
api_router.include_router(search.router)