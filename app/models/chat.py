# 채팅 데이터 요청 모델
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
from app.models.restaurant import Category, Restaurant

class ChatData(BaseModel):
    chatID: int
    category: Optional[Category] = None
    chat: Optional[str] = None

# /chat/chatting 응답 모델
class RestaurantResponse(BaseModel):
    httpStatusCode: int
    message: Optional[str] = None
    data: Optional[Dict[str, Union[int, str, List[Restaurant]]]] = None