import logging
from fastapi import APIRouter, HTTPException
from app.services.chat.chat_service import create_chat_session, save_chat_session
from app.models.chat import RestaurantResponse, ChatData
from app.api.deps import SessionDep

router = APIRouter(tags=["chat"])

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# /chat - 새로운 채팅방 생성
@router.post("/chat", response_model=RestaurantResponse, status_code=200)
async def create_chat(session: SessionDep):
    try:
        response = create_chat_session(session)
        return response

    except Exception as e:
        logger.exception("오류 발생")
        raise HTTPException(
            status_code=500,
            detail=RestaurantResponse(
                httpStatusCode=500,
                message="내부 서버 오류입니다.",
                data=None
            ).dict()
        )


# /chat/chatting - 유저 데이터 저장 후 추천 식당 정보 반환
@router.post("/chat/chatting", response_model=RestaurantResponse, status_code=200)
async def save_chat(chat_data: ChatData, session: SessionDep):
    try:
        response = save_chat_session(chat_data, session)
        return response

    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=RestaurantResponse(
                httpStatusCode=500,
                message="내부 서버 오류입니다.",
                data=None
            ).dict()
        )