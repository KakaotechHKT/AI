import json, logging

from app.api.deps import SessionDep
from app.core.chatbot.chatbot import ChatBot
from app.models.restaurant import Restaurant
from app.models.chat import RestaurantResponse, ChatData
from app.repositories.vector_db import faiss_store
from app.repositories.restaurant_db import insert_chat, insert_chat_log, get_restaurant_by_id

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

model = ChatBot()

def create_chat_session(session: SessionDep) -> RestaurantResponse:
    logger.info("채팅방 INSERT 쿼리 실행 전")
    try:
        # 생성된 chatID 가져오기
        chat_id = insert_chat(session)
        logger.info(f"채팅방 생성 성공 - chatID: {chat_id}")
    except Exception as e:
        logger.exception("채팅방 생성 쿼리 실패")
        raise

    # 식당 정보 가져오기
    restaurant_ids = []

    ##################################################
    ##### 광고 식당 ID #####
    suggest_restaurant_ids = []  # ID 입력
    ##################################################

    for id in suggest_restaurant_ids:
        restaurant_ids.append(id)

    place_list = []

    # restaurant_ids가 비어 있지 않을 때만 쿼리 실행
    if restaurant_ids:
        restaurants = get_restaurant_by_id(session, restaurant_ids)

        # 식당 정보 나열
        place_list = [
            Restaurant(
                id=restaurant["id"],
                name=restaurant["name"],
                mainCategory=restaurant["category1"],
                subCategory=restaurant["category2"],
                latitude=float(restaurant["latitude"]) if restaurant["latitude"] is not None else None,
                longitude=float(restaurant["longitude"]) if restaurant["longitude"] is not None else None,
                url=restaurant["kakao_link"],
                thumbnail=restaurant["thumbnail"] if restaurant["thumbnail"] is not None else None,
                menu=[{**item, "price": int(item["price"])} for item in json.loads(restaurant["menus"]) if
                    restaurant["menus"]]
            ) for restaurant in restaurants
        ]

    response = RestaurantResponse(
        httpStatusCode=200,
        message="채팅방 개설에 성공하였습니다.",
        data={
            "chatID": chat_id,
            "placeList": place_list
        }
    )
    return response

def handle_chat(query: str, chat_id: int, isKeyword: bool) -> str:
    response = model.ask(query, str(chat_id), isKeyword)
    return response

def save_chat_session(chat_data: ChatData, session: SessionDep) -> RestaurantResponse:
    chat_id = chat_data.chatID
    ctg1 = chat_data.category.main if chat_data.category and chat_data.category.main else None
    ctg2 = chat_data.category.keywords if chat_data.category and chat_data.category.keywords else None
    chat_text = chat_data.chat if chat_data.chat else None

    ##################################################
    ##### AI 모델 응답 - 채팅

    isKeyword = True if ctg1 else False
    query = ctg1 + ", " + ctg2 if ctg1 else chat_text
    
    ai_response = handle_chat(query, chat_id, isKeyword)
    ai_chat = ai_response["messages"]
    search_query = ai_response["search_query"] if ai_response["search_query"]!="" else ""

    logger.info("채팅 데이터 INSERT 쿼리 실행 전")
    try:
        ##################################################
        # 채팅 데이터 저장
        insert_chat_log(session, chat_id, ctg1, ctg2, ai_chat)

        logger.info("채팅 데이터 INSERT 쿼리 성공")
    except Exception as e:
        logger.exception("채팅 데이터 INSERT 쿼리 실패")
        raise
            
    ##################################################
    ##### AI 모델 응답 - 추천 식당 리스트
    ##################################################
    place_list=[]

    if search_query!="":
        # 식당 정보 가져오기
        restaurant_ids = []
        ids = faiss_store.search(search_query)
        if ids:
            restaurant_ids = [int(i) for i in ids]

        # for id in ids: restaurant_ids.append("id")  # ID 예시

        logger.info("식당 데이터 SELECT 쿼리 실행 전")
        try:
            restaurants = get_restaurant_by_id(session, restaurant_ids)
            logger.info("식당 데이터 SELECT 쿼리 성공")
        except Exception as e:
            logger.exception("식당 데이터 SELECT 쿼리 실행 실패")
            raise

        # 식당 정보 나열
        place_list = [
            Restaurant(
                id=restaurant["id"],
                name=restaurant["name"],
                mainCategory=restaurant["category1"],
                subCategory=restaurant["category2"],
                latitude=float(restaurant["latitude"]) if restaurant["latitude"] is not None else None,
                longitude=float(restaurant["longitude"]) if restaurant["longitude"] is not None else None,
                url=restaurant["kakao_link"],
                thumbnail=restaurant["thumbnail"] if restaurant["thumbnail"] is not None else None,
                menu=[{**item, "price": int(item["price"]) if item["price"] and item["price"] != "" else 0} for item in json.loads(restaurant["menus"]) if
                    restaurant["menus"]]
            ) for restaurant in restaurants
        ]

    response = RestaurantResponse(
        httpStatusCode=200,
        message="채팅 값 전달드립니다.",
        data={
            "chat": ai_chat if ai_chat else "",
            "placeList": place_list if place_list else []
        }
    )
    return response