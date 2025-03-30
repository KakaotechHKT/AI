import json, os, sys, logging
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
# from mysql.connector.pooling import MySQLConnectionPool
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker, Session
from elasticsearch import Elasticsearch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chatbot import ChatBot
from dotenv import load_dotenv
from utils.vector_db import search_vec

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)
app = FastAPI()
model = ChatBot()

es = Elasticsearch(
    hosts=["http://localhost:9200"],
    max_retries=3,
    retry_on_timeout=True,
    timeout=10
)

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 접근 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# DB 연결 설정
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# SQLAlchemy 설정 (15분 = 900초)
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}",
    pool_size=5,
    pool_recycle=900,
    pool_reset_on_return='commit'
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 식당 개별 정보 모델
class Restaurant(BaseModel):
    id: int
    name: str
    mainCategory: str # 리스트 변환 안함
    subCategory: str # 리스트 변환 안함
    latitude: Optional[float]  # 값이 NULL인 경우 발견 -> Optional 사용
    longitude: Optional[float]
    url: str
    thumbnail: Optional[str]
    menu: List[Dict[str, Union[str, int]]]  # JSON 리스트 형태로 변환

# /chat/chatting 응답 모델
class RestaurantResponse(BaseModel):
    httpStatusCode: int
    message: Optional[str] = None
    data: Optional[Dict[str, Union[int, str, List[Restaurant]]]] = None

# 카테고리 데이터 모델
class Category(BaseModel):
    main: Optional[str] = None
    keywords: Optional[str] = None

# 채팅 데이터 요청 모델
class ChatData(BaseModel):
    chatID: int
    category: Optional[Category] = None
    chat: Optional[str] = None

# ping 테스트
@app.get("/ping")
async def ping_test():
    return {"ping_test": "success"}

# /chat - 새로운 채팅방 생성
@app.post("/chat", response_model=RestaurantResponse, status_code=200)
async def create_chat(db: Session = Depends(get_db)):
    try:
        logger.info("채팅방 INSERT 쿼리 실행 전")
        try:
            # 새로운 채팅방 추가
            result = db.execute(text("INSERT INTO chat () VALUES ()"))
            db.commit()

            # 생성된 chatID 가져오기
            chat_id = result.lastrowid
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
            format_strings = ",".join(["%s"] * len(restaurant_ids))
            query = f"SELECT * FROM restaurant WHERE id IN ({format_strings})", restaurant_ids
            result = db.execute(text(query), suggest_restaurant_ids)
            restaurants = result.fetchall()

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
@app.post("/chat/chatting", response_model=RestaurantResponse, status_code=200)
async def save_chat(chat_data: ChatData, db: Session = Depends(get_db)):
    try:
        chat_id = chat_data.chatID
        ctg1 = chat_data.category.main if chat_data.category and chat_data.category.main else None
        ctg2 = chat_data.category.keywords if chat_data.category and chat_data.category.keywords else None
        chat_text = chat_data.chat if chat_data.chat else None

        ##################################################
        ##### AI 모델 응답 - 채팅

        isKeyword = True if ctg1 else False

        query = ctg1 + ", " + ctg2 if ctg1 else chat_text
        
        ai_response = model.ask(query, str(chat_id), isKeyword)
        ai_chat = ai_response["messages"]
        search_query = ai_response["search_query"] if ai_response["search_query"]!="" else ""

        logger.info("채팅 데이터 INSERT 쿼리 실행 전")
        try:
            ##################################################
            # 채팅 데이터 저장
            db.execute(
                text("INSERT INTO chat_chatting (chatID, ctg1, ctg2, chat) VALUES (:chatID, :ctg1, :ctg2, :chat)"),
                {"chatID": chat_id, "ctg1": ctg1, "ctg2": ctg2, "chat": ai_chat}
            )
            db.commit()
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
            ids = search_vec(search_query)
            if ids:
                restaurant_ids = [int(i) for i in ids]

            # for id in ids: restaurant_ids.append("id")  # ID 예시

            query = text("SELECT * FROM restaurant WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            )

            logger.info("식당 데이터 SELECT 쿼리 실행 전")
            try:
                result = db.execute(query, {"ids": restaurant_ids})
                restaurants = result.mappings().all()
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

@app.get("/search")
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
