import mysql.connector
import json
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
import chatbot, os

load_dotenv(dotenv_path=".env")
app = FastAPI()
model = chatbot()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 접근 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# DB 연결 설정 - 로컬
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# DB 연결 함수
def get_db_connection():
    return mysql.connector.connect(**db_config)

# 식당 개별 정보 모델
class Restaurant(BaseModel):
    id: int
    name: str
    mainCategory: str # 리스트 변환 안함
    subCategory: str # 리스트 변환 안함
    latitude: Optional[float]  # 값이 NULL인 경우 발견 -> Optional 사용
    longitude: Optional[float]
    url: str
    menu: List[Dict[str, str]]  # JSON 리스트 형태로 변환

# /chat 응답 모델
class ChatResponse(BaseModel):
    httpStatusCode: int
    message: str
    data: Optional[Dict[str, int]] = None

# /chat/chatting 응답 모델
class RestaurantResponse(BaseModel):
    httpStatusCode: int
    message: str
    data: Optional[Dict[str, Union[str, List[Restaurant]]]] = None

# 카테고리 데이터 모델
class Category(BaseModel):
    main: str
    keywords: str

# 채팅 데이터 요청 모델
class ChatData(BaseModel):
    chatID: int
    category: Optional[Category] = None
    chat: Optional[str] = None

# /chat - 새로운 채팅방 생성
@app.post("/chat", response_model=ChatResponse, status_code=200)
async def create_chat():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 새로운 채팅방 추가
        cursor.execute("INSERT INTO chat () VALUES ()")
        conn.commit()

        # 생성된 chatID 가져오기
        chat_id = cursor.lastrowid

        cursor.close()
        conn.close()

        response = ChatResponse(
            httpStatusCode=200,
            message="채팅방 개설에 성공하였습니다.",
            data={"chatID": chat_id}
        )
        return response

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=ChatResponse(
                httpStatusCode=500,
                message="내부 서버 오류입니다.",
                data=None
            ).dict()
        )

# /chat/chatting - 유저 데이터 저장 후 추천 식당 정보 반환
@app.post("/chat/chatting", response_model=RestaurantResponse, status_code=200)
async def save_chat(chat_data: ChatData):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        chat_id = chat_data.chatID
        ctg1 = chat_data.category.main if chat_data.category else None
        ctg2 = chat_data.category.keywords if chat_data.category else None
        chat_text = chat_data.chat if chat_data.chat else None

        # 채팅 데이터 저장
        cursor.execute(
            "INSERT INTO chat_chatting (chatID, ctg1, ctg2, chat) VALUES (%s, %s, %s, %s)",
            (chat_id, ctg1, ctg2, chat_text)
        )
        conn.commit()

        ########## 식당 선정 ##########
        query = ctg1 + ctg2 if ctg1 else chat_text
        output_text = model.ask(query, chat_id)
        ids = model.search_vec(query)

        # 식당 정보 가져오기
        restaurant_ids = []
        for id in ids: restaurant_ids.append(id)  # ID 예시
        format_strings = ",".join(["%s"] * len(restaurant_ids))
        cursor.execute(f"SELECT * FROM restaurant WHERE restaurant_id IN ({format_strings})", restaurant_ids)
        restaurants = cursor.fetchall()

        # 식당 정보 나열
        place_list = [
            Restaurant(
                id=restaurant["restaurant_id"],
                name=restaurant["name"],
                mainCategory=restaurant["category1"],
                subCategory=restaurant["category2"],
                latitude=float(restaurant["latitude"]) if restaurant["latitude"] is not None else None,
                longitude=float(restaurant["longitude"]) if restaurant["longitude"] is not None else None,
                url=restaurant["kakao_link"],
                menu=json.loads(restaurant["menus"]) if restaurant["menus"] else []
            ) for restaurant in restaurants
        ]

        cursor.close()
        conn.close()

        response = RestaurantResponse(
            httpStatusCode=200,
            message="채팅 값 전달드립니다.",
            data={
                "chat": output_text if output_text else "",
                "placeList": place_list if place_list else None
            }
        )
        return response

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=RestaurantResponse(
                httpStatusCode=500,
                message="내부 서버 오류입니다.",
                data=None
            ).dict()
        )