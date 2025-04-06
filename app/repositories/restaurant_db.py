import logging
from typing import List
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    f"mysql+mysqlconnector://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_NAME}",
    pool_size=5,
    pool_recycle=900,
    pool_reset_on_return='commit'
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def insert_chat(session) -> int:
    # 새로운 채팅방 추가
    result = session.execute(text("INSERT INTO chat () VALUES ()"))
    session.commit()

    # 생성된 chatID 가져오기
    return result.lastrowid

def insert_chat_log(session, chat_id, ctg1, ctg2, ai_chat):
    session.execute(
        text("INSERT INTO chat_chatting (chatID, ctg1, ctg2, chat) VALUES (:chatID, :ctg1, :ctg2, :chat)"),
        {"chatID": chat_id, "ctg1": ctg1, "ctg2": ctg2, "chat": ai_chat}
    )
    session.commit()

def get_restaurant_by_id(session, restaurant_ids: list[int]):
    query = text("SELECT * FROM restaurant WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    result = session.execute(query, {"ids": restaurant_ids})
    restaurants = result.mappings().all()

    return restaurants

def fetchall(param: List[int]) -> List[tuple[str, str, str, str]]:
    """식당 id를 가지고 식당 조회"""
    matched_ids = tuple(param)
    db = SessionLocal()
    sql = text("SELECT name, menus, category1, category2 FROM restaurant WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )

    ids = tuple(int(p) if isinstance(p, np.integer) else p for p in matched_ids)
    logger.info("식당 데이터 SELECT 쿼리 전")
    try:
        result = db.execute(sql, {"ids": ids})
        return result.fetchall()
    except Exception as e:
        logger.exception("식당 데이터 SELECT 쿼리 실패")
        raise
    finally:
        db.close()