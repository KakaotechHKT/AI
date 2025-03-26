# 식당 DB 연결 및 조회
# from mysql.connector.pooling import MySQLConnectionPool
import os, logging
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")

load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# DB 연결 설정 
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}",
    pool_size=5,
    pool_recycle=900,
    pool_reset_on_return='commit'
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# 식당 id를 가지고 식당 조희
def fetchall(param):
    db = SessionLocal()
    sql = text("SELECT name, menus, category1, category2 FROM restaurant WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    param = {"ids": [int(p) if isinstance(p, np.integer) else p for p in param]}
    logger.info("식당 데이터 SELECT 쿼리 전")
    try:
        result = db.execute(sql, param)
        return result.fetchall()
    except Exception as e:
        logger.exception("식당 데이터 SELECT 쿼리 실패")
    finally:
        db.close()