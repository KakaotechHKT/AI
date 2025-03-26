# 식당 DB 연결 및 조회
# from mysql.connector.pooling import MySQLConnectionPool
import os
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")

load_dotenv(dotenv_path=env_path)

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

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 식당 id를 가지고 식당 조희
def fetchall(param):
    db = SessionLocal()
    sql = text("SELECT name, menus, category1, category2 FROM restaurant WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    param = {"ids": [int(p) if isinstance(p, np.integer) else p for p in param]}

    try:
        result = db.execute(sql, param)
        return result.fetchall()
    finally:
        db.close()