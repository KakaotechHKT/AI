# 식당 DB 연결 및 조회
from mysql.connector.pooling import MySQLConnectionPool
import os
from dotenv import load_dotenv
import numpy as np

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

pool = MySQLConnectionPool(
    pool_name="gcp_pool",
    pool_size=5,
    pool_reset_session=True,  # 세션 초기화 여부
    **db_config
)

# 식당 id를 가지고 식당 조희
def fetchall(query, param):
    # conn = get_valid_connection()
    param = tuple(int(p) if isinstance(p, np.integer) else p for p in param)

    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, param)
        matched_restaurant = cursor.fetchall()

        cursor.close()
        return matched_restaurant
    finally:
        conn.close()