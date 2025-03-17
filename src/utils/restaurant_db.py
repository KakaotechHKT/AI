# 식당 DB 연결 및 조회
import pymysql
import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")

load_dotenv(dotenv_path=env_path)

# 식당 DB와 연결
def get_db_connection():
    return pymysql.connect(
    host= os.getenv("DB_HOST"),
    user= os.getenv("DB_USER"),
    password= os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    charset='utf8mb4',
    autocommit=False
)

# 식당 id를 가지고 식당 조희
def fetchall(query, param):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, param)
    matched_restaurant = cursor.fetchall()

    cursor.close()
    conn.close()
    return matched_restaurant