from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import lcc

app = FastAPI()

# 정적 페이지를 제공하기 위한 설정을 한다. 
# 이후 static 디렉토리에 파일을 두면 /static/ 밑으로 요청한다.
app.mount("/static", StaticFiles(directory="static"), name="static")

model = ChatBot()

# 모델에게 유저가 직접 질문
@app.get("/direct")
def direct_ask(text: str = Query()):
    response = model.direct_ask(text)
    return {"content" : response.content}

# 유저 선호 키워드 바탕 답변
@app.get("/default")
def default_ask(text: str = Query()):
    response = model.default_ask(text)
    return {"content" : response.content}

# 벡터 DB 적재 (name에는 벡터 디비 이름 설정)
@app.get("/ingest")
def ingest_json(name: str = Query()):
    model.ingest_json(name)