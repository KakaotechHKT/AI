from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Literal
from langchain_core.tools import tool
from dotenv import load_dotenv
import sys, os, json
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
import pymysql
import numpy as np
import pandas as pd
import openai
import faiss

openai_api_key=os.getenv("OPENAI_API_KEY")



def get_openai_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

@tool
def search(user_query: str): # state: MessagesState
    """유저의 쿼리와 유사한 식당을 벡터DB에서 찾는다."""
    index_file = "faiss_index.bin"
    index = faiss.read_index(index_file)

    query = get_openai_embedding(user_query)
    distances, indices = index.search(np.array([query]), 5) # XXX: 추천 식당 개수 최대 5개로 하드코딩

    matched_ids = []
    for i in range(5):  # XXX: 추천 식당 개수 최대 5개로 하드코딩
        idx = indices[0][i]
        similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

        # 유사도 임계값을 넘는 경우만 저장
        if similarity >= 0.7: # XXX: 유사도 0.7로 하드코딩
            matched_ids.append(idx)

    conn = get_db_connection()
    cursor = conn.cursor()

    format_strings = ','.join(['%s'] * len(matched_ids))
    sql = f"SELECT name, category1 FROM restaurant WHERE restaurant_id IN ({format_strings})"

    cursor.execute(sql, tuple(matched_ids))
    matched_restaurant = cursor.fetchall()

    cursor.close()
    conn.close()

    recommendation_prompt = f"유저가 {user_query}를 입력했으며, 이에 대한 식당을 추천해야 합니다.\n"
    recommendation_prompt += "추천 가능한 식당은 다음과 같습니다:\n"
    for restaurant in matched_restaurant:
        name, category = restaurant
        recommendation_prompt += f"-{name} {category}\n"

    return recommendation_prompt

class ChatBot:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.memory = MemorySaver()
        self.workflow = StateGraph(MessagesState)
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
        self.model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY")  # 환경변수에서 키 불러오기
        )
        self.tools = [self.search_response]
        self.model_with_tools = self.model.bind_tools(self.tools)
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("call_model", self.call_model)
        workflow.add_edge(START, "call_model")
        workflow.add_conditional_edges("call_model", self.should_recommend)
        workflow.add_node("send_response", lambda state: state)
        workflow.add_node("search_response", self.search_response)

        self.app = workflow.compile(checkpointer=self.memory)
    
    def search_vec(self, user_query):
        index_file = "faiss_index.bin"
        index = faiss.read_index(index_file)

        query = get_openai_embedding(user_query)
        distances, indices = index.search(np.array([query]), 5) # XXX: 추천 식당 개수 최대 5개로 하드코딩

        matched_ids = []
        for i in range(5):  # XXX: 추천 식당 개수 최대 5개로 하드코딩
            idx = indices[0][i]
            similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

            # 유사도 임계값을 넘는 경우만 저장
            if similarity >= 0.7: # XXX: 유사도 0.7로 하드코딩
                matched_ids.append(idx)

        return matched_ids   

    def ask(self, query, thread_id):
        config = {"configurable": {"thread_id": f"{thread_id}"}}
        user_message = HumanMessage(content=query)

        output = self.app.invoke({"messages": [user_message]}, config)
    
        return output["messages"][-1]
    
    def remake_message(self, state: MessagesState):
        formatted_messages = []
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})

        return formatted_messages
    
    def call_model(self, state: MessagesState):
        formatted_messages = self.remake_message(state)
        response = self.model_with_tools.invoke(formatted_messages)
    
        return {"messages": state["messages"] + [response]} # response[-1]을 써야 풀에 넘기기 좋을 듯
    
    def search_response(self, state: MessagesState):
        """
        유저가 입력한 문장이나, 키워드들 (양식, 파스타 등)과 유사도 높은 식당들을
        가지고 있는 vector database에서 찾아서 추천 프롬프트 만들어서 모델에 제공.
        모델이 유저에게 적절한 식당을 추천할 수 있게 한다.
        """
        user_query = state["messages"][-2]
        recommend = search.invoke({"user_query": user_query.content})

        formatted_messages = self.remake_message(state)
        formatted_messages.append({"role": "assistant", "content": recommend})
        response = self.model_with_tools.invoke(formatted_messages)
        return {"messages": state["messages"] + [response]}
    
    def should_recommend(self, state: MessagesState) -> Literal["send_response", "search_response"]:
        last_message = state["messages"][-1]
        if last_message.content:
            return "send_response"
        elif last_message.tool_calls is not None:
            return "search_response"

    def make_vecDB(self):
        df = pd.read_csv("restaurants_def.csv") # NOTE: vector DB에 저장할 파일명 하드코딩
        index_file = "faiss_index.bin"

        # description 컬럼에서 텍스트 데이터 추출
        text_column = "description"
        texts = df[text_column].dropna().astype(str).tolist()  # NaN 제거 및 문자열 변환

        # 모든 식당 설명(description) 임베딩 변환
        embeddings = np.array([get_openai_embedding(text) for text in texts])

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 인덱스 생성
        index.add(embeddings)  # FAISS에 임베딩 추가
        faiss.write_index(index, index_file)

        print("vector DB 저장 완료")

chatbot = ChatBot()
# chatbot.make_vecDB()
response1 = chatbot.ask("가성비 있는 한식 칼국수", "t1")
print("응답1: ", response1)

response2 = chatbot.ask("제 이름은 길동입니다. 저는 인도 음식이 먹고 싶어요", "t1")
print("응답2: ", response2)

response2 = chatbot.ask("제 이름이 뭐였죠?", "t1")
print("응답3: ", response2)