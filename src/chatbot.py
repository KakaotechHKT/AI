from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages, RemoveMessage
from typing import Literal
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
import numpy as np
import faiss
from utils.embedding import get_openai_embedding
from utils.recommendation import makeRecommendPrompt
from utils.cache_response import get_cached_response

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
index_path = os.path.join(base_dir, "vec_db", "faiss_index.bin")
cache_path = os.path.join(base_dir, "cache", "keyword_cache.db")
load_dotenv(dotenv_path=env_path)
openai_api_key=os.getenv("OPENAI_API_KEY")

cache_db_path = cache_path

@tool
def search(user_query: str):
    """유저의 쿼리와 유사한 식당을 벡터DB에서 찾는다."""
    index_file = index_path
    index = faiss.read_index(index_file)

    query = get_openai_embedding(user_query)
    distances, indices = index.search(np.array([query]), 5) # XXX: 추천 식당 개수 최대 5개로 하드코딩

    matched_ids = []
    for i in range(5):  # XXX: 추천 식당 개수 최대 5개로 하드코딩
        idx = indices[0][i]
        similarity = 1 / (1 + distances[0][i])  # 거리 → 유사도로 변환

        # 유사도 임계값을 넘는 경우만 저장
        if similarity >= 0.3: # XXX: 유사도 0.3로 하드코딩
            matched_ids.append(idx+1)

    return makeRecommendPrompt(matched_ids, user_query)

class ChatBot:
    def __init__(self):
        load_dotenv(dotenv_path=env_path)
        self.memory = MemorySaver()
        self.workflow = StateGraph(MessagesState)
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        self.model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY") 
        )
        self.tools = [self.search_response]
        self.model_with_tools = self.model.bind_tools(self.tools)
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("call_model", self.call_model)
        workflow.add_node("send_response", lambda state: state)
        workflow.add_node("search_response", self.search_response)
        workflow.add_node(self.delete_messages)

        workflow.add_edge(START, "delete_messages")
        workflow.add_edge("delete_messages", "call_model")
        workflow.add_conditional_edges("call_model", self.should_recommend)

        self.app = workflow.compile(checkpointer=self.memory)

    def ask(self, query, thread_id, isKeyword = False):
        config = {"configurable": {"thread_id": f"{thread_id}"}}
        user_message = HumanMessage(content=query)
        # 키워드에 대해서는 캐시 이용
        if isKeyword:
            cached_response = get_cached_response(query)
            # 캐시된 응답 이용 (추후 응답 여러 개 담아놓고 리스트 인덱스를 랜덤으로 뽑아서 이용하면 응답 다양성)
            response = AIMessage(content=cached_response[0])
            # state에 저장
            self.app.update_state(config, {"messages": [user_message, response]})
            state = self.app.get_state(config).values
            result = {"messages": state["messages"][-1].content, "isRecommend": True}
            return result
        
        output = self.app.invoke({"messages": [user_message]}, config)
        isRecommend = False
        if len(output["messages"]) >= 2 and "tool_calls" in output["messages"][-2].additional_kwargs:
            isRecommend = True
        result = {"messages": output["messages"][-1].content, "isRecommend": isRecommend}
        return result
    
    def delete_messages(self, state: MessagesState):
        messages = state["messages"]
        if len(messages) > 5:
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:-5]]}

    def remake_message(self, state: MessagesState):
        # 모델 입력 가능하도록 state["messages"] 조정
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
        # add trimmer
        state["messages"] = (state["messages"] + [response])[-5:]
        return {"messages": state["messages"]}

    def search_response(self, state: MessagesState):
        """
        유저가 입력한 문장이나, 키워드들 (양식, 파스타 등)과 유사도 높은 식당들을
        가지고 있는 vector database에서 찾아서 추천 프롬프트 만들어서 모델에 제공.
        모델이 유저에게 적절한 식당을 추천할 수 있게 한다.
        """

        user_query = state["messages"][-2]
        recommend = search.invoke({"user_query": user_query.content})

        formatted_messages = self.remake_message(state)
        formatted_messages.append({"role": "system", "content": recommend})

        response = self.model.invoke(formatted_messages)

        # delete message 노드를 만들어두긴 했는데 혹시 아래 코드가 필요한 부분이라면 되살리기
        state["messages"] = (state["messages"] + [response])[-5:]

        return {"messages": state["messages"]}
    
    def should_recommend(self, state: MessagesState) -> Literal["send_response", "search_response"]:
        last_message = state["messages"][-1]

        if last_message.content:
            return "send_response"
        elif last_message.tool_calls is not None:
            return "search_response"