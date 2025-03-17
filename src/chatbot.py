from langchain_core.messages import BaseMessage, AIMessage
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
import faiss, os, time
import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils.embedding import get_openai_embedding
from utils.recommendation import makeRecommendPrompt
from utils.cache_response import get_cached_response
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.runnables import ConfigurableFieldSpec

# 환경 변수 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
index_path = os.path.join(base_dir, "vec_db", "faiss_index.bin")
cache_path = os.path.join(base_dir, "cache", "keyword_cache.db")
load_dotenv(dotenv_path=env_path)
gemini_api_key = os.getenv("GEMINI_API_KEY")

@tool
def search(user_query: str):
    """
    유저가 식당 추천을 원하는 경우, 가지고 있는 벡터 데이터베이스에서 식당을 찾아서 반환합니다.
    모델은 search 함수에서 반환된 식당들은 유저에게 추천해야 합니다.
    """
    index = faiss.read_index(index_path)
    query = get_openai_embedding(user_query)
    distances, indices = index.search(np.array([query]), 5)

    matched_ids = []
    for i in range(5):
        idx = indices[0][i]
        similarity = 1 / (1 + distances[0][i])
        if similarity >= 0.3:
            matched_ids.append(idx + 1)

    recommend_text = makeRecommendPrompt(matched_ids, user_query)
    return recommend_text  # LangChain이 이 값을 자동으로 응답에 반영

# 세션별 대화 기록을 관리하는 인메모리 저장소
store = {}
SESSION_TTL = 600

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID를 기반으로 대화 기록을 가져오는 함수"""
    current_time = time.time()
    expired_sessions = [sid for sid, (_, access_time) in store.items() if current_time - access_time > SESSION_TTL]

    for sid in expired_sessions:
        del store[sid]
        print(f"Session {sid} expired and removed")

    if session_id not in store:
        store[session_id] = (InMemoryHistory(), current_time)
    return store[session_id][0]

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In-memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add messages to the history"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear the message history"""
        self.messages = []    

    class Config:
        arbitrary_types_allowed = True

class ChatBot:
    def __init__(self):
        load_dotenv(dotenv_path=env_path)

        # LLM 모델 설정
        self.model = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            temperature=0.5,
            google_api_key=gemini_api_key
        )

        self.tools = [search]
        self.model_with_tools = self.model.bind_tools(self.tools)

        # 프롬프트 정의
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that remembers past conversations and can find restaurants based on user preferences by utilizing tools to search and recommend relevant restaurants."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 에이전트 생성
        self.agent = create_tool_calling_agent(self.model_with_tools, self.tools, self.prompt)

        # 에이전트 실행기
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )

        # 세션별 대화 기록을 관리하는 RunnableWithMessageHistory 적용
        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.agent_executor,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the chat session.",
                    default="",
                    is_shared=True,
                )
            ],
        )

    def ask(self, query: str, session_id: str, isKeyword: bool = False):
        """
        사용자의 입력을 처리하고, 필요하면 캐시된 응답을 반환하거나 LangChain 에이전트를 호출.
        """
        config = {"configurable": {"session_id": session_id}}

        # 캐시된 응답 확인 (키워드 기반)
        if isKeyword:
            cached_response = get_cached_response(query)
            response_text = cached_response[0]
            
            history = get_session_history(session_id)
            history.add_messages([AIMessage(content=response_text)])

            return {"messages": response_text, "search_query": query}

        # LangChain 에이전트 호출
        output = self.agent_with_chat_history.invoke({"input": query}, config)

        search_query = ""
        # 툴 호출 여부 확인
        if output["intermediate_steps"] != []:
            tool_agent_action, _ = output["intermediate_steps"][0]
            search_query = tool_agent_action.tool_input["user_query"]

        return {"messages": output["output"], "search_query": search_query}