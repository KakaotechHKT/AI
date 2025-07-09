import time
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """인메모리 챗 히스토리 생성"""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """히스토리에 메시지 추가"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """메세지 히스토리 초기화"""
        self.messages = []    

    class Config:
        arbitrary_types_allowed = True

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