import logging
from langchain_core.messages import AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from app.core.config import settings
from app.core.chatbot.history import get_session_history
from app.core.chatbot.agent_factory import create_agent_executor
from app.repositories.cache_response import get_cached_response

# 환경 변수 로드
gemini_api_key = settings.GEMINI_API_KEY

logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        # 에이전트 실행기
        self.agent_executor = create_agent_executor(gemini_api_key)

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
            ctg = [item.strip() for item in query.split(',')]
            ctg1, ctg2 = ctg[0], ctg[1:]
            
            cached_response = get_cached_response(ctg1, ctg2)
            response_text = cached_response[0]
            
            history = get_session_history(session_id)
            history.add_messages([AIMessage(content=response_text)])

            return {"messages": response_text, "search_query": query}

        # LangChain 에이전트 호출
        output = self.agent_with_chat_history.invoke({"input": query}, config)
        search_query = output["intermediate_steps"][0][0].tool_input["query"] if output["intermediate_steps"] != [] else ""
        return {"messages": output["output"], "search_query": search_query}