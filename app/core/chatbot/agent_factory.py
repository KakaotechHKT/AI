from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from app.core.chatbot.search_tool import search

def create_agent_executor(api_key: str) -> AgentExecutor:
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0.5,
        google_api_key=api_key
    )

    tools = [search]
    model_with_tools = model.bind_tools(tools)

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that remembers past conversations and can find restaurants based on user preferences by utilizing tools to search and recommend relevant restaurants."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 에이전트 생성
    agent = create_tool_calling_agent(model_with_tools, tools, prompt)

    # 에이전트 실행기
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )