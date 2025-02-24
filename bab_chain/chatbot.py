from langchain_chroma import Chroma
# from langchain_community.chat_models import ChatOpenAI 
# NOTE: 더이상 지원되지 않는다고 해서 아래 대체되는 코드 넣어뒀습니다.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import List, Sequence
from typing_extensions import Annotated, TypedDict
import sys, os, json

 
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

class ChatBot:
	vector_store = None
	retriever = None
	chain = None
	# json 검색 용도 (custom chain)
	dataset = None
 
	def __init__(self):
		load_dotenv(dotenv_path=".env")
		self.model = ChatOpenAI(
			model_name="gpt-4",    # "gpt-3.5-turbo"  ""
			temperature=0.5,
			openai_api_key=os.getenv("OPENAI_API_KEY")  # 환경변수에서 키 불러오기
		)
		self.prompt_template = ChatPromptTemplate.from_messages(
			[
				MessagesPlaceholder(variable_name="messages"),
			]
		)
		# Loading embedding
		self.embedding = GPT4AllEmbeddings()
		
		with open("../store_data.json", 'r', encoding='utf-8') as file:
			self.dataset = json.load(file)
		
		self.prompts = {
			"direct": self.load_prompt("templates/direct_prompt.txt"),
			"keyword": self.load_prompt("templates/keyword_prompt.txt")
        }
		self.trimmer = trim_messages(
			max_tokens=65,
			strategy="last",
			token_counter=self.model,
			include_system=True,
			allow_partial=False,
			start_on="human",
		)

		def call_model(state: MessagesState):
			system_prompt = (
				"당신은 식당 추천봇입니다. "
				"최선을 다해 답해주세요."
            )
			messages = [SystemMessage(content=system_prompt)] + state["messages"]
			response = self.model.invoke(messages)
			return {"messages": response}  # ✅ 대화 내역 업데이트

		workflow = StateGraph(state_schema=State)
		workflow.add_edge(START, "model")
		workflow.add_node("model", call_model)
		self.memory = MemorySaver()
		self.app = workflow.compile(checkpointer = self.memory)

	def ask(self, thread_id, query, query_type, **kwargs):
		# NOTE: 벡터 디비 이름 변경하려면 아래 인자 수정 필요
		self.load("chroma_db")
		# config = {"configurable": {"thread_id": thread_id}}
		config = {"configurable": {"thread_id": f"{thread_id}"}}

		selected_prompt = self.prompts[query_type]
		search_results = self.search_restaurants(query)
		if query_type == "keyword": # query: 선택 키워드
			keywords = self.split_keyword(query)
			for i in range(len(keywords)):
				search_results[f"keyword{i+1}"] = keywords[i]
		elif query_type == "direct": # query: 유저 채팅
			search_results["input"] = query
			for i, (key, value) in enumerate(kwargs.items()):
				search_results[f"keyword{i+1}"] = value

		formatted_prompt = selected_prompt.format(**search_results)
		new_message = HumanMessage(content=formatted_prompt)

		output = self.app.invoke(
			{"messages": [new_message]}, config = config
		)

		return output["messages"]

		
    # 템플릿 파일을 불러오는 함수
	def load_prompt(self, file_path):
		with open(file_path, "r", encoding="utf-8") as file:
			return ChatPromptTemplate.from_template(file.read())
				
    # json을 텍스트로 변환해 벡터 DB에 적재
	def ingest_json(self, name):
		all_chunks = []
		for i in range(len(self.dataset)):
			data = self.dataset[i]
			text = ", ".join(data["ctg1"])
			text += ", "
			text = ", ".join(data["ctg2"])
			menus = data["menu"]
			menu_text = " "
			for j in range(len(data["menu"])):
				menu_text += ", " 
				menu_text += menus[j]["name"]
			text += menu_text # keyword + 메뉴 이름
			document = Document(page_content=text, metadata={"id": data["id"]})
			all_chunks.append(document)
		self.vector_store = Chroma.from_documents(documents=all_chunks, 
											embedding=self.embedding, 
											persist_directory=f"./{name}")   
	
    # 프롬프트 템플릿에 키워드가 담기지 않는 것을 방지하기 위해 초깃값 'N/A'를 넣어둠
	def default_keyword(self):
		restaurant_data = {}
		for i in range(3): #3개 식당
			restaurant_data[f"name{i+1}"] = "N/A"
			restaurant_data[f"ctg{i+1}1"] = "N/A"
			restaurant_data[f"ctg{i+1}2"] = "N/A"
			restaurant_data[f"isDiet{i+1}"] = False
			restaurant_data[f"isCheap{i+1}"] = False
			for j in range(3): #3개 메뉴
				restaurant_data[f"mainMenu{i+1}{j+1}"] = "N/A"
				restaurant_data[f"price{i+1}{j+1}"] = "N/A"
		return restaurant_data
	
	def search_restaurants(self, query: str):
		search_results = self.retriever.invoke(query)
		matched_ids = [rec.metadata["id"] for rec in search_results]
		return self.get_restaurant_info(matched_ids)
			
	# 유사도 높은 식당들 딕셔너리 저장	 
	def get_restaurant_info(self, matched_ids):
		matched_restaurant = [
			data for data in self.dataset if data["id"] in matched_ids
		]
		
		restaurant_data = self.default_keyword()
		for i, r in enumerate(matched_restaurant):
			menu_names = [menu["name"] for menu in r["menu"]]
			menu_prices = [menu["price"] for menu in r["menu"]]
			restaurant_data[f"name{i+1}"] = r["name"]
			restaurant_data[f"ctg{i+1}1"] = r["ctg1"]
			restaurant_data[f"ctg{i+1}2"] = r["ctg2"]
			restaurant_data[f"isDiet{i+1}"] = r["diet"]
			restaurant_data[f"isCheap{i+1}"] = r["cheap"]
			# 메뉴 상위 3개 이용
			for j in range(3):
				if j >= len(menu_names):
					restaurant_data[f"mainMenu{i+1}{j+1}"] = "N/A"
					restaurant_data[f"price{i+1}{j+1}"] = "N/A"
				else:
					restaurant_data[f"mainMenu{i+1}{j+1}"] = r["menu"][j]["name"]
					restaurant_data[f"price{i+1}{j+1}"] = r["menu"][j]["price"]
		return restaurant_data
	
	def get_restaurant_info_direct(self, query, matched_ids):
		restaurant_data = self.get_restaurant_info(matched_ids)
		restaurant_data["input"] = query
		return restaurant_data
	def get_restaurant_info_keyword(self, keywords, matched_ids):
		restaurant_data = self.get_restaurant_info(matched_ids)
		for i, kw in enumerate(keywords):
			restaurant_data[f"keyword{i+1}"] = kw
		return restaurant_data
			
	# 유사도 검색
	# NOTE: 일단은 기준을 매우 낮게 해둠. 
    # NOTE: 벡터DB 검색 성능 올리면 score_threshold를 0~1 사이 실수값 적절히 조정
	def load(self, name):
		if self.vector_store is None:
			self.vector_store = Chroma(persist_directory=f"./{name}", 
			embedding_function=self.embedding)
		if self.retriever is None:
			self.retriever = self.vector_store.as_retriever(
				search_type="similarity_score_threshold",
				search_kwargs={
					"k": 3,
					"score_threshold": 0.3,
				},
			)


	# default_ask 키워드 추출 함수
	def extract_keywords(self, keywords: List[str]):
		keywords_str=", ".join(kw for kw in keywords)
		return keywords_str
			
	def split_keyword(self, text):
		result = text.split(",")
		return result

if __name__ == "__main__":
	chatbot = ChatBot()
	if len(sys.argv) > 1 and sys.argv[1] == "-build":
		chatbot.ingest_json("chroma_db")
	else: 
		response1 = chatbot.ask(
			"thread-001", "한식, 칼국수, 가성비", "keyword"
		)
		print("응답1: ", response1)
		print("메모리 저장 내용: ")
		print()
		
		response2 = chatbot.ask(
			"thread-001", "제 이름은 길동입니다. 저는 인도 음식이 먹고 싶어요", "direct"
		)
		print("응답2: ", response2)
		print("메모리 저장 내용: ")
		print()
		
		response3 = chatbot.ask(
			"thread-001", "방금 추천해준 식당이 어디였죠?", "direct"
		)
		print("응답3: ", response3)
		print("메모리 저장 내용: ")
		print()
		
		response4 = chatbot.ask(
			"thread-002", "양식, 파스타, 분위기, 고급진", "keyword"
		)
		print("응답4: ", response4)
		print("메모리 저장 내용: ")
		print()
