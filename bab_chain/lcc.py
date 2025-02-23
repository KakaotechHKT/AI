from langchain_chroma import Chroma
# from langchain_community.chat_models import ChatOpenAI 
# NOTE: 더이상 지원되지 않는다고 해서 아래 대체되는 코드 넣어뒀습니다.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List
import sys, os, json
 
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
		# Loading embedding
		self.embedding = GPT4AllEmbeddings()
		
		with open("../store_data.json", 'r', encoding='utf-8') as file:
			self.dataset = json.load(file)

		self.default_prompt = self.load_prompt("templates/default_prompt.txt")
		self.keyword_prompt = self.load_prompt("templates/keyword_prompt.txt")
		self.direct_prompt = self.load_prompt("templates/direct_prompt.txt")

	def load_prompt(self, file_path):
		""" 템플릿 파일을 불러오는 함수 """
		with open(file_path, "r", encoding="utf-8") as file:
			return ChatPromptTemplate.from_template(file.read())
				
    # json을 텍스트로 변환해 벡터 DB에 적재
	def ingest_json(self, name):
		all_chunks = []
		for i in range(len(self.dataset)):
			data = self.dataset[i]
			text = ", ".join(data["keywords"]) 
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
											persist_directory="./{name}")   
	
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
			restaurant_data[f"isDiet{i+1}"] = r["isDiet"]
			restaurant_data[f"isCheap{i+1}"] = r["isCheap"]
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
	def get_restaurant_info_default(self, keywords, matched_ids):
		restaurant_data = self.get_restaurant_info(matched_ids)
		for i, kw in enumerate(keywords):
			restaurant_data[f"keyword{i+1}"] = kw
		return restaurant_data
	def get_restaurant_info_keyword(self, keywords, chosenKeywords, matched_ids):
		restaurant_data = self.get_restaurant_info(matched_ids)
		for i, kw in enumerate(keywords):
			restaurant_data[f"keyword{i+1}"] = kw
		for i, kw in enumerate(chosenKeywords):
			restaurant_data[f"chosenKeyword{i+1}"] = kw
		return restaurant_data
			
	# 유사도 검색
	# NOTE: 일단은 기준을 매우 낮게 해둠. 
    # NOTE: 벡터DB 검색 성능 올리면 score_threshold를 0~1 사이 실수값 적절히 조정
	def load(self, name):
		vector_store = Chroma(persist_directory="./{name}", 
			embedding_function=self.embedding)
		self.retriever = vector_store.as_retriever(
			search_type="similarity_score_threshold",
			search_kwargs={
				"k": 3,
				"score_threshold": 0.01,
			},
		)

	# default_ask 키워드 추출 함수
	def extract_keywords(self, keywords: List[str]):
		keywords_str=", ".join(kw for kw in keywords)
		return keywords_str

	# 전체 문장 기반
	def direct_ask(self, keywords: List[str], query: str):
		if not self.retriever:
			self.load("chroma_db")
		search = self.retriever.invoke(query)
		# NOTE: 벡터 DB 검색 결과 확인 용도
		print("벡터DB 결과 check: ", search)
		
		matched_ids = [rec.metadata["id"] for rec in search]
		# NOTE: 벡터 DB 검색 결과 ID만 퀵체크
		print("벡터DB 결과 ID check: ", matched_ids)
		restaurant_info = self.get_restaurant_info_direct(query, matched_ids)
		formatted_prompt = self.direct_prompt.format(
			**restaurant_info
			)
		
		response = self.model.invoke(formatted_prompt)
		print(response)
		return response

	# 초기 선호 기반
	def default_ask(self, keywords: List[str]):
		if not self.retriever:
			self.load("chroma_db")
		text_keywords=self.extract_keywords(keywords)
		search = self.retriever.invoke(text_keywords)
		# NOTE: 벡터 DB 검색 결과 확인 용도
		print("벡터DB 결과 check: ", search)

		matched_ids = [rec.metadata["id"] for rec in search]
		# NOTE: 벡터 DB 검색 결과 ID만 퀵체크
		print("벡터DB 결과 ID check: ", matched_ids)
		restaurant_info = self.get_restaurant_info_default(keywords, matched_ids)
		formatted_prompt = self.default_prompt.format(
			**restaurant_info
			)
		
		response = self.model.invoke(formatted_prompt)
		print(response)
		return response
	
	# 키워드 기반
	def keyword_ask(self, keywords: List[str], chosenKeywords: List[str]):
		if not self.retriever:
			self.load("chroma_db")
		text_keywords=self.extract_keywords(keywords)
		search = self.retriever.invoke(text_keywords)
		print("벡터DB 결과 check: ", search)

		matched_ids = [rec.metadata["id"] for rec in search]
		# NOTE: 벡터 DB 검색 결과 ID만 퀵체크
		print("벡터DB 결과 ID check: ", matched_ids)
		restaurant_info = self.get_restaurant_info_keyword(keywords, chosenKeywords, matched_ids)
		formatted_prompt = self.keyword_prompt.format(
			**restaurant_info
			)
		response = self.model.invoke(formatted_prompt)
		print(response)
		return response

# 챗봇 이용하는 경우
if sys.argv[1] == '-direct':
	chatbot = ChatBot()
	default_keyword = ["한식", "칼국수", "가성비"]
	query = "인도 음식이 먹고 싶어요! 주변에 카레 같은 인도 음식 파는 곳이 있나요?"
	chatbot.direct_ask(default_keyword, query)
# 초기 키워드 기반 추천
elif sys.argv[1] == '-default':
	chatbot = ChatBot()
	default_keyword = ["한식", "칼국수", "가성비"]
	chatbot.default_ask(default_keyword)
# 키워드 선택 기반 추천
elif sys.argv[1] == '-keyword':
	chatbot = ChatBot()
	default_keyword = ["한식", "칼국수", "가성비"]
	chosen_keyword = ["양식", "파스타", "분위기"]
	chatbot.keyword_ask(default_keyword, chosen_keyword)
elif sys.argv[1] == "-build":
	chatbot = ChatBot()
	chatbot.ingest_json()