from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import sys, os, json
 
class ChatBot:
	vector_store = None
	retriever = None
	chain = None
	# json 검색 용도 (custom chain)
	dataset = None
 
	def __init__(self):
		load_dotenv(dotenv_path="../.env")
		self.model = ChatOpenAI(
			model_name="gpt-4",    # "gpt-3.5-turbo"  ""
			temperature=0.5,
			openai_api_key=os.getenv("OPENAI_API_KEY")  # 환경변수에서 키 불러오기
		)
		# Loading embedding
		self.embedding = GPT4AllEmbeddings()
		
		with open("../store_data.json", 'r', encoding='utf-8') as file:
			self.dataset = json.load(file)
		# 재크롤링 후, 아래 템플릿으로 변경 예정
		self.prompt = ChatPromptTemplate.from_messages(
		[
			("system", 
"""
당신은 식당 추천 챗봇입니다.
아래의 REQUEST에 따라 식당을 한글로 추천해주세요.

REQUEST:
1. 유저가 선호하는 '키워드'의 식당들을 데이터베이스에서 찾아 '식당 정보'에 작성했습니다.
2. 해당 식당들을 유저에게 추천해주세요. 제시된 식당들은 모두 추천해주세요. 식당들은 추천 순으로 제시했으니 답변 시에도 이 순서를 지켜주세요.
3. 추천할때에는 '키워드'와 '식당 정보'를 모두 고려해주세요.
4. 각각의 식당들을 추천할 때는 먼저 식당의 '이름'을 두괄식으로 제공해주세요. 그리고 추천 이유를 제공해주세요.
4. 답변이 너무 길면 유저가 피로감을 느낄 수 있습니다. CONTEXT 중에 필수적이지 않은 내용은 포함하지 말아주세요.
 
CONTEXT:
키워드: {keyword1}, {keyword2}, {keyword3}
식당 정보:
식당1
이름: {name1}, 대분류: {ctg1}, 소분류: {ctg2}, {isDiet}, {isCheap}, 
{mainMenu1}은 {price1}원이고, {mainMenu2}은 {price2}원이고, {mainMenu3}은 {price3}원
식당2
이름: {name2}, 대분류: {ctg1}, 소분류: {ctg2}, {isDiet}, {isCheap}, 
{mainMenu1}은 {price1}원이고, {mainMenu2}은 {price2}원이고, {mainMenu3}은 {price3}원
식당3
이름: {name3}, 대분류: {ctg1}, 소분류: {ctg2}, {isDiet}, {isCheap}, 
{mainMenu1}은 {price1}원이고, {mainMenu2}은 {price2}원이고, {mainMenu3}은 {price3}원
 

"""),
			("human", "{input}"),
		]
	)
		# 현재 임시 템플릿
		self.test_prompt = ChatPromptTemplate.from_messages([("system", 
	"""
        당신은 식당 추천 챗봇입니다.
        아래의 REQUEST에 따라 식당을 한글로 추천해주세요.

        REQUEST:
        1. 유저가 선호하는 '키워드'의 식당들을 데이터베이스에서 찾아 '식당 정보'에 작성했습니다.
        2. 해당 식당들을 유저에게 추천해주세요. 제시된 식당들은 모두 추천해주세요. 식당들은 추천 순으로 제시했으니 답변 시에도 이 순서를 지켜주세요.
        3. 추천할때에는 '이름'과 '메뉴'를 고려해주세요.
        4. 각각의 식당들을 추천할 때는 먼저 식당의 '이름'을 두괄식으로 제공해주세요. 그리고 추천 이유를 제공해주세요.
        4. 답변이 너무 길면 유저가 피로감을 느낄 수 있습니다. CONTEXT 중에 필수적이지 않은 내용은 포함하지 말아주세요.
        
        CONTEXT:
        식당 정보:
        식당1
        이름: {name1},
        {mainMenu11}은 {price11}원이고, {mainMenu12}은 {price12}원이고, {mainMenu13}은 {price13}원
        식당2
        이름: {name2},
        {mainMenu21}은 {price21}원이고, {mainMenu22}은 {price22}원이고, {mainMenu23}은 {price23}원
        식당3
        이름: {name3},
        {mainMenu31}은 {price31}원이고, {mainMenu32}은 {price32}원이고, {mainMenu33}은 {price33}원
	""")])
		
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
			text += menu_text
			document = Document(page_content=text, metadata={"id": data["id"]})
			all_chunks.append(document)
		self.vector_store = Chroma.from_documents(documents=all_chunks, 
											embedding=self.embedding, 
											persist_directory="./{name}")   
	
    # 프롬프트 템플릿에 키워드가 담기지 않는 것을 방지하기 위해 초깃값 'N/A'를 넣어둠
	def default_keyword(self):
		restaurant_data = {}
		for i in range(3):
			restaurant_data[f"name{i+1}"] = "N/A"
			for j in range(3):
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
			# 메뉴 상위 3개 이용
			for j in range(3):
				if j >= len(menu_names):
					restaurant_data[f"mainMenu{i+1}{j+1}"] = "N/A"
					restaurant_data[f"price{i+1}{j+1}"] = "N/A"
				else:
					restaurant_data[f"mainMenu{i+1}{j+1}"] = r["menu"][j]["name"]
					restaurant_data[f"price{i+1}{j+1}"] = r["menu"][j]["price"]
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
		
	def direct_ask(self, query: str):
		if not self.retriever:
			self.load()
		search = self.retriever.invoke(query)
		# NOTE: 벡터 DB 검색 결과 확인 용도
		print("벡터DB 결과 check: ", search)
		
		matched_ids = [rec.metadata["id"] for rec in search]
		# NOTE: 벡터 DB 검색 결과 ID만 퀵체크
		print("벡터DB 결과 ID check: ", matched_ids)
		restaurant_info = self.get_restaurant_info(matched_ids)
		formatted_prompt = self.test_prompt.format(
			**restaurant_info
			)
		
		response = self.model.invoke(formatted_prompt)
		print(response)
		return response


if len(sys.argv) < 2:
	chatbot = ChatBot()
	test_query = "인도 음식이 먹고 싶어요! 주변에 카레 같은 인도 음식 파는 곳이 있나요?"
	chatbot.direct_ask(test_query)
elif sys.argv[1] == "--build":
	chatbot = ChatBot()
	chatbot.ingest_json("chroma_db")