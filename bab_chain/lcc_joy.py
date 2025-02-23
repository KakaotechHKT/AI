from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List
import os, json, sys


class ChatBot:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY")  # 환경변수에서 API 키 로드
        )

        # 임베딩 모델 로드
        self.embedding = GPT4AllEmbeddings()

        # 데이터셋 로드
        with open("../store_data.json", 'r', encoding='utf-8') as file:
            self.dataset = json.load(file)

        # 프롬프트 템플릿 로드
        self.default_prompt = self.load_prompt("templates/default_prompt.txt")
        self.keyword_prompt = self.load_prompt("templates/keyword_prompt.txt")
        self.direct_prompt = self.load_prompt("templates/direct_prompt.txt")

        # 벡터 DB 로드
        self.load_vector_store()


    # 템플릿 파일 불러오기
    def load_prompt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return ChatPromptTemplate.from_template(file.read())

    # 벡터 DB 로드
    def load_vector_store(self):
        """ 벡터 DB 로드 및 검색 설정 """
        self.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.01,  # 벡터 유사도 기준 (값 조정 가능)
            },
        )

	# 벡터 DB에서 유사한 식당 검색
    def search_restaurants(self, query: str):
        search_results = self.retriever.invoke(query)
        matched_ids = [rec.metadata["id"] for rec in search_results]
        return matched_ids



	# 벡터 DB에서 검색된 식당 정보를 가져와 프롬프트에 맞게 포맷팅
    def get_restaurant_info(self, matched_ids, keywords):
        matched_restaurant = [data for data in self.dataset if data["id"] in matched_ids]
		
		# 식당 정보를 저장할 딕셔너리 초기화
        restaurant_data = {}

        for i in range(3):
            if i < len(keywords):
                restaurant_data[f"keyword{i+1}"] = keywords[i]
            else:
                restaurant_data[f"keyword{i+1}"] = "N/A"

        for i, r in enumerate(matched_restaurant[:3]):  # 최대 3개 식당만 사용
            restaurant_data[f"name{i+1}"] = r.get("name", "N/A")

			# keywords에서 대분류(ctg1)와 소분류(ctg2)를 설정
            keywords = r.get("keywords", [])
            restaurant_data[f"ctg{i+1}1"] = keywords[0] if len(keywords) > 0 else "N/A"
            restaurant_data[f"ctg{i+1}2"] = keywords[1] if len(keywords) > 1 else "N/A"

			# 다이어트와 가성비 여부 (True/False 값을 문자열로 변환)
            restaurant_data[f"isDiet{i+1}"] = "Yes" if "다이어트" in keywords else "No"
            restaurant_data[f"isCheap{i+1}"] = "Yes" if "가성비" in keywords else "No"

			# 메뉴 정보 (최대 3개까지)
            menus = r.get("menu", [])
            for j in range(3):
                if j < len(menus):
                    restaurant_data[f"mainMenu{i+1}{j+1}"] = menus[j].get("name", "N/A")
                    restaurant_data[f"price{i+1}{j+1}"] = str(menus[j].get("price", "N/A"))  # 가격을 문자열로 변환
                else:
                    restaurant_data[f"mainMenu{i+1}{j+1}"] = "N/A"
                    restaurant_data[f"price{i+1}{j+1}"] = "N/A"

        return restaurant_data


	# 문장 입력 (query) 기반 식당 추천
    def direct_ask(self, query: str):
        if not self.retriever:
            self.load("chroma_db")

        search = self.retriever.invoke(query)
        print("벡터DB 검색 결과:", search)  # 검색 결과 확인

        matched_ids = [rec.metadata["id"] for rec in search]
        print("검색된 식당 ID 리스트:", matched_ids)  # 검색된 ID 리스트 확인

        # keywords=[] 빈 리스트 추가 (초기 선호 키워드 없음)
        restaurant_info = self.get_restaurant_info(matched_ids, keywords=[])
        restaurant_info["input"] = query
        print("생성된 restaurant_info:", restaurant_info)  # restaurant_info 확인

        formatted_prompt = self.direct_prompt.format(**restaurant_info)
        response = self.model.invoke(formatted_prompt)
    
        print(response)
        return response


	# 초기 선호 키워드 기반 식당 추천
    def default_ask(self, keywords: List[str]):
        if not self.retriever:
            self.load("chroma_db")

        text_keywords = ", ".join(keywords)
        search = self.retriever.invoke(text_keywords)
        print("벡터DB 검색 결과:", search)  # 검색 결과 확인

        matched_ids = [rec.metadata["id"] for rec in search]
        print("검색된 식당 ID 리스트:", matched_ids)  # 검색된 ID 리스트 확인

        # 키워드를 포함하여 식당 정보 가져오기
        restaurant_info = self.get_restaurant_info(matched_ids, keywords)
        print("생성된 restaurant_info:", restaurant_info)  # restaurant_info 확인

        formatted_prompt = self.default_prompt.format(**restaurant_info)
        response = self.model.invoke(formatted_prompt)
    
        print(response)
        return response



	# 입력 키워드 기반 식당 추천
    def keyword_ask(self, keywords: List[str], chosenKeywords: List[str]):
        matched_ids = self.search_restaurants(", ".join(keywords + chosenKeywords))
        if not self.retriever:
            self.load("chroma_db")

        text_keywords = ", ".join(keywords + chosenKeywords)
        search = self.retriever.invoke(text_keywords)
        print("벡터DB 검색 결과:", search)

        matched_ids = [rec.metadata["id"] for rec in search]
        print("검색된 식당 ID 리스트:", matched_ids)

        # keywords 전달 (keywords + chosenKeywords)
        restaurant_info = self.get_restaurant_info(matched_ids, keywords + chosenKeywords)
        for i in range(3):
            if i < len(chosenKeywords):
                restaurant_info[f"chosenKeyword{i+1}"] = chosenKeywords[i]
            else:
                restaurant_info[f"chosenKeyword{i+1}"] = "N/A"
            
        print("생성된 restaurant_info:", restaurant_info)

        formatted_prompt = self.keyword_prompt.format(**restaurant_info)
        response = self.model.invoke(formatted_prompt)
    
        print(response)
        return response


# 실행 방식에 따른 동작 설정
if __name__ == "__main__":
    chatbot = ChatBot()

    if len(sys.argv) < 2:
        print("사용법: python lcc_joy.py [옵션]")
        sys.exit(1)

    mode = sys.argv[1]
    
    if mode == '-direct':
        user_query = input("어떤 음식을 찾으시나요? : ")
        chatbot.direct_ask(user_query)

    elif mode == '-default':
        default_keywords = ["한식", "칼국수", "가성비"]
        chatbot.default_ask(default_keywords)

    elif mode == '-keyword':
        default_keywords = ["한식", "칼국수", "가성비"]
        chosen_keywords = ["양식", "파스타", "분위기"]
        chatbot.keyword_ask(default_keywords, chosen_keywords)