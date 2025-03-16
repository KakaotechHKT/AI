# 식당 id, 유저 쿼리가 제공되면 식당 DB 조회하여 추천 프롬프트 작성
from utils.restaurant_db import fetchall
import json

def makeRecommendPrompt(matched_ids, user_query):

    format_strings = ','.join(['%s'] * len(matched_ids))
    sql = f"SELECT name, menus, category1, category2 FROM restaurant WHERE restaurant_id IN ({format_strings})"
    matched_restaurant = fetchall(sql, tuple(matched_ids))

    recommendation_prompt = f"유저가 {user_query}를 입력했으며, 이에 대한 식당을 추천해야 합니다.\n"
    recommendation_prompt += "벡터 DB에서 주변에 추천 가능한 식당 정보를 찾아보니 다음과 같습니다. *** 이외의 식당은 절대 추천하지 마세요. ***\n"
    recommendation_prompt += "** 식당 정보 **\n"
    i = 1
    for restaurant in reversed(matched_restaurant):
        name, menus, category1, category2 = restaurant
        menus = json.loads(menus)
        recommendation_prompt += f"**식당{i}**\n"

        recommendation_prompt += f"- 이름: {name}, 대분류: {category1}, 소분류: {category2} \n"

        recommendation_prompt += "- 주요 메뉴: "
        recommendation_prompt += ", ".join(f"{m['name']}({m['price']}원)" for m in menus) + "\n"
        recommendation_prompt += "\n"

        i += 1
    return recommendation_prompt