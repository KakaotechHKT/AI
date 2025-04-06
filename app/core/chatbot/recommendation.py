import json
from typing import List
from app.repositories.restaurant_db import fetchall

def makeRecommendPrompt(matched_ids: List[int], query: str) -> str:
    """식당 id, 유저 쿼리가 제공되면 식당 DB 조회하여 추천 프롬프트 작성"""
    matched_restaurant = fetchall(matched_ids)

    recommendation_prompt = [
        f"유저가 {query}를 입력했으며, 이에 대한 식당을 추천해야 합니다.",
        "벡터 DB에서 주변에 추천 가능한 식당 정보를 찾아보니 다음과 같습니다. *** 이외의 식당은 절대 추천하지 마세요. ***",
        "** 식당 정보 **"
    ]

    for i, (name, menus_json, ctg1, ctg2) in enumerate(matched_restaurant, 1):
        menus = json.loads(menus_json)
        menu_line = ", ".join(f"{menu['name']}({menu['price']}원)" for menu in menus)

        restaurant_block = [
            f"**식당{i}**",
            f"- 이름: {name}, 대분류: {ctg1}, 소분류: {ctg2}",
            f"- 주요 메뉴: {menu_line}"
        ]
        recommendation_prompt.extend(restaurant_block)

    return "\n".join(recommendation_prompt)