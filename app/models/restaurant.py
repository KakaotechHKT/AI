from pydantic import BaseModel
from typing import Dict, Optional, List, Union

    # 카테고리 데이터 모델
class Category(BaseModel):
    main: Optional[str] = None
    keywords: Optional[str] = None


# 식당 개별 정보 모델
class Restaurant(BaseModel):
    id: int
    name: str
    mainCategory: str # 리스트 변환 안함
    subCategory: str # 리스트 변환 안함
    latitude: Optional[float]  # 값이 NULL인 경우 발견 -> Optional 사용
    longitude: Optional[float]
    url: str
    thumbnail: Optional[str]
    menu: List[Dict[str, Union[str, int]]]  # JSON 리스트 형태로 변환