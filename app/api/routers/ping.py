from fastapi import APIRouter

router = APIRouter(tags=["ping"])

# ping 테스트
@router.get("/ping")
async def ping_test():
    return {"ping_test": "success"}