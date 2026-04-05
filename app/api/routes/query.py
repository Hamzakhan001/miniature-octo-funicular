from fastapi import APIRouter

router = APIRouter()

@router.get("/query")
async def query_endpoint():
    return {"message": "Query endpoint"}