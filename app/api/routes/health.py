from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_endpoint():
    return {"status": "healthy"}