from fastapi import APIRouter

router = APIRouter()

@router.get("/evaluation")
async def evaluation_endpoint():
    return {"message": "Evaluation endpoint"}