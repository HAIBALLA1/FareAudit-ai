from fastapi import APIRouter

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)

@router.get("/health")
async def health_check():
    return {"status": "ok"}