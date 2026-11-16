from fastapi import APIRouter ,Body
from ..services.rag_service import get_retriever

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)

@router.post("/retriever" )
def get_retriever_response(query: str = Body(..., embed=True)):
    result=get_retriever(query)
    return result



