from fastapi import FastAPI
from src.fareaudit.routers.rag import router as rag_router
from src.fareaudit.routers.predictions import router as predictions_router

app = FastAPI(
    title="Fareaudit",
)

app.include_router(rag_router)
app.include_router(predictions_router)