from fastapi import FastAPI

app = FastAPI()

from .main import router as main_router

app.include_router(main_router)
