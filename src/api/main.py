from fastapi import HTTPException, status, APIRouter
from ..logic.process_data import ProcessData
from ..logic import book_recommender, docs

router = APIRouter()


@router.get("/books")
async def get_books():
    data_json = ProcessData.load_docs()
    return data_json


@router.get("/result/{query}")
async def get_result(query: str):
    data = book_recommender.recommend_books(query)

    result = sorted([(doc.title, doc_value[1]) for doc, doc_value in zip(docs, data) if doc_value[1] != 0.0],
                    key=lambda x: x[1], reverse=True)
    return result
