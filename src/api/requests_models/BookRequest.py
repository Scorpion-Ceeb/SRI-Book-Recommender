from pydantic import BaseModel
from typing import Optional


class BookRequest(BaseModel):
    title: str
    author: str
    description: str
    publish_year: int
    image_url: Optional[str] = None
