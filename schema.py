from pydantic import BaseModel
from typing import Optional, List

# Request Schema 
class RequestData(BaseModel):
    question: str
    image: Optional[str] = None

# Link Schema
class Link(BaseModel):
    url: str
    text: str

# Response Schema
class ResponseData(BaseModel):
    answer: str
    links: List[Link]  # List of dictionaries with "url" and "text" keys
