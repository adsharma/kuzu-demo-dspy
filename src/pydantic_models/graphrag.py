from pydantic import BaseModel, Field
from enum import Enum


class Query(BaseModel):
    cypher: str = Field(description="Valid Cypher query with no newlines")


class Tool(str, Enum):
    TEXT2CYPHER = "Text2Cypher"
    VECTORSEARCHSYMPTOMS = "VectorSearchSymptoms"
    VECTORSEARCHCONDITIONS = "VectorSearchConditions"
