from pydantic import BaseModel, Field
from typing import List


class Drug(BaseModel):
    generic_name: str
    brand_names: List[str] = Field(
        description="Strip the ® character at the end of the brand names"
    )


class ConditionAndDrug(BaseModel):
    condition: str
    drug: List[Drug]
    side_effects: List[str]
