from pydantic import BaseModel, Field
from typing import List


class Medication(BaseModel):
    name: str
    date: str = Field(description="Date format is YYYY-MM-DD")
    dosage: str = Field(description="Dosage of the medication")
    frequency: str = Field(description="Frequency of the medication")


class PatientInfo(BaseModel):
    patient_id: str
    medication: Medication
    side_effects: List[str] = Field(
        description="Do not list intensity or frequency of the side effect"
    )
