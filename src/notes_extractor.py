import json
import os
from pathlib import Path
from typing import List
import dspy
from pydantic_models.notes_extractor import PatientInfo, Medication

# Configure the language model
lm = dspy.LM("ollama_chat/qwen3:30b", api_base="http://192.168.68.54:11434", api_key="")
dspy.configure(lm=lm)


class ExtractPatientInfo(dspy.Signature):
    """Extract medication information from nurse's notes containing multiple patients. Include only documented side effects, not vital signs or observations. When listing side effects, do not describe intensity or frequency. Process ALL patients in the notes."""

    notes: str = dspy.InputField(
        desc="Nurse's notes containing medication information for multiple patients"
    )
    patients: list[PatientInfo] = dspy.OutputField(
        desc="List of patients with their medication details"
    )


class MedicationExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractPatientInfo)

    def forward(self, notes: str):
        return self.extract(notes=notes)


def extract_notes(notes: str) -> list[PatientInfo]:
    extractor = MedicationExtractor()
    result = extractor(notes=notes)

    return result.patients


if __name__ == "__main__":
    notes = Path("../data/text/notes_1.txt").read_text()
    # Model dump the result into a json file
    result = extract_notes(notes)
    output_path = Path("../data/extracted_data")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "notes.json", "w") as f:
        json.dump([item.model_dump() for item in result], f, indent=4)
