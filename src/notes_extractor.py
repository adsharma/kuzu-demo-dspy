import json
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_models.notes_extractor import PatientInfo
from openai import OpenAI

load_dotenv()

client = OpenAI()


def extract_notes(notes: str) -> list[PatientInfo]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": 'Extract the medication information from the following nurse\'s notes. Include only documented side effects, not vital signs or observations. When listing side effects, do not describe its intensity or frequency. ONLY list the name of the side effect. Respond with a JSON object that adheres to the following schema: \n{"properties": {"patient_id": {"title": "Patient ID", "type": "string"}, "medication": {"$ref": "#/definitions/Medication"}, "side_effects": {"description": "Do not list intensity or frequency of the side effect", "items": {"type": "string"}, "title": "Side Effects", "type": "array"}}, "required": ["patient_id", "medication", "side_effects"], "title": "PatientInfo", "type": "object", "definitions": {"Medication": {"properties": {"name": {"title": "Name", "type": "string"}, "date": {"description": "Date format is YYYY-MM-DD", "title": "Date", "type": "string"}, "dosage": {"description": "Dosage of the medication", "title": "Dosage", "type": "string"}, "frequency": {"description": "Frequency of the medication", "title": "Frequency", "type": "string"}}, "required": ["name", "date", "dosage", "frequency"], "title": "Medication", "type": "object"}}}',
            },
            {
                "role": "user",
                "content": notes,
            },
        ],
        temperature=0.1,
    )
    result = json.loads(response.choices[0].message.content)
    return [PatientInfo.model_validate(item) for item in result]


if __name__ == "__main__":
    notes = Path("../data/text/notes_1.txt").read_text()
    # Model dump the result into a json file
    result = extract_notes(notes)
    output_path = Path("../data/extracted_data")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "notes.json", "w") as f:
        json.dump([item.model_dump() for item in result], f, indent=4)
