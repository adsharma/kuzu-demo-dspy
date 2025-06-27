import json
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_models.image_extractor import ConditionAndDrug
import base64
import mimetypes
from openai import OpenAI

load_dotenv()

client = OpenAI()


def extract_from_base64(
    base64_str: str, mime_type: str = "image/png"
) -> list[ConditionAndDrug]:
    """Extract entities from a base64-encoded image"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": 'You are an expert at extracting healthcare and pharmaceutical information. Extract the condition, drug names and side effects from these columns: Reason for drug, Drug names: Generic name & (Brand name), Side effects. Respond with a JSON object that adheres to the following schema: \n{"properties": {"condition": {"title": "Condition", "type": "string"}, "drug": {"items": {"$ref": "#/definitions/Drug"}, "title": "Drug", "type": "array"}, "side_effects": {"items": {"type": "string"}, "title": "Side Effects", "type": "array"}}, "required": ["condition", "drug", "side_effects"], "title": "ConditionAndDrug", "type": "object", "definitions": {"Drug": {"properties": {"generic_name": {"title": "Generic Name", "type": "string"}, "brand_names": {"description": "Strip the \\u00ae character at the end of the brand names", "items": {"type": "string"}, "title": "Brand Names", "type": "array"}}, "required": ["generic_name", "brand_names"], "title": "Drug", "type": "object"}}}',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_str}"},
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    result = json.loads(response.choices[0].message.content)
    return [ConditionAndDrug.model_validate(item) for item in result]


def extract_from_file(file_path: Path) -> list[ConditionAndDrug]:
    """Extract entities from an image file"""
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    base64_str = base64.b64encode(image_bytes).decode()
    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
    return extract_from_base64(base64_str, mime_type)


def extract_from_bytes(
    image_bytes: bytes, mime_type: str = "image/png"
) -> list[ConditionAndDrug]:
    """Extract entities from a bytes object"""
    base64_str = base64.b64encode(image_bytes).decode()
    return extract_from_base64(base64_str, mime_type)


if __name__ == "__main__":
    input_dir = Path("../data/img")
    output_dir = Path("../data/extracted_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("drugs_*.png"))
    for file in files:
        # We are working with files, so we can pass in the file path directly
        result = extract_from_file(file)
        output_path = output_dir / file.with_suffix(".json").name

        with output_path.open("w") as f:
            json.dump([item.model_dump() for item in result], f, indent=4)

        print(f"Results written to {output_path}")
