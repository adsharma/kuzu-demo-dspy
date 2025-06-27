import json
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_models.image_extractor import ConditionAndDrug
import base64
import mimetypes
import dspy

load_dotenv()

# Configure the language model
lm = dspy.LM("ollama_chat/gemma3n:e4b", api_base="http://192.168.68.54:11434", api_key="")
dspy.configure(lm=lm)


class ExtractMedicalInfo(dspy.Signature):
    """Extract healthcare and pharmaceutical information from an image. Extract the condition, drug names and side effects from these columns: Reason for drug, Drug names: Generic name & (Brand name), Side effects."""

    image: str = dspy.InputField(
        desc="Base64-encoded image with medical information"
    )
    medical_data: list[ConditionAndDrug] = dspy.OutputField(
        desc="List of extracted medical conditions, drugs and side effects"
    )


class MedicalInfoExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractMedicalInfo)

    def forward(self, image: str):
        return self.extract(image=image)


def extract_from_base64(
    base64_str: str, mime_type: str = "image/png"
) -> list[ConditionAndDrug]:
    """Extract entities from a base64-encoded image"""
    extractor = MedicalInfoExtractor()
    # Create a data URL for the image
    image_data_url = f"data:{mime_type};base64,{base64_str}"
    result = extractor(image=image_data_url)
    return result.medical_data


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
