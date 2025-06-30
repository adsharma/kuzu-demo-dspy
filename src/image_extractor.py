import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_models.image_extractor import ConditionAndDrug
import dspy

load_dotenv()

# Configure the language model
lm = dspy.LM("openai/qwen2.5vl:7b", api_base="http://192.168.68.54:11434/v1", api_key="ollama")
dspy.configure(lm=lm)


class Describe(dspy.Signature):
    """Describe the image in detail. Respond only in English."""

    image: dspy.Image = dspy.InputField(desc="A photo")
    table: str = dspy.OutputField(desc="Table containing 3 columns with medical info")


class ExtractMedicalInfo(dspy.Signature):
    """Extract healthcare and pharmaceutical information from an image containing a table or list.
       Strip the ® character at the end of the brand names."""

    table: str = dspy.InputField(
        desc="A markdown table containing medical conditions, drugs, and side effects "
    )
    medical_data: list[ConditionAndDrug] = dspy.OutputField(
        desc="List of extracted medical conditions, drugs and side effects"
    )


class MedicalInfoExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractMedicalInfo)

    def forward(self, table: str):
        return self.extract(table=table)



def extract_from_file(file_path: Path) -> list[ConditionAndDrug]:
    """Extract entities from an image file."""

    p = dspy.Predict(Describe)
    result = p(image=dspy.Image.from_url(str(file_path)))
    # print("Extracted table:\n", result.table)
    extractor = MedicalInfoExtractor()
    return extractor(table=result.table).medical_data


if __name__ == "__main__":
    input_dir = Path("../data/img")
    output_dir = Path("../data/extracted_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("drugs_*.png"))
    for file in files:
        # We are working with files, so we can pass in the file path directly
        result = extract_from_file(file)
        # dspy.inspect_history(n=5)
        output_path = output_dir / file.with_suffix(".json").name

        with output_path.open("w") as f:
            json.dump([item.model_dump() for item in result], f, indent=4)

        print(f"Results written to {output_path}")
