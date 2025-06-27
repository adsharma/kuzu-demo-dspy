import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any


def load_json_file(file_path: str) -> List[dict]:
    """Load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_date(date_str: str) -> str:
    """Transform ISO date to 'Mon DD YYYY' format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d %Y")
    except ValueError:
        return date_str  # Return original if parsing fails


def evaluate_extraction(human_path: str, extracted_path: str) -> Dict[str, Any]:
    """Compare human annotated data with extracted data and return metrics."""
    human_data = load_json_file(human_path)
    extracted_data = load_json_file(extracted_path)

    exact_matches = set()
    missing_items = set()
    hallucinations = set()
    mismatches = []

    # Helper function to process items with the same structure
    def compare_items(human_items, extracted_items, find_similar=False):
        for h_item in human_items:
            if not h_item:  # Skip empty strings
                continue

            if h_item in extracted_items:
                exact_matches.add(h_item)
            elif find_similar:
                # Look for similar but not identical items
                similar = [e for e in extracted_items if e and e != h_item]
                if similar:
                    # If there are similar items, it's a mismatch
                    mismatches.append((h_item, similar[0]))
                else:
                    # If no similar items, it's missing
                    missing_items.add(h_item)
            else:
                missing_items.add(h_item)

        for e_item in extracted_items:
            if not e_item:  # Skip empty strings
                continue

            if e_item not in human_items and e_item not in [m[1] for m in mismatches]:
                hallucinations.add(e_item)

    # Process patient IDs
    human_patient_ids = {item["patient_id"] for item in human_data}
    extracted_patient_ids = {item["patient_id"] for item in extracted_data}
    compare_items(human_patient_ids, extracted_patient_ids)

    # Create lookup maps
    human_map = {item["patient_id"]: item for item in human_data}
    extracted_map = {item["patient_id"]: item for item in extracted_data}

    # Process matching patient records
    for patient_id in human_patient_ids & extracted_patient_ids:
        human_item = human_map[patient_id]
        extracted_item = extracted_map[patient_id]

        # Process medication names
        human_med_name = (
            human_item["medication"]["name"]
            if "medication" in human_item and "name" in human_item["medication"]
            else ""
        )
        extracted_med_name = (
            extracted_item["medication"]["name"]
            if "medication" in extracted_item and "name" in extracted_item["medication"]
            else ""
        )

        if human_med_name and extracted_med_name:
            if human_med_name == extracted_med_name:
                exact_matches.add(human_med_name)
            else:
                mismatches.append((human_med_name, extracted_med_name))
        elif human_med_name:
            missing_items.add(human_med_name)
        elif extracted_med_name:
            hallucinations.add(extracted_med_name)

        # Process medication dates
        human_date = (
            human_item["medication"]["date"]
            if "medication" in human_item and "date" in human_item["medication"]
            else ""
        )
        extracted_date = (
            extracted_item["medication"]["date"]
            if "medication" in extracted_item and "date" in extracted_item["medication"]
            else ""
        )

        # Format dates for comparison
        human_formatted_date = format_date(human_date) if human_date else ""
        extracted_formatted_date = format_date(extracted_date) if extracted_date else ""

        if human_formatted_date and extracted_formatted_date:
            if human_formatted_date == extracted_formatted_date:
                exact_matches.add(human_formatted_date)
            else:
                mismatches.append((human_formatted_date, extracted_formatted_date))
        elif human_formatted_date:
            missing_items.add(human_formatted_date)
        elif extracted_formatted_date:
            hallucinations.add(extracted_formatted_date)

        # Process side effects
        human_effects = set(human_item.get("side_effects", []))
        extracted_effects = set(extracted_item.get("side_effects", []))
        compare_items(human_effects, extracted_effects, find_similar=True)

    metrics = {
        "exact_match": len(exact_matches),
        "missing": len(missing_items),
        "potential_hallucination": len(hallucinations),
        "mismatch": len(mismatches),
        "hallucination_items": hallucinations,
        "missing_item_details": missing_items,
        "exact_match_items": exact_matches,
        "mismatch_pairs": mismatches,
    }

    return metrics


def run_evaluation(
    human_dir: Path = Path("../../data/human_annotated_data"),
    extracted_dir: Path = Path("../../data/extracted_data"),
) -> Dict[str, Dict[str, Any]]:
    """Run evaluation on all matching files with 'notes' prefix in both directories."""
    results = {}

    # Find all JSON files with 'notes' prefix in human_annotated_data
    for human_file in human_dir.glob("notes*.json"):
        filename = human_file.name
        extracted_file = extracted_dir / filename

        # Check if corresponding file exists in extracted_data
        if extracted_file.exists():
            results[filename] = evaluate_extraction(
                str(human_file), str(extracted_file)
            )

    return results


def inspect_hallucinations(results: Dict[str, Dict[str, Any]]) -> None:
    """Display detailed information about potential hallucinations."""
    print("\nPotential hallucination details:")
    print("================================")

    for filename, metrics in results.items():
        if "hallucination_items" in metrics and metrics["hallucination_items"]:
            print(f"\nFile: {filename}")
            print("  Potentially hallucinated items in extracted data (please verify):")
            for item in sorted(metrics["hallucination_items"]):
                print(f"    <Missing> (human annotated) --- '{item}' (extracted)")
        else:
            print(f"\nFile: {filename}")
            print("  No potential hallucinations detected.")


def inspect_missing_items(results: Dict[str, Dict[str, Any]]) -> None:
    """Display detailed information about missing items."""
    print("\nMissing items details:")
    print("=====================")

    for filename, metrics in results.items():
        if "missing_item_details" in metrics and metrics["missing_item_details"]:
            print(f"\nFile: {filename}")
            print("  Items missing from extracted data:")
            for item in sorted(metrics["missing_item_details"]):
                print(f"    '{item}' (human annotated) --- <Missing> (extracted)")
        else:
            print(f"\nFile: {filename}")
            print("  No missing items detected.")


def inspect_mismatches(results: Dict[str, Dict[str, Any]]) -> None:
    """Display detailed information about mismatched items."""
    print("\nMismatch details:")
    print("================")

    for filename, metrics in results.items():
        if "mismatch_pairs" in metrics and metrics["mismatch_pairs"]:
            print(f"\nFile: {filename}")
            print("  Mismatched items (different values for corresponding elements):")
            for human_item, extracted_item in metrics["mismatch_pairs"]:
                print(
                    f"    '{human_item}' (human annotated) --- '{extracted_item}' (extracted)"
                )
        else:
            print(f"\nFile: {filename}")
            print("  No mismatches detected.")


def main() -> Dict[str, Dict[str, Any]]:
    """Main function to run evaluation and print results."""
    results = run_evaluation()

    # Print results
    print("Notes evaluation results:")
    print("========================")

    for filename, metrics in results.items():
        total = (
            metrics["exact_match"]
            + metrics["missing"]
            + metrics["potential_hallucination"]
            + metrics["mismatch"]
        )
        print(f"\nFile: {filename}")
        print(
            f"  Exact matches: {metrics['exact_match']} ({metrics['exact_match']/total:.1%})"
        )
        print(f"  Missing items: {metrics['missing']} ({metrics['missing']/total:.1%})")
        print(
            f"  Potential hallucinations: {metrics['potential_hallucination']} ({metrics['potential_hallucination']/total:.1%})"
        )
        print(f"  Mismatches: {metrics['mismatch']} ({metrics['mismatch']/total:.1%})")

    # Calculate totals - only include count metrics, not the set objects
    totals = defaultdict(int)
    count_metrics = ["exact_match", "missing", "potential_hallucination", "mismatch"]
    for metrics in results.values():
        for metric in count_metrics:
            totals[metric] += metrics[metric]

    grand_total = sum(totals[metric] for metric in count_metrics)

    if grand_total > 0:  # Avoid division by zero
        print("\nTotals across all notes files:")
        print(
            f"  Total exact matches: {totals['exact_match']} ({totals['exact_match']/grand_total:.1%})"
        )
        print(
            f"  Total missing items: {totals['missing']} ({totals['missing']/grand_total:.1%})"
        )
        print(
            f"  Total potential hallucinations: {totals['potential_hallucination']} ({totals['potential_hallucination']/grand_total:.1%})"
        )
        print(
            f"  Total mismatches: {totals['mismatch']} ({totals['mismatch']/grand_total:.1%})"
        )
    else:
        print("\nNo matching notes files found for evaluation.")

    # Call inspect functions within main
    inspect_hallucinations(results)
    inspect_missing_items(results)
    inspect_mismatches(results)

    return results


if __name__ == "__main__":
    main()
