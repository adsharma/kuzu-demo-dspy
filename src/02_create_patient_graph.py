"""
This script is written to be run sequentially after 01_create_drug_graph.py
Do not run this script before running 01_create_drug_graph.py
"""

import kuzu
import polars as pl


def load_and_transform_data(file_path: str) -> pl.DataFrame:
    """Load and transform patient data"""
    df = pl.read_json(file_path)

    df_mod = df.with_columns(
        pl.col("medication").struct.field("name").str.to_lowercase().alias("drug_name"),
        pl.col("medication")
        .struct.field("date")
        .str.to_date(format="%Y-%m-%d")
        .alias("date"),
        pl.col("medication").struct.field("dosage").alias("dosage"),
        pl.col("medication").struct.field("frequency").alias("frequency"),
    )

    return df_mod


def create_schema(conn: kuzu.Connection) -> None:
    """Create the patient schema in the database"""
    # Create the patient table
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Patient(patient_id STRING PRIMARY KEY)"
    )

    # Create the patient_drug rel table (DrugGeneric node table must pre-exist)
    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS IS_PRESCRIBED(
            FROM Patient TO DrugGeneric,
            date DATE,
            dosage STRING,
            frequency STRING
        )
        """
    )
    # Create the patient_symptom rel table (Symptom node table must pre-exist)
    conn.execute("CREATE REL TABLE IF NOT EXISTS EXPERIENCES(FROM Patient TO Symptom)")


def merge_patient_nodes(conn: kuzu.Connection, df: pl.DataFrame) -> None:
    """Merge patient nodes and return count of merged nodes"""
    result = conn.execute(
        """
        LOAD FROM df
        WITH DISTINCT patient_id
        MERGE (p:Patient {patient_id: patient_id})
        RETURN COUNT(p) AS num_patients
        """
    )
    print(f"Merged {result.get_next()[0]} patient relationships")


def merge_prescription_rels(conn: kuzu.Connection, df: pl.DataFrame) -> None:
    """Merge prescription relationships and return count of merged relationships"""
    result = conn.execute(
        """
        LOAD FROM df
        MATCH (p:Patient {patient_id: patient_id}), (d:DrugGeneric {name: drug_name})
        MERGE (p)-[r:IS_PRESCRIBED]->(d)
           SET r.date = date,
               r.dosage = dosage,
               r.frequency = frequency
        RETURN COUNT(r) AS merged_count
        """
    )
    print(f"Merged {result.get_next()[0]} prescription relationships")


def merge_symptom_rels(conn: kuzu.Connection, df: pl.DataFrame) -> None:
    """Merge symptom relationships and return count of merged relationships"""
    result = conn.execute(
        """
        LOAD FROM df
        WITH DISTINCT patient_id, side_effects
        MATCH (p:Patient {patient_id: patient_id}), (s:Symptom {name: side_effects})
        MERGE (p)-[r:EXPERIENCES]->(s)
        RETURN COUNT(r) AS merged_count
        """
    )
    print(f"Merged {result.get_next()[0]} symptom relationships")


def main():
    # Connect to the database
    db = kuzu.Database("ex_kuzu_db")
    conn = kuzu.Connection(db)

    # Load and transform data
    df = load_and_transform_data("data/extracted_data/notes.json")
    # Create schema
    create_schema(conn)

    # Merge data and get counts
    merge_patient_nodes(conn, df.select("patient_id"))
    merge_prescription_rels(
        conn, df.select("patient_id", "drug_name", "date", "dosage", "frequency")
    )
    merge_symptom_rels(
        conn, df.select("patient_id", "side_effects").explode("side_effects")
    )


if __name__ == "__main__":
    main()
