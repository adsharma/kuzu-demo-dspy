"""
This script creates a vector index for the given nodes in the database.
"""
import kuzu
import polars as pl
from sentence_transformers import SentenceTransformer

db = kuzu.Database("ex_kuzu_db")
conn = kuzu.Connection(db)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Install and load the vector extension
conn.execute("INSTALL vector; LOAD vector;");

# Symptoms
symptom_ids = conn.execute("""
    MATCH (s:Symptom)
    RETURN s.name AS id
""")
symptom_ids = symptom_ids.get_as_pl() # type: ignore

# Embed symptoms
symptom_embeddings = model.encode(symptom_ids["id"].to_list()).tolist()
symptoms_df = pl.DataFrame({
    "id": symptom_ids["id"],
    "symptoms_embedding": symptom_embeddings
})
print("Finished creating symptom embeddings")

# Conditions
condition_ids = conn.execute("""
    MATCH (c:Condition)
    RETURN c.name AS id
""")
condition_ids = condition_ids.get_as_pl() # type: ignore

# Embed conditions
condition_embeddings = model.encode(condition_ids["id"].to_list()).tolist()
conditions_df = pl.DataFrame({
    "id": condition_ids["id"],
    "condition_embedding": condition_embeddings
})
print("Finished creating condition embeddings")

conn.execute(
    """
    ALTER TABLE Symptom ADD IF NOT EXISTS symptoms_embedding FLOAT[384];
    ALTER TABLE Condition ADD IF NOT EXISTS condition_embedding FLOAT[384];
    """
)
conn.execute(
    """
    LOAD FROM symptoms_df
    MATCH (s:Symptom {name: id})
    SET s.symptoms_embedding = symptoms_embedding
    """
)
conn.execute(
    """
    LOAD FROM conditions_df
    MATCH (c:Condition {name: id})
    SET c.condition_embedding = condition_embedding
    """
)
print("Finished loading embeddings into the database")

# Create a vector index on the product summary embedding
conn.execute(
    """
    CALL CREATE_VECTOR_INDEX(
        'Symptom',
        'symptoms_index',
        'symptoms_embedding'
    )
    """
)

# Create a vector index on the condition embedding
conn.execute(
    """
    CALL CREATE_VECTOR_INDEX(
        'Condition',
        'condition_index',
        'condition_embedding'
    )
    """
)

query_vector = model.encode("sleepiness").tolist()

response = conn.execute(
    """
    CALL QUERY_VECTOR_INDEX(
        'Symptom',
        'symptoms_index',
        $query_vector,
        100
    )
    RETURN node.name AS symptom, distance
    ORDER BY distance LIMIT 10
    """,
    {"query_vector": query_vector})

print(response.get_as_pl()) # type: ignore

query_vector = model.encode("acid reflux").tolist()

response = conn.execute(
    """
    CALL QUERY_VECTOR_INDEX(
        'Condition',
        'condition_index',
        $query_vector,
        100
    )
    RETURN node.name AS condition, distance
    ORDER BY distance LIMIT 10
    """,
    {"query_vector": query_vector})

print(response.get_as_pl()) # type: ignore