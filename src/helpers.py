import kuzu
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from baml_client import b, types

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


class DatabaseManager:
    """Manages Kuzu database connection and vector index setup."""
    
    def __init__(self, db_path: str = "ex_kuzu_db"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)
        self._setup_vector_extension()
    
    def _setup_vector_extension(self):
        """Install and load vector extension once."""
        self.conn.execute("INSTALL vector; LOAD vector;")
    
    def get_connection(self) -> kuzu.Connection:
        """Get the database connection."""
        return self.conn
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'db'):
            self.db.close()


def get_schema_dict(db_manager: DatabaseManager) -> dict[str, list[dict]]:
    # Get schema for LLM
    conn = db_manager.get_connection()
    nodes = conn._get_node_table_names()
    relationships = conn._get_rel_table_names()

    schema = {"nodes": [], "edges": []}

    for node in nodes:
        node_schema = {"label": node, "properties": []}
        node_properties = conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
        while node_properties.has_next():  # type: ignore
            row = node_properties.get_next()  # type: ignore
            node_schema["properties"].append({"name": row[1], "type": row[2]})
        schema["nodes"].append(node_schema)

    for rel in relationships:
        edge = {
            "label": rel["name"],
            "src": rel["src"],
            "dst": rel["dst"],
            "properties": [],
        }
        rel_properties = conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
        while rel_properties.has_next():  # type: ignore
            row = rel_properties.get_next()  # type: ignore
            edge["properties"].append({"name": row[1], "type": row[2]})
        schema["edges"].append(edge)

    return schema


def get_full_paths(schema: dict) -> str:
    """
    Obtain the schema as a set of full paths, including nodes, relationships and their associated
    property names and data types.
    """
    # Build a lookup for node properties
    node_props = {n["label"]: n["properties"] for n in schema.get("nodes", [])}
    lines = []
    for edge in schema.get("edges", []):
        src = edge["src"]
        dst = edge["dst"]
        edge_label = edge["label"]
        # Node properties
        def fmt_props(props):
            if not props:
                return ""
            return "{" + ", ".join(f"{p['name']}: {p['type'].lower()}" for p in props) + "}"
        src_props = fmt_props(node_props.get(src, []))
        dst_props = fmt_props(node_props.get(dst, []))
        # Edge properties
        edge_props = fmt_props(edge.get("properties", []))
        if edge_props:
            edge_str = f"-[:{edge_label} {edge_props}]->"
        else:
            edge_str = f"-[:{edge_label}]->"
        lines.append(f"(:{src} {src_props}) {edge_str} (:{dst} {dst_props})")
    return "\n\n".join(lines)


class GraphRAG:
    """
    Run Graph RAG on the Kuzu database
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.schema_dict = get_schema_dict(db_manager)
        self.schema = get_full_paths(self.schema_dict)

    def execute_query(self, question: str, cypher: str) -> types.Query:
        """Use the generated Cypher statement to query the graph database."""
        conn = self.db_manager.get_connection()
        response = conn.execute(cypher)
        result = []
        while response.has_next():  # type: ignore
            item = response.get_next()  # type: ignore
            if item not in result:
                result.extend(item)

        # Remove duplicates
        result_list = [x for i, x in enumerate(result) if x not in result[:i]]
        result_str = ", ".join(result_list)
        return types.Query(response=result_str)

    def retrieve(self, question: str) -> dict[str, str] | None:
        prompt_context = self.schema
        query = b.RAGText2Cypher(prompt_context, question, additional_context=None)
        result = {
            "question": question,
            "cypher": query.response,
            "response": "",
        }
        # Query the database
        query_response = self.execute_query(question, query.response)
        result["response"] = query_response.response
        n = 1
        MAX_RETRIES = 3
        while n <= MAX_RETRIES:
            # Limit the number of retries
            print(f"Iteration {n}\n---")
            if result["response"]:
                break
            else:
                next_tool = b.PickTool(self.schema, question)
                # Return the tool choice for the API to handle
                result["next_tool"] = next_tool.value if next_tool else ""
                break
        else:
            result["response"] = (
                "Unable to retrieve useful information - could you rephrase the question?"
            )
        return result

    def answer_question(self, response: dict[str, str]) -> str:
        question = response["question"]
        cypher = response["cypher"]
        context = response["response"]
        answer = b.RAGAnswerQuestion(question, cypher, context)
        return answer


def query_vector_index_symptoms(db_manager: DatabaseManager, query_string: str) -> list[str]:
    conn = db_manager.get_connection()
    query_vector = MODEL.encode(query_string).tolist()
    response = conn.execute(
        """
        CALL QUERY_VECTOR_INDEX(
            'Symptom',
            'symptoms_index',
            $query_vector,
            100
        )
        RETURN node.name AS symptom, distance
        ORDER BY distance LIMIT 1
        """,
        {"query_vector": query_vector},
    )
    topk_response = response.get_as_pl().select("symptom").to_series().to_list()  # type: ignore
    return topk_response


def query_vector_index_conditions(db_manager: DatabaseManager, query_string: str) -> list[str]:
    conn = db_manager.get_connection()
    query_vector = MODEL.encode(query_string).tolist()
    response = conn.execute(
        """
        CALL QUERY_VECTOR_INDEX(
            'Condition',
            'condition_index',
            $query_vector,
            100
        )
        RETURN node.name AS condition, distance
        ORDER BY distance LIMIT 1
        """,
        {"query_vector": query_vector},
    )
    topk_response = response.get_as_pl().select("condition").to_series().to_list()  # type: ignore
    return topk_response
