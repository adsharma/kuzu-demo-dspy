from textwrap import dedent
from typing import Optional

import kuzu
from sentence_transformers import SentenceTransformer

from baml_client import b, types

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --- Database ---


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
        if hasattr(self, "conn"):
            self.conn.close()
        if hasattr(self, "db"):
            self.db.close()


# --- Schema ---


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


def get_full_path_schema(schema: dict) -> str:
    """
    Obtain the graph schema as full paths, including nodes, relationships and their associated
    property names and data types.

    Example:
    ```
    (:Patient {patient_id: string}) -[:EXPERIENCES]-> (:Symptom {name: string})

    (:Patient {patient_id: string}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (:DrugGeneric {name: string})

    (:DrugGeneric {name: string}) -[:CAN_CAUSE]-> (:Symptom {name: string})

    (:DrugGeneric {name: string}) -[:HAS_BRAND]-> (:DrugBrand {name: string})

    (:Condition {name: string}) -[:IS_TREATED_BY]-> (:DrugGeneric {name: string})
    ```

    Benefit: This can help the LLM understand the properties of the nodes and relationships in the graph
    and use them to generate a more accurate and relevant Cypher query.
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


# --- Vector index search tools ---

def query_vector_index_symptoms(db_manager: DatabaseManager, query_string: str) -> list[str]:
    """
    Query the vector index for similar symptoms mentioned in the question
    """
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
    """
    Query the vector index for similar conditions (treated by a drug) mentioned in the question
    """
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


# --- Graph RAG ---


class GraphRAG:
    """
    Run Graph RAG on the Kuzu database
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.schema_dict = get_schema_dict(db_manager)
        self.schema = get_full_path_schema(self.schema_dict)

    def execute_query(self, cypher: str) -> str:
        """
        Use the generated Cypher statement to query the graph database
        """
        conn = self.db_manager.get_connection()
        response = conn.execute(cypher)
        result = []
        while response.has_next():  # type: ignore
            item = response.get_next()  # type: ignore
            if item not in result:
                result.extend(item)

        # Remove duplicates from the result list
        result_list = [x for i, x in enumerate(result) if x not in result[:i]]
        result_str = ", ".join(result_list)
        return result_str

    def retrieve(self, question: str, additional_context: Optional[str] = None) -> dict[str, str]:
        """
        Retrieve information from the graph database using text2cypher
        """
        prompt_context = self.schema
        query = b.RAGText2Cypher(prompt_context, question, additional_context)
        result = {
            "question": question,
            "cypher": query.cypher.replace("\n", " "),
        }
        # Query the database
        try:
            result_str = self.execute_query(query.cypher)
        except Exception as e:
            print(f"Error executing query: {e}")
            result_str = ""
        result["response"] = result_str
        return result

    def answer_question(self, response: dict[str, str]) -> str:
        """
        Answer a question in natural language based on the retrieved information
        """
        question = response["question"]
        cypher = response["cypher"]
        context = response["response"]
        answer = b.RAGAnswerQuestion(question, cypher, context)
        return answer


# --- Agent logic ---


class AgentOrchestrator:
    """
    Orchestrates the agentic workflow with retry logic and tool routing.
    """

    def __init__(self, rag: GraphRAG, db_manager: DatabaseManager, max_retries: int = 3):
        self.rag = rag
        self.db_manager = db_manager
        self.MAX_RETRIES = max_retries

    def _handle_vector_search(self, tool_name: str, question: str) -> str:
        """
        Handle vector search based on tool selection
        """
        if tool_name == "VectorSearchSymptoms":
            vector_response = query_vector_index_symptoms(self.db_manager, question)
            vector_query_result = ", ".join(vector_response)
            return f"Try querying for the most similar side effect or symptom: {vector_query_result}"

        elif tool_name == "VectorSearchConditions":
            vector_response = query_vector_index_conditions(self.db_manager, question)
            vector_query_result = ", ".join(vector_response)
            return f"Try querying for the most similar condition: {vector_query_result}"

        return "None"

    def _try_agentic_approach(self, question: str) -> dict[str, str] | None:
        """
        Try agentic approach with tool routing
        """
        # We'll use the BAML function to select the next tool to use
        next_tool = b.PickTool(self.rag.schema, question)

        if not next_tool or not next_tool.value:
            return None

        # If there's a tool selected to use, we'll add its context to the next prompt
        additional_context = self._handle_vector_search(next_tool.value, question)
        return self.rag.retrieve(question, additional_context)

    def run_agent(self, question: str) -> dict[str, str]:
        """
        Run an agentic router workflow with a set maximum number of retries
        """
        n = 0
        while n <= self.MAX_RETRIES:
            print(f"Iteration {n}\n---")

            # Try vanilla Graph RAG (text2cypher) first
            result = self.rag.retrieve(question)

            # If we get a response, we're done
            if result["response"]:
                return result

            # If we do not get a response, let's try an agent router approach
            agentic_result = self._try_agentic_approach(question)
            if agentic_result:
                return agentic_result
            n += 1

        # If we still don't have a response after all retries, provide a fallback response
        return {
            "question": question,
            "cypher": "",
            "response": dedent(
                """
                I'm sorry, I was unable to retrieve the relevant information based on
                the question you asked - could you please rephrase and try again?
                """
            ),
        }
