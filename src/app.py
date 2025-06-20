from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from helpers import DatabaseManager, GraphRAG, query_vector_index_symptoms, query_vector_index_conditions
from baml_client import b

# Load environment variables from .env file
load_dotenv()

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class VectorQueryRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    question: str
    cypher: str
    response: str
    answer: Optional[str] = None

class VectorQueryResponse(BaseModel):
    results: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    db_manager.close()


# Initialize FastAPI app and database manager
app = FastAPI(
    title="GraphRAG agentic API",
    description="Simple agentic Graph RAG system to query a graph database about patients, conditions, drugs and side effects",
    lifespan=lifespan
)
db_manager = DatabaseManager("ex_kuzu_db")
rag = GraphRAG(db_manager)

@app.get("/")
async def root():
    """Landing page with API information."""
    return {
        "message": "GraphRAG Agentic API",
        "description": "Simple agentic Graph RAG system to query a graph database about patients, conditions, drugs and side effects",
        "endpoints": {
            "POST /agent": "Run the GraphRAG agent workflow with a question",
            "POST /query_vector_index_symptoms": "Query vector index for similar symptoms",
            "POST /query_vector_index_conditions": "Query vector index for similar conditions",
            "GET /": "App information",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation"
        },
        "version": "1.0.0"
    }


@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: QuestionRequest):
    """Main agent endpoint that runs the GraphRAG workflow."""
    try:
        result = rag.retrieve(request.question)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve information")
        
        # Handle vector queries if needed
        if result.get("next_tool"):
            next_tool = result["next_tool"]
            if next_tool == "VectorSearchSymptoms":
                vector_response = query_vector_index_symptoms(db_manager, request.question)
                vector_query_result = ", ".join(vector_response)
                additional_context = f"Similar symptoms in the database: {vector_query_result}"
            elif next_tool == "VectorSearchConditions":
                vector_response = query_vector_index_conditions(db_manager, request.question)
                vector_query_result = ", ".join(vector_response)
                additional_context = f"Similar conditions in the database: {vector_query_result}"
            else:
                additional_context = None
            
            # Run the text2cypher again with the new context
            if additional_context:
                prompt_context = rag.schema
                query = b.RAGText2Cypher(prompt_context, request.question, additional_context)
                query_response = rag.execute_query(request.question, query.response)
                result["response"] = query_response.response
                result["cypher"] = query.response
        
        # Generate final answer
        answer = rag.answer_question(result)
        
        return AgentResponse(
            question=result["question"],
            cypher=result["cypher"],
            response=result["response"],
            answer=answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/query_vector_index_symptoms", response_model=VectorQueryResponse)
async def query_symptoms_endpoint(request: VectorQueryRequest):
    """Query the vector index for similar symptoms."""
    try:
        results = query_vector_index_symptoms(db_manager, request.query)
        return VectorQueryResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying symptoms: {str(e)}")


@app.post("/query_vector_index_conditions", response_model=VectorQueryResponse)
async def query_conditions_endpoint(request: VectorQueryRequest):
    """Query the vector index for similar conditions."""
    try:
        results = query_vector_index_conditions(db_manager, request.query)
        return VectorQueryResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying conditions: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
