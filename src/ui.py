import json

import requests
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Graph + vector RAG in Kuzu", page_icon="🔍", layout="wide"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# API configuration
API_BASE_URL = "http://localhost:8001"


def call_api_endpoint(endpoint: str, data: dict):
    """Helper function to call API endpoints."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None


def call_vector_endpoint(endpoint: str, query: str):
    """Helper function to call vector search endpoints."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}", json={"query": query}, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Vector search failed: {str(e)}")
        return None


# Function to process a question
def process_question(question):
    with st.spinner("Generating answer..."):
        try:
            # Call the agent endpoint
            result = call_api_endpoint("/agent", {"question": question})

            if result:
                # Add to chat history (most recent first)
                st.session_state.chat_history.insert(0, result)
                # Keep only the last 10 entries
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[:10]
            else:
                st.error("No result was returned. Please try a different question.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)


# Function to clear chat history
def clear_history():
    st.session_state.chat_history = []


# Function to test vector search
def test_vector_search(query: str, search_type: str):
    """Test vector search functionality."""
    endpoint = f"/query_vector_index_{search_type.lower()}"
    result = call_vector_endpoint(endpoint, query)
    if result:
        st.success(f"Vector search results for '{search_type}':")
        st.write(result["results"])
    return result


# App title
st.title("Graph + vector RAG in Kuzu")
st.markdown(
    "Ask questions about the data in your Kuzu database and get answers powered by RAG."
)

# User input
with st.form(key="question_form"):
    question = st.text_input(
        "Enter your question:", placeholder="What are the side effects of Morphine?"
    )
    submit_button = st.form_submit_button("Ask")

# Process the question when submitted
if submit_button and question:
    process_question(question)

# Clear history button
if st.session_state.chat_history:
    if st.button("Clear History"):
        clear_history()

# Sample questions below the text box
st.subheader("Sample Questions")
sample_questions = [
    "What condition is treated by the drug brand Xanax?",
    "Which patients are being given drugs to lower blood pressure, and what is the drug name, dosage and frequency?",
    "What drug is the patient B9P2T prescribed, and what is its dosage and frequency?",
]

# Custom CSS for left-aligned button text
st.markdown(
    """
<style>
.stButton > button {
    text-align: left;
    justify-content: flex-start;
    padding-left: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

for sample in sample_questions:
    if st.button(sample, key=f"sample_{sample}"):
        process_question(sample)

# Display the most recent result
if st.session_state.chat_history:
    st.subheader("Latest Response")
    latest = st.session_state.chat_history[0]

    # Display the question
    st.markdown(f"**Question:** {latest['question']}")

    # Display the Cypher query in a code block (full width)
    st.subheader("Generated Cypher Query")
    if latest["cypher"] and latest["cypher"] != "N/A":
        st.code(latest["cypher"], language="cypher")
    else:
        st.error(
            "No Cypher query was generated for this question. Try rephrasing your question."
        )

    # Display the answer (full width)
    st.subheader("Answer")
    if latest.get("answer"):
        st.markdown(latest["answer"])
    elif latest.get("response") and latest["response"] != "N/A":
        st.markdown(latest["response"])
    else:
        st.warning(
            "No answer was generated. This could be due to no results from the query or an error in processing."
        )

    # Divider
    st.divider()

# Display chat history
if len(st.session_state.chat_history) > 1:
    st.subheader("Chat History")

    for i, item in enumerate(st.session_state.chat_history[1:], 1):
        with st.expander(f"Question {i}: {item['question']}"):
            st.markdown(f"**Question:** {item['question']}")

            st.markdown("**Cypher Query:**")
            if item["cypher"] and item["cypher"] != "N/A":
                st.code(item["cypher"], language="cypher")
            else:
                st.error("No Cypher query was generated.")

            st.markdown("**Answer:**")
            if item.get("answer"):
                st.markdown(item["answer"])
            elif item.get("response") and item["response"] != "N/A":
                st.markdown(item["response"])
            else:
                st.warning("No answer was generated.")

# About section in the sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
    This application uses Graph RAG to:
    1. Convert natural language questions to Cypher queries
    2. Execute the queries against a Kuzu graph database
    3. Generate natural language answers
    """
    )

    st.markdown("---")
    st.markdown(
        "**Graph RAG** combines the power of graph databases with Retrieval Augmented Generation to provide answers based on the underlying data structure."
    )

    # API Status Check
    st.markdown("---")
    st.subheader("API Status")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Server is running")
        else:
            st.error("❌ API Server is not responding")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API Server")
        st.info("Make sure the FastAPI server is running on localhost:8001")

    # Debug mode toggle
    st.markdown("---")
    debug_mode = st.checkbox("Debug Mode", value=False)

    if debug_mode:
        st.subheader("Debug Information")

        if st.session_state.chat_history:
            st.write("Latest Result Object:")
            st.json(st.session_state.chat_history[0])

        # Vector search testing
        st.subheader("Test Vector Search")
        vector_query = st.text_input(
            "Enter query for vector search:", placeholder="sleepiness"
        )
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Test Symptoms"):
                test_vector_search(vector_query, "symptoms")

        with col2:
            if st.button("Test Conditions"):
                test_vector_search(vector_query, "conditions")

        # API Info
        st.subheader("API Information")
        try:
            api_info = requests.get(f"{API_BASE_URL}/", timeout=5)
            if api_info.status_code == 200:
                st.json(api_info.json())
        except requests.exceptions.RequestException:
            st.error("Could not fetch API information")
