# Graph extraction and Graph RAG with BAML and Kuzu

This repository contains a demonstration of transforming unstructured data from clinical notes
and drug side effects into a knowledge graph in Kuzu. The graph is then used to answer questions
about the data using a Graph RAG pipeline.

Tools used
- [BAML](https://github.com/boundaryml/baml): AI and LLM prompting framework
- [Kuzu](https://github.com/kuzudb/kuzu): Embedded, fast, scalable graph database
- [Streamlit](https://github.com/streamlit/streamlit): Visualization framework for building custom web apps

## Overview

The goal is to show how to build modular "agents" (i.e., prompt pipelines that accomplish a
subtask) that can then be strung together to accomplish more complex tasks. BAML allows users
to compose together these kinds of agents with testing and validation capabilities, by offering
a reliable means to generate structured outputs from LLMs.

Kuzu is used as a data store for the graph, and Streamlit is used to build a simple UI for
interacting with the graph.

The general components of the pipeline are shown in the diagram below.

![](./assets/drug-side-effects-graph-rag.png)

## Setup

Ensure you have Python 3.11+ installed.

1. Clone this repository
2. Install the required dependencies:
   ```bash
   # Install the uv package manager
   curl -fsSL https://get.uvm.dev | bash
   # Install the dependencies
   uv sync
   ```
3. Activate the environment and generate the BAML client files
```bash
cd
source .venv/bin/activate
baml-cli generate
```
This will generate the BAML client files in the `src/baml_client` directory.

## Extract data from images and text

To extract data from images and text, run the following command:

```bash
cd src
# Extract data from images that represent tables from the PDF of drugs and side effects
uv run image_extractor.py
# Extract data from the text of clinical notes
uv run notes_extractor.py
```

This will output JSON files into the `../data/extracted_data` directory.

## Creating the graph

To create the graph in Kuzu, run the following command:

```bash
cd src
uv run 01_create_drug_graph.py
```

This will persist the Kuzu graph locally in the `ex_kuzu_db` directory.

To add the patient data to the graph, run the following command:

```bash
cd src
uv run 02_create_patient_graph.py
```
This will augment the pre-existing graph (from the prior step) with the data from the
patient notes.

## Running the Graph RAG chatbot

To run the Streamlit application:

```bash
cd src
uv run streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501 by default.

## Sample questions

The application comes with several sample questions you can try:
- "What drug brands are there for lansoprazole?"
- "What are the side effects of morphine?"

## How it works

1. The user enters a question in natural language
2. BAML converts the question to a Cypher query
3. The Cypher query is executed against the Kuzu graph database
4. The results are processed and a natural language answer is generated
5. Both the query and answer are displayed to the user

## Notes

- The application maintains a history of up to 10 recent questions and answers
- You can clear the history using the "Clear History" button
- A debug mode is available in the sidebar that provides:
  - The GraphRAG schema used for generating Cypher queries
  - Detailed information about the latest result
  - A direct Cypher query execution interface for testing


> [!NOTE]
> The Graph RAG application is far from perfect and has room for improvement. For example,
> custom modules can be incorporated downstream of Text2Cypher to check the Cypher queries
> for syntax errors before execution. In addition, corrector agents could be added to the
> pipeline to increase robustness so that fewer questions result in an empty response.

---

## Evaluation

Part of the motivation for this project was to evaluate the performance of BAML and the given
LLM for the task of extracting data from unstructured text. We have two tasks to evaluate:

1. Extracting drugs and side effects from a table in a PDF
2. Extracting medications and side effects from clinical notes

To evaluate the performance of the image extractor, run the following command:

```bash
cd evals
uv run image_extractor_eval.py
```

| Model | Date[^3] | Exact Match | Mismatch | Missing | Potential<br> Hallucination | Cost | Cost<br> factor |
| --- | --- | :---: | :---: | :---: | :---: | ---: | ---: |
| `openai/gpt-4o-mini` | Mar 2025 | 170 | 0 | 2 | 2 | 0.0008 | 1.0 |
| `openai/gpt-4o` | Mar 2025 | 174 | 1 | 1 | 2 | $0.0277 | 35x |
| `anthropic/claude-3.5-sonnet` | Mar 2025 | 173 | 0 | 2 | 2 | $0.0551 | 69x |
| `google/gemini-2.0-flash` | Mar 2025 | 158 | 2 | 12 | 8 | Free tier | N/A |

Note that your costs, latency and results may differ based on when you run the code, as models
are being updated continually.

To evaluate the performance of the notes extractor, run the following command:

```bash
cd evals
uv run notes_extractor_eval.py
```

| Model | Date[^3] | Exact Match | Mismatch | Missing | Potential<br> Hallucination | Cost | Cost<br> factor |
| --- | --- | :---: | :---: | :---: | :---: | ---: | ---: |
| `openai/gpt-4o-mini` | Mar 2025 | 19 | 0 | 0 | 0 | $0.0003 | 1.0 |
| `openai/gpt-4o` | Mar 2025 | 19 | 0 | 0 | 0 | $0.0044 | 15x |
| `anthropic/claude-3.5-sonnet` | Mar 2025 | 19 | 0 | 0 | 0 | $0.0074 | 25x |
| `google/gemini-2.0-flash` | Mar 2025 | 19 | 0 | 0 | 0 | Free tier | N/A |

The text extraction task is well handled by all models tested!
