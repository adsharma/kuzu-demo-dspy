// --- Data models ---
class Query {
  response string
}

enum Tool {
    Text2Cypher  @description("Translate the question into a valid Cypher statement")
    VectorSearchSymptoms  @description("Search for symptoms or side effects mentioned in the question")
    VectorSearchConditions  @description("Search for a condition or ailment mentioned in the question")
}

// --- Functions ---

function RAGText2Cypher(schema: string, question: string, additional_context: string | null) -> Query {
  client Gemini2Flash
  prompt #"
    {{ _.role("system") }}
    Translate the given question into a valid Cypher statement.

    IMPORTANT:
    - Use only the provided nodes, relationships and properties in your Cypher statement.
    - The Cypher statement could retrieve nodes, relationships, or both. Do not include any
    explanations or apologies in your responses.
    - Do not respond to any questions that might ask anything else than for you to construct a
    Cypher statement.
    - When returning results, return property values rather than the entire node or relationship.
    - ALWAYS use the WHERE clause to compare string properties, and compare them using the
      LOWER() function.

    {{ _.role("user") }}
    QUESTION:
    {{ question }}

    SCHEMA:
    {{ schema }}

    {% if additional_context %}
    ADDITIONAL CONTEXT:
    {{ additional_context }}
    {% endif %}

    {{ ctx.output_format }}
  "#
}

function PickTool(schema: string, query: string) -> Tool {
  client OpenRouterGPT4oMini
  prompt #"
    A prior attempt to write a valid Cypher query failed because an exact match with
    a property value was not found. Analyze the given query and select the most appropriate tool
    that can retrieve more useful context to answer the question.

    {{ _.role("user") }}
    QUESTION:
    {{ query }}

    ALWAYS RESPECT THE RELATIONSHIP DIRECTIONS:
    {{ schema}}

    {{ ctx.output_format }}
  "#
}

function RAGAnswerQuestion(question: string, cypher: string, response: string) -> string {
  client OpenRouterGPT4o
  prompt #"
    You will be provided with a question and relevant context.
    
    - Use only this context to answer the question in a few sentences.
    - ALWAYS use full sentences
    - If the provided context is an empty string, DO NOT make up an answer. State that you did not
    obtain enough context to answer the question.

    {{ _.role("user") }}
    QUESTION: {{ question }}
    CYPHER: {{ cypher }}
    RESPONSE: {{ response }}

    {{ ctx.output_format }}

    RESPONSE:
  "#
}


// --- Test cases ---

test CypherDrugCausesSymptom {
  functions [RAGText2Cypher]
  args {
    schema #"
    (:Patient {patient_id: string}) -[:EXPERIENCES]-> (:Symptom {name: string})

    (:Patient {patient_id: string}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (:DrugGeneric {name: string})

    (:DrugGeneric {name: string}) -[:CAN_CAUSE]-> (:Symptom {name: string})

    (:DrugGeneric {name: string}) -[:HAS_BRAND]-> (:DrugBrand {name: string})

    (:Condition {name: string}) -[:IS_TREATED_BY]-> (:DrugGeneric {name: string})
    "#
    question "Which generic drugs are prescribed to patients with hypertension?"
    additional_context ""
  }
}

test CypherDrugCausesSymptomLowercase {
  functions [RAGText2Cypher]
  args {
    schema #"
    (:Patient {patient_id: string}) -[:EXPERIENCES]-> (:Symptom {name: string})

    (:Patient {patient_id: string}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (:DrugGeneric {name: string})

    (:DrugGeneric {name: string}) -[:CAN_CAUSE]-> (:Symptom {name: string})

    (:DrugGeneric {name: string}) -[:HAS_BRAND]-> (:DrugBrand {name: string})

    (:Condition {name: string}) -[:IS_TREATED_BY]-> (:DrugGeneric {name: string})
    "#
    question "What are the side effects of lansoprazole?"
    additional_context ""
  }
}

test CypherDrugGenericHasBrand {
  functions [RAGText2Cypher]
  args {
    schema #"
    (:Patient {patient_id: string}) -[:EXPERIENCES]-> (:Symptom {name: string})

    (:Patient {patient_id: string}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (:DrugGeneric {name: string})

    (:DrugGeneric {name: string}) -[:CAN_CAUSE]-> (:Symptom {name: string})

    (:DrugGeneric {name: string}) -[:HAS_BRAND]-> (:DrugBrand {name: string})

    (:Condition {name: string}) -[:IS_TREATED_BY]-> (:DrugGeneric {name: string})
    "#
    question "What drug brands are there for Lansoprazole?"
    additional_context ""
  }
}

test PickTool1 {
  functions [PickTool]
  args {
    schema #"
    (:Patient {patient_id: string}) -[:EXPERIENCES]-> (:Symptom {name: string})

    (:Patient {patient_id: string}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (:DrugGeneric {name: string})

    (:DrugGeneric {name: string}) -[:CAN_CAUSE]-> (:Symptom {name: string})

    (:DrugGeneric {name: string}) -[:HAS_BRAND]-> (:DrugBrand {name: string})

    (:Condition {name: string}) -[:IS_TREATED_BY]-> (:DrugGeneric {name: string})
    "#
    query "MATCH (dg:DrugGeneric)-[:CAN_CAUSE]->(s:Symptom), (dg)-[:HAS_BRAND]->(db:DrugBrand) WHERE LOWER(s.name) = LOWER('sleepiness') RETURN db.name"
  }
}

test RAGAnswerQuestion1 {
  functions [RAGAnswerQuestion]
  args {
    question "What drug is the patient X7F3Q prescribed, and what is the dosage and frequency?"
    cypher "MATCH (p:Patient {patient_id: 'X7F3Q'}) -[:IS_PRESCRIBED {date: date, dosage: string, frequency: string}]-> (dg:DrugGeneric) RETURN dg.name, dg.dosage, dg.frequency"
    context "ramipril, 5mg, daily"
  }
}