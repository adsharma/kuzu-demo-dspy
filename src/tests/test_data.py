# Shared test data for GraphRAG tests

# Test cases with questions and expected values to find in the natural language responses
test_cases = [
    {
        "question": "What is the drug brand Xanax used for?",
        "expected_values": ["sleepy", "calm", "nerves"]
    },
    {
        "question": "Which patients are being given drugs to lower blood pressure, and what is the drug name, dosage and frequency?",
        "expected_values": ["X7F3Q", "ramipril", "5mg", "daily"]
    },
    {
        "question": "What drug is the patient B9P2T prescribed, and what is its dosage and frequency?",
        "expected_values": ["lansoprazole", "30mg", "daily"]
    },
    {
        "question": "What are the side effects of the drug brand Ambien?",
        "expected_values": ["drowsiness", "dizziness", "confusion", "headache"]
    },
    {
        "question": "What drugs can cause sleepiness as a side effect?",
        "expected_values": ["diazepam", "morphine", "oxycodone"]
    },
    {
        "question": "Which patients experience sleepiness as a side effect?",
        "expected_values": ["L4D8Z"]
    },
    {
        "question": "Can Vancomycin cause vomiting as a side effect?",
        "expected_values": ["yes"]
    },
    {
        "question": "What are the side effects of drugs for conditions related to lowering cholestrol?",
        "expected_values": ["upset stomach", "headache", "muscle pain"]
    },
    {
        "question": "Which patients experience sleepiness as a side effect?",
        "expected_values": ["L4D8Z"]
    },
    {
        "question": "What drug brands treat the condition of irregular heart rhythm?",
        "expected_values": ["Digitek", "Cordarone", "Inderal", "Pacerone", "Lanoxin"]
    },
]