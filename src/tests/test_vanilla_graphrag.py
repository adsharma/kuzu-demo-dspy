import pytest
from dotenv import load_dotenv

from helpers import DatabaseManager, GraphRAG
from tests.test_data import test_cases

load_dotenv()


class TestAnswerQuestion:
    """Test suite for GraphRAG answer_question functionality."""

    @pytest.fixture(scope="class")
    def db_manager(self):
        """Initialize database manager for all tests."""
        manager = DatabaseManager()
        yield manager
        manager.close()

    @pytest.fixture(scope="class")
    def graph_rag(self, db_manager):
        """Initialize GraphRAG for all tests."""
        return GraphRAG(db_manager)

    @pytest.mark.parametrize("test_case", test_cases)
    def test_answer_question_contains_expected_values(self, graph_rag, test_case):
        """Test that the answer_question method returns responses containing expected values."""

        question = test_case["question"]
        expected_values = test_case["expected_values"]

        # First retrieve the data
        result = graph_rag.retrieve(question)

        # Then generate the natural language answer
        answer = graph_rag.answer_question(result)

        print(f"\nQuestion: {question}")
        print(f"Generated Cypher: {result['cypher']}")
        print(f"Raw response: {result['response']}")
        print(f"Natural language answer: {answer}")

        # Assert that the answer contains the expected values
        answer_lower = answer.lower()
        for expected_value in expected_values:
            assert (
                expected_value.lower() in answer_lower
            ), f"Expected '{expected_value}' not found in answer: {answer}"

        # Additional assertions
        assert result["cypher"], "Cypher query was generated"
        assert result["response"] is not None, "Query executed successfully"
        assert answer, "Natural language answer was generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
