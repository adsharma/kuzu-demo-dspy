import pytest
import requests
from dotenv import load_dotenv

from tests.test_data import test_cases

load_dotenv()
BASE_URL = "http://localhost:8001"


class TestAgentEndpoint:
    """Test suite for the /agent endpoint functionality."""

    @pytest.fixture(scope="class")
    def base_url(self):
        """Base URL for the API."""
        return BASE_URL

    @pytest.fixture(scope="class")
    def agent_url(self, base_url):
        """Full URL for the agent endpoint."""
        return f"{base_url}/agent"

    @pytest.mark.parametrize("test_case", test_cases)
    def test_agent_endpoint_contains_expected_values(self, agent_url, test_case):
        """Test that the /agent endpoint returns responses containing expected values."""

        question = test_case["question"]
        expected_values = test_case["expected_values"]

        # Make request to the agent endpoint
        payload = {"question": question}
        response = requests.post(agent_url, json=payload)

        print(f"\nQuestion: {question}")
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")

        # Assert successful response
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        # Parse response
        data = response.json()

        # Check that all required fields are present
        assert "question" in data, "Response missing 'question' field"
        assert "cypher" in data, "Response missing 'cypher' field"
        assert "response" in data, "Response missing 'response' field"
        assert "answer" in data, "Response missing 'answer' field"

        # Assert that the answer contains the expected values
        answer = data["answer"]
        assert answer, "Answer should not be empty"

        answer_lower = answer.lower()
        for expected_value in expected_values:
            assert (
                expected_value.lower() in answer_lower
            ), f"Expected '{expected_value}' not found in answer: {answer}"

        # Additional assertions
        assert data["cypher"], "Cypher query should be generated"
        assert data["response"] is not None, "Query response should not be None"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
