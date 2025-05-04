from unittest.mock import Mock
from pathlib import Path

def mock_post(*args, **kwargs):
    """Mock for client.post that returns a response similar to the real API."""
    # Read base64 data from file without exposing it
    base64_data = Path('base64_file.txt').read_text().strip()
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "images": [base64_data],
        "parameters": {
            "prompt": kwargs.get('json', {}).get('prompt', ''),
            "steps": kwargs.get('json', {}).get('steps', 0)
        },
        "info": "Mocked response info"
    }
    return mock_response

def setup_mock(monkeypatch):
    """Setup the mock for requests.post."""
    monkeypatch.setattr('requests.post', mock_post)