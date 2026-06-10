#!/usr/bin/env python3
import requests
import json
import sys

try:
    # Test health endpoint
    response = requests.get('http://localhost:8000/health', timeout=5)
    print("✓ Backend is responding!")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    sys.exit(0)
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to backend on http://localhost:8000")
    print("  Is the backend running?")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
