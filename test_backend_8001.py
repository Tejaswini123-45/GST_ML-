#!/usr/bin/env python3
import requests
import json
import sys

try:
    # Test health endpoint on PORT 8001
    response = requests.get('http://localhost:8001/health', timeout=5)
    print("✓ Backend (on 8001) is responding!")
    print(f"Status Code: {response.status_code}")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error connecting to localhost:8001: {e}")
    sys.exit(1)
