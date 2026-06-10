"""
Comprehensive integration tests for security, input validation, and endpoints.
"""

import os
import sys
import requests

BACKEND_URL = "http://localhost:8001"
API_KEY = os.getenv("API_KEY")

def test_endpoint(method, url, headers=None, json_data=None, expected_status=200):
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=json_data, timeout=5)
        
        print(f"[{method}] {url.replace(BACKEND_URL, '')} -> Status: {r.status_code} (Expected: {expected_status})")
        if r.status_code != expected_status:
            print(f"  FAILED: Unexpected response content: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  FAILED: Connection error to {url}: {e}")
        return False

def main():
    print("=== STARTING GST RECONCILIATION API INTEGRATION TESTS ===")
    
    # Check if backend is alive
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=3)
    except Exception:
        print(f"ERROR: Backend is not running at {BACKEND_URL}. Run scripts/entrypoint.py first!")
        sys.exit(1)

    passed = True

    # 1. Test public endpoints
    print("\n--- 1. Public Endpoints ---")
    passed &= test_endpoint("GET", f"{BACKEND_URL}/", expected_status=200)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/health", expected_status=200)

    # 2. Test security/authentication
    print("\n--- 2. Security & Auth ---")
    headers = {}
    if API_KEY:
        print(f"[INFO] API_KEY environment variable is set. Enforcing auth check.")
        # Request without key
        passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/taxpayers", expected_status=401)
        # Request with invalid key
        passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/taxpayers", headers={"X-API-Key": "wrong-key"}, expected_status=403)
        # Setup headers for subsequent calls
        headers = {"X-API-Key": API_KEY}
    else:
        print(f"[INFO] API_KEY environment variable is NOT set. Running in open dev-mode.")

    # 3. Test functional endpoints (must succeed with configured headers)
    print("\n--- 3. Functional Endpoints ---")
    passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/taxpayers", headers=headers, expected_status=200)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/ingest/status", headers=headers, expected_status=200)
    
    # 4. Test input validation (GSTIN)
    print("\n--- 4. GSTIN Parameter Validation ---")
    # Valid GSTIN (e.g. 11cddgT4839TZ2)
    valid_gstin = "11cddgT4839TZ2"
    passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/overview/{valid_gstin}", headers=headers, expected_status=200)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/vendor-network/{valid_gstin}", headers=headers, expected_status=200)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/graph/risk-score/{valid_gstin}", headers=headers, expected_status=200)
    
    # Invalid GSTIN format (too short/long/specials)
    invalid_gst = "invalid_gst"
    passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/overview/{invalid_gst}", headers=headers, expected_status=400)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/dashboard/vendor-network/{invalid_gst}", headers=headers, expected_status=400)
    passed &= test_endpoint("GET", f"{BACKEND_URL}/graph/risk-score/{invalid_gst}", headers=headers, expected_status=400)

    # 5. Test input validation (Invoice ID)
    print("\n--- 5. Invoice ID Parameter Validation ---")
    # Valid format (needs to be alphanumeric/hyphens/underscores)
    valid_invoice = "INV-15"
    passed &= test_endpoint("GET", f"{BACKEND_URL}/graph/audit/{valid_invoice}", headers=headers, expected_status=200)
    
    # Invalid format (spaces/invalid characters)
    invalid_invoice = "INV 15 #$$"
    passed &= test_endpoint("GET", f"{BACKEND_URL}/graph/audit/{invalid_invoice}", headers=headers, expected_status=400)

    # 6. Test Debug endpoints restricted in production
    print("\n--- 6. Debug / Test Endpoints ---")
    # In production/normal run (DEBUG=False in config), test endpoints should return 404
    # Let's check config state:
    try:
        r = requests.get(f"{BACKEND_URL}/openapi.json", headers=headers, timeout=3)
        openapi = r.json()
        paths = openapi.get("paths", {})
        has_test_routes = any(p.startswith("/test") for p in paths)
        print(f"[INFO] OpenAPI spec lists test routes: {has_test_routes}")
        
        # Test request directly
        r_mongo = requests.get(f"{BACKEND_URL}/test/mongo", headers=headers, timeout=3)
        if has_test_routes:
            print(f"GET /test/mongo -> Status: {r_mongo.status_code} (Debug mode enabled)")
        else:
            print(f"GET /test/mongo -> Status: {r_mongo.status_code} (Expected: 404 in production)")
            if r_mongo.status_code != 404:
                print("  FAILED: Debug routes should return 404 when DEBUG=False")
                passed = False
    except Exception as e:
        print(f"  FAILED during debug route check: {e}")
        passed = False

    print("\n========================================================")
    if passed:
        print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! [FAIL]")
        sys.exit(1)

if __name__ == "__main__":
    main()
