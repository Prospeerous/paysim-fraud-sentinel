"""
Error handling verification tests for the fraud detection FastAPI app.
Run after the server is started on port 8000.
"""
import requests
import json
import sys

base = "http://localhost:8000"

PASS = "PASS"
FAIL = "FAIL"

results = []

def check(label, status_code, body_text, expected_status, expected_body_substr):
    ok_status = (status_code == expected_status)
    ok_body   = (expected_body_substr.lower() in body_text.lower())
    verdict   = PASS if (ok_status and ok_body) else FAIL
    results.append((verdict, label, status_code, body_text[:300]))
    print(f"[{verdict}] {label}")
    print(f"       HTTP {status_code} (expected {expected_status})  body_contains={repr(expected_body_substr)}: {ok_body}")
    if verdict == FAIL:
        print(f"       BODY: {body_text[:300]}")
    print()

# ---------- Test 1: invalid type ----------
r = requests.post(f"{base}/predict", json={
    "step": 1, "type": "INVALID", "amount": 100,
    "oldbalanceOrg": 0, "newbalanceOrig": 0,
    "oldbalanceDest": 0, "newbalanceDest": 0
})
check("T1 - Invalid type (INVALID)", r.status_code, r.text, 422, "detail")

# ---------- Test 2: missing fields ----------
r = requests.post(f"{base}/predict", json={
    "step": 1, "type": "TRANSFER", "amount": 100
})
check("T2 - Missing fields", r.status_code, r.text, 422, "detail")

# ---------- Test 3: bad CSV (missing 'amount' column) ----------
with open("test_bad.csv", "rb") as f:
    r = requests.post(f"{base}/predict-batch", files={"file": ("test_bad.csv", f, "text/csv")})
check("T3 - Bad CSV (missing amount)", r.status_code, r.text, 400, "Missing columns")

# ---------- Test 4: non-CSV file ----------
with open("requirements.txt", "rb") as f:
    r = requests.post(f"{base}/predict-batch", files={"file": ("requirements.txt", f, "text/plain")})
check("T4 - Non-CSV file", r.status_code, r.text, 400, "Only CSV files are accepted")

# ---------- Test 5: valid single prediction ----------
r = requests.post(f"{base}/predict", json={
    "step": 1, "type": "TRANSFER", "amount": 181,
    "oldbalanceOrg": 181, "newbalanceOrig": 0,
    "oldbalanceDest": 0, "newbalanceDest": 0
})
check("T5 - Valid prediction", r.status_code, r.text, 200, "fraud_probability")

# ---------- Summary ----------
print("=" * 50)
passed = sum(1 for v, *_ in results if v == PASS)
print(f"Results: {passed}/{len(results)} tests passed")
failed = [(label, code, body) for v, label, code, body in results if v == FAIL]
if failed:
    print("FAILED tests:")
    for label, code, body in failed:
        print(f"  - {label}  (HTTP {code})")
    sys.exit(1)
else:
    print("All tests PASSED.")
    sys.exit(0)
