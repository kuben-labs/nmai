#!/usr/bin/env python3
"""Keep 1-2 submissions running at all times, forever."""

import time
import requests

SUBMIT_URL   = "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions"
STATUS_URL   = "https://api.ainm.no/tripletex/my/submissions"
ENDPOINT_URL = "https://689d-2001-700-1501-d124-99cb-97c5-ceb8-1585.ngrok-free.app/solve"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZTQ1NDRmYS00MzYyLTRjN2MtYmI5MS01MzI3MWZlNDU1ZmQiLCJlbWFpbCI6Imdvcm1lcnlrb21ib0BnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0NTA4Nn0.uVBow6pCnaKRfsb_v2Gm_2YKk6fxwZCSvK1uu4Et6Xc"

COOKIES       = {"access_token": ACCESS_TOKEN}
MAX_IN_FLIGHT = 1   # change to 2 if you want 2 at a time
POLL_INTERVAL = 10  # seconds


def count_pending():
    resp = requests.get(STATUS_URL, cookies=COOKIES)
    if resp.status_code != 200:
        return 0
    pending_statuses = {"queued", "running", "processing", "pending"}
    return sum(1 for s in resp.json() if s["status"].lower() in pending_statuses)


def submit():
    resp = requests.post(SUBMIT_URL, json={"endpoint_url": ENDPOINT_URL, "endpoint_api_key": None}, cookies=COOKIES)
    if resp.status_code == 201:
        d = resp.json()
        print(f"  submitted {d['id'][:8]}  (daily {d['daily_submissions_used']}/{d['daily_submissions_max']})")
    else:
        print(f"  submit failed ({resp.status_code}): {resp.text[:100]}")


print(f"Running forever — max {MAX_IN_FLIGHT} in flight. Ctrl+C to stop.\n")
while True:
    pending = count_pending()
    if pending < MAX_IN_FLIGHT:
        print(f"[{pending} pending] → submitting")
        submit()
    else:
        print(f"[{pending} pending] → waiting", end="\r", flush=True)
    time.sleep(POLL_INTERVAL)