#!/usr/bin/env python3
"""Submit the accounting agent endpoint to the competition API."""

import requests
import sys

SUBMIT_URL = "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions"
ENDPOINT_URL = "https://accounting-agent-502687038260.us-central1.run.app"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwYjM1ZDg2NC02YWU2LTQxNTAtOGIxMi04ZTQyN2QzZDAyYjgiLCJlbWFpbCI6InNhbmRlcndAa3ViZW5sYWJzLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTM1MzA5fQ.DApu5yR5OR-tO2i76IUxZ_uOwDgpjZm6alyGTJhqyVc"
NUM_SUBMISSIONS = 3


def submit(n=NUM_SUBMISSIONS):
    """Submit the endpoint URL n times."""
    cookies = {"access_token": ACCESS_TOKEN}
    payload = {"endpoint_url": ENDPOINT_URL}

    for i in range(n):
        resp = requests.post(SUBMIT_URL, json=payload, cookies=cookies)
        if resp.status_code == 201:
            data = resp.json()
            print(f"[{i+1}/{n}] Queued: {data['id']}  (daily: {data['daily_submissions_used']}/{data['daily_submissions_max']})")
        else:
            print(f"[{i+1}/{n}] FAILED ({resp.status_code}): {resp.text}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SUBMISSIONS
    submit(n)
