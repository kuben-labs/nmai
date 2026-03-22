#!/usr/bin/env python3
"""Submit the accounting agent endpoint and track submission results."""

import requests
import sys
import time

SUBMIT_URL = "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions"
STATUS_URL = "https://api.ainm.no/tripletex/my/submissions"
ENDPOINT_URL = "https://accounting-agent-502687038260.us-central1.run.app"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwYjM1ZDg2NC02YWU2LTQxNTAtOGIxMi04ZTQyN2QzZDAyYjgiLCJlbWFpbCI6InNhbmRlcndAa3ViZW5sYWJzLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTM1MzA5fQ.DApu5yR5OR-tO2i76IUxZ_uOwDgpjZm6alyGTJhqyVc"
NUM_SUBMISSIONS = 3
COOKIES = {"access_token": ACCESS_TOKEN}


def submit(n=NUM_SUBMISSIONS):
    """Submit the endpoint URL n times. Returns list of queued submission IDs."""
    payload = {"endpoint_url": ENDPOINT_URL}
    ids = []

    for i in range(n):
        if i > 0:
            time.sleep(1)
        resp = requests.post(SUBMIT_URL, json=payload, cookies=COOKIES)
        if resp.status_code == 201:
            data = resp.json()
            ids.append(data["id"])
            print(f"[{i+1}/{n}] Queued: {data['id']}  (daily: {data['daily_submissions_used']}/{data['daily_submissions_max']})")
        else:
            print(f"[{i+1}/{n}] FAILED ({resp.status_code}): {resp.text}")

    return ids


def check_status(ids=None):
    """Check submission statuses. If ids given, wait for them to complete."""
    resp = requests.get(STATUS_URL, cookies=COOKIES)
    if resp.status_code != 200:
        print(f"Failed to fetch status: {resp.status_code}")
        return []

    submissions = resp.json()

    if ids:
        # Filter to only our submissions
        id_set = set(ids)
        submissions = [s for s in submissions if s["id"] in id_set]

    for s in submissions:
        sid = s["id"][:8]
        status = s["status"]
        score = s.get("score_raw")
        score_max = s.get("score_max")
        norm = s.get("normalized_score")
        fb = s.get("feedback", {})
        comment = fb.get("comment", "")
        checks = fb.get("checks", [])
        failed = [c for c in checks if "failed" in c]

        if status == "completed":
            marker = "PASS" if score == score_max and score_max else "FAIL" if score == 0 else "PARTIAL"
            print(f"  {sid} [{marker}] {score}/{score_max} (norm={norm}) — {comment}")
            for c in failed:
                print(f"    {c}")
        else:
            print(f"  {sid} [{status}]")

    return submissions


def wait_for_results(ids, poll_interval=10, timeout=300):
    """Poll until all submission IDs are completed, then print results."""
    print(f"\nWaiting for {len(ids)} submissions to complete...")
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.get(STATUS_URL, cookies=COOKIES)
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue

        all_subs = resp.json()
        id_set = set(ids)
        ours = [s for s in all_subs if s["id"] in id_set]
        pending = [s for s in ours if s["status"] not in ("completed", "failed", "error")]

        if not pending:
            print(f"\nAll done ({int(time.time() - start)}s):\n")
            check_status(ids)

            # Summary
            scores = [s.get("score_raw", 0) or 0 for s in ours]
            maxes = [s.get("score_max", 0) or 0 for s in ours]
            total = sum(scores)
            total_max = sum(maxes)
            perfect = sum(1 for s in ours if s.get("score_raw") == s.get("score_max") and s.get("score_max"))
            print(f"\n  Total: {total}/{total_max}  |  {perfect}/{len(ours)} perfect  |  norm={sum(s.get('normalized_score', 0) or 0 for s in ours):.2f}")
            return ours

        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] {len(pending)} still processing...", end="\r")
        time.sleep(poll_interval)

    print(f"\nTimeout after {timeout}s. Checking current status:")
    check_status(ids)
    return []


def submit_and_wait(n=NUM_SUBMISSIONS):
    """Submit n times, then wait for all results."""
    ids = submit(n)
    if ids:
        return wait_for_results(ids)
    return []


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SUBMISSIONS
    submit_and_wait(n)
