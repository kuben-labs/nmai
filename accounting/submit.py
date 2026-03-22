#!/usr/bin/env python3
"""Submit the accounting agent endpoint and track submission results."""

import json
import os
import requests
import sys
import time
from datetime import datetime

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


RESULTS_LOG = os.path.join(os.path.dirname(__file__), "results_log.json")
GCP_PROJECT = "ai-nm26osl-1759"


def fetch_task_inputs(since_minutes=30):
    """Fetch recent TASK_INPUT logs from Cloud Run to get task prompts."""
    import subprocess
    cmd = [
        "gcloud", "logging", "read",
        f'resource.type="cloud_run_revision" AND resource.labels.service_name="accounting-agent" AND textPayload=~"TASK_INPUT"',
        "--project", GCP_PROJECT,
        "--limit", "50",
        "--freshness", f"{since_minutes}m",
        "--format", "json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  (Could not fetch Cloud Run logs: {result.stderr.strip()[:200]})")
            return []

        entries = json.loads(result.stdout) if result.stdout.strip() else []
        inputs = []
        for entry in entries:
            text = entry.get("textPayload", "")
            ts = entry.get("timestamp", "")
            # Extract the JSON part after the log prefix
            json_start = text.find("{")
            if json_start >= 0:
                try:
                    data = json.loads(text[json_start:])
                    if data.get("tag") == "TASK_INPUT":
                        inputs.append({
                            "timestamp": ts,
                            "prompt": data.get("prompt", ""),
                            "file_count": data.get("file_count", 0),
                            "file_names": data.get("file_names", []),
                        })
                except json.JSONDecodeError:
                    pass
        return inputs
    except Exception as e:
        print(f"  (Could not fetch Cloud Run logs: {e})")
        return []


def classify_prompt(prompt):
    """Extract a short task type label from the prompt using keywords."""
    p = prompt.lower()
    categories = [
        ("bank_reconciliation", ["bankavsteming", "reconciliation", "kontoutskrift", "bank statement"]),
        ("year_end_closing", ["årsavslutning", "year-end", "year end"]),
        ("depreciation", ["avskrivning", "depreciation"]),
        ("tax_provision", ["skattekostnad", "tax provision", "skatt"]),
        ("accrual", ["periodisering", "accrual"]),
        ("salary", ["lønn", "salary", "payroll", "lønnskjøring"]),
        ("travel_expense", ["reise", "travel", "diett", "per diem", "reiseregning"]),
        ("employee_onboarding", ["onboarding", "ansettelse"]),
        ("employee", ["ansatt", "employee", "empleado", "mitarbeiter"]),
        ("credit_note", ["kreditnota", "credit note"]),
        ("invoice_payment", ["betaling", "payment", "innbetaling"]),
        ("invoice", ["faktura", "invoice", "ordre", "order"]),
        ("voucher_correction", ["korreksjon", "correction", "feil"]),
        ("receipt_voucher", ["kvittering", "receipt", "bilag"]),
        ("voucher", ["bilag", "voucher", "journal", "razão", "ledger", "buchung"]),
        ("project", ["prosjekt", "project", "projeto", "proyecto", "projekt", "projet", "atividade", "activit"]),
        ("dimension", ["dimensjon", "dimension"]),
        ("supplier", ["leverandør", "supplier"]),
        ("customer", ["kunde", "customer"]),
        ("department", ["avdeling", "department"]),
        ("company", ["selskap", "company"]),
    ]
    for label, keywords in categories:
        if any(kw in p for kw in keywords):
            return label
    return "unknown"


def match_input_to_submission(submission, task_inputs):
    """Find the task input that matches a submission by timestamp overlap."""
    queued = submission.get("queued_at", "")
    completed = submission.get("completed_at", "")
    if not queued or not completed:
        return None

    # Find inputs whose timestamp falls between queued and completed
    for inp in task_inputs:
        ts = inp["timestamp"]
        if queued <= ts <= completed:
            return inp

    return None


def load_results_log():
    """Load the results log from disk."""
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            return json.load(f)
    return []


def save_results_log(log):
    """Save the results log to disk."""
    with open(RESULTS_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def log_results(submissions, task_inputs=None):
    """Append completed submission results to the log file, with task prompts from server logs."""
    log = load_results_log()
    existing_ids = {entry["id"] for entry in log}

    if task_inputs is None:
        task_inputs = fetch_task_inputs(since_minutes=60)

    for s in submissions:
        if s["id"] in existing_ids:
            continue
        if s["status"] not in ("completed", "failed", "error"):
            continue

        fb = s.get("feedback", {})
        checks = fb.get("checks", [])
        passed = [c for c in checks if "passed" in c]
        failed = [c for c in checks if "failed" in c]

        # Try to match server-side input log
        matched_input = match_input_to_submission(s, task_inputs)
        prompt = matched_input["prompt"] if matched_input else ""
        task_type = classify_prompt(prompt) if prompt else "unknown"
        file_count = matched_input.get("file_count", 0) if matched_input else 0

        entry = {
            "id": s["id"],
            "timestamp": s.get("completed_at", datetime.utcnow().isoformat() + "Z"),
            "status": s["status"],
            "score_raw": s.get("score_raw"),
            "score_max": s.get("score_max"),
            "normalized_score": s.get("normalized_score"),
            "comment": fb.get("comment", ""),
            "task_type": task_type,
            "prompt": prompt[:500],
            "file_count": file_count,
            "checks_passed": len(passed),
            "checks_total": len(checks),
            "failed_checks": failed,
            "result": "PASS" if s.get("score_raw") == s.get("score_max") and s.get("score_max") else
                      "FAIL" if s.get("score_raw") == 0 else "PARTIAL",
        }
        log.append(entry)

    save_results_log(log)
    return log


def show_progress(verbose=False):
    """Show summary of all tracked results — task types and pass rates."""
    log = load_results_log()
    if not log:
        print("No results logged yet.")
        return

    # Group by task type
    by_type = {}
    for entry in log:
        task_type = entry.get("task_type", "unknown")
        if task_type not in by_type:
            by_type[task_type] = {"pass": 0, "partial": 0, "fail": 0, "scores": [], "maxes": [], "entries": []}
        by_type[task_type][entry["result"].lower()] += 1
        by_type[task_type]["scores"].append(entry.get("score_raw", 0) or 0)
        by_type[task_type]["maxes"].append(entry.get("score_max", 0) or 0)
        by_type[task_type]["entries"].append(entry)

    print(f"\n{'Task Type':<30} {'Pass':>5} {'Part':>5} {'Fail':>5} {'Score':>12} {'Rate':>6}")
    print("-" * 70)

    total_score = 0
    total_max = 0
    total_pass = 0
    total_partial = 0
    total_fail = 0

    for task_type in sorted(by_type.keys()):
        d = by_type[task_type]
        s = sum(d["scores"])
        m = sum(d["maxes"])
        total_score += s
        total_max += m
        total_pass += d["pass"]
        total_partial += d["partial"]
        total_fail += d["fail"]
        rate = f"{s/m*100:.0f}%" if m else "N/A"
        print(f"{task_type[:30]:<30} {d['pass']:>5} {d['partial']:>5} {d['fail']:>5} {s:>5}/{m:<5} {rate:>6}")

        if verbose:
            for e in d["entries"]:
                marker = e["result"]
                score = f"{e.get('score_raw', 0)}/{e.get('score_max', 0)}"
                prompt_snip = e.get("prompt", "")[:80].replace("\n", " ")
                print(f"    [{marker:>7}] {score:>7}  {prompt_snip}")
                for fc in e.get("failed_checks", []):
                    print(f"             {fc}")

    print("-" * 70)
    overall_rate = f"{total_score/total_max*100:.0f}%" if total_max else "N/A"
    print(f"{'TOTAL':<30} {total_pass:>5} {total_partial:>5} {total_fail:>5} {total_score:>5}/{total_max:<5} {overall_rate:>6}")
    print(f"\n  {len(log)} submissions logged")


def submit_and_wait(n=NUM_SUBMISSIONS):
    """Submit n times, then wait for all results."""
    ids = submit(n)
    if ids:
        results = wait_for_results(ids)
        if results:
            log_results(results)
        return results
    return []


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "progress":
        verbose = "-v" in sys.argv
        show_progress(verbose=verbose)
    elif len(sys.argv) > 1 and sys.argv[1] == "backfill":
        # Backfill old submissions with task inputs from logs
        mins = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        inputs = fetch_task_inputs(since_minutes=mins)
        print(f"Found {len(inputs)} task inputs in last {mins}m")
        resp = requests.get(STATUS_URL, cookies=COOKIES)
        if resp.status_code == 200:
            subs = [s for s in resp.json() if s["status"] == "completed"]
            log_results(subs, task_inputs=inputs)
            print(f"Logged results. Run 'python submit.py progress' to view.")
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SUBMISSIONS
        submit_and_wait(n)
