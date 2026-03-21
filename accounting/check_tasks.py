#!/usr/bin/env python3
"""Fetch task results from Cloud Run logs and save as local files.

Usage:
    python check_tasks.py              # Fetch latest results
    python check_tasks.py --revision X # Fetch specific revision
    python check_tasks.py --clean      # Just clean old results
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "task_results")
FAILED_DIR = os.path.join(RESULTS_DIR, "failed")
SUCCESS_DIR = os.path.join(RESULTS_DIR, "success")
PROJECT = "ai-nm26osl-1759"
SERVICE = "accounting-agent"


def clean_results():
    """Remove old result files."""
    for d in [FAILED_DIR, SUCCESS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("Cleaned old results.")


def get_latest_revision():
    """Get the latest serving revision name."""
    result = subprocess.run(
        ["gcloud", "run", "revisions", "list",
         "--service", SERVICE, "--region", "us-central1",
         "--project", PROJECT, "--format", "value(metadata.name)",
         "--limit", "1", "--sort-by", "~metadata.creationTimestamp"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def fetch_task_results(revision=None):
    """Fetch TASK_RESULT logs from Cloud Run."""
    if not revision:
        revision = get_latest_revision()
    print(f"Fetching results for revision: {revision}")

    log_filter = (
        f'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{SERVICE}" '
        f'resource.labels.revision_name="{revision}" '
        f'textPayload=~"TASK_RESULT:"'
    )

    result = subprocess.run(
        ["gcloud", "logging", "read", log_filter,
         "--project", PROJECT, "--limit", "50", "--format", "json"],
        capture_output=True, text=True, timeout=60
    )

    if result.returncode != 0:
        print(f"Error fetching logs: {result.stderr}")
        return []

    logs = json.loads(result.stdout)
    tasks = []
    for log in logs:
        text = log.get("textPayload", "")
        idx = text.find("TASK_RESULT: ")
        if idx >= 0:
            try:
                summary = json.loads(text[idx + 13:])
                summary["log_timestamp"] = log.get("timestamp", "")
                tasks.append(summary)
            except json.JSONDecodeError:
                pass

    tasks.sort(key=lambda x: x.get("log_timestamp", ""))
    return tasks


def save_results(tasks):
    """Save task results as files."""
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(SUCCESS_DIR, exist_ok=True)

    failed = 0
    success = 0

    for i, task in enumerate(tasks):
        status = task.get("status", "UNKNOWN")
        prompt = task.get("prompt", "no prompt")
        # Create a short filename from the prompt
        slug = prompt[:80].replace(" ", "_").replace("/", "-").replace(":", "")
        slug = "".join(c for c in slug if c.isalnum() or c in "_-.")
        ts = task.get("timestamp", "")[:19].replace(":", "")
        filename = f"{ts}_{slug}.json"

        content = json.dumps(task, indent=2, ensure_ascii=False)

        if status == "FAILED":
            filepath = os.path.join(FAILED_DIR, filename)
            failed += 1
        else:
            filepath = os.path.join(SUCCESS_DIR, filename)
            success += 1

        with open(filepath, "w") as f:
            f.write(content)

    return success, failed


def print_summary(tasks):
    """Print a concise summary of results."""
    failed_tasks = [t for t in tasks if t.get("status") == "FAILED"]
    ok_tasks = [t for t in tasks if t.get("status") == "OK"]

    print(f"\n{'='*60}")
    print(f"  {len(ok_tasks)} OK  |  {len(failed_tasks)} FAILED  |  {len(tasks)} total")
    print(f"{'='*60}")

    if failed_tasks:
        print(f"\nFAILED TASKS:")
        print(f"{'-'*60}")
        for t in failed_tasks:
            prompt = t.get("prompt", "")[:100]
            iters = t.get("iterations", 0)
            errs = t.get("write_error_count", 0)
            print(f"\n  [{iters} iters, {errs} write errors] {prompt}...")

            # Show unique error messages
            seen = set()
            for e in t.get("errors", [])[:5]:
                body = e.get("error_body", "")
                # Extract validation message
                try:
                    body_parsed = json.loads(body)
                    msgs = body_parsed.get("validationMessages", [])
                    for m in msgs:
                        msg = m.get("message", "")[:120]
                        if msg not in seen:
                            seen.add(msg)
                            print(f"    -> {e['tool']} {e['endpoint']} [{e['status_code']}]: {msg}")
                except Exception:
                    key = f"{e['tool']} {e['endpoint']} [{e['status_code']}]"
                    if key not in seen:
                        seen.add(key)
                        print(f"    -> {key}: {body[:120]}")

    if ok_tasks:
        print(f"\nSUCCESSFUL TASKS:")
        print(f"{'-'*60}")
        for t in ok_tasks:
            prompt = t.get("prompt", "")[:100]
            iters = t.get("iterations", 0)
            writes = t.get("write_calls", 0)
            print(f"  [{iters} iters, {writes} writes] {prompt}...")

    print()


def main():
    parser = argparse.ArgumentParser(description="Check accounting agent task results")
    parser.add_argument("--revision", help="Specific revision to check")
    parser.add_argument("--clean", action="store_true", help="Clean old results only")
    parser.add_argument("--no-clean", action="store_true", help="Don't clean before fetching")
    args = parser.parse_args()

    if args.clean:
        clean_results()
        return

    if not args.no_clean:
        clean_results()

    tasks = fetch_task_results(args.revision)
    if not tasks:
        print("No task results found in logs.")
        return

    success, failed = save_results(tasks)
    print_summary(tasks)
    print(f"Saved: {success} to {SUCCESS_DIR}/")
    print(f"       {failed} to {FAILED_DIR}/")


if __name__ == "__main__":
    main()
