"""Ledger analysis handler — compare expense accounts across months."""

import calendar
import logging
from collections import defaultdict
from datetime import date

from ._helpers import get_all_accounts

logger = logging.getLogger(__name__)


def handle_ledger_analysis(tripletex, params):
    """Analyze ledger for expense account changes and create projects/activities."""
    today = date.today().isoformat()

    start_month = params["start_month"]
    end_month = params["end_month"]
    num_accounts = params.get("num_accounts", 3)
    direction = params.get("change_direction", "increase")

    start_year, start_m = int(start_month.split("-")[0]), int(start_month.split("-")[1])
    end_year, end_m = int(end_month.split("-")[0]), int(end_month.split("-")[1])
    date_from = f"{start_year}-{start_m:02d}-01"
    last_day = calendar.monthrange(end_year, end_m)[1]
    date_to = f"{end_year}-{end_m:02d}-{last_day:02d}"

    result = tripletex.get("/ledger/posting", {
        "fields": "id,date,account,amount",
        "dateFrom": date_from, "dateTo": date_to, "count": 10000,
    })
    if result["status_code"] != 200:
        return {"success": False, "error": "Could not fetch postings"}

    postings = result["body"].get("values", [])

    all_accounts = get_all_accounts(tripletex)
    id_to_num = {a["id"]: a.get("number") for a in all_accounts if a.get("number")}
    id_to_name = {a["id"]: a.get("name", "") for a in all_accounts}

    account_month_totals = defaultdict(lambda: defaultdict(float))
    account_info = {}

    for post in postings:
        p_date = post.get("date", "")
        if not p_date or len(p_date) < 7:
            continue
        month_key = p_date[:7]
        acct = post.get("account", {})
        acct_num = acct.get("number") or id_to_num.get(acct.get("id"))
        acct_num_str = str(acct_num) if acct_num else ""
        amount = post.get("amount", 0) or 0
        if acct_num_str:
            account_month_totals[acct_num_str][month_key] += amount
            acct_name = acct.get("name") or id_to_name.get(acct.get("id"), "")
            account_info[acct_num_str] = (acct_name, acct.get("id", ""))

    changes = []
    for acct_num, totals in account_month_totals.items():
        try:
            num = int(acct_num)
        except ValueError:
            continue
        if 4000 <= num <= 8999:
            val_start = totals.get(start_month, 0)
            val_end = totals.get(end_month, 0)
            diff = val_end - val_start
            name, aid = account_info.get(acct_num, ("", ""))
            changes.append((diff, acct_num, name, aid, val_start, val_end))

    if direction == "increase":
        changes.sort(reverse=True)
    else:
        changes.sort()

    top_accounts = changes[:num_accounts]
    logger.info(f"Top {num_accounts} accounts by {direction}: {[(c[1], c[2], round(c[0], 2)) for c in top_accounts]}")

    project_manager_id = None
    result = tripletex.get("/employee", {"fields": "id,firstName,lastName", "count": 1})
    if result["status_code"] == 200:
        emps = result["body"].get("values", [])
        if emps:
            project_manager_id = emps[0]["id"]

    if not project_manager_id:
        return {"success": False, "error": "No employee found for project manager"}

    is_internal = params.get("is_internal", True)
    create_projects = params.get("create_projects", True)
    create_activities = params.get("create_activities", True)

    for diff, acct_num, acct_name, acct_id, val_start, val_end in top_accounts:
        project_name = acct_name or f"Account {acct_num}"

        if create_projects:
            result = tripletex.post("/project", {
                "name": project_name,
                "projectManager": {"id": project_manager_id},
                "startDate": today,
                "isInternal": is_internal,
            })
            project_id = None
            if result["status_code"] in (200, 201):
                project_id = result["body"].get("value", {}).get("id")
            else:
                logger.warning(f"Project creation failed for {project_name}: {result['status_code']}")
                continue

            if create_activities and project_id:
                result = tripletex.post("/activity", {
                    "name": project_name,
                    "activityType": "PROJECT_GENERAL_ACTIVITY",
                })
                activity_id = None
                if result["status_code"] in (200, 201):
                    activity_id = result["body"].get("value", {}).get("id")

                if activity_id:
                    result = tripletex.post("/project/projectActivity", {
                        "activity": {"id": activity_id},
                        "project": {"id": project_id},
                    })
                    if result["status_code"] not in (200, 201):
                        logger.warning(f"Project-activity link failed: {result['status_code']}")

    return {"success": True, "top_accounts": [(c[1], c[2], round(c[0], 2)) for c in top_accounts]}
