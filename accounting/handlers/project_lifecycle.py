"""Project lifecycle handler — project, activity, timesheet, supplier cost, invoice."""

import json
import logging
from datetime import date

from ._helpers import (
    find_or_create_customer, find_or_create_employee, find_or_create_supplier,
    get_first_department_id, get_all_accounts, find_account_id, ensure_bank_account,
)

logger = logging.getLogger(__name__)


def handle_project_lifecycle(tripletex, params):
    """Execute complete project lifecycle: create project, log time, supplier cost, invoice."""
    today = date.today().isoformat()

    customer_id = find_or_create_customer(
        tripletex, params["customer_name"], params.get("customer_org_number")
    )
    if not customer_id:
        return {"success": False, "error": "Could not create/find customer"}

    team = []
    project_manager_id = None
    dept_id = get_first_department_id(tripletex)
    for member in params.get("team_members", []):
        emp_id = find_or_create_employee(
            tripletex, member["first_name"], member["last_name"],
            email=member.get("email"), department_id=dept_id,
        )
        if emp_id:
            team.append({"id": emp_id, "hours": member["hours"], "rate": member.get("hourly_rate")})
            if member.get("is_project_manager"):
                project_manager_id = emp_id

    if not project_manager_id and team:
        project_manager_id = team[0]["id"]
    if not project_manager_id:
        return {"success": False, "error": "No project manager found"}

    project_data = {
        "name": params["project_name"],
        "projectManager": {"id": project_manager_id},
        "startDate": today,
        "customer": {"id": customer_id},
    }
    if params.get("budget"):
        project_data["isFixedPrice"] = True
        project_data["fixedprice"] = params["budget"]

    result = tripletex.post("/project", project_data)
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Project POST failed: {json.dumps(result, default=str)[:500]}"}

    project_id = result["body"].get("value", {}).get("id")
    if not project_id:
        return {"success": False, "error": "Project created but no ID returned"}
    logger.info(f"Project created: id={project_id}")

    activity_name = params.get("activity_name") or f"{params['project_name']} - Work"
    result = tripletex.post("/activity", {
        "name": activity_name,
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

    if activity_id:
        for member in team:
            entry_data = {
                "employee": {"id": member["id"]},
                "activity": {"id": activity_id},
                "project": {"id": project_id},
                "date": today,
                "hours": member["hours"],
            }
            if member.get("rate"):
                entry_data["hourlyRate"] = member["rate"]
            result = tripletex.post("/timesheet/entry", entry_data)
            if result["status_code"] not in (200, 201):
                logger.warning(f"Timesheet entry failed for employee {member['id']}: {result['status_code']}")

    if params.get("supplier_name") and params.get("supplier_cost_amount"):
        supplier_id = find_or_create_supplier(
            tripletex, params["supplier_name"], params.get("supplier_org_number")
        )
        if supplier_id:
            accounts = get_all_accounts(tripletex)
            expense_acct_id = None
            for acct_num in [4300, 4390, 4500, 6590, 7770]:
                expense_acct_id = find_account_id(tripletex, acct_num, accounts)
                if expense_acct_id:
                    break
            if not expense_acct_id:
                for a in accounts:
                    num = a.get("number", 0)
                    if isinstance(num, int) and 4000 <= num <= 7999:
                        expense_acct_id = a["id"]
                        break

            ap_acct_id = find_account_id(tripletex, 2400, accounts)

            if expense_acct_id and ap_acct_id:
                amount = params["supplier_cost_amount"]
                desc = params.get("supplier_cost_description", f"Supplier cost - {params['supplier_name']}")
                result = tripletex.post("/ledger/voucher", {
                    "date": today,
                    "description": desc,
                    "postings": [
                        {
                            "account": {"id": expense_acct_id},
                            "amount": amount, "amountCurrency": amount,
                            "amountGross": amount, "amountGrossCurrency": amount,
                            "description": desc,
                            "project": {"id": project_id},
                        },
                        {
                            "account": {"id": ap_acct_id},
                            "amount": -amount, "amountCurrency": -amount,
                            "amountGross": -amount, "amountGrossCurrency": -amount,
                            "description": f"AP - {params['supplier_name']}",
                            "supplier": {"id": supplier_id},
                        },
                    ]
                })
                if result["status_code"] not in (200, 201):
                    logger.warning(f"Supplier cost voucher failed: {result['status_code']}")

    if params.get("create_customer_invoice"):
        ensure_bank_account(tripletex)
        result = tripletex.post("/product", {"name": params["project_name"]})
        product_id = None
        if result["status_code"] in (200, 201):
            product_id = result["body"].get("value", {}).get("id")

        if product_id:
            invoice_amount = params.get("budget")
            if not invoice_amount:
                invoice_amount = sum(
                    m.get("hours", 0) * m.get("rate", 0) for m in team if m.get("rate")
                )
            if not invoice_amount:
                invoice_amount = params.get("supplier_cost_amount", 0)
            result = tripletex.post("/order", {
                "customer": {"id": customer_id},
                "deliveryDate": today,
                "orderDate": today,
                "project": {"id": project_id},
                "orderLines": [{
                    "product": {"id": product_id},
                    "count": 1,
                    "unitPriceExcludingVatCurrency": invoice_amount,
                    "description": f"Project: {params['project_name']}",
                }]
            })
            order_id = None
            if result["status_code"] in (200, 201):
                order_id = result["body"].get("value", {}).get("id")

            if order_id:
                result = tripletex.post("/invoice", {
                    "invoiceDate": today,
                    "invoiceDueDate": today,
                    "orders": [{"id": order_id}],
                })
                if result["status_code"] not in (200, 201):
                    logger.warning(f"Invoice creation failed: {result['status_code']}")

    return {"success": True, "project_id": project_id}
