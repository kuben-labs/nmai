"""Travel expense handler."""

import json
import logging
from datetime import date

from ._helpers import (
    get_first_department_id, find_or_create_employee,
    get_enabled_zone, get_payment_type, get_cost_category,
)

logger = logging.getLogger(__name__)


def handle_travel_expense(tripletex, params):
    """Create travel expense with costs and per diem."""
    today = date.today().isoformat()

    employee_id = None
    result = tripletex.get("/employee", {"fields": "id,firstName,lastName,email", "count": 100})
    if result["status_code"] == 200:
        for e in result["body"].get("values", []):
            if (e.get("firstName", "").lower() == params["employee_first_name"].lower() and
                    e.get("lastName", "").lower() == params["employee_last_name"].lower()):
                employee_id = e["id"]
                break
            if params.get("employee_email") and e.get("email", "").lower() == params["employee_email"].lower():
                employee_id = e["id"]
                break

    if not employee_id:
        dept_id = get_first_department_id(tripletex)
        employee_id = find_or_create_employee(
            tripletex, params["employee_first_name"], params["employee_last_name"],
            email=params.get("employee_email"), department_id=dept_id,
        )
    if not employee_id:
        return {"success": False, "error": "Could not find/create employee"}

    zone_id = get_enabled_zone(tripletex)
    payment_type_id = get_payment_type(tripletex)

    dep_date = params.get("departure_date") or today
    ret_date = params.get("return_date") or dep_date
    destination = params.get("destination", "Norge")

    travel_data = {
        "employee": {"id": employee_id},
        "title": params["title"],
        "date": dep_date,
    }
    travel_data["travelDetails"] = {
        "departureDate": dep_date,
        "returnDate": ret_date,
        "destination": destination,
    }

    result = tripletex.post("/travelExpense", travel_data)
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Travel expense POST failed: {json.dumps(result, default=str)[:500]}"}

    travel_expense_id = result["body"].get("value", {}).get("id")
    if not travel_expense_id:
        return {"success": False, "error": "Travel expense created but no ID returned"}
    logger.info(f"Travel expense created: id={travel_expense_id}")

    for cost_item in params.get("costs", []):
        cost_data = {
            "travelExpense": {"id": travel_expense_id},
            "amountCurrencyIncVat": cost_item["amount"],
            "comments": cost_item.get("description", ""),
            "date": dep_date,
            "isPaidByEmployee": True,
        }
        if payment_type_id:
            cost_data["paymentType"] = {"id": payment_type_id}

        cost_cat_id = get_cost_category(tripletex, cost_item.get("description", ""))
        if cost_cat_id:
            cost_data["costCategory"] = {"id": cost_cat_id}

        result = tripletex.post("/travelExpense/cost", cost_data)
        if result["status_code"] not in (200, 201):
            logger.warning(f"Cost POST failed: {result['status_code']} — {json.dumps(result.get('body', {}), default=str)[:300]}")

    if params.get("per_diem_days") and params.get("per_diem_rate"):
        per_diem_data = {
            "travelExpense": {"id": travel_expense_id},
            "count": params["per_diem_days"],
            "rate": params["per_diem_rate"],
            "overnightAccommodation": "NONE",
            "location": f"{destination}, Norge" if destination and "Norge" not in destination and "Norway" not in destination else destination,
        }
        if zone_id:
            per_diem_data["travelExpenseZoneId"] = zone_id

        result = tripletex.post("/travelExpense/perDiemCompensation", per_diem_data)
        if result["status_code"] not in (200, 201):
            logger.warning(f"PerDiem failed ({result['status_code']}), retrying with zone only")
            per_diem_data.pop("countryCode", None)
            if zone_id:
                per_diem_data["travelExpenseZoneId"] = zone_id
            result = tripletex.post("/travelExpense/perDiemCompensation", per_diem_data)
            if result["status_code"] not in (200, 201):
                logger.warning(f"PerDiem retry also failed: {result['status_code']}")

    tripletex.get(f"/travelExpense/{travel_expense_id}", {"fields": "*"})
    logger.info(f"Travel expense completed: id={travel_expense_id}")
    return {"success": True, "travel_expense_id": travel_expense_id}
