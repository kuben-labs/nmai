"""Salary/payroll handler."""

import calendar
import json
import logging

from ._helpers import (
    get_first_department_id, find_or_create_employee,
    ensure_employee_date_of_birth,
)

logger = logging.getLogger(__name__)


def handle_salary(tripletex, params):
    """Run salary/payroll for an employee."""
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

    ensure_employee_date_of_birth(tripletex, employee_id)

    employment_id = None
    result = tripletex.get("/employee/employment", {"employeeId": employee_id, "fields": "id,division,version", "count": 1})
    if result["status_code"] == 200:
        existing = result["body"].get("values", [])
        if existing:
            employment_id = existing[0]["id"]

    if not employment_id:
        result = tripletex.post("/employee/employment", {
            "employee": {"id": employee_id},
            "startDate": f"{params['year']}-01-01",
        })
        if result["status_code"] in (200, 201):
            employment_id = result["body"].get("value", {}).get("id")

    division_id = None
    result = tripletex.get("/division", {"fields": "id,name", "count": 10})
    if result["status_code"] == 200:
        divs = result["body"].get("values", [])
        if divs:
            division_id = divs[0]["id"]

    if not division_id:
        result_muni = tripletex.get("/municipality", {"fields": "id,name", "count": 1})
        muni_id = None
        if result_muni["status_code"] == 200:
            munis = result_muni["body"].get("values", [])
            if munis:
                muni_id = munis[0]["id"]

        div_data = {
            "name": "Hovedkontor",
            "startDate": f"{params['year']}-01-01",
            "organizationNumber": "000000000",
        }
        if muni_id:
            div_data["municipality"] = {"id": muni_id}
        result = tripletex.post("/division", div_data)
        if result["status_code"] in (200, 201):
            division_id = result["body"].get("value", {}).get("id")

    if employment_id and division_id:
        result = tripletex.get(f"/employee/employment/{employment_id}", {"fields": "id,version,division,startDate"})
        if result["status_code"] == 200:
            emp_data = result["body"].get("value", {})
            current_div = emp_data.get("division")
            if not current_div or not current_div.get("id"):
                tripletex.put(f"/employee/employment/{employment_id}", {
                    "id": emp_data["id"],
                    "version": emp_data.get("version", 0),
                    "startDate": emp_data.get("startDate", f"{params['year']}-01-01"),
                    "division": {"id": division_id},
                })
                logger.info(f"Linked employment {employment_id} to division {division_id}")

    salary_type_id = None
    target_name = params.get("salary_type_name", "fastlønn").lower()
    result = tripletex.get("/salary/type", {"fields": "id,name,number", "count": 100})
    if result["status_code"] == 200:
        for st in result["body"].get("values", []):
            if target_name in (st.get("name") or "").lower():
                salary_type_id = st["id"]
                break
        if not salary_type_id and result["body"].get("values"):
            salary_type_id = result["body"]["values"][0]["id"]

    last_day = calendar.monthrange(params['year'], params['month'])[1]
    voucher_date = params.get("date") or f"{params['year']}-{params['month']:02d}-{last_day:02d}"

    salary_data = {
        "date": voucher_date,
        "month": params["month"],
        "year": params["year"],
    }

    gross = params.get("gross_salary")
    if gross and salary_type_id:
        salary_data["payslips"] = [{
            "employee": {"id": employee_id},
            "specifications": [{
                "salaryType": {"id": salary_type_id},
                "count": 1,
                "rate": gross,
                "amount": gross,
            }]
        }]

    result = tripletex.post("/salary/transaction", salary_data)
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Salary transaction failed: {json.dumps(result, default=str)[:500]}"}

    transaction_id = result["body"].get("value", {}).get("id")
    logger.info(f"Salary transaction created: id={transaction_id}")
    return {"success": True, "transaction_id": transaction_id}
