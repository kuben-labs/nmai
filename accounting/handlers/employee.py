"""Employee creation handler."""

import logging
from datetime import date

from ._helpers import find_department_id, get_first_department_id, find_or_create_employee

logger = logging.getLogger(__name__)


def handle_employee_creation(tripletex, params):
    """Create employee + employment + details + standard time."""
    today = date.today().isoformat()

    dept_id = find_department_id(tripletex, params.get("department_name"))
    if not dept_id:
        dept_id = get_first_department_id(tripletex)

    employee_id = find_or_create_employee(
        tripletex, params["first_name"], params["last_name"],
        email=params.get("email"),
        date_of_birth=params.get("date_of_birth"),
        department_id=dept_id,
        phoneNumberMobile=params.get("phone_mobile"),
        bankAccountNumber=params.get("bank_account"),
        nationalIdentityNumber=params.get("national_id"),
        employeeNumber=params.get("employee_number"),
    )
    if not employee_id:
        return {"success": False, "error": "Could not create employee"}

    start_date = params.get("start_date") or today
    percentage = params.get("percentage", 100.0)

    employment_id = None
    result = tripletex.get("/employee/employment", {"employeeId": employee_id, "fields": "id,startDate", "count": 10})
    if result["status_code"] == 200:
        existing = result["body"].get("values", [])
        if existing:
            employment_id = existing[0]["id"]

    if not employment_id:
        result = tripletex.post("/employee/employment", {
            "employee": {"id": employee_id},
            "startDate": start_date,
        })
        if result["status_code"] in (200, 201):
            employment_id = result["body"].get("value", {}).get("id")
        else:
            logger.warning(f"Employment POST failed: {result}")

    if employment_id:
        annual_salary = params.get("annual_salary")
        if not annual_salary and params.get("monthly_salary"):
            annual_salary = round(params["monthly_salary"] * 12, 2)

        if annual_salary or params.get("percentage"):
            details_data = {
                "employment": {"id": employment_id},
                "date": start_date,
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "remunerationType": "MONTHLY_WAGE",
                "workingHoursScheme": "NOT_SHIFT",
                "percentageOfFullTimeEquivalent": percentage,
            }
            if annual_salary:
                details_data["annualSalary"] = annual_salary

            result = tripletex.post("/employee/employment/details", details_data)
            if result["status_code"] not in (200, 201):
                logger.warning(f"Employment details POST failed: {result['status_code']}")

    if params.get("hours_per_day"):
        result = tripletex.post("/employee/standardTime", {
            "employee": {"id": employee_id},
            "fromDate": start_date,
            "hoursPerDay": params["hours_per_day"],
        })
        if result["status_code"] not in (200, 201):
            logger.warning(f"Standard time POST failed: {result['status_code']}")

    logger.info(f"Employee created: id={employee_id}, employment={employment_id}")
    return {"success": True, "employee_id": employee_id, "employment_id": employment_id}
