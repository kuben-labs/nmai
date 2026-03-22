"""Shared helpers for Tripletex API lookups and entity creation."""

import json
import logging
from datetime import date

logger = logging.getLogger(__name__)


def get_all_accounts(tripletex):
    result = tripletex.get("/ledger/account", {"fields": "id,number,name", "count": 1000})
    if result["status_code"] == 200:
        return result["body"].get("values", [])
    return []


def find_account_id(tripletex, account_number, accounts_cache=None):
    acct_str = str(account_number)
    if accounts_cache:
        for a in accounts_cache:
            if str(a.get("number")) == acct_str:
                return a["id"]
    result = tripletex.get("/ledger/account", {"number": acct_str, "fields": "id,number,name", "count": 1})
    if result["status_code"] == 200:
        values = result["body"].get("values", [])
        if values:
            return values[0]["id"]
    return None


def find_account_in_range(accounts_cache, target_num, range_start, range_end):
    """Find target account or closest account in range from cache. /ledger/account is GET-only."""
    for a in accounts_cache:
        if a.get("number") == target_num:
            return a["id"]
    best = None
    best_diff = float("inf")
    for a in accounts_cache:
        num = a.get("number", 0)
        if isinstance(num, int) and range_start <= num <= range_end:
            diff = abs(num - target_num)
            if diff < best_diff:
                best_diff = diff
                best = a["id"]
    return best


def find_department_id(tripletex, name):
    if not name:
        return None
    result = tripletex.get("/department", {"fields": "id,name", "count": 50})
    if result["status_code"] == 200:
        for d in result["body"].get("values", []):
            if name.lower() in d.get("name", "").lower():
                return d["id"]
    return None


def get_first_department_id(tripletex):
    result = tripletex.get("/department", {"fields": "id,name", "count": 1})
    if result["status_code"] == 200:
        depts = result["body"].get("values", [])
        if depts:
            return depts[0]["id"]
    return None


def find_or_create_supplier(tripletex, name, org_number=None):
    result = tripletex.get("/supplier", {"fields": "id,name,organizationNumber", "count": 100})
    if result["status_code"] == 200:
        for s in result["body"].get("values", []):
            if s.get("name", "").lower() == name.lower():
                return s["id"]
            if org_number and s.get("organizationNumber") == org_number:
                return s["id"]

    data = {"name": name}
    if org_number:
        data["organizationNumber"] = org_number
    result = tripletex.post("/supplier", data)
    if result["status_code"] in (200, 201):
        return result["body"].get("value", {}).get("id")
    logger.warning(f"Failed to create supplier: {result}")
    return None


def find_or_create_customer(tripletex, name, org_number=None, email=None):
    result = tripletex.get("/customer", {"fields": "id,name,organizationNumber", "count": 100})
    if result["status_code"] == 200:
        for c in result["body"].get("values", []):
            if c.get("name", "").lower() == name.lower():
                return c["id"]
            if org_number and c.get("organizationNumber") == org_number:
                return c["id"]

    data = {"name": name}
    if org_number:
        data["organizationNumber"] = org_number
    if email:
        data["email"] = email
    result = tripletex.post("/customer", data)
    if result["status_code"] in (200, 201):
        return result["body"].get("value", {}).get("id")
    return None


def find_or_create_employee(tripletex, first_name, last_name, email=None, date_of_birth=None, department_id=None, **extra):
    result = tripletex.get("/employee", {"fields": "id,firstName,lastName,email", "count": 100})
    if result["status_code"] == 200:
        for e in result["body"].get("values", []):
            if (e.get("firstName", "").lower() == first_name.lower() and
                    e.get("lastName", "").lower() == last_name.lower()):
                return e["id"]
            if email and e.get("email", "").lower() == email.lower():
                return e["id"]

    data = {"firstName": first_name, "lastName": last_name}
    if email:
        data["email"] = email
    if date_of_birth:
        data["dateOfBirth"] = date_of_birth
    if department_id:
        data["department"] = {"id": department_id}
    for key in ["phoneNumberMobile", "bankAccountNumber", "nationalIdentityNumber", "employeeNumber"]:
        if extra.get(key):
            data[key] = extra[key]

    result = tripletex.post("/employee", data)
    if result["status_code"] in (200, 201):
        return result["body"].get("value", {}).get("id")
    logger.warning(f"Failed to create employee: {result}")
    return None


def find_project_id(tripletex, name):
    if not name:
        return None
    result = tripletex.get("/project", {"fields": "id,name", "count": 50})
    if result["status_code"] == 200:
        for p in result["body"].get("values", []):
            if name.lower() in p.get("name", "").lower():
                return p["id"]
    return None


def find_vat_type_id(tripletex, target_pct):
    """Find VAT type ID for a given percentage (prefer input/inngående VAT)."""
    result = tripletex.get("/ledger/vatType", {"fields": "id,name,percentage", "count": 100})
    if result["status_code"] == 200:
        for vt in result["body"].get("values", []):
            pct = vt.get("percentage")
            name = (vt.get("name") or "").lower()
            if pct == target_pct and ("inngående" in name or "input" in name or "innk" in name):
                return vt["id"]
        for vt in result["body"].get("values", []):
            if vt.get("percentage") == target_pct:
                return vt["id"]
    return None


def get_enabled_zone(tripletex):
    """Find an enabled travel expense zone."""
    result = tripletex.get("/travelExpense/zone", {"isDisabled": "false", "fields": "id,zoneName", "count": 10})
    if result["status_code"] == 200:
        values = result["body"].get("values", [])
        if values:
            return values[0]["id"]
    return None


def get_cost_category(tripletex, hint=""):
    """Find a travel expense cost category."""
    result = tripletex.get("/travelExpense/costCategory", {"fields": "id,description", "count": 50})
    if result["status_code"] == 200:
        values = result["body"].get("values", [])
        if hint and values:
            h = hint.lower()
            for v in values:
                n = (v.get("description") or "").lower()
                if any(kw in n for kw in h.split()):
                    return v["id"]
        if values:
            return values[0]["id"]
    return None


def get_payment_type(tripletex):
    """Find a travel expense payment type."""
    result = tripletex.get("/travelExpense/paymentType", {"fields": "id,description", "count": 10})
    if result["status_code"] == 200:
        values = result["body"].get("values", [])
        if values:
            return values[0]["id"]
    return None


def ensure_bank_account(tripletex):
    """Ensure the company has a bank account number set (required for invoicing)."""
    result = tripletex.get("/ledger/account", {
        "isBankAccount": "true", "fields": "id,number,name,bankAccountNumber,version", "count": 10
    })
    if result["status_code"] != 200:
        return
    for acct in result["body"].get("values", []):
        ban = acct.get("bankAccountNumber")
        if ban and ban.strip():
            return
    for acct in result["body"].get("values", []):
        tripletex.put(f"/ledger/account/{acct['id']}", {
            "id": acct["id"],
            "version": acct.get("version", 0),
            "name": acct.get("name", "Bank"),
            "bankAccountNumber": "12345678903",
        })
        logger.info(f"Set bank account number on account {acct['id']}")
        return


def ensure_employee_date_of_birth(tripletex, employee_id):
    """Ensure employee has dateOfBirth set (required for salary processing)."""
    result = tripletex.get(f"/employee/{employee_id}", {"fields": "id,version,firstName,lastName,dateOfBirth"})
    if result["status_code"] != 200:
        return
    emp = result["body"].get("value", {})
    if emp.get("dateOfBirth"):
        return
    tripletex.put(f"/employee/{employee_id}", {
        "id": emp["id"],
        "version": emp.get("version", 0),
        "firstName": emp.get("firstName", ""),
        "lastName": emp.get("lastName", ""),
        "dateOfBirth": "1990-01-01",
    })
    logger.info(f"Set default dateOfBirth for employee {employee_id}")


def extract_structured(client, model, prompt, files, schema):
    """Use Claude tool_use to extract structured data from the prompt."""
    from agent import build_user_message

    user_content = build_user_message(prompt, files)

    system = (
        "You are a data extraction assistant. Extract the requested information from the task prompt. "
        "The prompt may be in Norwegian, English, Spanish, Portuguese, German, French, or Nynorsk. "
        "Extract ALL values exactly as stated. Use YYYY-MM-DD for all dates. "
        "If a value is not mentioned, omit it or use empty string."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        tools=[schema],
        tool_choice={"type": "tool", "name": schema["name"]},
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return None
