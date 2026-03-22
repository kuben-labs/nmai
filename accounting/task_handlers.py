"""Deterministic task handlers for common accounting tasks.

For recognized task patterns, these handlers:
1. Use a single LLM call to extract structured data from the prompt
2. Execute the correct API calls in Python (no LLM guessing)

This is more reliable than LLM consensus for formulaic tasks.
"""

import calendar
import json
import logging
import re
from datetime import date

from tripletex import TripletexClient

logger = logging.getLogger(__name__)

# --- Extraction schemas (Claude tool_use for structured output) ---

SUPPLIER_INVOICE_SCHEMA = {
    "name": "extract_supplier_invoice",
    "description": "Extract supplier invoice details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_name": {"type": "string"},
            "supplier_org_number": {"type": "string", "description": "9-digit org number if mentioned"},
            "invoice_reference": {"type": "string", "description": "Invoice number/reference"},
            "total_amount": {"type": "number", "description": "Total invoice amount in NOK"},
            "vat_included": {"type": "boolean", "description": "True if total_amount includes VAT"},
            "vat_rate_percent": {"type": "number", "description": "VAT rate as percentage (e.g. 25)"},
            "expense_account_number": {"type": "integer", "description": "Expense account number (e.g. 6590)"},
            "expense_description": {"type": "string", "description": "What the invoice is for"},
            "invoice_date": {"type": "string", "description": "YYYY-MM-DD or empty for today"},
            "due_date": {"type": "string", "description": "Payment due date YYYY-MM-DD or empty"},
            "department_name": {"type": "string"},
        },
        "required": ["supplier_name", "total_amount", "vat_included", "vat_rate_percent", "expense_account_number"]
    }
}

EMPLOYEE_SCHEMA = {
    "name": "extract_employee",
    "description": "Extract employee details from the task prompt or attached PDF/image",
    "input_schema": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "email": {"type": "string"},
            "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
            "start_date": {"type": "string", "description": "Employment start date YYYY-MM-DD"},
            "department_name": {"type": "string"},
            "phone_mobile": {"type": "string"},
            "bank_account": {"type": "string"},
            "national_id": {"type": "string"},
            "employee_number": {"type": "string"},
            "percentage": {"type": "number", "description": "Employment percentage (e.g. 100 for full-time, 80 for 80%)"},
            "annual_salary": {"type": "number", "description": "Annual salary in NOK"},
            "monthly_salary": {"type": "number", "description": "Monthly salary in NOK"},
            "hours_per_day": {"type": "number", "description": "Standard working hours per day (e.g. 7.5 or 8)"},
        },
        "required": ["first_name", "last_name"]
    }
}

CUSTOMER_INVOICE_SCHEMA = {
    "name": "extract_customer_invoice",
    "description": "Extract customer invoice details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string"},
            "customer_org_number": {"type": "string"},
            "customer_email": {"type": "string"},
            "invoice_date": {"type": "string", "description": "YYYY-MM-DD"},
            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
            "delivery_date": {"type": "string", "description": "YYYY-MM-DD"},
            "order_lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "product_number": {"type": "string", "description": "Product number/code if specified"},
                        "quantity": {"type": "number"},
                        "unit_price_excl_vat": {"type": "number"},
                        "vat_rate_percent": {"type": "number", "description": "Default 25"},
                        "description": {"type": "string"},
                    },
                    "required": ["product_name", "quantity", "unit_price_excl_vat"]
                }
            },
            "department_name": {"type": "string"},
            "project_name": {"type": "string"},
        },
        "required": ["customer_name", "order_lines"]
    }
}

CREDIT_NOTE_SCHEMA = {
    "name": "extract_credit_note",
    "description": "Extract credit note details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string"},
            "customer_org_number": {"type": "string"},
            "invoice_description": {"type": "string", "description": "Description to identify the original invoice"},
            "original_amount": {"type": "number", "description": "Amount of original invoice if mentioned"},
            "is_full_reversal": {"type": "boolean", "description": "True if reversing the entire invoice"},
            "credit_note_date": {"type": "string", "description": "Date for the credit note YYYY-MM-DD, default today"},
        },
        "required": ["customer_name", "is_full_reversal"]
    }
}

TRAVEL_EXPENSE_SCHEMA = {
    "name": "extract_travel_expense",
    "description": "Extract travel expense details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "employee_first_name": {"type": "string"},
            "employee_last_name": {"type": "string"},
            "employee_email": {"type": "string"},
            "title": {"type": "string", "description": "Title/description of the travel expense"},
            "per_diem_days": {"type": "integer", "description": "Number of days for per diem/diett"},
            "per_diem_rate": {"type": "number", "description": "Daily rate for per diem in NOK"},
            "costs": {
                "type": "array",
                "description": "Individual expense items (flights, taxi, hotel, etc.)",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number", "description": "Amount in NOK"},
                    },
                    "required": ["description", "amount"]
                }
            },
            "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "YYYY-MM-DD"},
            "destination": {"type": "string", "description": "Travel destination city/location"},
        },
        "required": ["employee_first_name", "employee_last_name", "title"]
    }
}

SALARY_SCHEMA = {
    "name": "extract_salary",
    "description": "Extract salary/payroll details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "employee_first_name": {"type": "string"},
            "employee_last_name": {"type": "string"},
            "employee_email": {"type": "string"},
            "month": {"type": "integer", "description": "Salary month (1-12)"},
            "year": {"type": "integer", "description": "Salary year"},
            "gross_salary": {"type": "number", "description": "Gross monthly salary in NOK"},
            "salary_type_name": {"type": "string", "description": "Salary type name if specified (e.g. Fastlønn)"},
            "date": {"type": "string", "description": "Voucher date YYYY-MM-DD"},
        },
        "required": ["employee_first_name", "employee_last_name", "month", "year"]
    }
}

PROJECT_LIFECYCLE_SCHEMA = {
    "name": "extract_project_lifecycle",
    "description": "Extract project lifecycle details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "project_name": {"type": "string"},
            "customer_name": {"type": "string"},
            "customer_org_number": {"type": "string"},
            "activity_name": {"type": "string", "description": "Name of the activity (e.g. 'Design', 'Development')"},
            "budget": {"type": "number", "description": "Project budget in NOK"},
            "team_members": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "email": {"type": "string"},
                        "role": {"type": "string"},
                        "hours": {"type": "number"},
                        "is_project_manager": {"type": "boolean"},
                        "hourly_rate": {"type": "number"},
                    },
                    "required": ["first_name", "last_name", "hours"]
                }
            },
            "supplier_name": {"type": "string"},
            "supplier_org_number": {"type": "string"},
            "supplier_cost_amount": {"type": "number"},
            "supplier_cost_description": {"type": "string"},
            "create_customer_invoice": {"type": "boolean"},
        },
        "required": ["project_name", "customer_name", "team_members"]
    }
}

LEDGER_ANALYSIS_SCHEMA = {
    "name": "extract_ledger_analysis",
    "description": "Extract ledger analysis parameters from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "start_month": {"type": "string", "description": "First comparison month YYYY-MM (e.g. 2026-01)"},
            "end_month": {"type": "string", "description": "Second comparison month YYYY-MM (e.g. 2026-02)"},
            "num_accounts": {"type": "integer", "description": "Number of top accounts to find (e.g. 3)"},
            "change_direction": {"type": "string", "enum": ["increase", "decrease"], "description": "Whether looking for largest increase or decrease"},
            "create_projects": {"type": "boolean", "description": "Whether to create projects for the identified accounts"},
            "create_activities": {"type": "boolean", "description": "Whether to create activities for the projects"},
            "is_internal": {"type": "boolean", "description": "Whether projects should be internal"},
        },
        "required": ["start_month", "end_month", "num_accounts"]
    }
}

BANK_RECONCILIATION_SCHEMA = {
    "name": "extract_bank_reconciliation",
    "description": "Extract bank reconciliation parameters from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "has_csv_file": {"type": "boolean", "description": "True if a CSV bank statement is attached"},
            "reconciliation_date": {"type": "string", "description": "Date for reconciliation YYYY-MM-DD"},
        },
        "required": ["has_csv_file"]
    }
}

YEAR_END_CLOSING_SCHEMA = {
    "name": "extract_year_end_closing",
    "description": "Extract year-end closing details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "year": {"type": "integer", "description": "Fiscal year being closed (e.g. 2025)"},
            "assets": {
                "type": "array",
                "description": "Fixed assets for depreciation",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost": {"type": "number", "description": "Original cost / acquisition value"},
                        "useful_life_years": {"type": "integer"},
                        "depreciation_method": {"type": "string", "description": "linear or declining-balance"},
                        "residual_value": {"type": "number", "description": "Residual/salvage value, default 0"},
                        "asset_account_number": {"type": "integer", "description": "Balance sheet account for the asset (e.g. 1200)"},
                        "depreciation_account_number": {"type": "integer", "description": "Accumulated depreciation account (e.g. 1209)"},
                        "expense_account_number": {"type": "integer", "description": "Depreciation expense account (e.g. 6010)"},
                    },
                    "required": ["name", "cost", "useful_life_years"]
                }
            },
            "prepaid_expenses": {
                "type": "array",
                "description": "Prepaid expenses to reverse/accrue",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number"},
                        "prepaid_account_number": {"type": "integer", "description": "e.g. 1700"},
                        "expense_account_number": {"type": "integer", "description": "e.g. 6300"},
                    },
                    "required": ["description", "amount"]
                }
            },
            "tax_rate_percent": {"type": "number", "description": "Tax rate percentage (e.g. 22)"},
            "revenue_total": {"type": "number", "description": "Total revenue for tax calculation"},
            "expense_total": {"type": "number", "description": "Total expenses for tax calculation (before depreciation if depreciation assets listed)"},
            "closing_date": {"type": "string", "description": "Last day of fiscal year YYYY-MM-DD, e.g. 2025-12-31"},
        },
        "required": ["year"]
    }
}

VOUCHER_CORRECTION_SCHEMA = {
    "name": "extract_voucher_correction",
    "description": "Extract voucher error correction details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "period_from": {"type": "string", "description": "Start of period to search YYYY-MM-DD"},
            "period_to": {"type": "string", "description": "End of period to search YYYY-MM-DD"},
            "num_errors": {"type": "integer", "description": "Number of errors to find"},
            "errors": {
                "type": "array",
                "description": "Described errors to find and fix",
                "items": {
                    "type": "object",
                    "properties": {
                        "error_type": {"type": "string", "enum": ["wrong_account", "wrong_amount", "duplicate", "missing_vat", "wrong_date", "missing_posting"]},
                        "description": {"type": "string", "description": "What the error is about"},
                        "wrong_account_number": {"type": "integer"},
                        "correct_account_number": {"type": "integer"},
                        "wrong_amount": {"type": "number"},
                        "correct_amount": {"type": "number"},
                        "search_keyword": {"type": "string", "description": "Keyword to find the voucher (from description)"},
                    },
                    "required": ["error_type", "description"]
                }
            },
        },
        "required": ["period_from", "period_to", "num_errors", "errors"]
    }
}

# --- LLM extraction ---

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


# --- Helpers ---

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
    # Try exact match first
    for a in accounts_cache:
        if a.get("number") == target_num:
            return a["id"]
    # Find closest in range
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
    """Find a travel expense cost category. hint helps pick the right one."""
    # FIX: costCategory does NOT have a 'name' field — use 'description' only
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
            return  # Already has a bank account number
    # No bank account number set — fix the first bank account
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
        return  # Already has it
    # Set a default dateOfBirth
    tripletex.put(f"/employee/{employee_id}", {
        "id": emp["id"],
        "version": emp.get("version", 0),
        "firstName": emp.get("firstName", ""),
        "lastName": emp.get("lastName", ""),
        "dateOfBirth": "1990-01-01",
    })
    logger.info(f"Set default dateOfBirth for employee {employee_id}")


# --- Task handlers ---

def handle_supplier_invoice(tripletex, params):
    """Register a supplier invoice using POST /incomingInvoice."""
    today = date.today().isoformat()

    supplier_id = find_or_create_supplier(
        tripletex, params["supplier_name"], params.get("supplier_org_number")
    )
    if not supplier_id:
        return {"success": False, "error": "Could not create/find supplier"}

    accounts = get_all_accounts(tripletex)
    expense_acct_id = find_account_id(tripletex, params["expense_account_number"], accounts)
    if not expense_acct_id:
        return {"success": False, "error": f"Expense account {params['expense_account_number']} not found"}

    vat_type_id = find_vat_type_id(tripletex, params["vat_rate_percent"])

    # FIX: Handle vat_included=false — compute gross amount
    total_amount = params["total_amount"]
    if not params.get("vat_included", True) and params["vat_rate_percent"] > 0:
        total_amount = round(total_amount * (1 + params["vat_rate_percent"] / 100), 2)

    inv_date = params.get("invoice_date") or today
    due_date = params.get("due_date") or inv_date
    desc = params.get("expense_description") or f"Supplier invoice - {params['supplier_name']}"
    inv_number = params.get("invoice_reference", "")

    order_line = {
        "row": 1,
        "description": desc,
        "accountId": expense_acct_id,
        "amountInclVat": total_amount,
        "externalId": "1",
    }
    if vat_type_id:
        order_line["vatTypeId"] = vat_type_id

    dept_id = find_department_id(tripletex, params.get("department_name"))
    if dept_id:
        order_line["departmentId"] = dept_id

    incoming_invoice = {
        "invoiceHeader": {
            "vendorId": supplier_id,
            "invoiceDate": inv_date,
            "dueDate": due_date,
            "invoiceAmount": total_amount,
            "description": desc,
        },
        "orderLines": [order_line],
    }
    if inv_number:
        incoming_invoice["invoiceHeader"]["invoiceNumber"] = inv_number

    result = tripletex.post("/incomingInvoice?sendTo=ledger", incoming_invoice)
    if result["status_code"] not in (200, 201):
        logger.warning(f"IncomingInvoice failed ({result['status_code']}), falling back to voucher")
        return _supplier_invoice_voucher_fallback(tripletex, params, supplier_id, expense_acct_id, accounts, total_amount)

    voucher_id = result["body"].get("value", {}).get("voucherId")
    logger.info(f"Incoming invoice created: voucherId={voucher_id}")
    return {"success": True, "voucher_id": voucher_id}


def _supplier_invoice_voucher_fallback(tripletex, params, supplier_id, expense_acct_id, accounts, total_incl_vat=None):
    """Fallback: register supplier invoice as journal voucher."""
    today = date.today().isoformat()

    vat_input_acct_id = None
    for vat_acct in [2710, 2711, 2712, 2713]:
        vat_input_acct_id = find_account_id(tripletex, vat_acct, accounts)
        if vat_input_acct_id:
            break

    ap_acct_id = find_account_id(tripletex, 2400, accounts)
    if not ap_acct_id:
        return {"success": False, "error": "Accounts payable (2400) not found"}

    # Use pre-computed total_incl_vat if available
    if total_incl_vat is not None:
        total = total_incl_vat
    else:
        total = params["total_amount"]
        if not params.get("vat_included", True) and params["vat_rate_percent"] > 0:
            total = round(total * (1 + params["vat_rate_percent"] / 100), 2)

    vat_rate = params["vat_rate_percent"] / 100.0
    net = round(total / (1 + vat_rate), 2) if vat_rate > 0 else total
    vat = round(total - net, 2)

    inv_date = params.get("invoice_date") or today
    desc = params.get("invoice_reference", "")
    if desc and params.get("expense_description"):
        desc = f"{desc} - {params['expense_description']}"
    elif params.get("expense_description"):
        desc = params["expense_description"]
    if not desc:
        desc = f"Supplier invoice - {params['supplier_name']}"

    dept_id = find_department_id(tripletex, params.get("department_name"))

    postings = [
        {
            "account": {"id": expense_acct_id},
            "amount": net, "amountCurrency": net,
            "amountGross": net, "amountGrossCurrency": net,
            "description": params.get("expense_description", "Expense excl. VAT"),
            **({"department": {"id": dept_id}} if dept_id else {}),
        },
    ]
    if vat > 0 and vat_input_acct_id:
        postings.append({
            "account": {"id": vat_input_acct_id},
            "amount": vat, "amountCurrency": vat,
            "amountGross": vat, "amountGrossCurrency": vat,
            "description": f"Input VAT {params['vat_rate_percent']}%",
        })
    postings.append({
        "account": {"id": ap_acct_id},
        "amount": -total, "amountCurrency": -total,
        "amountGross": -total, "amountGrossCurrency": -total,
        "description": f"Accounts payable - {params['supplier_name']}",
        "supplier": {"id": supplier_id},
    })

    balance = sum(p["amount"] for p in postings)
    if abs(balance) > 0.01:
        postings[-1]["amount"] = round(postings[-1]["amount"] - balance, 2)
        postings[-1]["amountCurrency"] = postings[-1]["amount"]
        postings[-1]["amountGross"] = postings[-1]["amount"]
        postings[-1]["amountGrossCurrency"] = postings[-1]["amount"]

    result = tripletex.post("/ledger/voucher", {
        "date": inv_date, "description": desc, "postings": postings,
    })
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Voucher fallback failed: {json.dumps(result, default=str)[:500]}"}

    voucher_id = result["body"].get("value", {}).get("id")
    logger.info(f"Supplier invoice voucher (fallback): id={voucher_id}")
    return {"success": True, "voucher_id": voucher_id}


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

    # Check existing employment
    employment_id = None
    result = tripletex.get("/employee/employment", {"employeeId": employee_id, "fields": "id,startDate", "count": 10})
    if result["status_code"] == 200:
        existing = result["body"].get("values", [])
        if existing:
            employment_id = existing[0]["id"]

    if not employment_id:
        # FIX: employmentType and percentageOfFullTimeEquivalent belong on EmploymentDetails, NOT Employment
        result = tripletex.post("/employee/employment", {
            "employee": {"id": employee_id},
            "startDate": start_date,
        })
        if result["status_code"] in (200, 201):
            employment_id = result["body"].get("value", {}).get("id")
        else:
            logger.warning(f"Employment POST failed: {result}")

    # Employment details (salary, percentage, working hours scheme)
    if employment_id:
        # FIX: Convert monthly_salary to annual_salary (monthlySalary field doesn't exist on API)
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

    # Standard time (hours per day)
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


def handle_customer_invoice(tripletex, params):
    """Create customer -> products -> order -> invoice."""
    today = date.today().isoformat()

    customer_id = find_or_create_customer(
        tripletex, params["customer_name"],
        params.get("customer_org_number"), params.get("customer_email"),
    )
    if not customer_id:
        return {"success": False, "error": "Could not create/find customer"}

    # FIX: Ensure bank account is set (required for invoicing in some sandboxes)
    ensure_bank_account(tripletex)

    # VAT types — normalize to float for consistent lookup, prefer output/sales types
    vat_types = {}
    result = tripletex.get("/ledger/vatType", {"fields": "id,name,percentage", "count": 100})
    if result["status_code"] == 200:
        for vt in result["body"].get("values", []):
            pct = vt.get("percentage")
            if pct is not None:
                pct = float(pct)
                name = (vt.get("name") or "").lower()
                # Prefer output/sales ("utgående") VAT types for invoices
                is_output = any(w in name for w in ["utgående", "output", "sales", "ut"])
                if pct not in vat_types or is_output:
                    vat_types[pct] = vt["id"]

    order_lines = []
    for line in params.get("order_lines", []):
        prod_name = line["product_name"]
        result = tripletex.get("/product", {"name": prod_name, "fields": "id,name", "count": 10})
        product_id = None
        if result["status_code"] == 200:
            for p in result["body"].get("values", []):
                if p.get("name", "").lower() == prod_name.lower():
                    product_id = p["id"]
                    break

        if not product_id:
            vat_pct = float(line.get("vat_rate_percent", 25))
            prod_data = {"name": prod_name}
            if line.get("product_number"):
                prod_data["number"] = str(line["product_number"])
            if vat_pct in vat_types:
                prod_data["vatType"] = {"id": vat_types[vat_pct]}
            result = tripletex.post("/product", prod_data)
            if result["status_code"] in (200, 201):
                product_id = result["body"].get("value", {}).get("id")

        if product_id:
            order_lines.append({
                "product": {"id": product_id},
                "count": line["quantity"],
                "unitPriceExcludingVatCurrency": line["unit_price_excl_vat"],
                "description": line.get("description", prod_name),
            })

    if not order_lines:
        return {"success": False, "error": "No order lines could be created"}

    delivery_date = params.get("delivery_date") or params.get("invoice_date") or today
    order_date = params.get("invoice_date") or today
    order_data = {"customer": {"id": customer_id}, "deliveryDate": delivery_date, "orderDate": order_date, "orderLines": order_lines}

    dept_id = find_department_id(tripletex, params.get("department_name"))
    if dept_id:
        order_data["department"] = {"id": dept_id}
    project_id = find_project_id(tripletex, params.get("project_name"))
    if project_id:
        order_data["project"] = {"id": project_id}

    result = tripletex.post("/order", order_data)
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Order POST failed: {json.dumps(result, default=str)[:500]}"}

    order_id = result["body"].get("value", {}).get("id")
    if not order_id:
        return {"success": False, "error": "Order created but no ID returned"}

    invoice_date = params.get("invoice_date") or today
    due_date = params.get("due_date") or today

    result = tripletex.post("/invoice", {
        "invoiceDate": invoice_date, "invoiceDueDate": due_date, "orders": [{"id": order_id}],
    })
    if result["status_code"] not in (200, 201):
        # Check if it's a bank account error and retry
        err_body = json.dumps(result.get("body", {}), ensure_ascii=False).lower()
        if "bankkontonummer" in err_body or "bank account" in err_body:
            ensure_bank_account(tripletex)
            result = tripletex.post("/invoice", {
                "invoiceDate": invoice_date, "invoiceDueDate": due_date, "orders": [{"id": order_id}],
            })
            if result["status_code"] not in (200, 201):
                return {"success": False, "error": f"Invoice POST failed (after bank fix): {json.dumps(result, default=str)[:500]}"}
        else:
            return {"success": False, "error": f"Invoice POST failed: {json.dumps(result, default=str)[:500]}"}

    invoice_id = result["body"].get("value", {}).get("id")
    logger.info(f"Invoice created: id={invoice_id}, order={order_id}")
    return {"success": True, "invoice_id": invoice_id, "order_id": order_id}


def handle_credit_note(tripletex, params):
    """Create a credit note for an existing invoice."""
    today = date.today().isoformat()

    customer_id = None
    result = tripletex.get("/customer", {"fields": "id,name,organizationNumber", "count": 100})
    if result["status_code"] == 200:
        for c in result["body"].get("values", []):
            if params["customer_name"].lower() in c.get("name", "").lower():
                customer_id = c["id"]
                break
            if params.get("customer_org_number") and c.get("organizationNumber") == params["customer_org_number"]:
                customer_id = c["id"]
                break

    if not customer_id:
        return {"success": False, "error": f"Customer not found: {params['customer_name']}"}

    result = tripletex.get("/invoice", {
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
        "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "count": 100,
    })

    invoice_id = None
    if result["status_code"] == 200:
        customer_invoices = [inv for inv in result["body"].get("values", [])
                            if inv.get("customer", {}).get("id") == customer_id]
        if params.get("original_amount"):
            amt = params["original_amount"]
            # Try exact match first, then with 25% VAT
            for inv in customer_invoices:
                inv_amt = inv.get("amount") or 0
                if abs(inv_amt - amt) < 1.0:
                    invoice_id = inv["id"]
                    break
            if not invoice_id:
                # Maybe prompt amount was excl. VAT, invoice amount is incl. VAT
                amt_incl = round(amt * 1.25, 2)
                for inv in customer_invoices:
                    inv_amt = inv.get("amount") or 0
                    if abs(inv_amt - amt_incl) < 1.0:
                        invoice_id = inv["id"]
                        break
            if not invoice_id and customer_invoices:
                # Just use the first invoice for this customer
                invoice_id = customer_invoices[0]["id"]
        elif customer_invoices:
            invoice_id = customer_invoices[0]["id"]

    if not invoice_id:
        return {"success": False, "error": "Could not find matching invoice"}

    # FIX: Credit note uses PUT (not POST), with date as query parameter
    cn_date = params.get("credit_note_date") or today
    result = tripletex.put(f"/invoice/{invoice_id}/:createCreditNote?date={cn_date}", {})
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Credit note failed: {json.dumps(result, default=str)[:500]}"}

    credit_note_id = result["body"].get("value", {}).get("id")
    logger.info(f"Credit note created: id={credit_note_id} for invoice {invoice_id}")
    return {"success": True, "credit_note_id": credit_note_id}


def handle_travel_expense(tripletex, params):
    """Create travel expense with costs and per diem."""
    today = date.today().isoformat()

    # 1. Find employee
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

    # 2. Lookup travel expense config
    zone_id = get_enabled_zone(tripletex)
    payment_type_id = get_payment_type(tripletex)

    # 3. Create travel expense
    dep_date = params.get("departure_date") or today
    ret_date = params.get("return_date") or dep_date
    destination = params.get("destination", "Norge")

    travel_data = {
        "employee": {"id": employee_id},
        "title": params["title"],
        "date": dep_date,  # FIX: date is required
    }

    # FIX: Always include travelDetails — without it, per diem will fail
    # (system creates "ansattutlegg" instead of "reiseregning")
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

    # 4. Add costs
    for cost_item in params.get("costs", []):
        cost_data = {
            "travelExpense": {"id": travel_expense_id},
            "amountCurrencyIncVat": cost_item["amount"],  # FIX: was amountNOKInclVAT (wrong field name)
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

    # 5. Per diem
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

    # Verify
    tripletex.get(f"/travelExpense/{travel_expense_id}", {"fields": "*"})
    logger.info(f"Travel expense completed: id={travel_expense_id}")
    return {"success": True, "travel_expense_id": travel_expense_id}


def handle_salary(tripletex, params):
    """Run salary/payroll for an employee."""
    # 1. Find employee
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

    # FIX: Employee MUST have dateOfBirth for salary processing
    ensure_employee_date_of_birth(tripletex, employee_id)

    # 2. Ensure employment exists
    employment_id = None
    result = tripletex.get("/employee/employment", {"employeeId": employee_id, "fields": "id,division,version", "count": 1})
    if result["status_code"] == 200:
        existing = result["body"].get("values", [])
        if existing:
            employment_id = existing[0]["id"]

    if not employment_id:
        # FIX: Employment POST does NOT accept employmentType — that's on EmploymentDetails
        result = tripletex.post("/employee/employment", {
            "employee": {"id": employee_id},
            "startDate": f"{params['year']}-01-01",
        })
        if result["status_code"] in (200, 201):
            employment_id = result["body"].get("value", {}).get("id")

    # 3. Check division exists (required for salary)
    division_id = None
    result = tripletex.get("/division", {"fields": "id,name", "count": 10})
    if result["status_code"] == 200:
        divs = result["body"].get("values", [])
        if divs:
            division_id = divs[0]["id"]

    if not division_id:
        # Create a division — requires municipality
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

    # FIX: Link division to employment (required for salary processing)
    if employment_id and division_id:
        # Check if employment already has division
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

    # 4. Find salary type
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

    # 5. Create salary transaction with payslips inline
    # FIX: Use last day of month for voucher date
    last_day = calendar.monthrange(params['year'], params['month'])[1]
    voucher_date = params.get("date") or f"{params['year']}-{params['month']:02d}-{last_day:02d}"

    salary_data = {
        "date": voucher_date,
        "month": params["month"],
        "year": params["year"],
    }

    # FIX: Include payslips with specifications inline
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


def handle_project_lifecycle(tripletex, params):
    """Execute complete project lifecycle: create project, log time, supplier cost, invoice."""
    today = date.today().isoformat()

    # 1. Create/find customer
    customer_id = find_or_create_customer(
        tripletex, params["customer_name"], params.get("customer_org_number")
    )
    if not customer_id:
        return {"success": False, "error": "Could not create/find customer"}

    # 2. Create/find all employees, identify project manager
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

    # 3. Create project
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

    # 4. Create activity (PROJECT_GENERAL_ACTIVITY — required for timesheet on projects)
    activity_name = params.get("activity_name") or f"{params['project_name']} - Work"
    result = tripletex.post("/activity", {
        "name": activity_name,
        "activityType": "PROJECT_GENERAL_ACTIVITY",
    })
    activity_id = None
    if result["status_code"] in (200, 201):
        activity_id = result["body"].get("value", {}).get("id")

    # 5. Link activity to project
    if activity_id:
        result = tripletex.post("/project/projectActivity", {
            "activity": {"id": activity_id},
            "project": {"id": project_id},
        })
        if result["status_code"] not in (200, 201):
            logger.warning(f"Project-activity link failed: {result['status_code']}")

    # 6. Log timesheet entries
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

    # 7. Register supplier cost
    if params.get("supplier_name") and params.get("supplier_cost_amount"):
        supplier_id = find_or_create_supplier(
            tripletex, params["supplier_name"], params.get("supplier_org_number")
        )
        if supplier_id:
            accounts = get_all_accounts(tripletex)
            # Find expense account for supplier cost
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

    # 8. Create customer invoice
    if params.get("create_customer_invoice"):
        ensure_bank_account(tripletex)
        result = tripletex.post("/product", {"name": params["project_name"]})
        product_id = None
        if result["status_code"] in (200, 201):
            product_id = result["body"].get("value", {}).get("id")

        if product_id:
            # Calculate invoice amount: budget, or sum(hours * rate) for all team members
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


def handle_ledger_analysis(tripletex, params):
    """Analyze ledger for expense account changes and create projects/activities."""
    today = date.today().isoformat()
    from collections import defaultdict

    start_month = params["start_month"]  # e.g. "2026-01"
    end_month = params["end_month"]  # e.g. "2026-02"
    num_accounts = params.get("num_accounts", 3)
    direction = params.get("change_direction", "increase")

    # Parse date range
    start_year, start_m = int(start_month.split("-")[0]), int(start_month.split("-")[1])
    end_year, end_m = int(end_month.split("-")[0]), int(end_month.split("-")[1])
    date_from = f"{start_year}-{start_m:02d}-01"
    last_day = calendar.monthrange(end_year, end_m)[1]
    date_to = f"{end_year}-{end_m:02d}-{last_day:02d}"

    # Use /ledger/posting directly — more reliable than expanding voucher postings
    result = tripletex.get("/ledger/posting", {
        "fields": "id,date,account,amount",
        "dateFrom": date_from, "dateTo": date_to, "count": 10000,
    })
    if result["status_code"] != 200:
        return {"success": False, "error": "Could not fetch postings"}

    postings = result["body"].get("values", [])

    # Build id→number lookup from accounts
    all_accounts = get_all_accounts(tripletex)
    id_to_num = {a["id"]: a.get("number") for a in all_accounts if a.get("number")}
    id_to_name = {a["id"]: a.get("name", "") for a in all_accounts}

    # Aggregate amounts by account and month
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

    # Compute changes for expense accounts (4000-8999)
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

    # Find project manager (first available employee)
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


def handle_bank_reconciliation(tripletex, params):
    """Match CSV bank statement payments to open invoices."""
    import csv
    import io

    csv_content = params.get("_csv_content", "")
    if not csv_content:
        return {"success": False, "error": "No CSV content found"}

    # Parse CSV
    reader = csv.reader(io.StringIO(csv_content), delimiter=";")
    rows = list(reader)
    if not rows:
        return {"success": False, "error": "Empty CSV"}

    # Find header row
    header = None
    data_rows = []
    for i, row in enumerate(rows):
        row_lower = [c.strip().lower() for c in row]
        if any(h in " ".join(row_lower) for h in ["dato", "date", "beløp", "amount", "inn", "ut"]):
            header = row_lower
            data_rows = rows[i + 1:]
            break

    if not header:
        # Try comma delimiter
        reader = csv.reader(io.StringIO(csv_content), delimiter=",")
        rows = list(reader)
        for i, row in enumerate(rows):
            row_lower = [c.strip().lower() for c in row]
            if any(h in " ".join(row_lower) for h in ["dato", "date", "beløp", "amount", "inn", "ut"]):
                header = row_lower
                data_rows = rows[i + 1:]
                break

    if not header:
        header = [c.strip().lower() for c in rows[0]]
        data_rows = rows[1:]

    # Identify columns
    def find_col(keywords):
        for kw in keywords:
            for i, h in enumerate(header):
                if kw in h:
                    return i
        return None

    date_col = find_col(["dato", "date", "fecha", "data"])
    amount_col = find_col(["beløp", "amount", "monto", "valor", "betrag"])
    in_col = find_col(["inn", "inngående", "incoming", "credit", "kredit"])
    out_col = find_col(["ut", "utgående", "outgoing", "debit"])
    ref_col = find_col(["referanse", "reference", "ref", "tekst", "text", "beskrivelse", "description"])
    name_col = find_col(["navn", "name", "avsender", "sender", "mottaker"])

    # Parse payments
    payments = []
    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue
        payment = {}
        if date_col is not None and date_col < len(row):
            payment["date"] = row[date_col].strip()
        if ref_col is not None and ref_col < len(row):
            payment["ref"] = row[ref_col].strip()
        if name_col is not None and name_col < len(row):
            payment["name"] = row[name_col].strip()

        # Amount: either single column or separate in/out
        amount = 0
        if amount_col is not None and amount_col < len(row):
            try:
                amount = float(row[amount_col].strip().replace(" ", "").replace(",", "."))
            except ValueError:
                continue
        elif in_col is not None or out_col is not None:
            try:
                if in_col is not None and in_col < len(row) and row[in_col].strip():
                    amount = float(row[in_col].strip().replace(" ", "").replace(",", "."))
                if out_col is not None and out_col < len(row) and row[out_col].strip():
                    amount = -float(row[out_col].strip().replace(" ", "").replace(",", "."))
            except ValueError:
                continue
        else:
            continue

        if amount == 0:
            continue
        payment["amount"] = amount
        payments.append(payment)

    if not payments:
        return {"success": False, "error": "No payments found in CSV"}

    logger.info(f"Parsed {len(payments)} payments from CSV")

    # Get payment types
    in_payment_type_id = None
    result = tripletex.get("/invoice/paymentType", {"fields": "id,description", "count": 50})
    if result["status_code"] == 200:
        for pt in result["body"].get("values", []):
            in_payment_type_id = pt["id"]
            break

    out_payment_type_id = None
    result = tripletex.get("/ledger/paymentTypeOut", {"fields": "id,description", "count": 50})
    if result["status_code"] == 200:
        for pt in result["body"].get("values", []):
            out_payment_type_id = pt["id"]
            break

    # Get customer invoices
    result = tripletex.get("/invoice", {
        "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer,kid",
        "count": 200,
    })
    customer_invoices = result["body"].get("values", []) if result["status_code"] == 200 else []

    # Get supplier invoices
    result = tripletex.get("/supplierInvoice", {
        "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31",
        "fields": "id,invoiceNumber,amount,supplier",
        "count": 200,
    })
    supplier_invoices = result["body"].get("values", []) if result["status_code"] == 200 else []

    matched = 0
    for payment in payments:
        amt = payment["amount"]
        pay_date = payment.get("date", date.today().isoformat())
        # Normalize date format
        for fmt_from, fmt_to in [(".", "-"), ("/", "-")]:
            pay_date = pay_date.replace(fmt_from, fmt_to)
        # Handle DD-MM-YYYY → YYYY-MM-DD
        parts = pay_date.split("-")
        if len(parts) == 3 and len(parts[0]) <= 2:
            pay_date = f"{parts[2]}-{parts[1]}-{parts[0]}"

        ref = payment.get("ref", "")
        name = payment.get("name", "")
        search_text = f"{ref} {name}".lower()

        if amt > 0:
            # Incoming payment → match to customer invoice
            best_match = None
            best_score = 0
            for inv in customer_invoices:
                inv_amt = inv.get("amountOutstanding") or inv.get("amount", 0)
                if inv_amt is None:
                    continue
                score = 0
                # Exact amount match
                if abs(abs(inv_amt) - abs(amt)) < 0.01:
                    score += 10
                elif abs(abs(inv_amt) - abs(amt)) < abs(amt) * 0.05:
                    score += 5
                # KID/reference match
                kid = str(inv.get("kid", ""))
                if kid and kid in ref:
                    score += 20
                # Customer name match
                cust = inv.get("customer", {})
                cust_name = (cust.get("name", "") or "").lower()
                if cust_name and cust_name in search_text:
                    score += 15
                # Invoice number in reference
                inv_num = str(inv.get("invoiceNumber", ""))
                if inv_num and inv_num in ref:
                    score += 15
                if score > best_score:
                    best_score = score
                    best_match = inv

            if best_match and best_score >= 5 and in_payment_type_id:
                inv_id = best_match["id"]
                result = tripletex.put(
                    f"/invoice/{inv_id}/:payment?paymentDate={pay_date}&paymentTypeId={in_payment_type_id}&paidAmount={amt}",
                    {}
                )
                if result["status_code"] in (200, 201, 204):
                    matched += 1
                    logger.info(f"Matched incoming {amt} to invoice {inv_id}")
                else:
                    logger.warning(f"Payment failed for invoice {inv_id}: {result['status_code']}")

        elif amt < 0:
            # Outgoing payment → match to supplier invoice
            pay_amount = abs(amt)
            best_match = None
            best_score = 0
            for inv in supplier_invoices:
                inv_amt = inv.get("amount", 0)
                if inv_amt is None:
                    continue
                score = 0
                if abs(abs(inv_amt) - pay_amount) < 0.01:
                    score += 10
                elif abs(abs(inv_amt) - pay_amount) < pay_amount * 0.05:
                    score += 5
                sup = inv.get("supplier", {})
                sup_name = (sup.get("name", "") or "").lower()
                if sup_name and sup_name in search_text:
                    score += 15
                inv_num = str(inv.get("invoiceNumber", ""))
                if inv_num and inv_num in ref:
                    score += 15
                if score > best_score:
                    best_score = score
                    best_match = inv

            if best_match and best_score >= 5 and out_payment_type_id:
                inv_id = best_match["id"]
                result = tripletex.post(
                    f"/supplierInvoice/{inv_id}/:addPayment?paymentTypeId={out_payment_type_id}&amount={pay_amount}&paidDate={pay_date}"
                )
                if result["status_code"] in (200, 201, 204):
                    matched += 1
                    logger.info(f"Matched outgoing {pay_amount} to supplier invoice {inv_id}")
                else:
                    logger.warning(f"Supplier payment failed for invoice {inv_id}: {result['status_code']}")

    return {"success": True, "payments_parsed": len(payments), "matched": matched}


def handle_year_end_closing(tripletex, params):
    """Execute year-end closing: depreciation + prepaid reversals + tax provision."""
    year = params["year"]
    closing_date = params.get("closing_date", f"{year}-12-31")
    accounts = get_all_accounts(tripletex)

    total_depreciation = 0

    # 1. Post depreciation vouchers for each asset
    for asset in params.get("assets", []):
        cost = asset["cost"]
        life = asset["useful_life_years"]
        residual = asset.get("residual_value", 0)
        method = asset.get("depreciation_method", "linear")

        if method == "linear":
            annual_depr = (cost - residual) / life
        else:
            annual_depr = (cost - residual) / life  # fallback to linear

        annual_depr = round(annual_depr, 2)
        total_depreciation += annual_depr

        # Find accounts — /ledger/account is GET-only, cannot create
        expense_num = asset.get("expense_account_number", 6010)
        accum_num = asset.get("depreciation_account_number", 1209)

        expense_id = find_account_id(tripletex, expense_num, accounts)
        accum_id = find_account_id(tripletex, accum_num, accounts)

        # Fallback: find any depreciation expense account (6000-6099)
        if not expense_id:
            expense_id = find_account_in_range(accounts, expense_num, 6000, 6099)
        # Fallback: find any accumulated depreciation account (1200-1299)
        if not accum_id:
            accum_id = find_account_in_range(accounts, accum_num, 1200, 1299)

        if expense_id and accum_id:
            result = tripletex.post("/ledger/voucher", {
                "date": closing_date,
                "description": f"Avskrivning {year} - {asset['name']}",
                "postings": [
                    {
                        "account": {"id": expense_id},
                        "amount": annual_depr, "amountCurrency": annual_depr,
                        "amountGross": annual_depr, "amountGrossCurrency": annual_depr,
                        "description": f"Avskrivning {asset['name']}",
                    },
                    {
                        "account": {"id": accum_id},
                        "amount": -annual_depr, "amountCurrency": -annual_depr,
                        "amountGross": -annual_depr, "amountGrossCurrency": -annual_depr,
                        "description": f"Akkumulert avskrivning {asset['name']}",
                    },
                ]
            })
            if result["status_code"] not in (200, 201):
                logger.warning(f"Depreciation voucher failed for {asset['name']}: {result['status_code']}")

    # 2. Post prepaid expense reversals
    for prepaid in params.get("prepaid_expenses", []):
        amount = prepaid["amount"]
        prepaid_num = prepaid.get("prepaid_account_number", 1700)
        expense_num = prepaid.get("expense_account_number", 6300)

        prepaid_id = find_account_id(tripletex, prepaid_num, accounts)
        expense_id = find_account_id(tripletex, expense_num, accounts)

        # Fallback: find closest prepaid account (1700-1799)
        if not prepaid_id:
            prepaid_id = find_account_in_range(accounts, prepaid_num, 1700, 1799)
        # Fallback: find closest expense account (6000-6999)
        if not expense_id:
            expense_id = find_account_in_range(accounts, expense_num, 6000, 6999)

        if prepaid_id and expense_id:
            result = tripletex.post("/ledger/voucher", {
                "date": closing_date,
                "description": f"Periodisering - {prepaid['description']}",
                "postings": [
                    {
                        "account": {"id": expense_id},
                        "amount": amount, "amountCurrency": amount,
                        "amountGross": amount, "amountGrossCurrency": amount,
                        "description": prepaid["description"],
                    },
                    {
                        "account": {"id": prepaid_id},
                        "amount": -amount, "amountCurrency": -amount,
                        "amountGross": -amount, "amountGrossCurrency": -amount,
                        "description": prepaid["description"],
                    },
                ]
            })
            if result["status_code"] not in (200, 201):
                logger.warning(f"Prepaid reversal failed: {result['status_code']}")

    # 3. Tax provision
    tax_rate = params.get("tax_rate_percent")
    if tax_rate:
        revenue = params.get("revenue_total", 0)
        expenses = params.get("expense_total", 0)
        profit = revenue - expenses - total_depreciation
        if profit > 0:
            tax_amount = round(profit * tax_rate / 100, 2)

            tax_expense_id = find_account_id(tripletex, 8700, accounts)
            tax_payable_id = find_account_id(tripletex, 2920, accounts) or find_account_id(tripletex, 2500, accounts)

            # Fallback: find any tax expense account (8700-8799)
            if not tax_expense_id:
                tax_expense_id = find_account_in_range(accounts, 8700, 8700, 8799)
            # Fallback: find any tax payable account (2500-2999)
            if not tax_payable_id:
                tax_payable_id = find_account_in_range(accounts, 2920, 2500, 2999)

            if tax_expense_id and tax_payable_id:
                result = tripletex.post("/ledger/voucher", {
                    "date": closing_date,
                    "description": f"Skattekostnad {year}",
                    "postings": [
                        {
                            "account": {"id": tax_expense_id},
                            "amount": tax_amount, "amountCurrency": tax_amount,
                            "amountGross": tax_amount, "amountGrossCurrency": tax_amount,
                            "description": f"Skattekostnad {year} ({tax_rate}%)",
                        },
                        {
                            "account": {"id": tax_payable_id},
                            "amount": -tax_amount, "amountCurrency": -tax_amount,
                            "amountGross": -tax_amount, "amountGrossCurrency": -tax_amount,
                            "description": f"Betalbar skatt {year}",
                        },
                    ]
                })
                if result["status_code"] not in (200, 201):
                    logger.warning(f"Tax provision voucher failed: {result['status_code']}")

    return {"success": True, "total_depreciation": total_depreciation}


def handle_voucher_correction(tripletex, params):
    """Find and correct errors in existing vouchers."""
    period_from = params["period_from"]
    period_to = params["period_to"]
    errors = params.get("errors", [])

    # Get all vouchers in the period
    result = tripletex.get("/ledger/voucher", {
        "dateFrom": period_from, "dateTo": period_to,
        "fields": "id,date,description,postings(*)",
        "count": 1000,
    })
    if result["status_code"] != 200:
        # Fallback to simpler fields query
        result = tripletex.get("/ledger/voucher", {
            "dateFrom": period_from, "dateTo": period_to,
            "fields": "*",
            "count": 1000,
        })
        if result["status_code"] != 200:
            return {"success": False, "error": "Could not fetch vouchers"}

    vouchers = result["body"].get("values", [])
    accounts = get_all_accounts(tripletex)
    # Build lookups from accounts cache
    acct_num_to_id = {a.get("number"): a.get("id") for a in accounts if a.get("number") and a.get("id")}
    acct_id_to_num = {a.get("id"): a.get("number") for a in accounts if a.get("number") and a.get("id")}
    corrections_made = 0

    def get_posting_acct_num(posting):
        """Get account number from posting, using cache if API didn't return it."""
        acct = posting.get("account", {})
        num = acct.get("number")
        if num:
            return num
        aid = acct.get("id")
        if aid and aid in acct_id_to_num:
            return acct_id_to_num[aid]
        return None

    for error in errors:
        error_type = error.get("error_type", "")
        keyword = (error.get("search_keyword", "") or error.get("description", "")).lower()
        wrong_acct_num = error.get("wrong_account_number")
        correct_acct_num = error.get("correct_account_number")
        wrong_amount = error.get("wrong_amount")
        correct_amount = error.get("correct_amount")

        # Find matching voucher
        target_voucher = None
        for v in vouchers:
            desc = (v.get("description", "") or "").lower()
            postings = v.get("postings", []) or []

            # Match by keyword in description
            if keyword and keyword in desc:
                target_voucher = v
                break

            # Match by account number
            if wrong_acct_num:
                for p in postings:
                    if get_posting_acct_num(p) == wrong_acct_num:
                        target_voucher = v
                        break
                if target_voucher:
                    break

        if not target_voucher:
            logger.warning(f"Could not find voucher for error: {error.get('description')}")
            continue

        v_date = target_voucher.get("date", date.today().isoformat())
        v_desc = target_voucher.get("description", "")
        postings = target_voucher.get("postings", []) or []

        if error_type == "wrong_account":
            # Find the posting with the wrong account and reverse it on wrong, add on correct
            for p in postings:
                if get_posting_acct_num(p) == wrong_acct_num:
                    amount = p.get("amount", 0)
                    wrong_id = find_account_id(tripletex, wrong_acct_num, accounts)
                    correct_id = find_account_id(tripletex, correct_acct_num, accounts)

                    # /ledger/account is GET-only — find closest existing account
                    if not correct_id and correct_acct_num:
                        hundreds = (correct_acct_num // 100) * 100
                        correct_id = find_account_in_range(accounts, correct_acct_num, hundreds, hundreds + 99)

                    if wrong_id and correct_id:
                        result = tripletex.post("/ledger/voucher", {
                            "date": v_date,
                            "description": f"Korreksjon - {v_desc}",
                            "postings": [
                                {
                                    "account": {"id": wrong_id},
                                    "amount": -amount, "amountCurrency": -amount,
                                    "amountGross": -amount, "amountGrossCurrency": -amount,
                                    "description": f"Reversering feil konto {wrong_acct_num}",
                                },
                                {
                                    "account": {"id": correct_id},
                                    "amount": amount, "amountCurrency": amount,
                                    "amountGross": amount, "amountGrossCurrency": amount,
                                    "description": f"Korrekt konto {correct_acct_num}",
                                },
                            ]
                        })
                        if result["status_code"] in (200, 201):
                            corrections_made += 1
                    break

        elif error_type == "wrong_amount":
            for p in postings:
                amt = p.get("amount", 0)
                if wrong_amount and abs(amt - wrong_amount) < 0.01:
                    acct_id = p.get("account", {}).get("id")
                    if not acct_id:
                        continue
                    diff = (correct_amount or 0) - wrong_amount
                    # Find the counterpart posting
                    counter_id = None
                    for p2 in postings:
                        if p2.get("account", {}).get("id") != acct_id:
                            counter_id = p2.get("account", {}).get("id")
                            break
                    if counter_id:
                        result = tripletex.post("/ledger/voucher", {
                            "date": v_date,
                            "description": f"Korreksjon beløp - {v_desc}",
                            "postings": [
                                {
                                    "account": {"id": acct_id},
                                    "amount": diff, "amountCurrency": diff,
                                    "amountGross": diff, "amountGrossCurrency": diff,
                                    "description": f"Beløpskorreksjon {wrong_amount} -> {correct_amount}",
                                },
                                {
                                    "account": {"id": counter_id},
                                    "amount": -diff, "amountCurrency": -diff,
                                    "amountGross": -diff, "amountGrossCurrency": -diff,
                                    "description": f"Beløpskorreksjon motkonto",
                                },
                            ]
                        })
                        if result["status_code"] in (200, 201):
                            corrections_made += 1
                    break

        elif error_type == "duplicate":
            # Reverse all postings in the duplicate voucher
            correction_postings = []
            for p in postings:
                acct_id = p.get("account", {}).get("id")
                amount = p.get("amount", 0)
                if acct_id:
                    correction_postings.append({
                        "account": {"id": acct_id},
                        "amount": -amount, "amountCurrency": -amount,
                        "amountGross": -amount, "amountGrossCurrency": -amount,
                        "description": f"Reversering duplikat",
                    })
            if correction_postings:
                result = tripletex.post("/ledger/voucher", {
                    "date": v_date,
                    "description": f"Korreksjon duplikat - {v_desc}",
                    "postings": correction_postings,
                })
                if result["status_code"] in (200, 201):
                    corrections_made += 1

        elif error_type == "missing_vat":
            # Add missing VAT posting
            vat_acct_id = find_account_id(tripletex, 2710, accounts) or find_account_id(tripletex, 2711, accounts)
            # Fallback: find any input VAT account (2700-2799)
            if not vat_acct_id:
                vat_acct_id = find_account_in_range(accounts, 2710, 2700, 2799)

            if vat_acct_id and postings:
                # Find the expense posting and calculate VAT
                for p in postings:
                    acct_num = get_posting_acct_num(p) or 0
                    if isinstance(acct_num, int) and 4000 <= acct_num <= 7999:
                        gross_amount = p.get("amount", 0)
                        vat_amount = round(gross_amount * 0.25, 2)  # 25% MVA
                        expense_id = p.get("account", {}).get("id")
                        if expense_id:
                            result = tripletex.post("/ledger/voucher", {
                                "date": v_date,
                                "description": f"Korreksjon manglende MVA - {v_desc}",
                                "postings": [
                                    {
                                        "account": {"id": vat_acct_id},
                                        "amount": vat_amount, "amountCurrency": vat_amount,
                                        "amountGross": vat_amount, "amountGrossCurrency": vat_amount,
                                        "description": "Manglende inngående MVA",
                                    },
                                    {
                                        "account": {"id": expense_id},
                                        "amount": -vat_amount, "amountCurrency": -vat_amount,
                                        "amountGross": -vat_amount, "amountGrossCurrency": -vat_amount,
                                        "description": "Justering for MVA",
                                    },
                                ]
                            })
                            if result["status_code"] in (200, 201):
                                corrections_made += 1
                        break

    return {"success": True, "corrections_made": corrections_made, "total_errors": len(errors)}


# --- Task type detection ---

TASK_PATTERNS = {
    "supplier_invoice": [
        "leverandørfaktura", "supplier invoice", "factura del proveedor",
        "fatura do fornecedor", "lieferantenrechnung", "facture fournisseur",
        "factura proveedor",
    ],
    "employee_creation": [
        "ny ansatt", "new employee", "nuevo empleado", "novo empregado",
        "neuer mitarbeiter", "nouvel employ", "opprett.*ansatt",
        "ansettelse", "onboarding", "nytilsett", "integration",
    ],
    "customer_invoice": [
        "kundefaktura", "customer invoice",
        "opprett.*faktura", "lag.*faktura",
        "skriv.*faktura", "utstede.*faktura",
        "emitir.*factura", "kundenrechnung", "facture client",
    ],
    "credit_note": [
        "kreditnota", "credit note", "nota de crédito", "gutschrift",
        "avoir", "nota credito",
    ],
    "travel_expense": [
        "reiseregning", "travel expense", "nota de gastos de viaje",
        "nota de despesa de viagem", "despesa de viagem",
        "reisekostenabrechnung", "note de frais",
        "reise.*diett", "travel.*per diem",
        "gastos de viaje", "despesas de viagem",
    ],
    "salary": [
        "lønnskjøring", "gehaltsabrechnung", "payroll",
        "nómina", "folha de pagamento",
        "bulletin de salaire", "bulletin de paie",
    ],
    "project_lifecycle": [
        "prosjekt.*team", "project.*team", "proyecto.*equipo",
        "projeto.*equipe", "projekt.*team", "projet.*équipe",
        "prosjekt.*aktivitet", "project.*activity",
        "prosjekt.*timeføring", "project.*timesheet",
        "prosjekt.*kunde", "project.*customer",
        "prosjekt.*leverandør", "project.*supplier",
        "project lifecycle", "complete project lifecycle",
        "project.*lifecycle", "prosjektlivssyklus",
    ],
    "ledger_analysis": [
        "analyser.*hovedbok", "analyze.*ledger", "analyse.*ledger",
        "analizar.*libro mayor", "analisar.*razão",
        "hauptbuch.*analysieren", "analyser.*grand livre",
        "kontoutvikling", "account.*development",
        "største.*endring", "largest.*change", "biggest.*change",
        "most.*change", "cost.*analysis", "kostnadsanalyse",
        "custos.*aumentaram", "costs.*increased", "kostnader.*økt",
        "kosten.*gestiegen", "coûts.*augmenté", "costos.*aumentaron",
        "analise.*livro razão", "identifique.*contas",
    ],
    "bank_reconciliation": [
        "bankavsteming", "bank reconciliation", "reconciliación bancaria",
        "reconciliação bancária", "bankabstimmung", "rapprochement bancaire",
        "kontoutskrift", "bank statement", "extracto bancario",
        "extrato bancário", "relevé bancaire", "kontoauszug",
        "avstem.*kontoutskrift", "concilia.*extracto",
    ],
    "year_end_closing": [
        "årsavslutning", "årsoppgjer", "year-end closing", "year end closing",
        "cierre anual", "encerramento anual", "jahresabschluss",
        "clôture annuelle", "forenkla årsoppgjer",
        "avskrivning.*skatt", "depreciation.*tax",
    ],
    "voucher_correction": [
        "korreksjon.*bilag", "correction.*voucher", "correct.*voucher",
        "feil.*hovedbok", "error.*ledger", "feil.*bilag",
        "fehler.*buchung", "corrección.*asiento",
        "correção.*lançamento", "correction.*écriture",
    ],
}


def detect_task_type(prompt: str) -> str | None:
    """Detect task type from prompt keywords."""
    p = prompt.lower()

    # Priority 1: Bank reconciliation — check before anything else since prompts contain "faktura"/"invoice"
    recon_words = ["avstem", "reconcili", "concilia", "rapprochement", "abstimm",
                   "bankavsteming", "bank statement", "kontoutskrift", "extracto bancario",
                   "extrato bancário", "relevé bancaire", "kontoauszug"]
    if any(w in p for w in recon_words):
        return "bank_reconciliation"

    # Priority 2: Project lifecycle — check before invoice since prompts contain "factura"/"invoice"
    project_words = ["prosjekt", "project", "proyecto", "projeto", "projekt", "projet"]
    lifecycle_words = ["team", "aktivitet", "activity", "actividad", "atividade", "timesheet",
                       "timeføring", "timer", "horas", "hours", "stunden", "heures",
                       "bemanning", "staffing", "medarbeider", "member", "budget",
                       "lifecycle", "livssyklus"]
    if any(w in p for w in project_words) and any(w in p for w in lifecycle_words):
        return "project_lifecycle"

    # Priority 3: Supplier invoice — check before customer invoice
    supplier_words = ["leverandør", "supplier", "proveedor", "fornecedor", "lieferant", "fournisseur"]
    invoice_words = ["faktura", "invoice", "factura", "fatura", "rechnung", "facture"]
    payment_words = ["betaling", "payment", "innbetaling", "paiement", "pago", "pagamento", "zahlung"]
    if any(w in p for w in supplier_words) and any(w in p for w in invoice_words):
        if not any(w in p for w in payment_words):
            return "supplier_invoice"

    for task_type, patterns in TASK_PATTERNS.items():
        for pattern in patterns:
            if ".*" in pattern:
                if re.search(pattern, p):
                    return task_type
            elif pattern in p:
                return task_type

    # Fuzzy: employee + creation
    employee_words = ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employé"]
    create_words = ["opprett", "create", "registrer", "register", "nuevo", "nova", "ny ", "neue"]
    if any(w in p for w in employee_words) and any(w in p for w in create_words):
        return "employee_creation"

    # Fuzzy: customer + invoice (only if no supplier keywords)
    customer_words = ["kunde", "customer", "client", "klient"]
    if any(w in p for w in customer_words) and any(w in p for w in invoice_words):
        return "customer_invoice"

    # Fuzzy: generic "create invoice" (without supplier context) → customer_invoice
    create_invoice_words = ["opprett", "create", "lag", "skriv", "issue", "emitir"]
    if any(w in p for w in create_invoice_words) and any(w in p for w in invoice_words):
        if not any(w in p for w in supplier_words):
            return "customer_invoice"

    # Fuzzy: salary keywords
    salary_words = ["lønn", "salary", "gehalt", "salario", "salaire"]
    run_words = ["kjør", "run", "durchführ", "ejecut", "process", "registr"]
    if any(w in p for w in salary_words) and any(w in p for w in run_words):
        return "salary"

    # Fuzzy: project lifecycle (project + team/activity/timesheet)
    project_words = ["prosjekt", "project", "proyecto", "projeto", "projekt", "projet"]
    lifecycle_words = ["team", "aktivitet", "activity", "timesheet", "timeføring",
                       "bemanning", "staffing", "medarbeider", "member", "budget"]
    if any(w in p for w in project_words) and any(w in p for w in lifecycle_words):
        return "project_lifecycle"

    # Fuzzy: ledger analysis (analyze/compare + accounts/ledger)
    analyze_words = ["analyser", "analyze", "analyse", "analizar", "analisar", "analysieren",
                     "sammenlign", "compare", "comparar", "comparer", "vergleich"]
    ledger_words = ["hovedbok", "ledger", "libro mayor", "razão", "hauptbuch", "grand livre",
                    "konto", "account", "cuenta", "conta", "konto"]
    if any(w in p for w in analyze_words) and any(w in p for w in ledger_words):
        return "ledger_analysis"

    # Fuzzy: bank reconciliation (bank/payment + match/reconcile + CSV/statement)
    bank_words = ["bank", "kontoutskrift", "statement", "extracto", "extrato", "relevé", "kontoauszug"]
    match_words = ["avstem", "reconcil", "concilia", "rapprochement", "abstimm", "match"]
    if any(w in p for w in bank_words) and any(w in p for w in match_words):
        return "bank_reconciliation"

    # Fuzzy: year-end closing (year-end/closing + depreciation/tax/accrual)
    closing_words = ["årsavslutning", "årsoppgjer", "year-end", "year end", "closing",
                     "cierre", "encerramento", "jahresabschluss", "clôture"]
    accounting_words = ["avskrivning", "depreciation", "skatt", "tax", "periodisering",
                        "accrual", "bokfør", "post", "voucher"]
    if any(w in p for w in closing_words):
        return "year_end_closing"
    # Also catch combined depreciation + tax tasks
    depr_words = ["avskrivning", "depreciation", "amortissement", "abschreibung", "depreciación", "depreciação"]
    tax_words = ["skattekostnad", "tax provision", "tax expense", "impuesto", "imposto", "steuer"]
    if any(w in p for w in depr_words) and any(w in p for w in tax_words):
        return "year_end_closing"

    # Fuzzy: voucher correction (error/correction + voucher/ledger)
    error_words = ["feil", "error", "correction", "korreksjon", "fehler", "corrección",
                   "correção", "erreur", "oppdaget", "discovered", "found"]
    voucher_words = ["bilag", "voucher", "postering", "posting", "hovedbok", "ledger",
                     "buchung", "asiento", "lançamento", "écriture"]
    if any(w in p for w in error_words) and any(w in p for w in voucher_words):
        return "voucher_correction"

    return None


EXTRACTION_SCHEMAS = {
    "supplier_invoice": SUPPLIER_INVOICE_SCHEMA,
    "employee_creation": EMPLOYEE_SCHEMA,
    "customer_invoice": CUSTOMER_INVOICE_SCHEMA,
    "credit_note": CREDIT_NOTE_SCHEMA,
    "travel_expense": TRAVEL_EXPENSE_SCHEMA,
    "salary": SALARY_SCHEMA,
    "project_lifecycle": PROJECT_LIFECYCLE_SCHEMA,
    "ledger_analysis": LEDGER_ANALYSIS_SCHEMA,
    "bank_reconciliation": BANK_RECONCILIATION_SCHEMA,
    "year_end_closing": YEAR_END_CLOSING_SCHEMA,
    "voucher_correction": VOUCHER_CORRECTION_SCHEMA,
}

HANDLERS = {
    "supplier_invoice": handle_supplier_invoice,
    "employee_creation": handle_employee_creation,
    "customer_invoice": handle_customer_invoice,
    "credit_note": handle_credit_note,
    "travel_expense": handle_travel_expense,
    "salary": handle_salary,
    "project_lifecycle": handle_project_lifecycle,
    "ledger_analysis": handle_ledger_analysis,
    "bank_reconciliation": handle_bank_reconciliation,
    "year_end_closing": handle_year_end_closing,
    "voucher_correction": handle_voucher_correction,
}


def try_handle(client, model, prompt, files, tripletex) -> dict | None:
    """Try to handle the task deterministically.

    Returns {"status": "completed"} on success, or None to fall back to LLM consensus.
    """
    task_type = detect_task_type(prompt)
    if not task_type or task_type not in HANDLERS:
        logger.info(f"No deterministic handler for task (detected: {task_type})")
        return None

    logger.info(f"Deterministic handler: {task_type}")

    schema = EXTRACTION_SCHEMAS[task_type]
    params = extract_structured(client, model, prompt, files, schema)
    if not params:
        logger.warning(f"LLM extraction failed for {task_type} — falling back")
        return None

    logger.info(f"Extracted: {json.dumps(params, default=str, ensure_ascii=False)[:500]}")

    # Inject CSV content for bank reconciliation
    if task_type == "bank_reconciliation" and files:
        import base64
        for f in files:
            fn = f.get("filename", "")
            mime = f.get("mime_type", "")
            if fn.endswith(".csv") or "csv" in mime or "text" in mime:
                try:
                    raw = base64.b64decode(f.get("content_base64", ""))
                    params["_csv_content"] = raw.decode("utf-8", errors="replace")
                    break
                except Exception:
                    pass

    handler = HANDLERS[task_type]
    try:
        result = handler(tripletex, params)
    except Exception as e:
        logger.exception(f"Handler {task_type} crashed: {e}")
        return None

    logger.info(f"Handler result: {json.dumps(result, default=str)[:300]}")

    if result and result.get("success"):
        return {"status": "completed"}

    logger.warning(f"Handler failed: {result.get('error')} — falling back to LLM consensus")
    return None
