"""Supplier invoice handler — POST /incomingInvoice with voucher fallback."""

import json
import logging
from datetime import date

from ._helpers import (
    find_or_create_supplier, get_all_accounts, find_account_id,
    find_vat_type_id, find_department_id,
)

logger = logging.getLogger(__name__)


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
        return _voucher_fallback(tripletex, params, supplier_id, expense_acct_id, accounts, total_amount)

    voucher_id = result["body"].get("value", {}).get("voucherId")
    logger.info(f"Incoming invoice created: voucherId={voucher_id}")
    return {"success": True, "voucher_id": voucher_id}


def _voucher_fallback(tripletex, params, supplier_id, expense_acct_id, accounts, total_incl_vat=None):
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
