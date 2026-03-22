"""Credit note handler."""

import json
import logging
from datetime import date

logger = logging.getLogger(__name__)


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
            for inv in customer_invoices:
                inv_amt = inv.get("amount") or 0
                if abs(inv_amt - amt) < 1.0:
                    invoice_id = inv["id"]
                    break
            if not invoice_id:
                amt_incl = round(amt * 1.25, 2)
                for inv in customer_invoices:
                    inv_amt = inv.get("amount") or 0
                    if abs(inv_amt - amt_incl) < 1.0:
                        invoice_id = inv["id"]
                        break
            if not invoice_id and customer_invoices:
                invoice_id = customer_invoices[0]["id"]
        elif customer_invoices:
            invoice_id = customer_invoices[0]["id"]

    if not invoice_id:
        return {"success": False, "error": "Could not find matching invoice"}

    cn_date = params.get("credit_note_date") or today
    result = tripletex.put(f"/invoice/{invoice_id}/:createCreditNote?date={cn_date}", {})
    if result["status_code"] not in (200, 201):
        return {"success": False, "error": f"Credit note failed: {json.dumps(result, default=str)[:500]}"}

    credit_note_id = result["body"].get("value", {}).get("id")
    logger.info(f"Credit note created: id={credit_note_id} for invoice {invoice_id}")
    return {"success": True, "credit_note_id": credit_note_id}
