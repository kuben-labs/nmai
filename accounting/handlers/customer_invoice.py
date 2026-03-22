"""Customer invoice handler — customer -> products -> order -> invoice."""

import json
import logging
from datetime import date

from ._helpers import (
    find_or_create_customer, ensure_bank_account,
    find_department_id, find_project_id,
)

logger = logging.getLogger(__name__)


def handle_customer_invoice(tripletex, params):
    """Create customer -> products -> order -> invoice."""
    today = date.today().isoformat()

    customer_id = find_or_create_customer(
        tripletex, params["customer_name"],
        params.get("customer_org_number"), params.get("customer_email"),
    )
    if not customer_id:
        return {"success": False, "error": "Could not create/find customer"}

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

    # Send/issue the invoice if requested
    if params.get("send_invoice") and invoice_id:
        send_result = tripletex.put(
            f"/invoice/{invoice_id}/:send?sendType=EMAIL&overrideEmailAddress={params.get('customer_email', '')}",
            {}
        )
        if send_result["status_code"] not in (200, 201, 204):
            # Try alternative: just mark as sent
            send_result = tripletex.put(
                f"/invoice/{invoice_id}/:send?sendType=EFAKTURA",
                {}
            )
            if send_result["status_code"] not in (200, 201, 204):
                logger.warning(f"Invoice send failed: {send_result['status_code']}")

    return {"success": True, "invoice_id": invoice_id, "order_id": order_id}
