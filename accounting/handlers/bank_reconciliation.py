"""Bank reconciliation handler — match CSV bank statement to invoices."""

import csv
import io
import logging
from datetime import date

logger = logging.getLogger(__name__)


def handle_bank_reconciliation(tripletex, params):
    """Match CSV bank statement payments to open invoices."""
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

    # Get supplier invoices (no amountOutstanding field on SupplierInvoiceDTO)
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
        for fmt_from, fmt_to in [(".", "-"), ("/", "-")]:
            pay_date = pay_date.replace(fmt_from, fmt_to)
        parts = pay_date.split("-")
        if len(parts) == 3 and len(parts[0]) <= 2:
            pay_date = f"{parts[2]}-{parts[1]}-{parts[0]}"

        ref = payment.get("ref", "")
        name = payment.get("name", "")
        search_text = f"{ref} {name}".lower()

        if amt > 0:
            best_match = None
            best_score = 0
            for inv in customer_invoices:
                inv_amt = inv.get("amountOutstanding") or inv.get("amount", 0)
                if inv_amt is None:
                    continue
                score = 0
                if abs(abs(inv_amt) - abs(amt)) < 0.01:
                    score += 10
                elif abs(abs(inv_amt) - abs(amt)) < abs(amt) * 0.05:
                    score += 5
                kid = str(inv.get("kid", ""))
                if kid and kid in ref:
                    score += 20
                cust = inv.get("customer", {})
                cust_name = (cust.get("name", "") or "").lower()
                if cust_name and cust_name in search_text:
                    score += 15
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
