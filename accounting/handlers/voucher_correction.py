"""Voucher correction handler — find and fix errors in existing vouchers."""

import logging
from datetime import date

from ._helpers import get_all_accounts, find_account_id, find_account_in_range

logger = logging.getLogger(__name__)


def handle_voucher_correction(tripletex, params):
    """Find and correct errors in existing vouchers."""
    period_from = params["period_from"]
    period_to = params["period_to"]
    errors = params.get("errors", [])

    result = tripletex.get("/ledger/voucher", {
        "dateFrom": period_from, "dateTo": period_to,
        "fields": "id,date,description,postings(*)",
        "count": 1000,
    })
    if result["status_code"] != 200:
        result = tripletex.get("/ledger/voucher", {
            "dateFrom": period_from, "dateTo": period_to,
            "fields": "*",
            "count": 1000,
        })
        if result["status_code"] != 200:
            return {"success": False, "error": "Could not fetch vouchers"}

    vouchers = result["body"].get("values", [])
    accounts = get_all_accounts(tripletex)
    acct_num_to_id = {a.get("number"): a.get("id") for a in accounts if a.get("number") and a.get("id")}
    acct_id_to_num = {a.get("id"): a.get("number") for a in accounts if a.get("number") and a.get("id")}
    corrections_made = 0

    def get_posting_acct_num(posting):
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

        target_voucher = None
        for v in vouchers:
            desc = (v.get("description", "") or "").lower()
            postings = v.get("postings", []) or []

            if keyword and keyword in desc:
                target_voucher = v
                break

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
            for p in postings:
                if get_posting_acct_num(p) == wrong_acct_num:
                    amount = p.get("amount", 0)
                    wrong_id = find_account_id(tripletex, wrong_acct_num, accounts)
                    correct_id = find_account_id(tripletex, correct_acct_num, accounts)

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
            vat_acct_id = find_account_id(tripletex, 2710, accounts) or find_account_id(tripletex, 2711, accounts)
            if not vat_acct_id:
                vat_acct_id = find_account_in_range(accounts, 2710, 2700, 2799)

            if vat_acct_id and postings:
                for p in postings:
                    acct_num = get_posting_acct_num(p) or 0
                    if isinstance(acct_num, int) and 4000 <= acct_num <= 7999:
                        gross_amount = p.get("amount", 0)
                        vat_amount = round(gross_amount * 0.25, 2)
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
