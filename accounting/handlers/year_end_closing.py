"""Year-end closing handler — depreciation, prepaid reversals, tax provision."""

import logging
from datetime import date

from ._helpers import get_all_accounts, find_account_id, find_account_in_range

logger = logging.getLogger(__name__)


def handle_year_end_closing(tripletex, params):
    """Execute year-end closing: depreciation + prepaid reversals + tax provision."""
    year = params["year"]
    closing_date = params.get("closing_date", f"{year}-12-31")
    accounts = get_all_accounts(tripletex)

    total_depreciation = 0

    for asset in params.get("assets", []):
        cost = asset["cost"]
        life = asset["useful_life_years"]
        residual = asset.get("residual_value", 0)
        method = asset.get("depreciation_method", "linear")

        if method == "linear":
            annual_depr = (cost - residual) / life
        else:
            annual_depr = (cost - residual) / life

        annual_depr = round(annual_depr, 2)
        total_depreciation += annual_depr

        expense_num = asset.get("expense_account_number", 6010)
        accum_num = asset.get("depreciation_account_number", 1209)

        expense_id = find_account_id(tripletex, expense_num, accounts)
        accum_id = find_account_id(tripletex, accum_num, accounts)

        if not expense_id:
            expense_id = find_account_in_range(accounts, expense_num, 6000, 6099)
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

    for prepaid in params.get("prepaid_expenses", []):
        amount = prepaid["amount"]
        prepaid_num = prepaid.get("prepaid_account_number", 1700)
        expense_num = prepaid.get("expense_account_number", 6300)

        prepaid_id = find_account_id(tripletex, prepaid_num, accounts)
        expense_id = find_account_id(tripletex, expense_num, accounts)

        if not prepaid_id:
            prepaid_id = find_account_in_range(accounts, prepaid_num, 1700, 1799)
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

    tax_rate = params.get("tax_rate_percent")
    if tax_rate:
        revenue = params.get("revenue_total", 0)
        expenses = params.get("expense_total", 0)

        # If revenue/expenses not provided, fetch from ledger
        if not revenue and not expenses:
            result = tripletex.get("/ledger/posting", {
                "dateFrom": f"{year}-01-01", "dateTo": closing_date,
                "fields": "id,account,amount", "count": 10000,
            })
            if result["status_code"] == 200:
                for post in result["body"].get("values", []):
                    acct = post.get("account", {})
                    acct_num = acct.get("number")
                    if not acct_num:
                        aid = acct.get("id")
                        if aid:
                            for a in accounts:
                                if a.get("id") == aid:
                                    acct_num = a.get("number")
                                    break
                    if acct_num:
                        amt = post.get("amount", 0) or 0
                        if 3000 <= acct_num <= 3999:
                            revenue += abs(amt)
                        elif 4000 <= acct_num <= 7999:
                            expenses += abs(amt)

        profit = revenue - expenses - total_depreciation
        if profit > 0:
            tax_amount = round(profit * tax_rate / 100, 2)

            tax_expense_id = find_account_id(tripletex, 8700, accounts)
            tax_payable_id = find_account_id(tripletex, 2920, accounts) or find_account_id(tripletex, 2500, accounts)

            if not tax_expense_id:
                tax_expense_id = find_account_in_range(accounts, 8700, 8700, 8799)
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
