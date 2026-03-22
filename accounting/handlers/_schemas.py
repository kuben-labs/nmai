"""Extraction schemas for Claude tool_use structured output."""

SUPPLIER_INVOICE_SCHEMA = {
    "name": "extract_supplier_invoice",
    "description": "Extract supplier invoice details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_name": {"type": "string"},
            "supplier_org_number": {"type": "string"},
            "invoice_reference": {"type": "string"},
            "invoice_date": {"type": "string", "description": "YYYY-MM-DD"},
            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
            "expense_description": {"type": "string"},
            "total_amount": {"type": "number"},
            "vat_included": {"type": "boolean", "description": "True if total_amount includes VAT"},
            "vat_rate_percent": {"type": "number", "description": "VAT rate (e.g. 25)"},
            "expense_account_number": {"type": "integer", "description": "Norwegian chart of accounts number (e.g. 6340)"},
            "department_name": {"type": "string"},
        },
        "required": ["supplier_name", "total_amount", "vat_rate_percent", "expense_account_number"]
    }
}

EMPLOYEE_SCHEMA = {
    "name": "extract_employee",
    "description": "Extract employee creation details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "email": {"type": "string"},
            "phone_mobile": {"type": "string"},
            "bank_account": {"type": "string"},
            "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "department_name": {"type": "string"},
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
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number"},
                        "prepaid_account_number": {"type": "integer"},
                        "expense_account_number": {"type": "integer"},
                    },
                    "required": ["description", "amount"]
                }
            },
            "tax_rate_percent": {"type": "number", "description": "Tax rate (e.g. 22)"},
            "revenue_total": {"type": "number", "description": "Total revenue for the year"},
            "expense_total": {"type": "number", "description": "Total expenses for the year (before depreciation)"},
            "closing_date": {"type": "string", "description": "Closing date YYYY-MM-DD, defaults to YYYY-12-31"},
        },
        "required": ["year"]
    }
}

VOUCHER_CORRECTION_SCHEMA = {
    "name": "extract_voucher_correction",
    "description": "Extract voucher correction details from the task prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "period_from": {"type": "string", "description": "Start of period to search YYYY-MM-DD"},
            "period_to": {"type": "string", "description": "End of period YYYY-MM-DD"},
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "error_type": {"type": "string", "enum": ["wrong_account", "wrong_amount", "duplicate", "missing_vat"]},
                        "search_keyword": {"type": "string", "description": "Keyword to find the voucher by description"},
                        "description": {"type": "string"},
                        "wrong_account_number": {"type": "integer"},
                        "correct_account_number": {"type": "integer"},
                        "wrong_amount": {"type": "number"},
                        "correct_amount": {"type": "number"},
                    },
                    "required": ["error_type"]
                }
            },
        },
        "required": ["period_from", "period_to", "errors"]
    }
}

SCHEMAS = {
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
