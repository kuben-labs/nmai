"""Task type detection from multilingual prompts."""

import re

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
        "korreksjon.*bilag", "korriger.*bilag", "correction.*voucher", "correct.*voucher",
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

    # Priority 2: Credit note — check before invoice patterns
    credit_words = ["kreditnota", "credit note", "nota de crédito", "gutschrift",
                    "avoir", "nota credito", "kreditnotiz", "nota de crédito"]
    if any(w in p for w in credit_words):
        return "credit_note"

    # Priority 3: Project lifecycle — check before invoice since prompts contain "factura"/"invoice"
    project_words = ["prosjekt", "project", "proyecto", "projeto", "projekt", "projet"]
    lifecycle_words = ["team", "aktivitet", "activity", "actividad", "atividade", "timesheet",
                       "timeføring", "timer", "horas", "hours", "stunden", "heures",
                       "bemanning", "staffing", "medarbeider", "member", "budget",
                       "lifecycle", "livssyklus"]
    if any(w in p for w in project_words) and any(w in p for w in lifecycle_words):
        return "project_lifecycle"

    # Priority 4: Supplier invoice — check before customer invoice
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

    # Fuzzy: year-end closing
    closing_words = ["årsavslutning", "årsoppgjer", "year-end", "year end", "closing",
                     "cierre", "encerramento", "jahresabschluss", "clôture"]
    if any(w in p for w in closing_words):
        return "year_end_closing"
    depr_words = ["avskrivning", "depreciation", "amortissement", "abschreibung", "depreciación", "depreciação"]
    tax_words = ["skattekostnad", "tax provision", "tax expense", "impuesto", "imposto", "steuer"]
    if any(w in p for w in depr_words) and any(w in p for w in tax_words):
        return "year_end_closing"

    # Fuzzy: voucher correction (error/correction + voucher/ledger)
    error_words = ["feil", "error", "correction", "korreksjon", "korriger", "fehler", "corrección",
                   "correção", "erreur", "oppdaget", "discovered", "found", "corrigir", "corriger"]
    voucher_words = ["bilag", "voucher", "postering", "posting", "hovedbok", "ledger",
                     "buchung", "asiento", "lançamento", "écriture"]
    if any(w in p for w in error_words) and any(w in p for w in voucher_words):
        return "voucher_correction"

    return None
