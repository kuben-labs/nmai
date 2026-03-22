"""Tripletex API client - thin wrapper for making authenticated API calls."""

import httpx
import json
import logging

logger = logging.getLogger(__name__)


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.client = httpx.Client(
            timeout=30.0,
            auth=self.auth,
            headers={"Content-Type": "application/json"},
        )
        self._enabled_zone_id = None
        self._dimension_value_to_index = {}  # value_id -> dimensionIndex
    def _get_enabled_zone_id(self) -> int | None:
        """Find an enabled travel expense zone, cached."""
        if self._enabled_zone_id is not None:
            return self._enabled_zone_id
        result = self.get("/travelExpense/zone", {"isDisabled": "false", "fields": "id,zoneName,countryCode", "count": 10})
        if result["status_code"] == 200:
            values = result["body"].get("values", [])
            if values:
                self._enabled_zone_id = values[0]["id"]
                logger.info(f"Found enabled zone: id={values[0]['id']}, name={values[0].get('zoneName')}")
                return self._enabled_zone_id
        return None

    def _fix_per_diem(self, data: dict | None) -> dict | None:
        """Auto-set travelExpenseZoneId to an enabled zone (always override)."""
        if not data:
            return data
        zone_id = self._get_enabled_zone_id()
        if zone_id:
            data["travelExpenseZoneId"] = zone_id
        return data

    def _fix_voucher_postings(self, data: dict | None) -> dict | None:
        """Fix voucher postings: set row, remove vatType, fix dimension field names."""
        if not data or "postings" not in data:
            return data
        postings = data.get("postings")
        if not isinstance(postings, list):
            return data
        for i, posting in enumerate(postings):
            posting["row"] = i + 1  # Row 0 is reserved for system use
            posting.pop("vatType", None)  # Let account-locked vatType apply
            # Auto-fix freeAccountingDimension field based on actual dimension index
            self._fix_dimension_field(posting)
        return data

    def _fix_dimension_field(self, posting: dict) -> None:
        """Move freeAccountingDimension to the correct field based on tracked dimension index."""
        for dim_key in ["freeAccountingDimension1", "freeAccountingDimension2", "freeAccountingDimension3"]:
            dim_val = posting.get(dim_key)
            if dim_val and isinstance(dim_val, dict) and "id" in dim_val:
                value_id = dim_val["id"]
                if value_id in self._dimension_value_to_index:
                    correct_index = self._dimension_value_to_index[value_id]
                    correct_key = f"freeAccountingDimension{correct_index}"
                    if correct_key != dim_key:
                        logger.info(f"  Auto-fix dimension: {dim_key} -> {correct_key} (value {value_id} has index {correct_index})")
                        posting[correct_key] = posting.pop(dim_key)

    def _track_dimension_value(self, response_body: dict) -> None:
        """Track dimension value IDs and their indices from POST responses."""
        value = response_body.get("value", {})
        if isinstance(value, dict) and "id" in value and "dimensionIndex" in value:
            self._dimension_value_to_index[value["id"]] = value["dimensionIndex"]
            logger.info(f"  Tracked dimension value {value['id']} -> index {value['dimensionIndex']}")

    def _validate_post(self, endpoint: str, data: dict | None) -> str | None:
        """Validate POST data before sending. Returns error message or None if valid.

        This catches common LLM mistakes BEFORE making the API call,
        preventing 4xx errors and wasted write calls.
        """
        ep = endpoint.rstrip("/")

        # Employee: requires firstName + lastName
        if ep.endswith("/employee"):
            if not data:
                return "Missing data for POST /employee"
            if not data.get("firstName"):
                return "Missing required field 'firstName' for POST /employee"
            if not data.get("lastName"):
                return "Missing required field 'lastName' for POST /employee"

        # Customer: requires name
        elif ep.endswith("/customer"):
            if not data or not data.get("name"):
                return "Missing required field 'name' for POST /customer"

        # Supplier: requires name
        elif ep.endswith("/supplier"):
            if not data or not data.get("name"):
                return "Missing required field 'name' for POST /supplier"

        # Order: requires customer + deliveryDate
        elif ep.endswith("/order"):
            if not data:
                return "Missing data for POST /order"
            if not data.get("customer"):
                return "Missing required field 'customer' (object with id) for POST /order"
            if not data.get("deliveryDate"):
                return "Missing required field 'deliveryDate' for POST /order"

        # Invoice: requires invoiceDate + invoiceDueDate + orders
        elif ep.endswith("/invoice"):
            if not data:
                return "Missing data for POST /invoice"
            if not data.get("invoiceDate"):
                return "Missing required field 'invoiceDate' for POST /invoice"
            if not data.get("invoiceDueDate"):
                return "Missing required field 'invoiceDueDate' for POST /invoice"
            if not data.get("orders"):
                return "Missing required field 'orders' (array of {id}) for POST /invoice. Create an order first."

        # Voucher: requires date + postings that balance
        elif "ledger/voucher" in ep and "list" not in ep:
            if not data:
                return "Missing data for POST /ledger/voucher"
            if not data.get("date"):
                return "Missing required field 'date' for POST /ledger/voucher"
            postings = data.get("postings", [])
            if not postings:
                return "Missing required field 'postings' for POST /ledger/voucher"
            # Check balance
            total = sum(p.get("amount", 0) for p in postings if isinstance(p, dict))
            if abs(total) > 0.01:
                return f"Voucher postings do not balance: sum={total}. Debit (positive) and credit (negative) must sum to 0."
            # Check each posting has account + amount fields
            for i, p in enumerate(postings):
                if not isinstance(p, dict):
                    continue
                if not p.get("account"):
                    return f"Posting {i+1} missing 'account' (object with id)"
                if "amount" not in p:
                    return f"Posting {i+1} missing 'amount'"

        # Employment: requires employee + startDate
        elif "employee/employment" in ep and "details" not in ep and "occupationCode" not in ep:
            if not data:
                return "Missing data for POST /employee/employment"
            if not data.get("employee"):
                return "Missing required field 'employee' (object with id) for POST /employee/employment"
            if not data.get("startDate"):
                return "Missing required field 'startDate' for POST /employee/employment"

        # Travel expense: requires employee + title
        elif ep.endswith("/travelExpense"):
            if not data:
                return "Missing data for POST /travelExpense"
            if not data.get("employee"):
                return "Missing required field 'employee' (object with id) for POST /travelExpense"

        # Product: requires name
        elif ep.endswith("/product"):
            if not data or not data.get("name"):
                return "Missing required field 'name' for POST /product"

        # Department: requires name + departmentNumber
        elif ep.endswith("/department"):
            if not data:
                return "Missing data for POST /department"
            if not data.get("name"):
                return "Missing required field 'name' for POST /department"

        # Project: requires name + projectManager + startDate
        elif ep.endswith("/project") and "activity" not in ep:
            if not data:
                return "Missing data for POST /project"
            if not data.get("name"):
                return "Missing required field 'name' for POST /project"
            if not data.get("projectManager"):
                return "Missing required field 'projectManager' (object with id, must be employee) for POST /project"

        return None  # Valid

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"GET {url} params={params}")
        resp = self.client.get(url, params=params)
        logger.info(f"  -> {resp.status_code}")
        body = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            logger.warning(f"  ERROR BODY: {json.dumps(body, default=str, ensure_ascii=False)[:1000]}")
        return {"status_code": resp.status_code, "body": body}

    def post(self, endpoint: str, data: dict | None = None) -> dict:
        # Pre-flight validation — catch mistakes before making the API call
        validation_error = self._validate_post(endpoint, data)
        if validation_error:
            logger.warning(f"  VALIDATION BLOCKED: {validation_error}")
            return {"status_code": 400, "body": {"validation_error": validation_error,
                    "_note": "This error was caught locally before calling the API. Fix the data and retry."}}

        if "ledger/voucher" in endpoint and data:
            data = self._fix_voucher_postings(data)
        if endpoint.rstrip("/").endswith("/employee") and data and "userType" not in data:
            data["userType"] = "NO_ACCESS"
        if "perDiemCompensation" in endpoint and data:
            data = self._fix_per_diem(data)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"POST {url} data={json.dumps(data, default=str)[:500]}")
        resp = self.client.post(url, json=data)
        logger.info(f"  -> {resp.status_code}")
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        if resp.status_code >= 400:
            logger.warning(f"  ERROR BODY: {json.dumps(body, default=str, ensure_ascii=False)[:1000]}")
        if "accountingDimension" in endpoint and resp.status_code in (200, 201):
            self._track_dimension_value(body)
        return {"status_code": resp.status_code, "body": body}

    def put(self, endpoint: str, data: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"PUT {url} data={json.dumps(data, default=str)[:500]}")
        resp = self.client.put(url, json=data)
        logger.info(f"  -> {resp.status_code}")
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        if resp.status_code >= 400:
            logger.warning(f"  ERROR BODY: {json.dumps(body, default=str, ensure_ascii=False)[:1000]}")
        return {"status_code": resp.status_code, "body": body}

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"DELETE {url}")
        resp = self.client.delete(url)
        logger.info(f"  -> {resp.status_code}")
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {"raw": resp.text}
        return {"status_code": resp.status_code, "body": body}

    def close(self):
        self.client.close()
