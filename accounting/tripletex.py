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
