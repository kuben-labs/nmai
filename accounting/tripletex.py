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
    def _fix_voucher_postings(self, data: dict | None) -> dict | None:
        """Fix voucher postings: set row starting from 1, remove vatType to let account defaults apply."""
        if not data or "postings" not in data:
            return data
        postings = data.get("postings")
        if not isinstance(postings, list):
            return data
        for i, posting in enumerate(postings):
            posting["row"] = i + 1  # Row 0 is reserved for system use
            posting.pop("vatType", None)  # Let account-locked vatType apply
        return data

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
