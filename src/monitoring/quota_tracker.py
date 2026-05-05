"""
Quota tracking utilities for NVIDIA Build free-tier monitoring.

Tracks per-day and per-month request volumes, emits alerts, and optionally
blocks new requests once limits are exceeded.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NVIDIAQuotaTracker:
    """Persistent tracker enforcing NVIDIA API quotas."""

    FREE_TIER_MONTHLY = 10_000
    FREE_TIER_DAILY = 333
    ALERT_THRESHOLD_DEFAULT = 0.90

    def __init__(
        self,
        usage_file: Path | str | None = None,
        alert_threshold: float = ALERT_THRESHOLD_DEFAULT,
        daily_limit: int | None = None,
        monthly_limit: int | None = None,
        enforcement_enabled: bool = True,
        alert_callback: Any | None = None,
    ) -> None:
        self.usage_file = Path(usage_file or Path("logs") / "nvidia_usage.json")
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self.alert_threshold = alert_threshold
        self.daily_limit = daily_limit or self.FREE_TIER_DAILY
        self.monthly_limit = monthly_limit or self.FREE_TIER_MONTHLY
        self.enforcement_enabled = enforcement_enabled
        self.alert_callback = alert_callback
        self._usage = self._load_usage()
        self._ensure_period_entries()
        self._save_usage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_request(self, model: str, tokens: int, query_type: str | None = None) -> None:
        """Record a successful API request and update counters."""
        self._ensure_period_entries()
        day_key = self._current_day_key()
        month_key = self._current_month_key()

        daily_entry = self._usage["daily"][day_key]
        monthly_entry = self._usage["monthly"][month_key]

        self._increment_entry(daily_entry, model, tokens, query_type)
        self._increment_entry(monthly_entry, model, tokens, query_type)

        self._save_usage()
        self._check_thresholds(daily_entry, monthly_entry)

    def check_quota_available(self, required_requests: int = 1) -> bool:
        """Return True when quota remains for the requested operations."""
        self._ensure_period_entries()
        day_key = self._current_day_key()
        month_key = self._current_month_key()
        daily = self._usage["daily"][day_key]["requests"]
        monthly = self._usage["monthly"][month_key]["requests"]

        if not self.enforcement_enabled:
            return True

        daily_ok = daily + required_requests <= self.daily_limit
        monthly_ok = monthly + required_requests <= self.monthly_limit
        if not daily_ok or not monthly_ok:
            logger.warning(
                "Quota exceeded (daily=%s/%s, monthly=%s/%s)",
                daily,
                self.daily_limit,
                monthly,
                self.monthly_limit,
            )
            return False
        return True

    def get_usage_summary(self) -> dict[str, Any]:
        """Return current quota utilization metrics."""
        self._ensure_period_entries()
        day_key = self._current_day_key()
        month_key = self._current_month_key()
        daily = self._usage["daily"][day_key]
        monthly = self._usage["monthly"][month_key]

        daily_remaining = max(self.daily_limit - daily["requests"], 0)
        monthly_remaining = max(self.monthly_limit - monthly["requests"], 0)
        daily_utilization = (daily["requests"] / self.daily_limit) * 100 if self.daily_limit else 0.0
        monthly_utilization = (monthly["requests"] / self.monthly_limit) * 100 if self.monthly_limit else 0.0

        days_elapsed = max(self._current_day_index(), 1)
        avg_daily = monthly["requests"] / days_elapsed
        estimated_monthly_total = round(avg_daily * 30)

        return {
            "requests_today": daily["requests"],
            "requests_this_month": monthly["requests"],
            "daily_remaining": daily_remaining,
            "monthly_remaining": monthly_remaining,
            "daily_utilization_percent": round(daily_utilization, 2),
            "monthly_utilization_percent": round(monthly_utilization, 2),
            "estimated_monthly_total": estimated_monthly_total,
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit,
        }

    def reset_daily_counters(self) -> None:
        """Reset daily counters manually."""
        self._usage["daily"] = {}
        self._ensure_period_entries(force=True)
        self._save_usage()

    def reset_monthly_counters(self) -> None:
        """Reset monthly counters manually."""
        self._usage["monthly"] = {}
        self._ensure_period_entries(force=True)
        self._save_usage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _increment_entry(self, entry: dict[str, Any], model: str, tokens: int, query_type: str | None) -> None:
        entry["requests"] = entry.get("requests", 0) + 1
        entry["tokens"] = entry.get("tokens", 0) + max(tokens, 0)

        models = entry.setdefault("models", {})
        models[model] = models.get(model, 0) + 1

        if query_type:
            query_types = entry.setdefault("query_types", {})
            query_types[query_type] = query_types.get(query_type, 0) + 1

    def _check_thresholds(self, daily_entry: dict[str, Any], monthly_entry: dict[str, Any]) -> None:
        daily_ratio = daily_entry["requests"] / self.daily_limit if self.daily_limit else 0.0
        monthly_ratio = monthly_entry["requests"] / self.monthly_limit if self.monthly_limit else 0.0

        alerts = []
        if daily_ratio >= self.alert_threshold:
            alerts.append("daily")
        if monthly_ratio >= self.alert_threshold:
            alerts.append("monthly")

        if not alerts:
            return

        for scope in alerts:
            message = (
                f"Approaching {scope} quota: "
                f"{daily_entry['requests'] if scope == 'daily' else monthly_entry['requests']} requests used."
            )
            logger.warning(message)
            if self.alert_callback:
                try:
                    self.alert_callback(scope, {"message": message})
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Quota alert callback failed: %s", exc)

    def _ensure_period_entries(self, force: bool = False) -> None:
        day_key = self._current_day_key()
        month_key = self._current_month_key()

        self._usage.setdefault("daily", {})
        self._usage.setdefault("monthly", {})

        if force or day_key not in self._usage["daily"]:
            self._usage["daily"][day_key] = {"requests": 0, "tokens": 0, "models": {}, "query_types": {}}
        if force or month_key not in self._usage["monthly"]:
            self._usage["monthly"][month_key] = {"requests": 0, "tokens": 0, "models": {}, "query_types": {}}

    def _load_usage(self) -> dict[str, Any]:
        if not self.usage_file.exists():
            return {"daily": {}, "monthly": {}}

        try:
            with self.usage_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if "daily" not in data or "monthly" not in data:
                    date_pattern = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
                    migrated_daily = {
                        key: value for key, value in data.items() if isinstance(key, str) and date_pattern.match(key)
                    }
                    data = {
                        "daily": data.get("daily", migrated_daily),
                        "monthly": data.get("monthly", {}),
                    }
                return {"daily": dict(data.get("daily", {})), "monthly": dict(data.get("monthly", {}))}
        except Exception as exc:
            logger.warning("Failed to load quota usage file (%s). Resetting counters. Error: %s", self.usage_file, exc)
            return {"daily": {}, "monthly": {}}

    def _save_usage(self) -> None:
        try:
            with self.usage_file.open("w", encoding="utf-8") as handle:
                json.dump(self._usage, handle, indent=2)
        except Exception as exc:
            logger.error("Failed to persist quota usage: %s", exc)

    @staticmethod
    def _current_day_key() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    @staticmethod
    def _current_month_key() -> str:
        return datetime.utcnow().strftime("%Y-%m")

    @staticmethod
    def _current_day_index() -> int:
        return int(datetime.utcnow().strftime("%d"))
