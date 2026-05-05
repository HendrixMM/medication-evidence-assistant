"""Utility helpers supporting the advanced PubMed caching layer.

The helpers in this module are intentionally self-contained so they can be
reused by both the caching implementation and tests.  While the new caching
features are opinionated, each utility exposes lightweight primitives that can
be swapped out in the future when more sophisticated backends (for example
Redis) are introduced.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache key normalisation and hashing utilities
# ---------------------------------------------------------------------------


class CacheKeyNormalizer:
    """Generate deterministic cache keys with schema versioning."""

    DEFAULT_VERSION = "1"

    @staticmethod
    def normalize(
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        version: str | int = DEFAULT_VERSION,
        salt: str | None = None,
    ) -> str:
        """Return a stable cache key for the given query/parameter pair.

        The function lowercases and strips the query, sorts the parameters and
        hashes everything together.  A short prefix with the schema version is
        added so future migrations can invalidate old entries quickly.
        """

        normalized_query = CacheKeyNormalizer._prepare_query(query)
        serialized_params = CacheKeyNormalizer._prepare_params(params)

        fingerprint_source = json.dumps(
            {
                "query": normalized_query,
                "params": serialized_params,
                "salt": salt or "",
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        digest = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return f"v{version}-{digest}"

    @staticmethod
    def _prepare_query(query: str) -> str:
        normalized = (query or "").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    @staticmethod
    def _prepare_params(params: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not params:
            return {}
        normalized: dict[str, Any] = {}
        for key in sorted(params.keys()):
            value = params[key]
            normalized[key] = CacheKeyNormalizer._normalize_value(value)
        return normalized

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: CacheKeyNormalizer._normalize_value(value[k]) for k in sorted(value.keys())}
        if isinstance(value, (list, tuple, set)):
            return [CacheKeyNormalizer._normalize_value(v) for v in value]
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        return str(value)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


class QueryTypeClassifier:
    """Lightweight classifier that labels queries for cache policy tuning."""

    _PHARMA_KEYWORDS = {
        "drug",
        "interaction",
        "pharma",
        "dose",
        "pharmacokinetics",
        "pharmacodynamics",
        "metab",
        "cyp",
    }

    _REVIEW_KEYWORDS = {
        "review",
        "meta-analysis",
        "systematic",
    }

    @classmethod
    def classify(cls, query: str) -> str:
        lowered = (query or "").lower()
        if any(token in lowered for token in cls._PHARMA_KEYWORDS):
            return "pharmaceutical"
        if any(token in lowered for token in cls._REVIEW_KEYWORDS):
            return "review"
        if len(lowered) < 40:
            return "short"
        return "general"


class CacheMetadataExtractor:
    """Create metadata payloads for cached PubMed results."""

    @staticmethod
    def extract(
        *,
        original_query: str,
        enhanced_query: str,
        parameters: Mapping[str, Any],
        results: Sequence[Any] | None = None,
        status: str = "success",
    ) -> dict[str, Any]:
        result_count = len(results or [])
        metadata = {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "parameters": dict(parameters),
            "result_count": result_count,
            "status": status,
            "query_type": QueryTypeClassifier.classify(enhanced_query or original_query),
            "created_at": datetime.now(UTC).isoformat(),
        }
        metadata["pharmaceutical_focus"] = metadata["query_type"] == "pharmaceutical"
        metadata["empty"] = result_count == 0
        return metadata


# ---------------------------------------------------------------------------
# Compression and validation
# ---------------------------------------------------------------------------


class CacheCompressionUtils:
    """Helpers for transparently compressing cached payloads."""

    @staticmethod
    def compress_json(payload: Any, enabled: bool) -> tuple[str, str]:
        """Return a base64 encoded representation of the payload.

        Returns a tuple of ``(encoded_payload, encoding)`` where ``encoding`` is
        either ``"gzip"`` or ``"plain"``.
        """

        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if not enabled:
            return base64.b64encode(raw).decode("ascii"), "plain"
        compressed = gzip.compress(raw)
        return base64.b64encode(compressed).decode("ascii"), "gzip"

    @staticmethod
    def decompress_json(encoded_payload: str, encoding: str) -> Any:
        raw = base64.b64decode(encoded_payload.encode("ascii"))
        if encoding == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


class CacheValidationUtils:
    """Centralise cache entry validation to simplify upgrades."""

    REQUIRED_FIELDS = {
        "schema_version",
        "created_at",
        "expires_at",
        "grace_expires_at",
        "hit_count",
        "payload",
        "payload_encoding",
        "metadata",
    }

    @classmethod
    def validate_entry(cls, entry: Mapping[str, Any]) -> None:
        missing = cls.REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Cache entry missing required fields: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Compliance helpers
# ---------------------------------------------------------------------------


class NCBIComplianceChecker:
    """Perform lightweight validation of cache policies against NCBI guidance."""

    MIN_TTL_HOURS = 24

    @classmethod
    def validate_ttl(cls, ttl_hours: float) -> bool:
        if ttl_hours < cls.MIN_TTL_HOURS:
            logger.warning(
                "NCBI recommends caching responses for at least 24 hours; configured TTL is %.2f hours.",
                ttl_hours,
            )
            return False
        return True


class CacheTTLCalculator:
    """Compute TTL and grace periods based on metadata and configuration."""

    def __init__(
        self,
        default_ttl_hours: float,
        grace_period_hours: float,
        empty_results_ttl_minutes: float,
        error_results_ttl_minutes: float,
        *,
        pharma_ttl_bonus_hours: float = 12.0,
        strict_ncbi_ttl: bool = False,
    ) -> None:
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.grace_period = timedelta(hours=grace_period_hours)
        self.empty_ttl = timedelta(minutes=empty_results_ttl_minutes)
        self.error_ttl = timedelta(minutes=error_results_ttl_minutes)
        bonus = max(pharma_ttl_bonus_hours, 0.0)
        self.pharma_bonus = timedelta(hours=bonus)
        self.strict_ncbi_ttl = bool(strict_ncbi_ttl)

    def calculate(self, metadata: Mapping[str, Any]) -> tuple[timedelta, timedelta]:
        status = metadata.get("status", "success")
        result_count = metadata.get("result_count", 0)
        pharma_focus = metadata.get("pharmaceutical_focus", False)

        ttl = self.default_ttl
        if status == "error":
            ttl = self.error_ttl
        elif result_count == 0:
            ttl = self.empty_ttl
        elif pharma_focus and self.pharma_bonus:
            ttl = max(ttl, self.default_ttl + self.pharma_bonus)

        if self.strict_ncbi_ttl and ttl < self.default_ttl:
            ttl = self.default_ttl

        return ttl, self.grace_period


# ---------------------------------------------------------------------------
# Background scheduling and analytics utilities
# ---------------------------------------------------------------------------


@dataclass
class WarmingDecision:
    cache_key: str
    metadata: dict[str, Any]
    decision: str  # "due", "defer", "skip"
    reason: str
    priority: int = 0
    suggested_defer_time: float | None = None


class CacheWarmingScheduler:
    """Track stale keys that should be refreshed in the background."""

    def __init__(self, rate_limiter=None) -> None:
        self._pending: dict[str, dict[str, Any]] = {}
        self._decisions: dict[str, WarmingDecision] = {}
        self._lock = threading.Lock()
        self._rate_limiter = rate_limiter
        self._optimal_timing_threshold = 0.7  # Priority multiplier for optimal timing

    def schedule(self, cache_key: str, metadata: Mapping[str, Any]) -> None:
        with self._lock:
            self._pending[cache_key] = dict(metadata)

    def get_warming_decisions(self, now: datetime | None = None) -> list[WarmingDecision]:
        """Return list of warming decisions with due/defer logic."""
        if now is None:
            now = datetime.now(UTC)

        with self._lock:
            decisions = []

            for cache_key, metadata in self._pending.items():
                decision = self._evaluate_warming_decision(cache_key, metadata, now)
                self._decisions[cache_key] = decision
                decisions.append(decision)

            # Sort by priority (descending) and decision type
            decisions.sort(key=lambda d: (-d.priority, 0 if d.decision == "due" else 1))
            return decisions

    def _evaluate_warming_decision(self, cache_key: str, metadata: Mapping[str, Any], now: datetime) -> WarmingDecision:
        """Evaluate whether to warm, defer, or skip a cache entry."""
        # Calculate base priority from metadata
        priority = 0

        # Pharmaceutical queries get higher priority
        if metadata.get("pharmaceutical_focus"):
            priority += 100

        # Queries with more results get higher priority
        result_count = metadata.get("result_count", 0)
        priority += min(result_count, 50)  # Cap at 50 points

        # Queries with recent access get higher priority
        last_accessed = metadata.get("last_accessed_at")
        if last_accessed:
            try:
                access_time = datetime.fromisoformat(last_accessed)
                if access_time.tzinfo is None:
                    access_time = access_time.replace(tzinfo=UTC)
                age_hours = (now - access_time).total_seconds() / 3600
                if age_hours < 24:
                    priority += 25
            except (ValueError, TypeError):
                pass

        # Check rate limiter status if available
        is_optimal_timing = False
        wait_time = 0.0
        if self._rate_limiter:
            try:
                status = self._rate_limiter.get_status()
                is_optimal_timing = status.optimal_timing
                wait_time = status.seconds_until_next_request
            except Exception:
                # If rate limiter fails, proceed without timing info
                pass

        # Apply optimal timing bonus
        if is_optimal_timing:
            priority = int(priority * self._optimal_timing_threshold)

        # Make decision
        if wait_time > 300:  # More than 5 minutes wait
            return WarmingDecision(
                cache_key=cache_key,
                metadata=dict(metadata),
                decision="defer",
                reason=f"Rate limited, wait {wait_time:.1f}s",
                priority=priority,
                suggested_defer_time=wait_time,
            )
        elif is_optimal_timing or priority >= 75:
            return WarmingDecision(
                cache_key=cache_key,
                metadata=dict(metadata),
                decision="due",
                reason="High priority or optimal timing",
                priority=priority,
            )
        else:
            return WarmingDecision(
                cache_key=cache_key,
                metadata=dict(metadata),
                decision="defer",
                reason="Low priority, suboptimal timing",
                priority=priority,
                suggested_defer_time=300.0,  # Defer 5 minutes
            )

    def pop_due_entries(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        """Return entries marked as 'due' for warming."""
        decisions = self.get_warming_decisions(now)
        due_entries = {}

        with self._lock:
            for decision in decisions:
                if decision.decision == "due":
                    due_entries[decision.cache_key] = decision.metadata
                    self._pending.pop(decision.cache_key, None)

        return due_entries

    def pop_all(self) -> dict[str, dict[str, Any]]:
        """Legacy method - returns all pending entries."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
            self._decisions.clear()
            return pending

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def get_decision_summary(self) -> dict[str, int]:
        """Return summary of current decisions."""
        with self._lock:
            summary = {"due": 0, "defer": 0, "skip": 0}
            for decision in self._decisions.values():
                summary[decision.decision] += 1
            return summary

    def compute_optimal_warming_timing(
        self, cache_key: str, metadata: Mapping[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Compute optimal warming timing for a cache entry.

        Returns:
            Dict with:
            - due_now: bool - whether entry should be warmed immediately
            - defer_until: Optional[datetime] - when to retry if deferred
            - reason: str - explanation of decision
            - priority: int - computed priority score
        """
        if now is None:
            now = datetime.now(UTC)

        # Calculate base priority (same logic as _evaluate_warming_decision)
        priority = 0
        if metadata.get("pharmaceutical_focus"):
            priority += 100
        result_count = metadata.get("result_count", 0)
        priority += min(result_count, 50)
        last_accessed = metadata.get("last_accessed_at")
        if last_accessed:
            try:
                access_time = datetime.fromisoformat(last_accessed)
                if access_time.tzinfo is None:
                    access_time = access_time.replace(tzinfo=UTC)
                age_hours = (now - access_time).total_seconds() / 3600
                if age_hours < 24:
                    priority += 25
            except (ValueError, TypeError):
                pass

        # Check rate limiter and compute optimal timing
        defer_until = None
        due_now = False
        reason = ""

        if self._rate_limiter:
            try:
                status = self._rate_limiter.get_status()

                if status.seconds_until_next_request <= 0:
                    # Can proceed immediately
                    due_now = priority >= 50  # Minimum priority threshold
                    reason = "Rate limiter available" if due_now else "Below priority threshold"
                else:
                    # Need to defer - find optimal window
                    wait_time = status.seconds_until_next_request
                    base_priority = priority

                    # Look ahead up to 48 hours for optimal timing windows
                    max_lookahead = timedelta(hours=48)
                    check_time = now
                    best_time = None
                    best_priority = 0

                    # Check current time + regular intervals
                    for i in range(0, int(max_lookahead.total_seconds()), 300):  # Every 5 minutes
                        check_time = now + timedelta(seconds=i)

                        # Simulate priority decay over time
                        time_decay = max(0, 1 - (i / max_lookahead.total_seconds()))
                        adjusted_priority = int(base_priority * time_decay)

                        # Check if this would be an optimal window
                        if adjusted_priority >= 50:  # Minimum threshold
                            if not best_time or adjusted_priority > best_priority:
                                best_time = check_time
                                best_priority = adjusted_priority

                    if best_time:
                        defer_until = best_time
                        reason = f"Deferred to optimal window at {best_time.isoformat()}"
                    else:
                        # No suitable window found - use standard delay
                        defer_until = now + timedelta(seconds=min(wait_time, 3600))
                        reason = f"No optimal window found, deferred {wait_time:.1f}s"

            except Exception:
                # Rate limiter failed - use fallback logic
                due_now = priority >= 75
                reason = "Rate limiter unavailable" if due_now else "Below priority threshold"
        else:
            # No rate limiter - use simple priority-based decision
            due_now = priority >= 75
            reason = "High priority" if due_now else "Below priority threshold"

        return {"due_now": due_now, "defer_until": defer_until, "reason": reason, "priority": priority}

    def get_warming_schedule(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        """Get complete warming schedule with timing decisions for all entries.

        Returns:
            Dict mapping cache keys to warming decisions with timing info
        """
        if now is None:
            now = datetime.now(UTC)

        schedule = {}

        with self._lock:
            for cache_key, metadata in self._pending.items():
                timing = self.compute_optimal_warming_timing(cache_key, metadata, now=now)
                schedule[cache_key] = {"metadata": metadata, **timing}

        return schedule


class CacheCleanupScheduler:
    """Decide when automatic cleanup should fire."""

    def __init__(self, interval_hours: float, *, run_on_start: bool = False) -> None:
        self.interval = timedelta(hours=interval_hours)
        if run_on_start:
            self.last_run: datetime | None = None
        else:
            self.last_run = datetime.now(UTC)
        self._lock = threading.Lock()

    def should_run(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        with self._lock:
            if self.last_run is None:
                return True
            return now - self.last_run >= self.interval

    def mark_ran(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        with self._lock:
            self.last_run = now


@dataclass
class CacheStatistics:
    hits: int = 0
    stale_hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    warmed: int = 0
    skipped_rate_limit: int = 0
    exported: int = 0
    imported: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "stale_hits": self.stale_hits,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "warmed": self.warmed,
            "skipped_rate_limit": self.skipped_rate_limit,
            "exported": self.exported,
            "imported": self.imported,
        }


class CacheAnalytics:
    """Collect cache statistics in a threadsafe manner."""

    def __init__(self) -> None:
        self._stats = CacheStatistics()
        self._lock = threading.Lock()

    def record_hit(self, *, stale: bool) -> None:
        with self._lock:
            if stale:
                self._stats.stale_hits += 1
            else:
                self._stats.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self._stats.misses += 1

    def record_write(self) -> None:
        with self._lock:
            self._stats.writes += 1

    def record_eviction(self) -> None:
        with self._lock:
            self._stats.evictions += 1

    def record_warming(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._stats.warmed += count

    def record_rate_limit_skip(self) -> None:
        with self._lock:
            self._stats.skipped_rate_limit += 1

    def record_export(self, count: int) -> None:
        with self._lock:
            self._stats.exported += count

    def record_import(self, count: int) -> None:
        with self._lock:
            self._stats.imported += count

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return self._stats.to_dict()


class CacheOptimizer:
    """Naive optimiser that provides hints for cache maintenance."""

    def suggest_evictions(
        self,
        entries: Sequence[tuple[Path, Mapping[str, Any]]],
        *,
        max_entries: int,
        max_size_bytes: int,
    ) -> list[Path]:
        if len(entries) <= max_entries and self._total_size(entries) <= max_size_bytes:
            return []

        # Prefer evicting least-recently-used entries when metadata is available.
        sorted_entries = sorted(
            entries,
            key=lambda item: self._timestamp_for_entry(item[1]),
        )
        to_remove: list[Path] = []
        total_size = self._total_size(entries)
        for path, _metadata in sorted_entries:
            if len(entries) - len(to_remove) <= max_entries and total_size <= max_size_bytes:
                break
            to_remove.append(path)
            total_size -= path.stat().st_size if path.exists() else 0
        return to_remove

    @staticmethod
    def _total_size(entries: Sequence[tuple[Path, Mapping[str, Any]]]) -> int:
        total = 0
        for path, _metadata in entries:
            if path.exists():
                total += path.stat().st_size
        return total

    @staticmethod
    def _timestamp_for_entry(entry: Mapping[str, Any]) -> datetime:
        for candidate in (entry.get("last_accessed_at"), entry.get("created_at")):
            if not candidate:
                continue
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed
            except (TypeError, ValueError):
                continue
        return datetime.min.replace(tzinfo=UTC)


class CacheMigrationUtils:
    """Helpers for coordinating cache schema migrations."""

    @staticmethod
    def needs_migration(entry_version: str, current_version: str) -> bool:
        return str(entry_version) != str(current_version)


class CacheExportImportUtils:
    """Export or import cache entries for backup or migration."""

    @staticmethod
    def export_entries(cache_dir: Path, destination: Path) -> int:
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry in cache_dir.glob("*.json"):
            target = destination / entry.name
            target.write_bytes(entry.read_bytes())
            count += 1
        return count

    @staticmethod
    def import_entries(source: Path, cache_dir: Path, *, target_schema_version: str = "1") -> int:
        cache_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry_path in source.glob("*.json"):
            try:
                # Load and validate JSON content
                with entry_path.open("r", encoding="utf-8") as fh:
                    entry_data = json.load(fh)

                # Validate entry structure
                CacheValidationUtils.validate_entry(entry_data)

                # Check schema version compatibility
                entry_version = str(entry_data.get("schema_version", "unknown"))
                if entry_version != target_schema_version:
                    logger.warning(
                        "Skipping cache entry %s: schema version %s != target %s",
                        entry_path.name,
                        entry_version,
                        target_schema_version,
                    )
                    continue

                # Copy validated entry
                target = cache_dir / entry_path.name
                target.write_bytes(entry_path.read_bytes())
                count += 1

            except Exception as exc:
                logger.warning("Skipping invalid cache entry %s: %s", entry_path.name, exc)
                continue

        return count


class RateLimitCacheIntegrator:
    """Provide glue logic between the cache layer and rate limiter."""

    def __init__(self, analytics: CacheAnalytics) -> None:
        self._analytics = analytics

    def notify_cache_hit(self, *, stale: bool) -> None:
        # Both fresh and stale hits skip API calls, so record rate limit skip
        self._analytics.record_rate_limit_skip()


class PharmaceuticalCacheOptimizer:
    """Tune cache metadata for pharmaceutical queries."""

    KEY_TERMS = {
        "drug interaction",
        "drug-drug",
        "pharmacokinetics",
        "pharmacodynamics",
        "cyp",
    }

    def enrich_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        enhanced_query = metadata.get("enhanced_query", "").lower()
        metadata["pharmaceutical_focus"] = metadata.get("pharmaceutical_focus") or any(
            term in enhanced_query for term in self.KEY_TERMS
        )
        return metadata


class CacheConfigValidator:
    """Validate cache configuration values and provide defaults."""

    @staticmethod
    def validate(config: Mapping[str, Any]) -> None:
        numeric_fields = {
            "default_ttl_hours",
            "grace_period_hours",
            "empty_results_ttl_minutes",
            "error_results_ttl_minutes",
            "cache_max_size_mb",
            "cache_max_entries",
            "pharma_ttl_bonus_hours",
        }
        for field in numeric_fields:
            value = config.get(field)
            if value is None:
                continue
            if isinstance(value, (int, float)) and value >= 0:
                continue
            raise ValueError(f"Cache configuration field '{field}' must be a non-negative number")

        backend = config.get("cache_backend")
        if backend not in {None, "file"}:
            raise ValueError(
                f"Unsupported cache backend '{backend}'. Supported backends: {{'file'}}. "
                "Contact development team to add support for additional backends."
            )

        ttl_hours = float(config.get("default_ttl_hours", 24))
        NCBIComplianceChecker.validate_ttl(ttl_hours)


__all__ = [
    "CacheAnalytics",
    "CacheCleanupScheduler",
    "CacheCompressionUtils",
    "CacheConfigValidator",
    "CacheExportImportUtils",
    "CacheKeyNormalizer",
    "CacheMetadataExtractor",
    "CacheMigrationUtils",
    "CacheOptimizer",
    "CacheTTLCalculator",
    "CacheValidationUtils",
    "CacheWarmingScheduler",
    "NCBIComplianceChecker",
    "PharmaceuticalCacheOptimizer",
    "QueryTypeClassifier",
    "RateLimitCacheIntegrator",
    "CacheStatistics",
    "WarmingDecision",
]
