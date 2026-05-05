"""
Enhanced Credit Tracking System for Pharmaceutical Research

Extends existing NVIDIABuildCreditsMonitor via composition pattern to provide:
- Pharmaceutical-specific metrics tracking
- Multi-tier alerting system (daily/weekly/monthly)
- Research project cost budgeting
- Usage pattern optimization for free tier
- Cost-per-query pharmaceutical analysis

Design Principles:
- Composition over inheritance for clean extension
- Pharmaceutical domain-specific metrics
- Intelligent alerting with research workflow awareness
- Cost optimization recommendations

Integration:
- Works with existing credit monitoring infrastructure
- Adds pharmaceutical research context
- Provides actionable cost optimization insights
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import existing credit monitor for composition
try:
    from ..nvidia_build_client import NVIDIABuildCreditsMonitor
except ImportError:
    try:
        from src.nvidia_build_client import NVIDIABuildCreditsMonitor
    except ImportError:
        # Fallback if not available
        class NVIDIABuildCreditsMonitor:
            def __init__(self, api_key=None):
                self.credits_used = 0
                self.credits_remaining = 10000

            def track_request(self):
                self.credits_used += 1
                self.credits_remaining = max(0, 10000 - self.credits_used)

            def get_usage_summary(self):
                return {"requests_this_month": self.credits_used}


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Compatibility alert levels for test-facing daily usage checks."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CreditUsage:
    """Test-facing unit of credit usage."""

    timestamp: datetime
    credits_used: int
    query_type: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PharmaceuticalQuery:
    """Represents a pharmaceutical research query for cost tracking."""

    query_type: str  # "drug_interaction", "pharmacokinetics", "mechanism", "clinical"
    model_used: str
    tokens_consumed: int
    response_time_ms: int
    cost_tier: str  # "free_tier", "infrastructure"
    timestamp: datetime
    research_context: Optional[str] = None


@dataclass
class ResearchProject:
    """Represents a pharmaceutical research project for budget tracking."""

    project_id: str
    name: str
    description: str
    budget_limit: int  # Max requests allowed
    queries_used: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class PharmaceuticalCreditTracker:
    """
    Enhanced credit tracking system for pharmaceutical research applications.

    Extends NVIDIABuildCreditsMonitor via composition to provide domain-specific
    analytics and cost optimization for drug research workflows.
    """

    def __init__(
        self,
        base_monitor: Optional[NVIDIABuildCreditsMonitor] = None,
        cache_dir: Optional[str] = None,
        enable_project_tracking: bool = True,
        alert_config_path: Optional[str] = None,
    ):
        """
        Initialize pharmaceutical credit tracker.

        Args:
            base_monitor: Base credit monitor (created if None)
            cache_dir: Directory for persistent storage
            enable_project_tracking: Enable research project budgeting
        """
        # Composition: Use existing monitor as base
        # Initialize base monitor with default monthly limit; it will be updated
        # from centralized config below when available.
        self.base_monitor = base_monitor or NVIDIABuildCreditsMonitor(api_key=os.getenv("NVIDIA_API_KEY"))
        # Attach self to base monitor when supported for rich callbacks
        try:
            if hasattr(self.base_monitor, "attach_pharma_tracker"):
                self.base_monitor.attach_pharma_tracker(self)
        except Exception:
            pass

        # Pharmaceutical-specific tracking
        self.pharmaceutical_queries: List[PharmaceuticalQuery] = []
        self.research_projects: Dict[str, ResearchProject] = {}
        self.enable_project_tracking = enable_project_tracking
        self.alert_config_path = alert_config_path
        self.alert_manager = type("AlertManagerStub", (), {"send_alert": lambda self, alert: None})()
        self.usage_log: List[CreditUsage] = []
        self.current_daily_usage = 0
        self.pharmaceutical_optimized = True
        self.pharmaceutical_metrics: Dict[str, Any] = {}

        # Cache for persistence
        self.cache_dir = Path(cache_dir or "./cache/pharmaceutical_analytics")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Alert thresholds (pharmaceutical research optimized)
        self.alert_thresholds = {
            "daily_burn_rate": 0.05,  # 5% of monthly limit per day (150 requests)
            "weekly_burn_rate": 0.20,  # 20% of monthly limit per week (2000 requests)
            "monthly_usage": 0.80,  # 80% of monthly limit (8000 requests)
            "research_project_budget": 0.75,  # 75% of project budget used
        }
        # Centralized monthly free-tier limit (default 10k), may be overridden by alerts.yaml
        self._monthly_free_requests: int = 10000
        self.monthly_limit = self._monthly_free_requests
        self.daily_limit = 333
        # Optionally merge thresholds from centralized YAML
        self._load_alert_thresholds_from_yaml()
        self._load_test_alert_config()
        # Propagate configured monthly free requests into base monitor if supported
        try:
            if hasattr(self.base_monitor, "set_monthly_free_requests"):
                self.base_monitor.set_monthly_free_requests(self._monthly_free_requests)
        except Exception:
            pass

        # Load cached data
        self._load_cached_data()

        logger.info("Pharmaceutical credit tracker initialized")

    def _load_test_alert_config(self) -> None:
        if not self.alert_config_path:
            return
        path = Path(self.alert_config_path)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:
            return
        credit_cfg = payload.get("credit_monitoring", {})
        free_tier = credit_cfg.get("free_tier_limits", {})
        self.monthly_limit = int(free_tier.get("monthly_requests", self.monthly_limit) or self.monthly_limit)
        self.daily_limit = int(free_tier.get("daily_average", self.daily_limit) or self.daily_limit)
        self.pharmaceutical_metrics = dict(credit_cfg.get("pharmaceutical_metrics", {}) or {})

    def calculate_daily_allocation(self, monthly_limit: int, days_in_month: int) -> int:
        if days_in_month <= 0:
            raise ValueError("days_in_month must be positive")
        return monthly_limit // days_in_month

    def calculate_burst_capacity(self, daily_allocation: int) -> int:
        return math.ceil(daily_allocation * 1.5)

    def calculate_weighted_cost(self, base_cost: float, query_type: str) -> float:
        metrics = self.pharmaceutical_metrics.get(query_type, {})
        multiplier = float(metrics.get("cost_multiplier", 1.0) or 1.0)
        return base_cost * multiplier

    def assess_daily_burn_rate(self, usage: List[CreditUsage]) -> AlertLevel:
        total = sum(item.credits_used for item in usage)
        if self.daily_limit <= 0:
            return AlertLevel.NORMAL
        percentage = round(total / self.daily_limit * 100)
        if percentage >= 90:
            return AlertLevel.CRITICAL
        if percentage >= 80:
            return AlertLevel.WARNING
        return AlertLevel.NORMAL

    def analyze_pharmaceutical_usage(self, usage: List[CreditUsage]) -> Dict[str, Any]:
        total = sum(item.credits_used for item in usage)
        breakdown: Dict[str, int] = {}
        for item in usage:
            breakdown[item.query_type] = breakdown.get(item.query_type, 0) + item.credits_used
        safety_total = breakdown.get("drug_safety_queries", 0)
        return {
            "total_credits_used": total,
            "query_type_breakdown": breakdown,
            "pharmaceutical_categories": sorted(breakdown),
            "cost_efficiency_score": max(0.0, min(100.0, 100.0 - total / 10.0)),
            "safety_query_percentage": (safety_total / total * 100.0) if total else 0.0,
        }

    def generate_cost_optimization_recommendations(self, usage: List[CreditUsage]) -> Dict[str, Any]:
        total = sum(item.credits_used for item in usage)
        return {
            "batch_processing": {"recommended": total >= 250, "potential_savings": max(0, total * 0.15)},
            "query_optimization": {"recommended": True, "focus": "high-volume query consolidation"},
            "timing_optimization": {"recommended": True, "focus": "smooth daily usage across time windows"},
        }

    async def record_usage_async(
        self,
        credits_used: int,
        query_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = CreditUsage(
            timestamp=datetime.now(),
            credits_used=credits_used,
            query_type=query_type,
            success=True,
            metadata=metadata or {},
        )
        self.usage_log.append(record)
        self.current_daily_usage += credits_used
        level = self.assess_daily_burn_rate([CreditUsage(datetime.now(), self.current_daily_usage, query_type, True)])
        if level != AlertLevel.NORMAL:
            self.alert_manager.send_alert({"level": level.value, "current_daily_usage": self.current_daily_usage})

    def generate_pharmaceutical_cost_report(self, usage: List[CreditUsage], period: str) -> Dict[str, Any]:
        total = sum(item.credits_used for item in usage)
        by_day: Dict[str, int] = {}
        for item in usage:
            key = item.timestamp.date().isoformat()
            by_day[key] = by_day.get(key, 0) + item.credits_used
        first = next(iter(by_day.values()), 0)
        last = list(by_day.values())[-1] if by_day else 0
        direction = "stable"
        if last > first:
            direction = "increasing"
        elif last < first:
            direction = "decreasing"
        return {
            "period": period,
            "total_cost": total,
            "daily_breakdown": by_day,
            "pharmaceutical_insights": self.analyze_pharmaceutical_usage(usage),
            "optimization_opportunities": ["batch similar safety queries", "redistribute usage across days"],
            "trend_analysis": {"direction": direction},
        }

    def generate_monthly_optimization_plan(
        self,
        target_queries: int,
        pharmaceutical_mix: Dict[str, float],
    ) -> Dict[str, Any]:
        allocation = {key: int(target_queries * weight) for key, weight in pharmaceutical_mix.items()}
        return {
            "daily_allocation": self.calculate_daily_allocation(target_queries, 30),
            "query_type_allocation": allocation,
            "buffer_recommendations": {"reserve_queries": max(0, self.monthly_limit - target_queries)},
            "cost_projection": {"total": min(target_queries, self.monthly_limit)},
        }

    def assess_burst_usage(self, usage: List[CreditUsage]) -> Dict[str, Any]:
        total = sum(item.credits_used for item in usage)
        burst_limit = self.calculate_burst_capacity(self.daily_limit)
        return {
            "within_burst_limit": total <= burst_limit,
            "burst_utilization": (total / burst_limit) if burst_limit else 0.0,
            "burst_limit": burst_limit,
            "total_usage": total,
        }

    def plan_burst_recovery(self, burst_status: Dict[str, Any]) -> Dict[str, Any]:
        utilization = float(burst_status.get("burst_utilization", 0.0))
        return {
            "recommended_cooldown_hours": 24 if utilization > 1.0 else 12,
            "reduced_usage_target": max(0, self.daily_limit - int(self.daily_limit * max(0.0, utilization - 1.0))),
        }

    def get_daily_status(self) -> Dict[str, Any]:
        total = sum(item.credits_used for item in self.usage_log)
        breakdown = self.analyze_pharmaceutical_usage(self.usage_log)
        return {
            "total_usage": total,
            "remaining_capacity": max(0, self.daily_limit - total),
            "pharmaceutical_breakdown": {
                "drug_safety_percentage": breakdown["safety_query_percentage"],
                "clinical_research_percentage": (
                    breakdown["query_type_breakdown"].get("clinical_research", 0) / total * 100.0 if total else 0.0
                ),
            },
        }

    def track_pharmaceutical_query(
        self,
        query_type: str,
        model_used: str,
        tokens_consumed: int,
        response_time_ms: int,
        cost_tier: str = "free_tier",
        research_context: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """
        Track a pharmaceutical research query with detailed analytics.

        Args:
            query_type: Type of pharmaceutical query
            model_used: Model that processed the query
            tokens_consumed: Total tokens used
            response_time_ms: Query response time
            cost_tier: Cost tier used
            research_context: Additional research context
            project_id: Associated research project ID
        """
        # Track with base monitor (prefer pharma-aware API when available)
        try:
            if hasattr(self.base_monitor, "log_api_call_pharma"):
                self.base_monitor.log_api_call_pharma(
                    service=model_used,
                    # Log exactly one request to align with free-tier tracking
                    tokens_used=1,
                    query_text=research_context or "",
                    query_type=query_type,
                )
            elif hasattr(self.base_monitor, "track_request"):
                self.base_monitor.track_request()
        except Exception:
            try:
                self.base_monitor.track_request()
            except Exception:
                pass

        # Create pharmaceutical query record
        query = PharmaceuticalQuery(
            query_type=query_type,
            model_used=model_used,
            tokens_consumed=tokens_consumed,
            response_time_ms=response_time_ms,
            cost_tier=cost_tier,
            timestamp=datetime.now(),
            research_context=research_context,
        )

        self.pharmaceutical_queries.append(query)

        # Track project usage if enabled
        if self.enable_project_tracking and project_id:
            self._track_project_usage(project_id, query)

        # Check alert thresholds
        self._check_pharmaceutical_alerts()

        # Persist data
        self._save_cached_data()

        logger.debug(f"Pharmaceutical query tracked: {query_type} using {model_used}")

    def create_research_project(
        self, project_id: str, name: str, description: str, budget_limit: int, tags: Optional[List[str]] = None
    ) -> ResearchProject:
        """
        Create a new research project for budget tracking.

        Args:
            project_id: Unique project identifier
            name: Project name
            description: Project description
            budget_limit: Maximum requests allowed
            tags: Project tags for categorization

        Returns:
            Created ResearchProject instance
        """
        if project_id in self.research_projects:
            raise ValueError(f"Research project {project_id} already exists")

        project = ResearchProject(
            project_id=project_id, name=name, description=description, budget_limit=budget_limit, tags=tags or []
        )

        self.research_projects[project_id] = project
        self._save_cached_data()

        logger.info(f"Research project created: {name} ({project_id})")
        return project

    def get_pharmaceutical_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive pharmaceutical research analytics.

        Returns:
            Dictionary with detailed pharmaceutical usage analytics
        """
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)

        # Filter queries by time period
        today_queries = [q for q in self.pharmaceutical_queries if q.timestamp >= today]
        week_queries = [q for q in self.pharmaceutical_queries if q.timestamp >= week_start]
        month_queries = [q for q in self.pharmaceutical_queries if q.timestamp >= month_start]

        # Query type analysis
        query_types = {}
        for query in month_queries:
            query_types[query.query_type] = query_types.get(query.query_type, 0) + 1

        # Model usage analysis
        model_usage = {}
        for query in month_queries:
            model_usage[query.model_used] = model_usage.get(query.model_used, 0) + 1

        # Cost tier analysis
        free_tier_queries = len([q for q in month_queries if q.cost_tier == "free_tier"])
        infrastructure_queries = len([q for q in month_queries if q.cost_tier == "infrastructure"])

        # Performance metrics
        avg_response_time = sum(q.response_time_ms for q in month_queries) / max(len(month_queries), 1)
        total_tokens = sum(q.tokens_consumed for q in month_queries)

        result = {
            "time_period_analysis": {
                "today": len(today_queries),
                "this_week": len(week_queries),
                "this_month": len(month_queries),
            },
            "query_type_distribution": query_types,
            "model_usage_distribution": model_usage,
            "cost_analysis": {
                "free_tier_queries": free_tier_queries,
                "infrastructure_queries": infrastructure_queries,
                "cost_optimization_percentage": (free_tier_queries / max(len(month_queries), 1) * 100),
            },
            "performance_metrics": {
                "avg_response_time_ms": int(avg_response_time),
                "total_tokens_consumed": total_tokens,
                "avg_tokens_per_query": int(total_tokens / max(len(month_queries), 1)),
            },
            "research_projects": {
                "active_projects": len(self.research_projects),
                "total_budget": sum(p.budget_limit for p in self.research_projects.values()),
                "budget_used": sum(p.queries_used for p in self.research_projects.values()),
            },
            "base_monitor_summary": self.base_monitor.get_usage_summary()
            if hasattr(self.base_monitor, "get_usage_summary")
            else {},
        }
        # Include daily burn snapshot if the base monitor provides it
        try:
            if hasattr(self.base_monitor, "daily_burn_rate"):
                result["daily_burn"] = self.base_monitor.daily_burn_rate()
        except Exception:
            pass
        return result

    def get_cost_optimization_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate cost optimization recommendations for pharmaceutical research.

        Returns:
            List of actionable cost optimization recommendations
        """
        recommendations = []
        analytics = self.get_pharmaceutical_analytics()

        # Free tier optimization
        free_tier_percentage = analytics["cost_analysis"]["cost_optimization_percentage"]
        if free_tier_percentage < 80:
            recommendations.append(
                {
                    "category": "cost_optimization",
                    "priority": "high",
                    "title": "Increase Free Tier Usage",
                    "description": f"Currently using free tier for {free_tier_percentage:.1f}% of queries. "
                    "Consider prioritizing cloud endpoints to maximize free tier benefits.",
                    "action": "Enable cloud-first configuration and batch queries for efficiency",
                }
            )

        # Query efficiency
        avg_tokens = analytics["performance_metrics"]["avg_tokens_per_query"]
        if avg_tokens > 1000:
            recommendations.append(
                {
                    "category": "efficiency",
                    "priority": "medium",
                    "title": "Optimize Query Length",
                    "description": f"Average tokens per query: {avg_tokens}. Consider more concise queries.",
                    "action": "Use specific pharmaceutical terminology and shorter context windows",
                }
            )

        # Response time optimization
        avg_response_time = analytics["performance_metrics"]["avg_response_time_ms"]
        if avg_response_time > 3000:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "medium",
                    "title": "Improve Response Times",
                    "description": f"Average response time: {avg_response_time}ms. Consider endpoint optimization.",
                    "action": "Enable batch processing and request optimization features",
                }
            )

        # Budget utilization
        if analytics["research_projects"]["active_projects"] > 0:
            projects_over_budget = []
            for project in self.research_projects.values():
                usage_percentage = project.queries_used / project.budget_limit * 100
                if usage_percentage > 75:
                    projects_over_budget.append(project.name)

            if projects_over_budget:
                recommendations.append(
                    {
                        "category": "budget_management",
                        "priority": "high",
                        "title": "Project Budget Alert",
                        "description": f"Projects near budget limit: {', '.join(projects_over_budget)}",
                        "action": "Review project budgets and consider optimization strategies",
                    }
                )

        return recommendations

    def generate_monthly_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monthly report for pharmaceutical research.

        Returns:
            Detailed monthly usage and cost analysis report
        """
        analytics = self.get_pharmaceutical_analytics()
        recommendations = self.get_cost_optimization_recommendations()

        return {
            "report_generated": datetime.now().isoformat(),
            "report_period": "monthly",
            "summary": {
                "total_queries": analytics["time_period_analysis"]["this_month"],
                "cost_optimization": f"{analytics['cost_analysis']['cost_optimization_percentage']:.1f}%",
                "avg_response_time": f"{analytics['performance_metrics']['avg_response_time_ms']}ms",
                "active_projects": analytics["research_projects"]["active_projects"],
            },
            "detailed_analytics": analytics,
            "optimization_recommendations": recommendations,
            "pharmaceutical_insights": self._generate_pharmaceutical_insights(),
            "next_steps": self._generate_next_steps(recommendations),
        }

    def _track_project_usage(self, project_id: str, query: PharmaceuticalQuery) -> None:
        """Track usage for a specific research project."""
        if project_id not in self.research_projects:
            logger.warning(f"Unknown project ID: {project_id}")
            return

        project = self.research_projects[project_id]
        project.queries_used += 1

        # Check project budget alerts
        usage_percentage = project.queries_used / project.budget_limit
        if usage_percentage >= self.alert_thresholds["research_project_budget"]:
            logger.warning(f"Project '{project.name}' has used {usage_percentage:.1%} of budget")

    def _check_pharmaceutical_alerts(self) -> None:
        """Check and emit pharmaceutical research alerts based on thresholds."""
        try:
            # Prefer base monitor daily burn if available
            daily = None
            if hasattr(self.base_monitor, "daily_burn_rate"):
                daily = self.base_monitor.daily_burn_rate()
            # Daily burn threshold is defined as a fraction of the monthly limit in alerts.yaml
            threshold = float(self.alert_thresholds.get("daily_burn_rate", 0.05))
            if isinstance(daily, dict):
                burn = float(daily.get("burn_rate", 0.0) or 0.0)
                if burn >= threshold:
                    logger.info(
                        "Daily burn rate alert: burn_rate=%.4f used_today=%s (threshold=%.4f of monthly)",
                        burn,
                        daily.get("used_today"),
                        threshold,
                    )
                    return
            # Fallback: compute against configured monthly denominator
            # Prefer configured monthly_free_requests; fall back to base monitor's summary
            monthly_limit = self._monthly_free_requests
            try:
                base_summary = getattr(self.base_monitor, "get_usage_summary", lambda: {})() or {}
                monthly_limit = int(base_summary.get("monthly_limit", monthly_limit) or monthly_limit)
            except Exception:
                pass
            today = [q for q in self.pharmaceutical_queries if q.timestamp.date() == datetime.now().date()]
            threshold_count = int(monthly_limit * threshold)
            if len(today) >= threshold_count:
                logger.info(
                    "Daily burn rate alert: %s queries today (threshold=%s of monthly=%s)",
                    len(today),
                    threshold_count,
                    monthly_limit,
                )
        except Exception:
            pass

    def _load_alert_thresholds_from_yaml(self) -> None:
        """Merge thresholds from config/alerts.yaml when available (best-effort)."""
        try:
            import yaml  # type: ignore
        except Exception:
            return
        try:
            cfg_path = Path("config/alerts.yaml")
            if not cfg_path.exists():
                return
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            nb = cfg.get("nvidia_build", {}) or {}
            nv = nb.get("usage_alerts", {}) if isinstance(nb, dict) else {}
            if nv:
                self.alert_thresholds["daily_burn_rate"] = float(
                    nv.get("daily_burn_rate", self.alert_thresholds["daily_burn_rate"])
                )
                self.alert_thresholds["weekly_burn_rate"] = float(
                    nv.get("weekly_burn_rate", self.alert_thresholds["weekly_burn_rate"])
                )
            # Pick up the configured monthly free requests if present
            try:
                mfr = int(nb.get("monthly_free_requests", self._monthly_free_requests))
                if mfr > 0:
                    self._monthly_free_requests = mfr
            except Exception:
                pass
            pharma = cfg.get("pharmaceutical", {}) or {}
            proj = pharma.get("project_budget", {}) or {}
            if proj:
                self.alert_thresholds["research_project_budget"] = float(
                    proj.get("warning_threshold", self.alert_thresholds["research_project_budget"])
                )
        except Exception:
            # best-effort only
            pass

    def _generate_pharmaceutical_insights(self) -> List[Dict[str, str]]:
        """Generate pharmaceutical-specific insights from usage data."""
        analytics = self.get_pharmaceutical_analytics()
        insights = []

        # Query type insights
        query_types = analytics["query_type_distribution"]
        if query_types:
            most_common_type = max(query_types, key=query_types.get)
            insights.append(
                {
                    "category": "research_focus",
                    "insight": f"Most common query type: {most_common_type}",
                    "implication": "Consider optimizing workflows for this query type",
                }
            )

        # Model efficiency insights
        model_usage = analytics["model_usage_distribution"]
        if len(model_usage) > 1:
            insights.append(
                {
                    "category": "model_optimization",
                    "insight": f"Using {len(model_usage)} different models",
                    "implication": "Evaluate model performance vs cost for pharmaceutical queries",
                }
            )

        return insights

    def _generate_next_steps(self, recommendations: List[Dict[str, str]]) -> List[str]:
        """Generate actionable next steps based on recommendations."""
        next_steps = []

        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        if high_priority:
            next_steps.extend([r["action"] for r in high_priority])

        # Always include monitoring step
        next_steps.append("Continue monitoring pharmaceutical query patterns and costs")

        return next_steps

    def _save_cached_data(self) -> None:
        """Save tracking data to cache for persistence."""
        try:
            # Save pharmaceutical queries
            queries_file = self.cache_dir / "pharmaceutical_queries.json"
            queries_data = [
                {
                    "query_type": q.query_type,
                    "model_used": q.model_used,
                    "tokens_consumed": q.tokens_consumed,
                    "response_time_ms": q.response_time_ms,
                    "cost_tier": q.cost_tier,
                    "timestamp": q.timestamp.isoformat(),
                    "research_context": q.research_context,
                }
                for q in self.pharmaceutical_queries[-1000:]  # Keep last 1000 queries
            ]

            with open(queries_file, "w") as f:
                json.dump(queries_data, f, indent=2)

            # Save research projects
            projects_file = self.cache_dir / "research_projects.json"
            projects_data = {
                pid: {
                    "project_id": p.project_id,
                    "name": p.name,
                    "description": p.description,
                    "budget_limit": p.budget_limit,
                    "queries_used": p.queries_used,
                    "created_at": p.created_at.isoformat(),
                    "tags": p.tags,
                }
                for pid, p in self.research_projects.items()
            }

            with open(projects_file, "w") as f:
                json.dump(projects_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cached data: {str(e)}")

    def _load_cached_data(self) -> None:
        """Load tracking data from cache."""
        try:
            # Load pharmaceutical queries
            queries_file = self.cache_dir / "pharmaceutical_queries.json"
            if queries_file.exists():
                with open(queries_file) as f:
                    queries_data = json.load(f)

                self.pharmaceutical_queries = [
                    PharmaceuticalQuery(
                        query_type=q["query_type"],
                        model_used=q["model_used"],
                        tokens_consumed=q["tokens_consumed"],
                        response_time_ms=q["response_time_ms"],
                        cost_tier=q["cost_tier"],
                        timestamp=datetime.fromisoformat(q["timestamp"]),
                        research_context=q.get("research_context"),
                    )
                    for q in queries_data
                ]

            # Load research projects
            projects_file = self.cache_dir / "research_projects.json"
            if projects_file.exists():
                with open(projects_file) as f:
                    projects_data = json.load(f)

                self.research_projects = {
                    pid: ResearchProject(
                        project_id=p["project_id"],
                        name=p["name"],
                        description=p["description"],
                        budget_limit=p["budget_limit"],
                        queries_used=p["queries_used"],
                        created_at=datetime.fromisoformat(p["created_at"]),
                        tags=p["tags"],
                    )
                    for pid, p in projects_data.items()
                }

        except Exception as e:
            logger.error(f"Failed to load cached data: {str(e)}")


# Convenience function for pharmaceutical research
def create_pharmaceutical_tracker(enable_project_tracking: bool = True) -> PharmaceuticalCreditTracker:
    """
    Create pharmaceutical credit tracker with default configuration.

    Args:
        enable_project_tracking: Enable research project budget tracking

    Returns:
        Configured PharmaceuticalCreditTracker instance
    """
    return PharmaceuticalCreditTracker(enable_project_tracking=enable_project_tracking)


if __name__ == "__main__":
    # Quick test of pharmaceutical tracking
    tracker = create_pharmaceutical_tracker()

    # Simulate pharmaceutical queries
    tracker.track_pharmaceutical_query(
        query_type="drug_interaction",
        model_used="nvidia/nv-embedqa-e5-v5",
        tokens_consumed=150,
        response_time_ms=800,
        research_context="metformin drug interactions study",
    )

    # Generate analytics
    analytics = tracker.get_pharmaceutical_analytics()
    print("Pharmaceutical Analytics:")
    print(json.dumps(analytics, indent=2))
