"""
Alert Management System Test Suite

Comprehensive testing of the pharmaceutical alert management system with:
- Multi-tier alert configuration (daily/weekly/monthly)
- Drug safety alert prioritization
- Real-time notification system
- Integration with YAML configuration
- Cost monitoring alert workflows

Tests validate alert system for pharmaceutical research cost optimization.
"""
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import yaml

# Import modules under test
try:
    pass

    from src.monitoring.alert_manager import Alert, AlertManager, AlertSeverity, AlertType
    from src.monitoring.credit_tracker import CreditUsage
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from src.monitoring.alert_manager import Alert, AlertManager, AlertSeverity, AlertType
    from src.monitoring.credit_tracker import CreditUsage


class TestAlertManager:
    """Test suite for pharmaceutical alert management system."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with mock alert configuration."""
        # Create comprehensive alert configuration
        self.alert_config = {
            "credit_monitoring": {
                "daily_burn_rate": {
                    "warning_threshold": 0.8,
                    "critical_threshold": 0.9,
                    "alert_channels": ["email", "console", "webhook"],
                },
                "weekly_trends": {"warning_threshold": 0.75, "critical_threshold": 0.85},
                "monthly_limits": {"free_tier_warning": 0.8, "free_tier_critical": 0.95},
            },
            "pharmaceutical_alerts": {
                "drug_safety_queries": {"priority": "high", "immediate_notification": True, "escalation_threshold": 10},
                "drug_interactions": {
                    "priority": "critical",
                    "immediate_notification": True,
                    "escalation_threshold": 5,
                },
                "clinical_research": {"priority": "medium", "batch_notification": True, "batch_interval_minutes": 30},
            },
            "notification_channels": {
                "email": {
                    "enabled": True,
                    "recipients": ["pharma-team@research.org"],
                    "severity_filter": ["warning", "critical"],
                },
                "console": {
                    "enabled": True,
                    "format": "structured",
                    "severity_filter": ["info", "warning", "critical"],
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://api.slack.com/incoming/webhook",
                    "severity_filter": ["critical"],
                },
            },
        }

        # Write to temporary file
        self.temp_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(self.alert_config, self.temp_config_file)
        self.temp_config_file.close()

        yield

        # Cleanup
        os.unlink(self.temp_config_file.name)

    def test_alert_manager_initialization(self):
        """Test alert manager initialization with pharmaceutical configuration."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        assert alert_manager is not None
        assert hasattr(alert_manager, "config")
        assert hasattr(alert_manager, "pharmaceutical_optimized")
        assert alert_manager.pharmaceutical_optimized == True

        # Should load pharmaceutical-specific alert rules
        assert hasattr(alert_manager, "pharmaceutical_rules")
        assert "drug_safety_queries" in alert_manager.pharmaceutical_rules
        assert "drug_interactions" in alert_manager.pharmaceutical_rules

    def test_alert_severity_classification(self):
        """Test alert severity classification for pharmaceutical queries."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test drug safety query severity
        safety_alert = Alert(
            alert_type=AlertType.PHARMACEUTICAL_SAFETY,
            severity=AlertSeverity.HIGH,
            message="High-priority drug safety query detected",
            metadata={"drug": "warfarin", "interaction_risk": "high"},
        )

        severity = alert_manager.classify_severity(safety_alert)
        assert severity == AlertSeverity.HIGH

        # Test drug interaction severity (should be critical)
        interaction_alert = Alert(
            alert_type=AlertType.DRUG_INTERACTION,
            severity=AlertSeverity.CRITICAL,
            message="Critical drug interaction detected",
            metadata={"drug_1": "warfarin", "drug_2": "aspirin", "risk_level": "major"},
        )

        severity = alert_manager.classify_severity(interaction_alert)
        assert severity == AlertSeverity.CRITICAL

        # Test general pharmaceutical query
        general_alert = Alert(
            alert_type=AlertType.PHARMACEUTICAL_GENERAL,
            severity=AlertSeverity.MEDIUM,
            message="General pharmaceutical query processed",
            metadata={"query_type": "mechanism_of_action"},
        )

        severity = alert_manager.classify_severity(general_alert)
        assert severity == AlertSeverity.MEDIUM

    def test_daily_burn_rate_alerts(self):
        """Test daily credit burn rate alert generation."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test normal usage (no alert)
        normal_usage = 200  # 60% of 333 daily limit
        alert = alert_manager.check_daily_burn_rate(normal_usage, 333)
        assert alert is None

        # Test warning threshold (80%)
        warning_usage = 266  # 80% of 333 daily limit
        alert = alert_manager.check_daily_burn_rate(warning_usage, 333)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert "daily burn rate" in alert.message.lower()
        assert "80%" in alert.message or "0.8" in alert.message

        # Test critical threshold (90%)
        critical_usage = 300  # 90% of 333 daily limit
        alert = alert_manager.check_daily_burn_rate(critical_usage, 333)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "daily burn rate" in alert.message.lower()

    def test_pharmaceutical_priority_escalation(self):
        """Test pharmaceutical priority escalation system."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Track drug safety queries
        safety_queries = []
        for i in range(12):  # Over escalation threshold of 10
            query = CreditUsage(
                timestamp=datetime.now(),
                credits_used=15,
                query_type="drug_safety_queries",
                success=True,
                metadata={"drug": f"drug_{i}", "safety_check": True},
            )
            safety_queries.append(query)

        # Check for escalation
        escalation_alert = alert_manager.check_pharmaceutical_escalation(safety_queries)
        assert escalation_alert is not None
        assert escalation_alert.severity == AlertSeverity.HIGH
        assert "escalation" in escalation_alert.message.lower()
        assert "drug safety" in escalation_alert.message.lower()

    def test_batch_notification_system(self):
        """Test batch notification system for non-urgent alerts."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Create multiple medium-priority alerts
        clinical_alerts = []
        for i in range(5):
            alert = Alert(
                alert_type=AlertType.PHARMACEUTICAL_RESEARCH,
                severity=AlertSeverity.MEDIUM,
                message=f"Clinical research query {i} processed",
                metadata={"study": f"study_{i}", "phase": "iii"},
            )
            clinical_alerts.append(alert)

        # Should batch these alerts
        batch_result = alert_manager.process_batch_alerts(clinical_alerts)
        assert batch_result["batched"] == True
        assert batch_result["batch_size"] == 5
        assert batch_result["next_send_time"] is not None

        # Should not batch critical alerts
        critical_alert = Alert(
            alert_type=AlertType.DRUG_INTERACTION,
            severity=AlertSeverity.CRITICAL,
            message="Critical drug interaction detected",
            metadata={"immediate": True},
        )

        batch_result = alert_manager.process_batch_alerts([critical_alert])
        assert batch_result["batched"] == False
        assert batch_result["sent_immediately"] == True

    @pytest.mark.asyncio
    async def test_real_time_alert_processing(self):
        """Test real-time alert processing and notification."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Mock notification channels
        with patch.object(alert_manager, "send_to_channels") as mock_send:
            # Process high-priority drug safety alert
            safety_alert = Alert(
                alert_type=AlertType.PHARMACEUTICAL_SAFETY,
                severity=AlertSeverity.HIGH,
                message="High-risk drug interaction query detected",
                metadata={
                    "drug": "warfarin",
                    "interaction_drug": "aspirin",
                    "risk_score": 0.85,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            await alert_manager.process_alert_async(safety_alert)

            # Should send immediately to appropriate channels
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0]
            assert safety_alert in call_args
            assert "email" in call_args or "console" in call_args

    def test_alert_channel_routing(self):
        """Test alert routing to appropriate notification channels."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test info-level alert routing
        info_alert = Alert(
            alert_type=AlertType.SYSTEM_INFO, severity=AlertSeverity.INFO, message="System status update", metadata={}
        )

        channels = alert_manager.determine_notification_channels(info_alert)
        assert "console" in channels  # Info goes to console
        assert "email" not in channels  # Info doesn't go to email (warning+ only)

        # Test critical alert routing
        critical_alert = Alert(
            alert_type=AlertType.DRUG_INTERACTION,
            severity=AlertSeverity.CRITICAL,
            message="Critical system alert",
            metadata={},
        )

        channels = alert_manager.determine_notification_channels(critical_alert)
        assert "email" in channels
        assert "console" in channels
        # Webhook disabled in test config, so shouldn't be included

    def test_pharmaceutical_alert_aggregation(self):
        """Test pharmaceutical alert aggregation and summarization."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Create diverse pharmaceutical alerts over time period
        alerts = []
        base_time = datetime.now() - timedelta(hours=1)

        # Drug safety alerts
        for i in range(8):
            alerts.append(
                Alert(
                    alert_type=AlertType.PHARMACEUTICAL_SAFETY,
                    severity=AlertSeverity.HIGH,
                    message=f"Drug safety query {i}",
                    timestamp=base_time + timedelta(minutes=i * 5),
                    metadata={"drug": f"drug_{i % 3}", "safety_level": "high"},
                )
            )

        # Clinical research alerts
        for i in range(5):
            alerts.append(
                Alert(
                    alert_type=AlertType.PHARMACEUTICAL_RESEARCH,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Clinical research query {i}",
                    timestamp=base_time + timedelta(minutes=i * 10),
                    metadata={"study_type": "phase_iii", "indication": "diabetes"},
                )
            )

        # Generate aggregation report
        aggregation = alert_manager.aggregate_pharmaceutical_alerts(alerts, period="hourly")

        assert "total_alerts" in aggregation
        assert aggregation["total_alerts"] == 13

        assert "alert_type_breakdown" in aggregation
        breakdown = aggregation["alert_type_breakdown"]
        assert "pharmaceutical_safety" in breakdown
        assert "pharmaceutical_research" in breakdown

        assert "severity_distribution" in aggregation
        severity_dist = aggregation["severity_distribution"]
        assert "high" in severity_dist
        assert "medium" in severity_dist

        # Should identify patterns
        assert "patterns" in aggregation
        patterns = aggregation["patterns"]
        assert len(patterns) > 0

    def test_cost_monitoring_alert_integration(self):
        """Test integration with cost monitoring alerts."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test free tier warning alert
        free_tier_alert = alert_manager.create_cost_monitoring_alert(
            alert_type="free_tier_warning", current_usage=8000, limit=10000, period="monthly"
        )

        assert free_tier_alert is not None
        assert free_tier_alert.alert_type == AlertType.COST_MONITORING
        assert free_tier_alert.severity == AlertSeverity.WARNING
        assert "80%" in free_tier_alert.message or "8000" in free_tier_alert.message

        # Test daily limit critical alert
        daily_critical_alert = alert_manager.create_cost_monitoring_alert(
            alert_type="daily_critical", current_usage=310, limit=333, period="daily"
        )

        assert daily_critical_alert is not None
        assert daily_critical_alert.severity == AlertSeverity.CRITICAL
        assert "daily" in daily_critical_alert.message.lower()

    def test_alert_suppression_and_deduplication(self):
        """Test alert suppression and deduplication logic."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Create duplicate alerts
        original_alert = Alert(
            alert_type=AlertType.PHARMACEUTICAL_SAFETY,
            severity=AlertSeverity.HIGH,
            message="Repeated drug safety query",
            metadata={"drug": "metformin", "query_hash": "abc123"},
        )

        duplicate_alert = Alert(
            alert_type=AlertType.PHARMACEUTICAL_SAFETY,
            severity=AlertSeverity.HIGH,
            message="Repeated drug safety query",
            metadata={"drug": "metformin", "query_hash": "abc123"},
        )

        # Process original alert
        should_send_1 = alert_manager.should_send_alert(original_alert)
        assert should_send_1 == True

        # Process duplicate alert (should be suppressed)
        should_send_2 = alert_manager.should_send_alert(duplicate_alert)
        assert should_send_2 == False

        # Test time-based suppression reset
        time.sleep(1)  # Brief pause
        alert_manager.suppression_window = 0.5  # 0.5 second window

        should_send_3 = alert_manager.should_send_alert(duplicate_alert)
        assert should_send_3 == True  # Should send after suppression window

    def test_pharmaceutical_alert_metrics(self):
        """Test pharmaceutical alert metrics and reporting."""
        alert_manager = AlertManager(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Process various alerts to build metrics
        test_alerts = [
            Alert(AlertType.PHARMACEUTICAL_SAFETY, AlertSeverity.HIGH, "Safety 1"),
            Alert(AlertType.PHARMACEUTICAL_SAFETY, AlertSeverity.CRITICAL, "Safety 2"),
            Alert(AlertType.DRUG_INTERACTION, AlertSeverity.CRITICAL, "Interaction 1"),
            Alert(AlertType.PHARMACEUTICAL_RESEARCH, AlertSeverity.MEDIUM, "Research 1"),
            Alert(AlertType.COST_MONITORING, AlertSeverity.WARNING, "Cost 1"),
        ]

        for alert in test_alerts:
            alert_manager.record_alert_metrics(alert)

        # Generate metrics report
        metrics = alert_manager.get_pharmaceutical_metrics()

        assert "total_alerts_processed" in metrics
        assert metrics["total_alerts_processed"] == 5

        assert "pharmaceutical_alert_percentage" in metrics
        # 4 out of 5 alerts are pharmaceutical-related (80%)
        assert metrics["pharmaceutical_alert_percentage"] == 80.0

        assert "average_severity_score" in metrics
        assert metrics["average_severity_score"] > 0

        assert "alert_type_distribution" in metrics
        distribution = metrics["alert_type_distribution"]
        assert distribution["pharmaceutical_safety"] == 2
        assert distribution["drug_interaction"] == 1


class TestIntegratedAlertWorkflows:
    """Integration tests for complete alert workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_pharmaceutical_alert_workflow(self):
        """Test complete pharmaceutical alert workflow from detection to notification."""

        # Create temporary config with all channels enabled
        alert_config = {
            "pharmaceutical_alerts": {"drug_safety_queries": {"priority": "high", "immediate_notification": True}},
            "notification_channels": {
                "console": {"enabled": True, "severity_filter": ["high", "critical"]},
                "email": {"enabled": True, "recipients": ["test@pharma.org"]},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            yaml.dump(alert_config, config_file)
            config_path = config_file.name

        try:
            alert_manager = AlertManager(config_path=config_path, pharmaceutical_optimized=True)

            # Mock notification system
            notifications_sent = []

            def mock_send_notification(channel, alert, recipients=None):
                notifications_sent.append(
                    {"channel": channel, "alert": alert, "recipients": recipients, "timestamp": datetime.now()}
                )

            with patch.object(alert_manager, "send_notification", side_effect=mock_send_notification):
                # Simulate high-priority drug safety alert
                safety_alert = Alert(
                    alert_type=AlertType.PHARMACEUTICAL_SAFETY,
                    severity=AlertSeverity.HIGH,
                    message="Critical drug safety query detected: warfarin interaction risk",
                    metadata={
                        "drug_primary": "warfarin",
                        "drug_secondary": "aspirin",
                        "interaction_severity": "major",
                        "patient_risk_factors": ["elderly", "kidney_disease"],
                        "recommended_action": "immediate_review",
                    },
                )

                # Process alert through complete workflow
                await alert_manager.process_alert_async(safety_alert)

                # Validate notifications were sent
                assert len(notifications_sent) > 0

                # Should send to multiple channels for high-priority
                channels_used = [notif["channel"] for notif in notifications_sent]
                assert "console" in channels_used

                # Validate alert content preservation
                sent_alert = notifications_sent[0]["alert"]
                assert sent_alert.message == safety_alert.message
                assert sent_alert.metadata["drug_primary"] == "warfarin"

            print("✅ End-to-end pharmaceutical alert workflow successful")
            print(f"   Notifications sent: {len(notifications_sent)}")
            print(f"   Channels used: {{notif['channel'] for notif in notifications_sent}}")

        finally:
            os.unlink(config_path)
