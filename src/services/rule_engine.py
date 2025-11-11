"""
Rule engine for automatic priority assignment in the GEX priority system.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..lib.logging import get_logger
from ..lib.exceptions import GEXPriorityError
from ..models.priority_rule import PriorityRule
from ..models.priority_request import PriorityRequest
from ..models.data_source import DataSource
from ..models.enums import PriorityLevel

logger = get_logger(__name__)


class RuleEngine:
    """Engine for evaluating priority rules and automatic assignment."""

    def __init__(self):
        self._rules_cache: Dict[str, PriorityRule] = {}
        self._last_cache_update: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)  # Cache rules for 5 minutes

    def evaluate_priority_request(
        self,
        request: PriorityRequest,
        data_source: DataSource,
        active_rules: List[PriorityRule]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate a priority request against all active rules.

        Args:
            request: The priority request to evaluate
            data_source: The data source for this request
            active_rules: List of active priority rules

        Returns:
            Tuple of (calculated_priority_score, list_of_matched_rule_names)
        """
        try:
            # Build evaluation context
            context = self._build_evaluation_context(request, data_source)

            matched_rules = []
            max_score = 0.0

            # Evaluate each rule
            for rule in active_rules:
                if rule.evaluate(context):
                    matched_rules.append(rule.name)
                    max_score = max(max_score, rule.priority_score)
                    logger.debug(f"Rule '{rule.name}' matched for request {request.request_id}")

            # If no rules matched, use default low priority
            if not matched_rules:
                logger.info(f"AUDIT: No rules matched for request {request.request_id}, using default low priority (0.1)")
                return 0.1, []  # Low priority score

            assigned_level = self.get_priority_level_from_score(max_score)
            logger.info(f"AUDIT: Request {request.request_id} matched {len(matched_rules)} rules {matched_rules}, final score: {max_score:.3f}, level: {assigned_level.value}")
            return max_score, matched_rules

        except Exception as e:
            logger.error(f"Failed to evaluate priority request {request.request_id}: {e}")
            raise GEXPriorityError(f"Rule evaluation failed: {str(e)}") from e

    def _build_evaluation_context(
        self,
        request: PriorityRequest,
        data_source: DataSource
    ) -> Dict[str, Any]:
        """
        Build the evaluation context for rule conditions.

        Args:
            request: The priority request
            data_source: The data source

        Returns:
            Dictionary containing all variables available to rule conditions
        """
        # Current time for freshness calculations
        now = datetime.utcnow()

        # Calculate data age if available
        data_age_seconds = None
        if request.metadata and 'data_timestamp' in request.metadata:
            try:
                data_timestamp = datetime.fromisoformat(request.metadata['data_timestamp'])
                data_age_seconds = (now - data_timestamp).total_seconds()
            except (ValueError, TypeError):
                pass

        # Calculate derived scores
        market_impact_score = self.evaluate_market_impact(request)
        freshness_score = self.evaluate_freshness(request)
        volatility_score = self.evaluate_market_volatility(request)

        context = {
            # Request attributes
            'data_type': request.data_type.value,
            'priority_level': request.priority_level.value,
            'market_symbol': getattr(request, 'market_symbol', None) or 'UNKNOWN',

            # Data source attributes
            'source_reliability': data_source.reliability_score,
            'total_requests': data_source.total_requests,
            'successful_requests': data_source.successful_requests,
            'average_response_time': data_source.average_response_time.total_seconds() if data_source.average_response_time else None,

            # Time-based attributes
            'current_hour': now.hour,
            'current_day_of_week': now.weekday(),  # 0=Monday, 6=Sunday
            'is_market_hours': self._is_market_hours(now),
            'data_age_seconds': data_age_seconds,
            'data_age_minutes': data_age_seconds / 60 if data_age_seconds else None,
            'data_age_hours': data_age_seconds / 3600 if data_age_seconds else None,

            # Derived evaluation scores
            'market_impact_score': market_impact_score,
            'freshness_score': freshness_score,
            'volatility_score': volatility_score,

            # Metadata attributes (if available)
            'has_metadata': bool(request.metadata),
            'metadata_keys': list(request.metadata.keys()) if request.metadata else [],
        }

        # Add metadata values to context
        if request.metadata:
            for key, value in request.metadata.items():
                # Convert key to valid Python identifier
                safe_key = key.replace('-', '_').replace(' ', '_')
                context[f'metadata_{safe_key}'] = value

        return context

    def _is_market_hours(self, dt: datetime) -> bool:
        """
        Check if the given datetime is during market hours (9:30 AM - 4:00 PM ET, weekdays).

        Args:
            dt: Datetime to check

        Returns:
            True if during market hours, False otherwise
        """
        # Convert to Eastern Time (simplified - assuming UTC input)
        # In production, you'd want proper timezone handling
        et_hour = (dt.hour - 4) % 24  # UTC-4 for EDT

        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        is_weekday = dt.weekday() < 5  # 0-4 = Mon-Fri
        is_market_time = 9.5 <= et_hour <= 16.0

        return is_weekday and is_market_time

    def validate_rule_condition(self, condition: str) -> bool:
        """
        Validate that a rule condition is syntactically correct and safe.

        Args:
            condition: The condition string to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to compile the condition
            compile(condition, '<string>', 'eval')

            # Test with a sample context
            test_context = {
                'data_type': 'tick_data',
                'source_reliability': 0.8,
                'current_hour': 12,
                'is_market_hours': True,
                'data_age_seconds': 300,
            }

            # Try to evaluate with safe globals
            safe_globals = {
                '__builtins__': {
                    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
                    'enumerate': enumerate, 'float': float, 'int': int, 'len': len,
                    'list': list, 'max': max, 'min': min, 'range': range, 'round': round,
                    'set': set, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
                }
            }

            eval(condition, safe_globals, test_context)
            return True

        except Exception as e:
            logger.warning(f"Rule condition validation failed: {e}")
            return False

    def get_priority_level_from_score(self, score: float) -> PriorityLevel:
        """
        Convert a priority score to a priority level.

        Args:
            score: Priority score (0.0-1.0)

        Returns:
            Corresponding PriorityLevel enum value
        """
        if score >= 0.9:
            return PriorityLevel.CRITICAL
        elif score >= 0.7:
            return PriorityLevel.HIGH
        elif score >= 0.4:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    def evaluate_market_impact(self, request: PriorityRequest) -> float:
        """
        Evaluate market impact based on volume and open interest data.

        Args:
            request: The priority request to evaluate

        Returns:
            Market impact score (0.0-1.0)
        """
        if not request.metadata:
            return 0.0

        volume = request.metadata.get('volume', 0)
        open_interest = request.metadata.get('open_interest', 0)

        # High volume threshold (adjust based on market)
        high_volume_threshold = 100000
        high_oi_threshold = 200000

        volume_score = min(volume / high_volume_threshold, 1.0)
        oi_score = min(open_interest / high_oi_threshold, 1.0)

        # Combined market impact score
        market_impact = (volume_score * 0.6) + (oi_score * 0.4)

        return market_impact

    def evaluate_freshness(self, request: PriorityRequest) -> float:
        """
        Evaluate data freshness based on age.

        Args:
            request: The priority request to evaluate

        Returns:
            Freshness score (0.0-1.0, higher is fresher)
        """
        if not request.metadata or 'data_timestamp' not in request.metadata:
            return 0.0

        try:
            data_timestamp = datetime.fromisoformat(request.metadata['data_timestamp'])
            age_seconds = (datetime.utcnow() - data_timestamp).total_seconds()

            # Freshness decay: 100% at 0s, 50% at 300s, 0% at 3600s+
            if age_seconds <= 0:
                return 1.0
            elif age_seconds <= 300:  # 5 minutes
                return 1.0 - (age_seconds / 600)  # Linear decay to 0.5
            elif age_seconds <= 3600:  # 1 hour
                return 0.5 * (1.0 - ((age_seconds - 300) / 3300))  # Decay to 0
            else:
                return 0.0

        except (ValueError, TypeError):
            return 0.0

    def evaluate_market_volatility(self, request: PriorityRequest) -> float:
        """
        Evaluate market volatility based on price movements and spreads.

        Args:
            request: The priority request to evaluate

        Returns:
            Volatility score (0.0-1.0, higher indicates more volatile)
        """
        if not request.metadata:
            return 0.0

        # Look for volatility indicators in metadata
        price_change_pct = abs(request.metadata.get('price_change_pct', 0))
        bid_ask_spread = request.metadata.get('bid_ask_spread', 0)
        vix_level = request.metadata.get('vix_level', 20)  # Default VIX ~20

        # Normalize components
        price_volatility = min(price_change_pct / 5.0, 1.0)  # 5% change = max volatility
        spread_volatility = min(bid_ask_spread / 0.05, 1.0)  # 5 cent spread = max
        vix_volatility = min(vix_level / 50.0, 1.0)  # VIX 50 = max volatility

        # Combined volatility score
        volatility = (price_volatility * 0.4) + (spread_volatility * 0.3) + (vix_volatility * 0.3)

        return volatility

    def validate_rules(self, rules: List[PriorityRule]) -> List[str]:
        """
        Validate a list of priority rules for conflicts and issues.

        Args:
            rules: List of rules to validate

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        # Check for duplicate rule names
        names = [rule.name for rule in rules]
        duplicates = set([name for name in names if names.count(name) > 1])
        if duplicates:
            errors.append(f"Duplicate rule names found: {', '.join(duplicates)}")

        # Validate each rule individually
        for rule in rules:
            rule_errors = self._validate_single_rule(rule)
            errors.extend([f"Rule '{rule.name}': {err}" for err in rule_errors])

        # Check for conflicting rules (same conditions, different priorities)
        conflicts = self._detect_rule_conflicts(rules)
        if conflicts:
            errors.extend([f"Rule conflict: {desc}" for desc in conflicts])

        return errors

    def _validate_single_rule(self, rule: PriorityRule) -> List[str]:
        """
        Validate a single priority rule.

        Args:
            rule: The rule to validate

        Returns:
            List of validation errors for this rule
        """
        errors = []

        # Validate condition syntax
        if not self.validate_rule_condition(rule.condition):
            errors.append(f"Invalid condition syntax: '{rule.condition}'")

        # Validate priority score range
        if not (0.0 <= rule.priority_score <= 1.0):
            errors.append(f"Priority score {rule.priority_score} is not between 0.0 and 1.0")

        # Check for empty or whitespace-only fields
        if not rule.name or not rule.name.strip():
            errors.append("Rule name cannot be empty")
        if not rule.condition or not rule.condition.strip():
            errors.append("Rule condition cannot be empty")
        if not rule.description or not rule.description.strip():
            errors.append("Rule description cannot be empty")

        return errors

    def _detect_rule_conflicts(self, rules: List[PriorityRule]) -> List[str]:
        """
        Detect potential conflicts between rules.

        Args:
            rules: List of rules to check for conflicts

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Check for rules with very similar conditions but different priorities
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                if abs(rule1.priority_score - rule2.priority_score) > 0.3:  # Significant priority difference
                    # Simple heuristic: if conditions are very similar in length and contain same keywords
                    cond1_words = set(rule1.condition.lower().replace(' ', '').split())
                    cond2_words = set(rule2.condition.lower().replace(' ', '').split())

                    similarity = len(cond1_words.intersection(cond2_words)) / len(cond1_words.union(cond2_words))

                    if similarity > 0.7:  # High similarity
                        conflicts.append(
                            f"Rules '{rule1.name}' and '{rule2.name}' have similar conditions "
                            f"but significantly different priorities ({rule1.priority_score:.2f} vs {rule2.priority_score:.2f})"
                        )

        return conflicts

    def resolve_rule_conflicts(self, rules: List[PriorityRule]) -> List[PriorityRule]:
        """
        Resolve conflicts in a list of rules by adjusting priorities.

        Args:
            rules: List of rules that may have conflicts

        Returns:
            List of rules with conflicts resolved
        """
        # Sort rules by priority score (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.priority_score, reverse=True)

        # For now, simple resolution: keep the highest priority rule when conflicts exist
        # In a more sophisticated system, this could use rule specificity, recency, etc.

        resolved_rules = []
        seen_conditions = set()

        for rule in sorted_rules:
            # Simple condition fingerprint (could be improved)
            condition_fingerprint = rule.condition.lower().replace(' ', '')

            if condition_fingerprint not in seen_conditions:
                resolved_rules.append(rule)
                seen_conditions.add(condition_fingerprint)
            else:
                logger.warning(f"Skipping conflicting rule '{rule.name}' - higher priority rule already exists")

        logger.info(f"Resolved rule conflicts: {len(rules)} -> {len(resolved_rules)} rules")
        return resolved_rules

    def create_default_rules(self) -> List[PriorityRule]:
        """
        Create a set of default priority rules based on the guideline definitions.

        Returns:
            List of default PriorityRule instances
        """
        default_rules = [
            PriorityRule(
                name="Critical Market Hours High Volume",
                description="Critical priority for high-volume data during market hours",
                condition="is_market_hours and source_reliability > 0.8 and (metadata_volume > 50000 or metadata_open_interest > 100000)",
                priority_score=0.95
            ),
            PriorityRule(
                name="High Priority Real-time Data",
                description="High priority for real-time data with good source reliability",
                condition="data_age_seconds is not None and data_age_seconds < 300 and source_reliability > 0.7",
                priority_score=0.85
            ),
            PriorityRule(
                name="Medium Priority Fresh Data",
                description="Medium priority for fresh data during market hours",
                condition="is_market_hours and data_age_seconds is not None and data_age_seconds < 1800",
                priority_score=0.65
            ),
            PriorityRule(
                name="Low Priority Stale Data",
                description="Low priority for stale data or low reliability sources",
                condition="data_age_seconds is None or data_age_seconds > 3600 or source_reliability < 0.5",
                priority_score=0.15
            ),
        ]

        # Validate the rules
        validation_errors = self.validate_rules(default_rules)
        if validation_errors:
            logger.error(f"Validation errors in default rules: {validation_errors}")
            # In production, you might want to raise an exception or handle this differently
            for error in validation_errors:
                logger.error(f"Rule validation error: {error}")

        # Resolve any conflicts
        resolved_rules = self.resolve_rule_conflicts(default_rules)

        logger.info(f"Created {len(resolved_rules)} validated default priority rules")
        return resolved_rules