"""
API routes for priority rules management.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from ...lib.logging import get_logger
from ...models.priority_rule import PriorityRule
from ...services.rule_engine import RuleEngine

logger = get_logger(__name__)

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("/", response_model=List[PriorityRule])
async def get_priority_rules():
    """
    Get all active priority rules.

    Returns:
        List of active priority rules used for automatic assignment
    """
    try:
        rule_engine = RuleEngine()
        rules = rule_engine.create_default_rules()

        logger.info(f"Retrieved {len(rules)} active priority rules")
        return rules

    except Exception as e:
        logger.error(f"Failed to retrieve priority rules: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve priority rules: {str(e)}"
        )


@router.get("/evaluate")
async def evaluate_rules_sample():
    """
    Get a sample evaluation showing how rules work.

    This endpoint demonstrates the rule evaluation process with sample data.
    """
    try:
        from ...models.priority_request import PriorityRequest
        from ...models.data_source import DataSource
        from ...models.enums import DataType, PriorityLevel
        from datetime import datetime, timedelta

        # Create sample request
        sample_request = PriorityRequest(
            data_type=DataType.TICK_DATA,
            priority_level=PriorityLevel.MEDIUM,
            market_symbol="SPY",
            metadata={
                "volume": 75000,
                "open_interest": 150000,
                "data_timestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat()
            }
        )

        # Create sample data source
        sample_source = DataSource(
            name="Sample Source",
            endpoint_url="https://api.example.com",
            api_key="sample_key"
        )
        # Simulate some history
        sample_source.record_request(True, timedelta(seconds=0.5))
        sample_source.record_request(True, timedelta(seconds=0.3))

        # Evaluate with rule engine
        rule_engine = RuleEngine()
        rules = rule_engine.create_default_rules()
        score, matched_rules = rule_engine.evaluate_priority_request(
            sample_request, sample_source, rules
        )

        # Build evaluation context for demonstration
        context = rule_engine._build_evaluation_context(sample_request, sample_source)

        result = {
            "sample_request": {
                "data_type": sample_request.data_type.value,
                "market_symbol": sample_request.market_symbol,
                "metadata": sample_request.metadata
            },
            "sample_source": {
                "name": sample_source.name,
                "reliability_score": sample_source.reliability_score,
                "total_requests": sample_source.total_requests
            },
            "evaluation_context": context,
            "calculated_score": score,
            "matched_rules": matched_rules,
            "assigned_priority": rule_engine.get_priority_level_from_score(score).value
        }

        logger.info("Generated sample rule evaluation")
        return result

    except Exception as e:
        logger.error(f"Failed to generate sample evaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sample evaluation: {str(e)}"
        )