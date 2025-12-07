"""
PriorityRule model for the GEX priority system.
"""

from datetime import datetime
from typing import Any, Dict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from lib.logging import get_logger

logger = get_logger(__name__)


class PriorityRule(BaseModel):
    """Model representing a rule for automatic priority assignment."""

    rule_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    condition: str = Field(..., min_length=1)  # Python expression to evaluate
    priority_score: float = Field(..., ge=0.0, le=1.0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }

    @validator('condition')
    def validate_condition(cls, v):
        """Validate that condition is a safe Python expression."""
        # Basic validation - ensure it doesn't contain dangerous operations
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__']
        for keyword in dangerous_keywords:
            if keyword in v.lower():
                raise ValueError(f'Condition cannot contain dangerous keyword: {keyword}')
        return v

    @validator('updated_at')
    def validate_updated_at(cls, v, values):
        """Validate that updated_at is not before created_at."""
        if values.get('created_at') and v < values['created_at']:
            raise ValueError('updated_at cannot be before created_at')
        return v

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the rule condition against a context.

        Args:
            context: Dictionary containing variables for condition evaluation

        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Create a restricted globals dict for safe evaluation
            safe_globals = {
                '__builtins__': {
                    'abs': abs,
                    'all': all,
                    'any': any,
                    'bool': bool,
                    'dict': dict,
                    'enumerate': enumerate,
                    'float': float,
                    'int': int,
                    'len': len,
                    'list': list,
                    'max': max,
                    'min': min,
                    'range': range,
                    'round': round,
                    'set': set,
                    'sorted': sorted,
                    'str': str,
                    'sum': sum,
                    'tuple': tuple,
                    'zip': zip,
                }
            }

            # Evaluate the condition
            result = eval(self.condition, safe_globals, context)
            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate rule {self.rule_id} condition: {e}")
            return False

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for database storage."""
        data = self.dict()
        data['rule_id'] = str(data['rule_id'])
        return data

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'PriorityRule':
        """Create instance from database dictionary."""
        if 'rule_id' in data and isinstance(data['rule_id'], str):
            data['rule_id'] = UUID(data['rule_id'])
        return cls(**data)

    def __str__(self) -> str:
        return f"PriorityRule(id={self.rule_id}, name={self.name}, score={self.priority_score}, active={self.is_active})"
