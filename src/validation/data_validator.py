from typing import Dict, Any, List


class DataValidator:
    @staticmethod
    def validate_timestamp(timestamp: Any) -> bool:
        """Validate timestamp format."""
        from datetime import datetime

        if isinstance(timestamp, datetime):
            return True
        elif isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return True
            except ValueError:
                return False
        elif isinstance(timestamp, (int, float)):
            return timestamp > 0
        return False

    @staticmethod
    def validate_numeric(
        value: Any, min_val: float = None, max_val: float = None
    ) -> bool:
        """Validate numeric value with optional range."""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return False
            if max_val is not None and num > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], required_fields: List[str]
    ) -> bool:
        """Check if all required fields are present and not None."""
        return all(
            field in data and data[field] is not None for field in required_fields
        )

    @staticmethod
    def check_duplicates(
        data: List[Dict[str, Any]], key_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Find duplicate records based on key fields."""
        seen = set()
        duplicates = []
        for record in data:
            key = tuple(record.get(field) for field in key_fields)
            if key in seen:
                duplicates.append(record)
            else:
                seen.add(key)
        return duplicates
