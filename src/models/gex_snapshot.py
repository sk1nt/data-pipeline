from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class GEXSnapshot:
    """Represents a Gamma Exposure (GEX) snapshot for options data."""

    timestamp: datetime
    ticker: str
    spot_price: float
    zero_gamma: float
    net_gex: float
    min_dte: Optional[int] = None
    sec_min_dte: Optional[int] = None
    major_pos_vol: Optional[float] = None
    major_pos_oi: Optional[float] = None
    major_neg_vol: Optional[float] = None
    major_neg_oi: Optional[float] = None
    sum_gex_vol: Optional[float] = None
    sum_gex_oi: Optional[float] = None
    delta_risk_reversal: Optional[float] = None
    max_priors: Optional[str] = None
    strike_data: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.strike_data is None:
            self.strike_data = []

    def validate_strike_data(self) -> List[str]:
        """Validate the strike data structure."""
        errors = []

        if not isinstance(self.strike_data, list):
            errors.append("strike_data must be a list")
            return errors

        for i, strike in enumerate(self.strike_data):
            if not isinstance(strike, dict):
                errors.append(f"strike_data[{i}] must be a dict")
                continue

            required_fields = ['strike', 'gamma']
            for field in required_fields:
                if field not in strike:
                    errors.append(f"strike_data[{i}] missing required field '{field}'")
                elif not isinstance(strike[field], (int, float)):
                    errors.append(f"strike_data[{i}]['{field}'] must be numeric")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert the snapshot to a dictionary."""
        return {
            'timestamp': self.timestamp,
            'ticker': self.ticker,
            'spot_price': self.spot_price,
            'zero_gamma': self.zero_gamma,
            'net_gex': self.net_gex,
            'min_dte': self.min_dte,
            'sec_min_dte': self.sec_min_dte,
            'major_pos_vol': self.major_pos_vol,
            'major_pos_oi': self.major_pos_oi,
            'major_neg_vol': self.major_neg_vol,
            'major_neg_oi': self.major_neg_oi,
            'sum_gex_vol': self.sum_gex_vol,
            'sum_gex_oi': self.sum_gex_oi,
            'delta_risk_reversal': self.delta_risk_reversal,
            'max_priors': self.max_priors,
            'strike_data': self.strike_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GEXSnapshot':
        """Create a GEXSnapshot from a dictionary."""
        # Handle strike_data deserialization if it's a string
        strike_data = data.get('strike_data', [])
        if isinstance(strike_data, str):
            try:
                strike_data = json.loads(strike_data)
            except json.JSONDecodeError:
                strike_data = []

        return cls(
            timestamp=data['timestamp'],
            ticker=data['ticker'],
            spot_price=data['spot_price'],
            zero_gamma=data['zero_gamma'],
            net_gex=data['net_gex'],
            min_dte=data.get('min_dte'),
            sec_min_dte=data.get('sec_min_dte'),
            major_pos_vol=data.get('major_pos_vol'),
            major_pos_oi=data.get('major_pos_oi'),
            major_neg_vol=data.get('major_neg_vol'),
            major_neg_oi=data.get('major_neg_oi'),
            sum_gex_vol=data.get('sum_gex_vol'),
            sum_gex_oi=data.get('sum_gex_oi'),
            delta_risk_reversal=data.get('delta_risk_reversal'),
            max_priors=data.get('max_priors'),
            strike_data=strike_data
        )