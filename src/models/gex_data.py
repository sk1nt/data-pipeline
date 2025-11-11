"""
GEX data models for Gamma Exposure calculations and market data.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from ..lib.logging import get_logger
from ..lib.utils import generate_uuid
from ..models.base import BaseModel

logger = get_logger(__name__)


class GEXStrike(BaseModel):
    """Model representing a single strike in GEX calculations."""

    def __init__(
        self,
        strike_id: Optional[UUID] = None,
        snapshot_id: Optional[UUID] = None,
        strike_price: float = 0.0,
        call_open_interest: int = 0,
        put_open_interest: int = 0,
        call_volume: int = 0,
        put_volume: int = 0,
        call_gamma: float = 0.0,
        put_gamma: float = 0.0,
        total_gamma: float = 0.0,
        gamma_flip_price: Optional[float] = None,
        expiration_date: Optional[datetime] = None,
        days_to_expiration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a GEX strike.

        Args:
            strike_id: Unique identifier for the strike
            snapshot_id: ID of the parent snapshot
            strike_price: Strike price
            call_open_interest: Call open interest
            put_open_interest: Put open interest
            call_volume: Call volume
            put_volume: Put volume
            call_gamma: Call gamma exposure
            put_gamma: Put gamma exposure
            total_gamma: Total gamma exposure
            gamma_flip_price: Price where gamma flips from positive to negative
            expiration_date: Option expiration date
            days_to_expiration: Days until expiration
            metadata: Additional strike metadata
        """
        self.strike_id = strike_id or generate_uuid()
        self.snapshot_id = snapshot_id
        self.strike_price = strike_price
        self.call_open_interest = call_open_interest
        self.put_open_interest = put_open_interest
        self.call_volume = call_volume
        self.put_volume = put_volume
        self.call_gamma = call_gamma
        self.put_gamma = put_gamma
        self.total_gamma = total_gamma
        self.gamma_flip_price = gamma_flip_price
        self.expiration_date = expiration_date
        self.days_to_expiration = days_to_expiration
        self.metadata = metadata or {}

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'strike_id': str(self.strike_id),
            'snapshot_id': str(self.snapshot_id) if self.snapshot_id else None,
            'strike_price': self.strike_price,
            'call_open_interest': self.call_open_interest,
            'put_open_interest': self.put_open_interest,
            'call_volume': self.call_volume,
            'put_volume': self.put_volume,
            'call_gamma': self.call_gamma,
            'put_gamma': self.put_gamma,
            'total_gamma': self.total_gamma,
            'gamma_flip_price': self.gamma_flip_price,
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'days_to_expiration': self.days_to_expiration,
            'metadata': self.metadata,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'GEXStrike':
        """Create instance from database dictionary."""
        # Convert string UUIDs back to UUID objects
        strike_id = UUID(data['strike_id']) if data.get('strike_id') else None
        snapshot_id = UUID(data['snapshot_id']) if data.get('snapshot_id') else None

        # Convert ISO strings back to datetime
        expiration_date = datetime.fromisoformat(data['expiration_date']) if data.get('expiration_date') else None

        return cls(
            strike_id=strike_id,
            snapshot_id=snapshot_id,
            strike_price=data.get('strike_price', 0.0),
            call_open_interest=data.get('call_open_interest', 0),
            put_open_interest=data.get('put_open_interest', 0),
            call_volume=data.get('call_volume', 0),
            put_volume=data.get('put_volume', 0),
            call_gamma=data.get('call_gamma', 0.0),
            put_gamma=data.get('put_gamma', 0.0),
            total_gamma=data.get('total_gamma', 0.0),
            gamma_flip_price=data.get('gamma_flip_price'),
            expiration_date=expiration_date,
            days_to_expiration=data.get('days_to_expiration'),
            metadata=data.get('metadata', {}),
        )

    def calculate_gamma_exposure(self, spot_price: float) -> float:
        """
        Calculate gamma exposure for this strike at a given spot price.

        Args:
            spot_price: Current spot price

        Returns:
            Gamma exposure contribution
        """
        # Simplified gamma exposure calculation
        # In practice, this would involve Black-Scholes or similar models
        distance_from_spot = abs(self.strike_price - spot_price) / spot_price

        # Gamma decays with distance from spot
        gamma_decay = max(0.0, 1.0 - distance_from_spot * 2)

        # Weight by open interest
        total_oi = self.call_open_interest + self.put_open_interest
        if total_oi == 0:
            return 0.0

        return self.total_gamma * gamma_decay * (total_oi / 1000)  # Scale factor

    def is_itm(self, spot_price: float) -> bool:
        """
        Check if this strike is in-the-money.

        Args:
            spot_price: Current spot price

        Returns:
            True if ITM, False if OTM
        """
        # This is a simplification - in reality, calls and puts have different ITM conditions
        return abs(self.strike_price - spot_price) / spot_price < 0.02  # Within 2%

    def __str__(self) -> str:
        """String representation of the strike."""
        return f"GEXStrike(strike={self.strike_price}, total_gamma={self.total_gamma:.2f})"


class GEXSnapshot(BaseModel):
    """Model representing a snapshot of GEX data at a point in time."""

    def __init__(
        self,
        snapshot_id: Optional[UUID] = None,
        market_symbol: str = "SPY",
        spot_price: float = 0.0,
        timestamp: Optional[datetime] = None,
        total_gamma: float = 0.0,
        gamma_flip_price: Optional[float] = None,
        max_gamma_price: Optional[float] = None,
        strikes: Optional[List[GEXStrike]] = None,
        data_source: Optional[str] = None,
        processing_job_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a GEX snapshot.

        Args:
            snapshot_id: Unique identifier for the snapshot
            market_symbol: Market symbol (e.g., 'SPY')
            spot_price: Current spot price
            timestamp: When this snapshot was taken
            total_gamma: Total gamma exposure across all strikes
            gamma_flip_price: Price where gamma flips from positive to negative
            max_gamma_price: Price with maximum gamma exposure
            strikes: List of individual strikes
            data_source: Source of the data
            processing_job_id: ID of the job that created this snapshot
            metadata: Additional snapshot metadata
        """
        self.snapshot_id = snapshot_id or generate_uuid()
        self.market_symbol = market_symbol
        self.spot_price = spot_price
        self.timestamp = timestamp or datetime.utcnow()
        self.total_gamma = total_gamma
        self.gamma_flip_price = gamma_flip_price
        self.max_gamma_price = max_gamma_price
        self.strikes = strikes or []
        self.data_source = data_source
        self.processing_job_id = processing_job_id
        self.metadata = metadata or {}

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'snapshot_id': str(self.snapshot_id),
            'market_symbol': self.market_symbol,
            'spot_price': self.spot_price,
            'timestamp': self.timestamp.isoformat(),
            'total_gamma': self.total_gamma,
            'gamma_flip_price': self.gamma_flip_price,
            'max_gamma_price': self.max_gamma_price,
            'strikes_count': len(self.strikes),
            'data_source': self.data_source,
            'processing_job_id': str(self.processing_job_id) if self.processing_job_id else None,
            'metadata': self.metadata,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'GEXSnapshot':
        """Create instance from database dictionary."""
        # Convert string UUIDs back to UUID objects
        snapshot_id = UUID(data['snapshot_id']) if data.get('snapshot_id') else None
        processing_job_id = UUID(data['processing_job_id']) if data.get('processing_job_id') else None

        # Convert ISO strings back to datetime
        timestamp = datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None

        return cls(
            snapshot_id=snapshot_id,
            market_symbol=data.get('market_symbol', 'SPY'),
            spot_price=data.get('spot_price', 0.0),
            timestamp=timestamp,
            total_gamma=data.get('total_gamma', 0.0),
            gamma_flip_price=data.get('gamma_flip_price'),
            max_gamma_price=data.get('max_gamma_price'),
            strikes=[],  # Strikes would be loaded separately
            data_source=data.get('data_source'),
            processing_job_id=processing_job_id,
            metadata=data.get('metadata', {}),
        )

    def add_strike(self, strike: GEXStrike) -> None:
        """Add a strike to this snapshot."""
        strike.snapshot_id = self.snapshot_id
        self.strikes.append(strike)

        # Recalculate totals
        self._recalculate_totals()

    def remove_strike(self, strike_id: UUID) -> bool:
        """Remove a strike from this snapshot."""
        for i, strike in enumerate(self.strikes):
            if strike.strike_id == strike_id:
                self.strikes.pop(i)
                self._recalculate_totals()
                return True
        return False

    def _recalculate_totals(self) -> None:
        """Recalculate total gamma and flip prices."""
        if not self.strikes:
            self.total_gamma = 0.0
            self.gamma_flip_price = None
            self.max_gamma_price = None
            return

        # Calculate total gamma
        self.total_gamma = sum(strike.total_gamma for strike in self.strikes)

        # Find gamma flip price (simplified - where gamma changes sign)
        gamma_values = [(strike.strike_price, strike.total_gamma) for strike in self.strikes]
        gamma_values.sort(key=lambda x: x[0])

        # Simple flip detection: where gamma crosses zero
        self.gamma_flip_price = None
        for i in range(len(gamma_values) - 1):
            price1, gamma1 = gamma_values[i]
            price2, gamma2 = gamma_values[i + 1]
            if gamma1 * gamma2 < 0:  # Sign change
                # Linear interpolation
                self.gamma_flip_price = price1 + (price2 - price1) * (0 - gamma1) / (gamma2 - gamma1)
                break

        # Find price with maximum absolute gamma
        max_gamma_strike = max(self.strikes, key=lambda s: abs(s.total_gamma))
        self.max_gamma_price = max_gamma_strike.strike_price

    def get_gamma_at_price(self, target_price: float) -> float:
        """
        Calculate total gamma exposure at a given price.

        Args:
            target_price: Price to calculate gamma for

        Returns:
            Total gamma exposure at the target price
        """
        total_gamma = 0.0
        for strike in self.strikes:
            total_gamma += strike.calculate_gamma_exposure(target_price)
        return total_gamma

    def get_strikes_in_range(self, min_price: float, max_price: float) -> List[GEXStrike]:
        """
        Get strikes within a price range.

        Args:
            min_price: Minimum strike price
            max_price: Maximum strike price

        Returns:
            List of strikes in the range
        """
        return [strike for strike in self.strikes
                if min_price <= strike.strike_price <= max_price]

    def is_market_neutral(self, tolerance: float = 0.1) -> bool:
        """
        Check if the market is gamma neutral.

        Args:
            tolerance: Tolerance for neutrality (as fraction of total gamma)

        Returns:
            True if gamma is neutral within tolerance
        """
        return abs(self.total_gamma) < abs(self.total_gamma) * tolerance

    def __str__(self) -> str:
        """String representation of the snapshot."""
        return f"GEXSnapshot(symbol={self.market_symbol}, spot={self.spot_price}, gamma={self.total_gamma:.2f}, strikes={len(self.strikes)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"GEXSnapshot(snapshot_id={self.snapshot_id!r}, market_symbol={self.market_symbol!r}, "
                f"spot_price={self.spot_price!r}, total_gamma={self.total_gamma!r}, "
                f"strikes_count={len(self.strikes)!r})")