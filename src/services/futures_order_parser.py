from enum import Enum


class FuturesAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"


class FuturesOrderParams:
    def __init__(
        self,
        action: FuturesAction,
        symbol: str,
        tp_ticks: float,
        quantity: int,
        mode: str,
    ):
        self.action = action
        self.symbol = symbol
        self.tp_ticks = tp_ticks
        self.quantity = quantity
        self.mode = mode  # 'dry' or 'live'


class FuturesOrderParser:
    def parse(
        self, action: str, symbol: str, tp_ticks: float, quantity: int, mode: str
    ) -> FuturesOrderParams:
        """Parse and validate futures order parameters."""

        # Validate action
        try:
            action_enum = FuturesAction(action.upper())
        except ValueError:
            raise ValueError(f"Invalid action: {action}. Must be buy, sell, or flat.")

        # Validate symbol (basic check for futures format)
        if not symbol or not symbol.startswith("/"):
            raise ValueError(
                f"Invalid symbol: {symbol}. Futures symbols should start with '/'."
            )

        # Validate tp_ticks
        if tp_ticks <= 0:
            raise ValueError(f"Invalid tp_ticks: {tp_ticks}. Must be positive.")

        # Validate quantity
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}. Must be positive.")

        # Validate mode
        if mode not in ["dry", "live"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'dry' or 'live'.")

        return FuturesOrderParams(action_enum, symbol, tp_ticks, quantity, mode)
