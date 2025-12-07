# Data Model: Automated Alert Trading

## Entities

- **AlertMessage**
  - raw_message: str
  - user_id: str
  - channel_id: str
  - posted_at: datetime (ISO8601)
  - message_id: str

- **ParsedAlert**
  - action: str (BTO/STC or BUY/SELL)
  - symbol: str (e.g., UBER, /NQZ5)
  - option_type: str (PUT or CALL)
  - strike: Decimal
  - expiry: date (YYYY-MM-DD or contract string)
  - quantity: int or None
  - price: Decimal or None

- **OrderRecord**
  - order_id: str
  - account_id: str
  - legs: list
  - order_type: str
  - price: Decimal
  - quantity: int
  - status: str (open/filled/canceled)
  - fills: list of fill objects

- **AuditRecord**
  - timestamp: datetime
  - user_id: str
  - channel_id: str
  - alert_message: str
  - parsed_alert: ParsedAlert
  - computed_quantity: int
  - entry_price: float
  - order_id: str
  - exit_order_id: str (optional)
  - error: str (optional)

## Validation rules
- `ParsedAlert.quantity` must be >= 1 when set
- `ParsedAlert.symbol` must be non-empty and validated against instrument lookup
- `entry_price` must be numeric and > 0 for limit orders
- `computed_quantity` must not exceed max configured value (per-account config)
- `audit` must persist for every attempted action, even if failed

## Relationships
- `AlertMessage` -> `ParsedAlert` (1:1)
- `ParsedAlert` -> `OrderRecord` (1:N) (entry + partial fills + exit orders)
- `OrderRecord` -> `AuditRecord` (1:1) (for each automated attempt)
