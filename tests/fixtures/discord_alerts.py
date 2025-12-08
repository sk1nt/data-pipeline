from datetime import datetime

# Example Discord alert messages for tests
ALERT_SIMPLE_BTO_UBER = "Alert: BTO UBER 78p 12/05 @ 0.75"
ALERT_NO_PRICE = "Alert: BTO UBER 78p 12/05"
ALERT_STC = "Alert: STC UBER 78p 12/05 @ 1.50"

def fake_alert_message(raw_text: str, user_id: str = "U123", channel_id: str = "C123"):
    return {
        "raw_message": raw_text,
        "user_id": user_id,
        "channel_id": channel_id,
        "posted_at": datetime.utcnow().isoformat(),
        "message_id": "m-123",
    }
