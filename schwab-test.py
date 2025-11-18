import os
from schwab import auth

# Load credentials from environment or .env
client_id = "StA39T6AYDRUAPKfKbWpqOlI19jxK31B"
client_secret = "yyLulGkqXwMm1v1N"
# refresh_token = os.environ["SCHWAB_REFRESH_TOKEN"]

# Authenticate and get access token
client = auth.easy_client(
    api_key=client_id,
    app_secret=client_secret,
    callback_url="https://127.0.0.1:8182",
    token_path=".tokens/schwab_token.json",
    interactive=False,
)

#client.session.refresh_token(client.token_url)
client.session.refresh_token("https://api.schwab.com/v1/oauth/token")


# Fetch quote for AAPL
quote = client.get_quote("AAPL")
print("AAPL price:", quote["AAPL"]["lastPrice"])
