from schwab.auth import easy_client
from schwab.streaming import StreamClient
import json
import time

# Assumes you've already created a token. See the authentication page for more information.
client = easy_client(
    api_key='StA39T6AYDRUAPKfKbWpqOlI19jxK31B',
    app_secret='yyLulGkqXwMm1v1N',
    callback_url="https://127.0.0.1:8182",
    token_path=".tokens/schwab_token.json"
)

# Print account information and account_id
try:
    response = client.get_accounts()
    account_info = response.json()
    print("Account Info:", json.dumps(account_info, indent=4))
    # Try to extract account_id from the response
    if isinstance(account_info, dict):
        accounts = account_info.get('accounts') or account_info.get('data')
        if accounts and isinstance(accounts, list) and len(accounts) > 0:
            account_id = accounts[0].get('accountId') or accounts[0].get('account_id')
            print("Account ID:", account_id)
        else:
            print("Account ID not found in account info.")
    else:
        print("Account info format not recognized.")
except Exception as e:
    print(f"Error fetching account info: {e}")

def handle_futures_data(message):
    print("Received futures data:", message)

import asyncio

stream_client = StreamClient(client)
stream_client.add_level_one_futures_handler(handle_futures_data)
futures_symbols = ["/MNQZ25"]

async def main():
    await stream_client.login()
    await stream_client.level_one_futures_subs(futures_symbols)
    print("Streaming started for futures symbols:", futures_symbols)
    try:
        for _ in range(60):
            await stream_client.handle_message()
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Streaming error: {e}")
    print("Streaming stopped.")

asyncio.run(main())
