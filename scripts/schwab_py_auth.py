#!/usr/bin/env python3
"""
Script to perform Schwab OAuth login using schwab-py and save tokens.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from schwab import auth
from src.config import settings

def main():
    print("Starting Schwab OAuth login with schwab-py...")
    client = auth.easy_client(
        api_key=settings.schwab_client_id,
        app_secret=settings.schwab_client_secret,
        callback_url='http://127.0.0.1:8182',
        token_path='.tokens/schwab_token.json'
    )
    print("Login completed. Tokens saved to .tokens/schwab_token.json")
    print("Access token:", client.access_token[:20] + "...")

if __name__ == "__main__":
    main()