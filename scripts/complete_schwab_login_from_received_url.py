#!/usr/bin/env python3
"""Deprecated helper — use `scripts/schwab_token_manager.py exchange-url` instead.

This file is intentionally minimal to prevent duplicate code paths.
"""
import sys
print('DEPRECATED: use scripts/schwab_token_manager.py exchange-url --url "<callback_url>"')
sys.exit(1)
import sys
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
from src.config import settings
from schwab import auth


def write_token_to_path(token: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as fh:
        json.dump(token, fh)
    print('Wrote token to', p)


def main(received_url: str, token_path: Optional[str] = None):
    api_key = settings.schwab_client_id
    app_secret = settings.schwab_client_secret
    callback_url = settings.schwab_redirect_uri or 'https://127.0.0.1:8182'
    tok_path = Path(token_path) if token_path else Path(__file__).resolve().parents[1] / '.tokens' / 'schwab_token.json'

    # Construct auth context and exchange the received URL for token
    # If the received URL contains a 'state' param, pass it into get_auth_context
    qs = parse_qs(urlparse(received_url).query)
    state = qs.get('state', [None])[0]
    auth_context = auth.get_auth_context(api_key, callback_url, state=state)
    # define a write func to persist tokens
    def token_write_func(token, *args, **kwargs):
        write_token_to_path(token, tok_path)

    client = auth.client_from_received_url(api_key, app_secret, auth_context, received_url, token_write_func)
    print('Client created; access_token:', getattr(client, 'access_token', None))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--received-url', required=True, help='Full received callback URL in browser after login')
    ap.add_argument('--token-path', required=False, help='Optional path to write token')
    args = ap.parse_args()
    main(args.received_url, args.token_path)
