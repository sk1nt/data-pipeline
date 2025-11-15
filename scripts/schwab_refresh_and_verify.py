"""Refresh access token using refresh token in .env or .schwab_tokens.json, then verify API access.

This script will:
- Read `SCHWAB_REFRESH_TOKEN`, `SCHWAB_CLIENT_ID`, and `SCHWAB_CLIENT_SECRET` from `.env`.
- POST to the token endpoint with `grant_type=refresh_token` to obtain a new access token.
- Update `.schwab_tokens.json` with the latest token response.
- Optionally make a simple authenticated GET request to a Schawb protected endpoint to verify the token works.

Usage:
  python3 scripts/schwab_refresh_and_verify.py

Note: This script does not start a live streaming websocket. If `Schwabdev` is installed and available in PYTHONPATH,
it will attempt to import and call a `stream` helper; otherwise it will perform a simple API request to check token validity.
"""

import os
import sys
import json
import base64
import time
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(BASE_DIR, '.env')
# Use .tokens directory for token storage
TOK_DIR = os.path.join(BASE_DIR, '.tokens')
TOK_PATH = os.path.join(TOK_DIR, 'schwab_token.json')


def load_env(path=ENV_PATH):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            if '=' in ln:
                k, v = ln.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def write_tokens(tokens: dict, path=TOK_PATH):
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    merged = existing.copy()
    merged.update({'updated_at': int(time.time()), 'tokens': tokens})
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)
    print('Updated', path)


def refresh(refresh_token, client_id, client_secret):
    endpoints = ['https://api.schwabapi.com/v1/oauth/token','https://api.schwab.com/v1/oauth/token']
    headers = {'Accept': 'application/json'}
    for url in endpoints:
        payload = {'grant_type': 'refresh_token', 'refresh_token': refresh_token}
        try:
            r = requests.post(url, data=payload, auth=(client_id, client_secret), headers=headers, timeout=20)
        except Exception as e:
            print('Request error to', url, e)
            continue
        try:
            body = r.json()
        except Exception:
            body = r.text
        print('Tried', url, '=>', r.status_code)
        print(body)
        if r.status_code == 200 and isinstance(body, dict):
            return body
    return None


def verify_access(access_token):
    # Try a simple protected read to validate token. The exact Schwab endpoint may vary; we'll use a generic profile endpoint.
    url = 'https://api.schwabapi.com/v1/accounts'  # conservative choice; if invalid, switch to another test endpoint
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        print('Request error during verification:', e)
        return False, None
    print('Verify status:', r.status_code)
    try:
        body = r.json()
    except Exception:
        body = r.text
    print('Verify response (first 500 chars):')
    print(json.dumps(body, indent=2)[:500] if isinstance(body, dict) else str(body)[:500])
    return r.status_code == 200, body


def main():
    env = load_env()
    client_id = env.get('SCHWAB_CLIENT_ID')
    client_secret = env.get('SCHWAB_CLIENT_SECRET')
    refresh_token = env.get('SCHWAB_REFRESH_TOKEN')

    if not client_id or not client_secret:
        print('Missing SCHWAB_CLIENT_ID or SCHWAB_CLIENT_SECRET in .env')
        sys.exit(1)

    if not refresh_token:
        # try reading tokens file
        if os.path.exists(TOK_PATH):
            try:
                with open(TOK_PATH, 'r') as f:
                    tok = json.load(f)
                    rt = tok.get('tokens', {}).get('refresh_token')
                    if rt:
                        refresh_token = rt
            except Exception:
                pass

    if not refresh_token:
        print('No refresh token found in .env or tokens file. Run exchange step first.')
        sys.exit(1)

    token_response = refresh(refresh_token, client_id, client_secret)
    if not token_response:
        print('Refresh failed for all endpoints')
        sys.exit(2)

    # write tokens file
    write_tokens(token_response)

    access_token = token_response.get('access_token')
    if not access_token:
        print('No access_token in response; cannot verify')
        sys.exit(0)

    ok, body = verify_access(access_token)
    if ok:
        print('\nAccess token is valid; API verification succeeded.')
    else:
        print('\nAccess token verification failed. If streaming is desired and Schwabdev is installed, run streamer separately.')


if __name__ == '__main__':
    main()
