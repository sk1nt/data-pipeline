"""Exchange an authorization code for tokens and store them.

Usage:
  python3 scripts/schwab_exchange_and_store.py --url "<full redirect url>"
or
  python3 scripts/schwab_exchange_and_store.py --code "<authorization code>"

On success this writes `.schwab_tokens.json` in the project root and appends
`SCHWAB_REFRESH_TOKEN` to `.env` if not present.
"""

import os
import sys
import json
import argparse
from urllib.parse import urlparse, parse_qs, unquote
import base64
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(BASE_DIR, '.env')
# Store tokens inside the `.tokens` directory to match project layout
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


def append_refresh_to_env(refresh_token, path=ENV_PATH):
    env = load_env(path)
    if 'SCHWAB_REFRESH_TOKEN' in env:
        print('SCHWAB_REFRESH_TOKEN already present in .env; not overwriting')
        return
    with open(path, 'a') as f:
        f.write('\nSCHWAB_REFRESH_TOKEN="%s"\n' % refresh_token)
    print('Appended SCHWAB_REFRESH_TOKEN to .env')


def extract_code_from_url(url: str) -> str | None:
    p = urlparse(url)
    q = parse_qs(p.query)
    code = q.get('code')
    if code:
        val = code[0]
    else:
        # fallback to fragment
        frag = p.fragment
        q2 = parse_qs(frag)
        code = q2.get('code')
        if code:
            val = code[0]
        else:
            return None
    val = unquote(val)
    # some redirect URLs append %40 representing '@' — keep it if present
    # but if it ends with a stray '@' the server expects it usually
    return val


def exchange_code(client_id, client_secret, code, redirect_uri):
    token_url_candidates = [
        'https://api.schwabapi.com/v1/oauth/token',
        'https://api.schwab.com/v1/oauth/token'
    ]
    headers = {'Accept': 'application/json'}
    for token_url in token_url_candidates:
        auth = (client_id, client_secret)
        payload = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
        }
        try:
            r = requests.post(token_url, data=payload, auth=auth, headers=headers, timeout=20)
        except Exception as e:
            print('Request error to', token_url, e)
            continue
        try:
            body = r.json()
        except Exception:
            body = r.text
        print('Tried', token_url, '=>', r.status_code)
        print(body)
        if r.status_code == 200 and isinstance(body, dict):
            return token_url, body
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', help='Full redirect URL received from Schwab after authorization')
    p.add_argument('--code', help='Authorization code directly (if you extracted it)')
    args = p.parse_args()

    env = load_env()
    client_id = env.get('SCHWAB_CLIENT_ID')
    client_secret = env.get('SCHWAB_CLIENT_SECRET')
    redirect_uri = env.get('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')

    if not client_id or not client_secret:
        print('Missing SCHWAB_CLIENT_ID or SCHWAB_CLIENT_SECRET in .env')
        sys.exit(1)

    code = None
    if args.code:
        code = args.code
    elif args.url:
        code = extract_code_from_url(args.url)
    else:
        print('Provide --url or --code')
        sys.exit(1)

    if not code:
        print('Could not extract code from provided URL')
        sys.exit(1)

    token_url, token_body = exchange_code(client_id, client_secret, code, redirect_uri)
    if not token_body:
        print('Exchange failed for all endpoints. See printed responses.')
        sys.exit(2)

    # ensure token dir exists and write tokens file
    os.makedirs(TOK_DIR, exist_ok=True)
    tokens = {'endpoint': token_url, 'tokens': token_body}
    with open(TOK_PATH, 'w') as f:
        json.dump(tokens, f, indent=2)
    print('Wrote tokens to', TOK_PATH)

    rt = token_body.get('refresh_token')
    if rt:
        append_refresh_to_env(rt)

    print('Done')


if __name__ == '__main__':
    main()
