"""Auto-refresh Schwab access token and rotate refresh token.

Behavior:
- Refresh the access token every 29 minutes (1740 seconds).
- Ensure a refresh-token rotation request is made at least every 6 days.
- Store tokens in `.schwab_tokens.json` and keep `SCHWAB_REFRESH_TOKEN` up to date in `.env`.

Run:
  python3 scripts/schwab_autorefresh.py

Recommend running under systemd or supervising process for production.
"""

import os
import sys
import time
import json
import signal
from typing import Optional
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(BASE_DIR, '.env')
# Use .tokens directory for token storage
TOK_DIR = os.path.join(BASE_DIR, '.tokens')
TOK_PATH = os.path.join(TOK_DIR, 'schwab_token.json')

# intervals
ACCESS_REFRESH_INTERVAL = 29 * 60        # 29 minutes
REFRESH_ROTATE_INTERVAL = 6 * 24 * 3600   # 6 days

running = True


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


def write_env_replace(key: str, value: str, path=ENV_PATH):
    # Replace existing key in .env or append if missing
    lines = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.read().splitlines()
    found = False
    out_lines = []
    for ln in lines:
        if ln.strip().startswith(f"{key}="):
            out_lines.append(f'{key}="{value}"')
            found = True
        else:
            out_lines.append(ln)
    if not found:
        out_lines.append(f'{key}="{value}"')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'Updated {key} in .env')


def load_tokens(path=TOK_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def write_tokens(tokens: dict, path=TOK_PATH, refresh_rotated=False):
    existing = load_tokens(path)
    merged = existing.copy()
    merged.update({
        'updated_at': int(time.time()),
        'tokens': tokens,
    })
    if refresh_rotated:
        merged['refresh_rotated_at'] = int(time.time())
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)
    print('Wrote tokens to', path)


def refresh(refresh_token: str, client_id: str, client_secret: str) -> Optional[dict]:
    endpoints = ['https://api.schwabapi.com/v1/oauth/token', 'https://api.schwab.com/v1/oauth/token']
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


def handle_sigterm(signum, frame):
    global running
    print('Received signal, shutting down...')
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    env = load_env()
    client_id = env.get('SCHWAB_CLIENT_ID')
    client_secret = env.get('SCHWAB_CLIENT_SECRET')
    refresh_token = env.get('SCHWAB_REFRESH_TOKEN')
    redirect_uri = env.get('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')

    if not client_id or not client_secret:
        print('Missing client credentials in .env; aborting')
        sys.exit(1)

    tokens_file = load_tokens()
    last_rotate = tokens_file.get('refresh_rotated_at') or tokens_file.get('updated_at') or 0

    if not refresh_token:
        # try reading tokens file
        if os.path.exists(TOK_PATH):
            try:
                with open(TOK_PATH, 'r') as f:
                    tok = json.load(f)
                    rt = tok.get('refresh_token')
                    if rt:
                        refresh_token = rt
            except Exception:
                pass

    if not refresh_token:
        print('No refresh token available; run the exchange step first and populate .env or tokens file')
        sys.exit(1)

    print('Starting auto-refresh loop: access interval', ACCESS_REFRESH_INTERVAL, 's, rotate interval', REFRESH_ROTATE_INTERVAL, 's')

    # Run an initial refresh immediately to populate access token
    try:
        resp = refresh(refresh_token, client_id, client_secret)
        if resp:
            # update tokens file
            has_new_rt = 'refresh_token' in resp and resp['refresh_token'] and resp['refresh_token'] != refresh_token
            write_tokens(resp, refresh_rotated=bool(has_new_rt))
            if has_new_rt:
                refresh_token = resp['refresh_token']
                write_env_replace('SCHWAB_REFRESH_TOKEN', refresh_token)
                last_rotate = int(time.time())
    except Exception as e:
        print('Initial refresh failed:', e)

    # Main loop: refresh access token every ACCESS_REFRESH_INTERVAL
    # and ensure we force a rotate attempt at least every REFRESH_ROTATE_INTERVAL
    next_rotate_deadline = last_rotate + REFRESH_ROTATE_INTERVAL

    while running:
        start = time.time()
        try:
            resp = refresh(refresh_token, client_id, client_secret)
            if resp:
                # write tokens; if server provided a new refresh token, update .env
                has_new_rt = 'refresh_token' in resp and resp['refresh_token'] and resp['refresh_token'] != refresh_token
                write_tokens(resp, refresh_rotated=bool(has_new_rt))
                if has_new_rt:
                    refresh_token = resp['refresh_token']
                    write_env_replace('SCHWAB_REFRESH_TOKEN', refresh_token)
                    next_rotate_deadline = int(time.time()) + REFRESH_ROTATE_INTERVAL
                    print('Refresh token rotated and saved')
                else:
                    # if we arrive at the rotate deadline and server did not return new RT, log it
                    if time.time() >= next_rotate_deadline:
                        print('Rotate deadline reached; server did not return a new refresh token on last refresh')
                        # still update next deadline so we don't spam
                        next_rotate_deadline = int(time.time()) + REFRESH_ROTATE_INTERVAL
        except Exception as e:
            print('Error during refresh attempt:', e)

        # sleep until next scheduled access refresh (respecting time spent)
        elapsed = time.time() - start
        to_sleep = max(1, ACCESS_REFRESH_INTERVAL - int(elapsed))
        slept = 0
        # break sleep into short chunks to be responsive to signals
        while slept < to_sleep and running:
            chunk = min(10, to_sleep - slept)
            time.sleep(chunk)
            slept += chunk

    print('Auto-refresh stopped')


if __name__ == '__main__':
    main()
