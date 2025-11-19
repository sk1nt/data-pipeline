#!/usr/bin/env python3
"""Consolidated Schwab token manager: exchange, persist, rotate, verify.

Commands:
  persist-env    Persist SCHWAB_TOKENS_JSON or SCHWAB_REFRESH_TOKEN -> .tokens/schwab_token.json
  exchange-url   Exchange a received callback URL (copy from browser) and persist tokens
  rotate         Force refresh/rotate of refresh token using SchwabAuthClient
  verify         Print current token details via SchwabAuthClient

This script consolidates smaller helper scripts so there's a single place to
manage refresh/token persistence in CI and on dev machines.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_schwab_dependencies():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from src.config import settings as cfg
    from src.services.schwab_streamer import SchwabAuthClient as AuthClient
    from schwab import auth as schwab_auth

    return cfg, AuthClient, schwab_auth


settings, SchwabAuthClient, auth = _load_schwab_dependencies()


def persist_env(token_path: Path | None = None, dry_run: bool = False):
    tok_dir = PROJECT_ROOT / '.tokens'
    tok_path = token_path or (tok_dir / 'schwab_token.json')
    json_tok = os.environ.get('SCHWAB_TOKENS_JSON')
    if json_tok:
        try:
            parsed = json.loads(json_tok)
            if isinstance(parsed, dict) and 'tokens' in parsed:
                parsed = parsed['tokens']
            tokens = parsed
        except Exception as e:  # pragma: no cover - defensive
            print('Failed parse SCHWAB_TOKENS_JSON:', e)
            return 2
    else:
        rt = os.environ.get('SCHWAB_REFRESH_TOKEN')
        if not rt:
            print('Missing SCHWAB_REFRESH_TOKEN in env; nothing to persist')
            return 1
        tokens = {
            'access_token': os.environ.get('SCHWAB_ACCESS_TOKEN', ''),
            'refresh_token': rt,
            'token_type': 'Bearer',
            'expires_in': int(os.environ.get('SCHWAB_EXPIRES_IN', '0')),
        }
    if dry_run:
        print(json.dumps(tokens, indent=2))
        return 0
    tok_dir.mkdir(parents=True, exist_ok=True)
    tmp = tok_path.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        json.dump(tokens, fh)
    os.replace(tmp, tok_path)
    print('Persisted tokens to', tok_path)
    return 0


def exchange_url(received_url: str, token_path: Path | None = None, append_to_env: bool = False):
    api_key = settings.schwab_client_id
    app_secret = settings.schwab_client_secret
    callback = settings.schwab_redirect_uri or 'https://127.0.0.1:8182'
    if not (api_key and app_secret):
        print('SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET must be set in .env or environment')
        return 1
    tok_path = token_path or (PROJECT_ROOT / '.tokens' / 'schwab_token.json')
    # Build auth context correctly, passing state if present
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(received_url)
    state = parse_qs(parsed.query).get('state', [None])[0]
    auth_ctx = auth.get_auth_context(api_key, callback, state=state)
    def write_fn(tokens, *args, **kwargs):
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tok_path, 'w') as fh:
            json.dump(tokens, fh)
        print('Wrote tokens to', tok_path)
    client = auth.client_from_received_url(api_key, app_secret, auth_ctx, received_url, write_fn)
    if append_to_env and hasattr(client, 'tokens'):
        rt = client.tokens.get('refresh_token')
        if rt:
            env_path = PROJECT_ROOT / '.env'
            with open(env_path, 'a') as fh:
                fh.write('\nSCHWAB_REFRESH_TOKEN="%s"\n' % rt)
            print('Appended refresh token to .env')
    return 0


def rotate_and_print(force: bool = False):
    if not (settings.schwab_client_id and settings.schwab_client_secret):
        print('Missing client id/secret in settings')
        return 1
    client = SchwabAuthClient(
        client_id=settings.schwab_client_id,
        client_secret=settings.schwab_client_secret,
        refresh_token=settings.schwab_refresh_token or '',
        rest_url=settings.schwab_rest_url,
        interactive=False,
    )
    tokens = client.refresh_refresh_token() if force else client.refresh_tokens()
    print('Refreshed tokens:', tokens)
    return 0


def verify_print():
    client = SchwabAuthClient(
        client_id=settings.schwab_client_id,
        client_secret=settings.schwab_client_secret,
        refresh_token=settings.schwab_refresh_token or '',
        rest_url=settings.schwab_rest_url,
        interactive=False,
    )
    token = client.get_token()
    print('Access token present:', bool(token.access_token))
    print('Expires at:', token.expires_at.isoformat())
    print('Refresh token:', token.refresh_token)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd')
    sp.add_parser('persist-env')
    eu = sp.add_parser('exchange-url')
    eu.add_argument('--url', required=True)
    eu.add_argument('--append-env', action='store_true')
    ro = sp.add_parser('rotate')
    ro.add_argument('--force', action='store_true')
    sp.add_parser('verify')
    ap.add_argument('--token-path', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'persist-env':
        return persist_env(token_path=Path(args.token_path) if args.token_path else None, dry_run=args.dry_run)
    elif args.cmd == 'exchange-url':
        return exchange_url(args.url, token_path=Path(args.token_path) if args.token_path else None, append_to_env=args.append_env)
    elif args.cmd == 'rotate':
        return rotate_and_print(force=args.force)
    elif args.cmd == 'verify':
        return verify_print()
    else:
        ap.print_help()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
