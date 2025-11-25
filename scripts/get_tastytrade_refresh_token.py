#!/usr/bin/env python3
"""Start an OAuth flow to retrieve a new TastyTrade refresh token (sandbox/test).

This script opens a browser and starts a local server to capture the code.
Run with `python scripts/get_tastytrade_refresh_token.py --sandbox` to use the sandbox/cert URL.
"""
import argparse
from tastytrade.oauth import login

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve a new TastyTrade refresh token (interactive)')
    parser.add_argument('--sandbox', action='store_true', help='Use TastyTrade sandbox environment')
    return parser.parse_args()

def main():
    args = parse_args()
    print('Starting OAuth flow; browser will open. Copy the printed refresh_token and update your .env')
    login(is_test=args.sandbox)

if __name__ == '__main__':
    main()
