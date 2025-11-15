import os
import json
import base64
import requests
from loguru import logger

# Load credentials from .env
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(BASE_DIR, '.env')
TOK_PATH = os.path.join(BASE_DIR, '.tokens', 'schwab_token.json')

def load_env():
    env = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                if '=' in ln:
                    k, v = ln.split('=', 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def load_refresh_token():
    if os.path.exists(TOK_PATH):
        try:
            with open(TOK_PATH, 'r') as f:
                data = json.load(f)
                return data.get('refresh_token')
        except Exception:
            pass
    return None

def refresh_tokens():
    logger.info("Initializing...")

    env = load_env()
    app_key = env.get('SCHWAB_CLIENT_ID')
    app_secret = env.get('SCHWAB_CLIENT_SECRET')
    refresh_token_value = load_refresh_token()

    if not app_key or not app_secret or not refresh_token_value:
        logger.error("Missing credentials or refresh token")
        return None

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token_value,
    }
    headers = {
        "Authorization": f'Basic {base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()}',
        "Content-Type": "application/x-www-form-urlencoded",
    }

    refresh_token_response = requests.post(
        url="https://api.schwabapi.com/v1/oauth/token",
        headers=headers,
        data=payload,
    )
    if refresh_token_response.status_code == 200:
        logger.info("Retrieved new tokens successfully using refresh token.")
    else:
        logger.error(
            f"Error refreshing access token: {refresh_token_response.text}"
        )
        return None

    refresh_token_dict = refresh_token_response.json()

    logger.debug(refresh_token_dict)

    # Save new tokens to .tokens/schwab_token.json
    os.makedirs(os.path.dirname(TOK_PATH), exist_ok=True)
    with open(TOK_PATH, 'w') as f:
        json.dump(refresh_token_dict, f, indent=2)
    logger.info("Saved new tokens to .tokens/schwab_token.json")

    logger.info("Token dict refreshed.")

    return refresh_token_dict

if __name__ == "__main__":
    refresh_tokens()