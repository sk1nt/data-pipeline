import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv(dotenv_path=Path('../.env'))
    sys.path.insert(0, '../discord-bot')
    from bot.tastytrade_client import TastyTradeClient

    try:
        TastyTradeClient()
        print('Client initialized successfully')
    except Exception as e:
        print(f'Failed to initialize TastyTrade client: {e}')


if __name__ == "__main__":
    main()
