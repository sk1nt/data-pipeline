import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bot.trade_bot import TradeBot


class Config(SimpleNamespace):
    pass


def test_format_supported_tickers_message_includes_aliases():
    bot = TradeBot(Config())
    sample = {
        'futures': ['NQ_NDX', 'ES_SPX'],
        'equity': ['SPY', 'QQQ']
    }
    msg = bot._format_supported_tickers_message(sample, alias_map=bot.ticker_aliases)
    assert 'Futures' in msg or 'futures' in msg
    assert 'NQ_NDX' in msg
    # Alias mapping should be present
    assert 'Aliases' in msg or 'NQ/' in msg
