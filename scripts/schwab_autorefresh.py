#!/usr/bin/env python3
"""Deprecated helper — token refresh/rotation is handled by `SchwabAuthClient` in `src/services/schwab_streamer.py`.

Use `scripts/schwab_token_manager.py rotate --force` to exercise rotate, or `run_schwab_streamer_with_tastytrade_symbols.py` to start the streamer.
"""
import sys
print('DEPRECATED: prefer the `SchwabAuthClient` API or `scripts/schwab_token_manager.py rotate`')
sys.exit(1)
