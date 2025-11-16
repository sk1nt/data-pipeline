import asyncio
import logging
import os
from bot.config import create_config_from_env
from bot.trade_bot import TradeBot
from env_loader import load_env_file

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('discord.client').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)

    load_env_file()
    config = create_config_from_env()
    bot = TradeBot(config)

    print('Starting Discord trading bot...')
    print(f'ThinkorSwim: {"Enabled" if config.thinkorswim_allocation.enabled else "Disabled"} ({config.thinkorswim_allocation.percentage}%)')
    print(f'TastyTrade: {"Enabled" if config.tastytrade_allocation.enabled else "Disabled"} ({config.tastytrade_allocation.percentage}%)')

    if config.tastytrade_credentials:
        env_label = 'sandbox' if config.tastytrade_credentials.use_sandbox_environment else 'cert' if config.tastytrade_credentials.use_cert_environment else 'live'
        print(f'TastyTrade credentials loaded ({"dry run" if config.tastytrade_credentials.dry_run else "live"}) [{env_label}]')
        if config.tastytrade_credentials.account_whitelist:
            print(f'TastyTrade account whitelist: {", ".join(config.tastytrade_credentials.account_whitelist)}')
        if config.tastytrade_credentials.default_account:
            print(f'TastyTrade default account: {config.tastytrade_credentials.default_account}')
    else:
        if config.tastytrade_allocation.enabled:
            print('Warning: TastyTrade trading is enabled but credentials are missing')

    if config.allowed_channel_ids:
        print(f'Allowed Discord channels: {", ".join(str(cid) for cid in config.allowed_channel_ids)}')

    await bot.start(config.discord_token)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nBot stopped by user')
    except Exception as exc:
        print(f'Error starting bot: {exc}')
