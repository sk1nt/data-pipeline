# Quickstart: Automated Alert Trading (testing/dry-run)

Requirements
- Python 3.11, repo dependencies installed (pip install -r requirements.txt)
- Redis running and accessible via REDIS_* env
- Discord bot tokens set in `.env` if running the Discord bot

Steps
1. Configure `.env` with TastyTrade refresh token and `TASTYTRADE_DRY_RUN=true` for safe testing.
2. Start Redis locally (or point `REDIS_HOST` to an existing instance).
3. Start the Discord bot (or use the admin API) in dry-run mode:
   ```bash
   python discord-bot/run_discord_bot.py
   ```
4. Use the admin API to simulate an alert:
   ```bash
   curl -X POST http://localhost:8877/admin/alerts/process -H 'Content-Type: application/json' -d '{"message":"Alert: BTO UBER 78p 12/05 @ 0.75", "channel_id":"123", "user_id":"456", "dry_run":true}'
   ```
5. Check recent audits via admin API or Redis `lrange audit:automated_alerts 0 50` to confirm the event was logged.

Notes
- To go live, set `TASTYTRADE_DRY_RUN=false`, ensure a valid refresh token, and test with restricted allowlist users and channels.
- For production safety, rotate the TastyTrade refresh token and verify `!tt auth status` before toggling live mode.
