# Quickstart: Automated Alert Trading (testing/dry-run)

Requirements
- Python 3.11, repo dependencies installed (`pip install -e .` to match dev tooling)
- Redis running and accessible via REDIS_* env
- Discord bot tokens set in `.env` if running the Discord bot; Tastytrade refresh token present

Steps
1. Configure `.env` with Tastytrade refresh token, allowlist IDs, and `TASTYTRADE_DRY_RUN=true` for safe testing. Keep `.env.back` out of git.
2. Start Redis locally (or point `REDIS_HOST` to an existing instance).
3. Run the orchestration API + monitoring UI:
   ```bash
   python data-pipeline.py --host 0.0.0.0 --port 8877
   ```
4. Start the Discord bot (optional if using admin API only):
   ```bash
   python discord-bot/run_discord_bot.py
   ```
5. Use the admin API to simulate an alert (dry-run recommended):
   ```bash
   curl -X POST http://localhost:8877/admin/alerts/process \
     -H 'Content-Type: application/json' \
     -H "x-api-key: $ADMIN_API_KEY" \
     -d '{"message":"Alert: BTO UBER 78p 12/05 @ 0.75", "channel_id":"123", "user_id":"456", "dry_run":true}'
   ```
6. Check recent audits via admin API or Redis `lrange audit:automated_alerts 0 50` to confirm the event was logged.

Environment toggles & how to run safe vs live
- `TASTYTRADE_DRY_RUN=true` and `TASTYTRADE_USE_SANDBOX=true` — safe testing mode, no live orders placed and sandbox account used when possible.
- `TASTYTRADE_DRY_RUN=false` and `TASTYTRADE_USE_SANDBOX=false` — live mode; verify `!tt auth status` and rotate refresh tokens as needed.
- Allowlist: `ALLOWED_USERS`, `ALLOWED_CHANNELS` in `.env`; admin API requires `ADMIN_API_KEY` header.

Testing notes
- Use the admin API to validate the full alert flow without sending Discord messages when testing.
- Local E2E test harness: `pytest tests/e2e/test_alert_e2e_flow.py -q` verifies alert processing and audit logging with simulated broker responses.
- Pending coverage: entry→fill→exit integration (tests/integration/test_alert_to_exit_flow.py) and insufficient buying power notification path.

Expected responses
- Successful dry-run: JSON includes parsed alert, computed quantity, and simulated order IDs; audit entry is written.
- Insufficient buying power: `AutomatedOptionsService` returns an error payload, alerts channel/admin via notification hook, and still writes an audit record with `error` populated; no exit order is attempted.
- Default account: set `TASTYTRADE_ACCOUNT` (current: `5WT31673`) and ensure allowlist (`TASTYTRADE_ACCOUNT_WHITELIST`) aligns.

Notes
- To go live, set `TASTYTRADE_DRY_RUN=false`, ensure a valid refresh token, and test with restricted allowlist users and channels.
- For production safety, rotate the TastyTrade refresh token and verify `!tt auth status` before toggling live mode.
