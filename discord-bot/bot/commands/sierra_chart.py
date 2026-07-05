"""sierra_chart.py -- Discord Cog for Sierra Chart DM trading context.

Users switch into SC context by typing !sc in a DM.  Subsequent shorthand
commands are routed here instead of the TastyTrade shorthand handler:

    b <qty> <tp_ticks>  --  market buy <qty> MNQ, TP at <tp_ticks> above fill
    s <qty> <tp_ticks>  --  market sell <qty> MNQ, TP at <tp_ticks> below fill
    f                   --  flatten all SC MNQ positions / cancel open TPs

Prefix commands (available in DM or allowed channels):
    !sc                 --  switch DM context to Sierra Chart, show help
    !sc dry on|off      --  toggle dry-run for this user
    !sc status          --  show current context and dry-run state

Signals are published to Redis channel sc:trade:signal.
Fill confirmations arrive on sc:trade:ack (handled in trade_bot.py).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from discord_bot.bot.trade_bot import TradeBot  # type: ignore


SC_TRADE_SIGNAL_CHANNEL = os.getenv("SC_TRADE_SIGNAL_CHANNEL", "sc:trade:signal")
SC_SYMBOL               = os.getenv("SC_DTC_SYMBOL", "MNQM26_FUT_CME")
SC_DISPLAY_SYMBOL       = os.getenv("SC_DOM_SYMBOL", "MNQ")
SC_TICK_SIZE            = float(os.getenv("MNQ_TICK_VALUE", "0.25"))

_HELP = (
    "**Sierra Chart context active** — commands sent directly (no `!` prefix needed):\n\n"
    "• `b <qty> <tp_ticks>` — market buy `<qty>` MNQ, TP `<tp_ticks>` ticks above fill\n"
    "• `s <qty> <tp_ticks>` — market sell `<qty>` MNQ, TP `<tp_ticks>` ticks below fill\n"
    "• `f` — flatten MNQ position + cancel open TPs\n\n"
    "**Context commands:**\n"
    "• `!sc dry on|off` — toggle dry-run (current session)\n"
    "• `!sc status` — show active context and dry-run state\n"
    "• `!tt` — switch back to TastyTrade context\n\n"
    "_Dry-run is **on** by default — signals are published but SC ignores them until dry-run is off._"
)


class SierraChartCog(commands.Cog, name="SierraChart"):
    """Handles Sierra Chart DM trading context and order dispatch."""

    def __init__(self, bot: "TradeBot") -> None:
        self.bot = bot
        self._signal_channel: str = getattr(
            bot.config, "sc_trade_signal_channel", SC_TRADE_SIGNAL_CHANNEL
        )

    # ------------------------------------------------------------------
    # Prefix commands
    # ------------------------------------------------------------------

    @commands.group(name="sc", invoke_without_command=True)
    async def sc_cmd(self, ctx: commands.Context) -> None:
        """Switch DM context to Sierra Chart."""
        if not self._is_privileged(ctx.author):
            await ctx.send("You are not authorized for Sierra Chart trading.")
            return
        user_id = ctx.author.id
        self.bot._dm_trading_context[user_id] = "sc"
        # Default dry-run on for new SC users unless already set
        if user_id not in self.bot._sc_dry_run_overrides:
            self.bot._sc_dry_run_overrides[user_id] = bool(
                getattr(self.bot.config, "sc_dry_run", True)
            )
        dry_state = "🟡 **dry-run ON**" if self.bot._sc_dry_run_overrides.get(user_id, True) else "🔴 **dry-run OFF**"
        await ctx.send(f"{_HELP}\n\nStatus: {dry_state}")

    @sc_cmd.command(name="dry")
    async def sc_dry(self, ctx: commands.Context, state: str) -> None:
        """Toggle dry-run: !sc dry on|off"""
        if not self._is_privileged(ctx.author):
            await ctx.send("Not authorized.")
            return
        state = state.strip().lower()
        if state not in ("on", "off"):
            await ctx.send("Usage: `!sc dry on` or `!sc dry off`")
            return
        dry = state == "on"
        self.bot._sc_dry_run_overrides[ctx.author.id] = dry
        label = "🟡 dry-run **ON** — signals publish but SC will not execute" if dry else "🔴 dry-run **OFF** — orders will execute in Sierra Chart"
        await ctx.send(label)

    @sc_cmd.command(name="status")
    async def sc_status(self, ctx: commands.Context) -> None:
        """Show current DM trading context."""
        ctx_name = self.bot._dm_trading_context.get(ctx.author.id, "tt").upper()
        dry = self.bot._sc_dry_run_overrides.get(ctx.author.id, True)
        dry_label = "🟡 ON" if dry else "🔴 OFF"
        await ctx.send(
            f"Context: **{ctx_name}** | Dry-run: {dry_label} | "
            f"Symbol: `{SC_DISPLAY_SYMBOL}` | Signal channel: `{self._signal_channel}`"
        )

    # ------------------------------------------------------------------
    # DM shorthand parser  (called from TradeBot.on_message)
    # ------------------------------------------------------------------

    async def handle_sc_shorthand(self, message: discord.Message) -> bool:
        """Parse and dispatch SC shorthand commands from a DM.

        Returns True if the message was handled (consumed), False otherwise.
        """
        content = message.content.strip()
        lower = content.lower()

        if not self._is_privileged(message.author):
            if re.match(r"^([bs]\s+\d+\s+\d+|f)$", lower):
                await message.channel.send("Not authorized for SC trading.")
                return True
            return False

        # f — flatten
        if lower == "f":
            return await self._dispatch(message, action="flatten", quantity=0, tp_ticks=0)

        # b <qty> <tp>  or  s <qty> <tp>
        m = re.match(r"^([bs])\s+(\d+)\s+(\d+)$", lower)
        if m:
            action = "buy" if m.group(1) == "b" else "sell"
            qty    = int(m.group(2))
            tp     = int(m.group(3))
            if qty <= 0:
                await message.channel.send("Quantity must be > 0.")
                return True
            if tp <= 0:
                await message.channel.send("TP ticks must be > 0.")
                return True
            return await self._dispatch(message, action=action, quantity=qty, tp_ticks=tp)

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        message: discord.Message,
        action: str,
        quantity: int,
        tp_ticks: int,
    ) -> bool:
        user_id  = message.author.id
        dry_run  = self.bot._sc_dry_run_overrides.get(user_id, True)
        ts_ms    = int(time.time() * 1000)
        req_id   = f"sc_{user_id}_{ts_ms}"

        payload = {
            "request_id":     req_id,
            "user_id":        str(user_id),
            "action":         action,
            "symbol":         SC_SYMBOL,
            "display_symbol": SC_DISPLAY_SYMBOL,
            "quantity":       quantity,
            "tp_ticks":       tp_ticks,
            "tick_size":      SC_TICK_SIZE,
            "dry_run":        dry_run,
            "ts_ms":          ts_ms,
        }

        try:
            self.bot.redis_client.publish(self._signal_channel, json.dumps(payload))
        except Exception as exc:
            await message.channel.send(f"❌ Failed to publish SC signal: {exc}")
            return True

        if action == "flatten":
            label = f"⚡ Flatten signal sent{' [DRY RUN]' if dry_run else ''} → `{self._signal_channel}`"
        else:
            tp_pts = tp_ticks * SC_TICK_SIZE
            label = (
                f"{'🟢 BUY' if action == 'buy' else '🔴 SELL'} {quantity} {SC_DISPLAY_SYMBOL} "
                f"@ market | TP {tp_ticks}t ({tp_pts:.2f} pts)"
                f"{' [DRY RUN]' if dry_run else ''} → `{self._signal_channel}`"
            )
        await message.channel.send(label)
        return True

    def _is_privileged(self, user: discord.abc.User) -> bool:
        if hasattr(self.bot, "_is_privileged_user_by_message"):
            # Duck-type: create a minimal object with .author
            class _M:
                author = user
            return self.bot._is_privileged_user_by_message(_M())
        return user.id in getattr(self.bot, "command_admin_ids", set())
