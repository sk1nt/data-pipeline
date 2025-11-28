import discord
from discord.ext import commands
from ..services.auth_service import AuthService
from ..services.futures_order_parser import FuturesOrderParser
from ..services.futures_order_service import FuturesOrderService


class TradingBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

        # Allowed channels
        self.allowed_channels = [
            "XXXXXXXXXXXXXXXXXX",  # control channel
            "1255265167113978008",  # buy message channel
        ]

        self.futures_parser = FuturesOrderParser()
        self.futures_service = FuturesOrderService()

    async def on_ready(self):
        print(f"Bot logged in as {self.user}")

    async def on_message(self, message):
        # Ignore own messages
        if message.author == self.user:
            return

        # Process commands
        await self.process_commands(message)

    @commands.command(name="tt")
    async def tt_command(
        self,
        ctx,
        action: str = None,
        symbol: str = None,
        tp_ticks: float = None,
        quantity: int = 1,
        mode: str = "dry",
    ):
        """Handle !tt buy/sell/flat commands for futures."""

        # Check channel
        if str(ctx.channel.id) not in self.allowed_channels:
            await ctx.send("Command not allowed in this channel.")
            return

        # Check user permissions for futures
        if not AuthService.verify_user_for_futures(str(ctx.author.id)):
            await ctx.send("You do not have permission to place futures orders.")
            return

        # Validate action
        if action not in ["buy", "sell", "flat"]:
            await ctx.send(
                "Usage: !tt <buy/sell/flat> <symbol> <tp_ticks> [quantity] [mode]"
            )
            return

        # Parse and validate parameters
        try:
            parsed = self.futures_parser.parse(action, symbol, tp_ticks, quantity, mode)
        except ValueError as e:
            await ctx.send(f"Invalid parameters: {e}")
            return

        # Place order
        try:
            result = await self.futures_service.place_order(parsed, str(ctx.author.id))
            await ctx.send(f"Order placed: {result}")
        except Exception as e:
            await ctx.send(f"Order failed: {e}")


# Bot instance
bot = TradingBot()
