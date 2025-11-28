from discord.ext import commands


class ErrorHandler:
    @staticmethod
    async def handle_tt_command_error(ctx: commands.Context, error: Exception):
        """Handle errors from !tt commands."""
        if isinstance(error, commands.BadArgument):
            await ctx.send(
                "Invalid command arguments. Usage: !tt <buy/sell/flat> <symbol> <tp_ticks> [quantity] [mode]"
            )
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                "Missing required arguments. Usage: !tt <buy/sell/flat> <symbol> <tp_ticks> [quantity] [mode]"
            )
        elif isinstance(error, ValueError):
            await ctx.send(f"Validation error: {error}")
        else:
            await ctx.send(f"An unexpected error occurred: {error}")

    @staticmethod
    async def handle_validation_errors(ctx: commands.Context, errors: list):
        """Send validation error messages."""
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            await ctx.send(error_msg)

    @staticmethod
    async def handle_api_error(ctx: commands.Context, error: Exception):
        """Handle API-related errors."""
        await ctx.send(f"API error: {error}. Please try again later.")


# Error handling setup for bot
def setup_error_handling(bot: commands.Bot):
    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return  # Ignore unknown commands
        elif isinstance(error, commands.CheckFailure):
            await ctx.send("You don't have permission to use this command.")
        else:
            await ErrorHandler.handle_tt_command_error(ctx, error)
