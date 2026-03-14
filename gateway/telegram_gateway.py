"""
gateway/telegram_gateway.py — Optional Telegram bot interface for NeuroClaw_Bot.

Allows you to control the coding agent remotely from your phone.
Architecture: Phone → Telegram → Bot → Agent Loop → Project Files

Setup:
1. Create a bot with @BotFather → get TOKEN
2. Get your Telegram user ID from @userinfobot
3. Set telegram_token and telegram_allowed_user_ids in config.yaml
4. Run: python agent.py --telegram
"""
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent_loop_dual import DualModelAgentLoop

logger = logging.getLogger(__name__)


class TelegramGateway:
    """
    Connects the NeuroClaw_Bot agent to a Telegram bot.

    Security model (same as OpenClaw's Telegram channel):
    - Only messages from ALLOWED_USER_IDS are processed.
    - All other messages receive an unauthorized response.
    """

    def __init__(self, token: str, agent: "AgentLoop", allowed_user_ids: list[int]):
        self.token = token
        self.agent = agent
        self.allowed_user_ids = set(allowed_user_ids)

    def run(self):
        """Start the Telegram bot (blocking)."""
        try:
            from telegram import Update  # type: ignore
            from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes  # type: ignore
        except ImportError:
            raise ImportError(
                "python-telegram-bot is not installed. Run:\n"
                "  pip install python-telegram-bot"
            )

        async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id
            username = update.effective_user.username or str(user_id)

            if user_id not in self.allowed_user_ids:
                await update.message.reply_text("⛔ Unauthorized. Your user ID is not in the allow list.")
                logger.warning(f"Unauthorized access attempt from user_id={user_id}")
                return

            task = update.message.text
            logger.info(f"[Telegram] Task from {username}: {task[:80]}")
            await update.message.reply_text("🤔 Working on it… (this may take a minute)")

            try:
                # Run blocking agent in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.agent.run, task)
            except Exception as e:
                result = f"❌ Agent error: {type(e).__name__}: {e}"

            # Telegram message limit is 4096 chars
            MAX_LEN = 4000
            for i in range(0, len(result), MAX_LEN):
                await update.message.reply_text(result[i: i + MAX_LEN])

        async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "👋 I'm NeuroClaw_Bot — your local AI coding agent.\n"
                "Send me a coding task and I'll work on your project files.\n\n"
                "Examples:\n• Add type hints to all functions in utils.py\n"
                "• Fix the bug in main.py\n• List all files in the project"
            )

        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", handle_start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print("[Telegram] 🤖 Bot is running. Send messages to your bot on Telegram.")
        app.run_polling(drop_pending_updates=True)
