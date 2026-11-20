"""Interactive CLI entry point for Ms. Minute AI Chatbot."""

from __future__ import annotations

import sys

from chatbot import ChatbotError, GeminiChatbot, NetworkTimeoutError, RateLimitError
from config import ConfigurationError, Settings

BOT_NAME = "Ms. Minute"

HELP_TEXT = f"""
Available commands:
  /help               Show this help message
  /clear              Clear conversation history
  /history            Display the current conversation history
  /system <prompt>    Update {BOT_NAME}'s system instruction (persona)
  /exit               Exit {BOT_NAME}
""".strip()


def print_banner(bot: GeminiChatbot) -> None:
    """Print startup information to the terminal."""
    print("=" * 60)
    print(f"{BOT_NAME} | Model: {bot.model_name}")
    print(f"Memory Window: {bot.memory.max_messages} messages")
    print("Type /help for commands, or /exit to quit.")
    print("=" * 60)


def handle_command(raw_input: str, bot: GeminiChatbot) -> bool:
    """Process a slash command.

    Args:
        raw_input: Raw user input starting with '/'.
        bot: Active GeminiChatbot instance.

    Returns:
        False if the application should exit, True otherwise.
    """
    parts = raw_input.strip().split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command == "/exit":
        print(f"{BOT_NAME}: Goodbye!")
        return False

    if command == "/help":
        print(f"\n{HELP_TEXT}\n")
        return True

    if command == "/clear":
        bot.clear_history()
        print("\n[Memory cleared]\n")
        return True

    if command == "/history":
        print(f"\n{bot.memory.format_history()}\n")
        return True

    if command == "/system":
        if not argument:
            print(f"\n[Current system instruction]: {bot.system_instruction}")
            print("Usage: /system <new system prompt>\n")
            return True
        bot.set_system_instruction(argument)
        print("\n[System instruction updated]\n")
        return True

    print(f"\n[Unknown command: {command}] Type /help to see available commands.\n")
    return True


def run_chat_loop(bot: GeminiChatbot) -> None:
    """Run the main interactive read-eval-print loop."""
    print_banner(bot)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            print(f"\n{BOT_NAME}: Goodbye!")
            break
        except KeyboardInterrupt:
            print("\n\n[Interrupted] Type /exit or press Ctrl+D to quit.")
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, bot):
                break
            continue

        print(f"\n{BOT_NAME}: ", end="", flush=True)
        try:
            for chunk in bot.stream_reply(user_input):
                print(chunk, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n[Response generation interrupted]")
        except (RateLimitError, NetworkTimeoutError, ChatbotError) as exc:
            print(f"\n[Error] {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Unexpected error] {exc}")


def main() -> int:
    """Application entry point."""
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    bot = GeminiChatbot(settings=settings)
    run_chat_loop(bot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())