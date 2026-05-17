"""Tiny interactive REPL for the QuickPay agent.

Run `python cli.py` to chat with the agent locally. Type `exit`, `quit`, or
Ctrl-D to end the session.
"""

from __future__ import annotations

import sys

from agent import Agent


def main() -> int:
    agent = Agent()
    print("QuickPay agent ready. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                user_input = input("you > ").strip()
            except EOFError:
                print()
                return 0
            if user_input.lower() in {"exit", "quit"}:
                return 0
            if not user_input:
                continue
            response = agent.next(user_input)
            print(f"bot > {response.get('message', '')}\n")
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
