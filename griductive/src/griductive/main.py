"""Application entrypoint for the Griductive deduction puzzle game.

Wires together the authoritative `GameEngine`, the isolated `LogicAgent`,
and the Pygame `GriductiveGUI` in Borderless Windowed Fullscreen mode.
"""

from __future__ import annotations

import argparse

from agent import LogicAgent
from engine import GameEngine
from gui import GriductiveGUI
from models import VALID_GRID_SIZES


def parse_args() -> argparse.Namespace:
    """Parses optional CLI flags for launching Griductive."""
    parser = argparse.ArgumentParser(
        prog="griductive",
        description="Griductive — Logical Grid Deduction Puzzle Game (Vice Sunset Edition)",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=sorted(VALID_GRID_SIZES),
        default=4,
        help="Initial N x N puzzle grid dimension (3, 4, or 5; default: 4).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Start in a standard resizable window instead of Borderless Windowed Fullscreen.",
    )
    return parser.parse_args()


def main() -> None:
    """Initializes GameEngine, LogicAgent, and launches GriductiveGUI."""
    args = parse_args()

    engine = GameEngine(grid_size=args.size)
    sample_puzzle = GameEngine.create_sample_puzzle(grid_size=args.size)
    engine.load_puzzle(sample_puzzle)

    agent = LogicAgent()

    gui = GriductiveGUI(
        engine=engine,
        agent=agent,
        start_windowed_fullscreen=not args.windowed,
    )
    gui.run()


if __name__ == "__main__":
    main()
