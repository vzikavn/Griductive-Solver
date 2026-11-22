"""Application entrypoint for the Griductive deduction puzzle game."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import LogicAgent
from engine import GameEngine
from gui import GriductiveGUI
from models import VALID_GRID_SIZES
from puzzle_loader import PuzzleLoader


def parse_args() -> argparse.Namespace:
    """Parses optional CLI flags for launching Griductive."""
    parser = argparse.ArgumentParser(
        prog="griductive",
        description="Griductive — Logical Grid Deduction Puzzle Game & DPLL SAT Solver",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=sorted(VALID_GRID_SIZES),
        default=4,
        help="Fallback N x N grid size when puzzles/ has no complete JSON (default: 4).",
    )
    parser.add_argument(
        "--puzzle",
        type=Path,
        default=None,
        help="Optional path to a specific JSON puzzle file.",
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
    puzzle_paths = PuzzleLoader.discover_puzzles()

    if args.puzzle is not None:
        initial_puzzle = PuzzleLoader.load_from_json(args.puzzle)
    elif puzzle_paths:
        initial_puzzle = PuzzleLoader.load_from_json(puzzle_paths[0])
    else:
        initial_puzzle = GameEngine.create_sample_puzzle(grid_size=args.size)

    engine = GameEngine(grid_size=initial_puzzle.grid_size)
    engine.load_puzzle(initial_puzzle)

    agent = LogicAgent()
    gui = GriductiveGUI(
        engine=engine,
        agent=agent,
        puzzle_paths=puzzle_paths,
        start_windowed_fullscreen=not args.windowed,
    )
    gui.run()


if __name__ == "__main__":
    main()