# Griductive (Vice Sunset Edition)

A modular grid-deduction puzzle game built for an Artificial Intelligence course project.

## Architecture
- **`models.py`**: Strictly typed domain models (`CharacterCell`, `Clue`, `ClueRegion`, `PublicCellView`, `PublicKnowledgeState`, `VerdictResult`).
- **`engine.py`**: Authoritative `GameEngine` owning ground-truth hidden labels and unrevealed clues. Never exposes `CharacterCell` outside the engine.
- **`agent.py`**: Isolated `LogicAgent` that receives only `PublicKnowledgeState` snapshots. Contains `NotImplementedError` placeholders for CNF encoding and DPLL SAT solving.
- **`gui.py`**: Interactive `pygame` presentation layer with Borderless Windowed Fullscreen (`pygame.NOFRAME` at desktop resolution), GTA 6 Vice Sunset gradient, and custom character-spaced Pricedown typography.
- **`main.py`**: Clean application entrypoint (`python3 src/griductive/main.py`).

## Quick Start

```bash
# Install dependencies
pip install pygame mypy ruff pytest

# Run in Borderless Windowed Fullscreen (default)
PYTHONPATH=src/griductive python3 src/griductive/main.py

# Run in a standard resizable window
PYTHONPATH=src/griductive python3 src/griductive/main.py --windowed --size 4

# Run strict type checking
mypy --strict src/griductive
```
