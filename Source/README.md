# Griductive Solver

A modular, no-guess Griductive deduction puzzle game and automated logic solver built for **CSC14003 — Introduction to Artificial Intelligence (Project 2)**. The system combines propositional logic, automatic CNF encoding, and a custom DPLL SAT solver under a strict architectural separation between game truth and public knowledge.

## Project Structure

```text
Source/
├── main.py               # Top-level application entrypoint
├── pyproject.toml        # Project metadata and linter/type-checker/test configuration
├── requirements.txt      # Runtime and testing dependencies
├── puzzles/              # JSON puzzle definitions (3x3, 4x4, 5x5)
├── src/
│   ├── models.py         # Strictly typed domain models & extensions
│   ├── encoder.py        # Automatic CNFEncoder & direct SemanticEvaluator
│   ├── solver.py         # Custom DPLL SAT solver with runtime/search metrics
│   ├── agent.py          # Isolated LogicAgent (entailment, hint, trace, uniqueness)
│   ├── engine.py         # Authoritative GameEngine controlling hidden board state
│   ├── puzzle_loader.py  # JSON puzzle loader and directory discovery
│   └── gui.py            # Interactive Pygame presentation layer
└── tests/
    ├── test_models.py
    ├── test_engine_boundary.py
    └── test_agent_solver.py
```

## Architecture

- **`src/models.py`**: Strictly typed domain models (`CharacterCell`, `Clue`, `ClueRegion`, `PublicCellView`, `PublicKnowledgeState`, `VerdictResult`, `CellEntailment`, `DeductionStep`, `PuzzleDefinition`). Supports the 6 core clue templates (`FACT`, `SAME`, `DIFFERENT`, `EXACTLY`, `AT_LEAST`, `AT_MOST`) and 2 extensions (advanced regions `CORNERS`/`BOUNDARY` and `PARITY` constraints).
- **`src/encoder.py`**: Houses `CNFEncoder` (deterministic variable mapping, region/parameter validation, automatic clue-to-CNF compilation, and variable/clause statistics) alongside `SemanticEvaluator` for direct semantic verification without CNF.
- **`src/solver.py`**: Custom `DPLLSolver` implementing unit propagation, conflict detection, deterministic variable selection, recursive backtracking, optional pure-literal elimination, and search instrumentation (`decisions`, `propagations`, `backtracks`, `runtime_sec`).
- **`src/agent.py`**: Isolated `LogicAgent` operating exclusively on immutable `PublicKnowledgeState` snapshots. Performs SAT-refutation entailment checks ($KB_t \wedge \neg C_i$ / $KB_t \wedge C_i$), deterministic no-guess deductions, hint generation, deduction tracing, and complete-puzzle uniqueness verification.
- **`src/engine.py`**: Authoritative `GameEngine` owning ground-truth hidden labels and unrevealed clues. Never exposes `CharacterCell` outside the engine and validates verdicts (`ACCEPTED`, `NOT_PROVABLE`, `CONTRADICTED`) strictly against public knowledge.
- **`src/puzzle_loader.py`**: Loads and validates structured JSON scenarios from `puzzles/`, supporting both alphanumeric (`"A1"`) and 0-based (`[0, 0]`) coordinates.
- **`src/gui.py`**: Interactive `pygame` presentation layer featuring Borderless Windowed Fullscreen (`pygame.NOFRAME`), a 7-stop sunset gradient theme, clue region spotlight highlighting, and `Load`, `Restart`, `Hint`, and continuous `Auto Solve` controls.
- **`main.py`**: Clean top-level CLI entrypoint.

## Quick Start

Requires **Python 3.12+**. Run all commands from the `Source/` directory:

```bash
# 1. Install runtime dependencies
pip install -r requirements.txt

# (Optional) Install editable package with dev tools (mypy, ruff, pytest)
pip install -e ".[dev]"

# 2. Run in Borderless Windowed Fullscreen (default)
python main.py

# 3. Run in a standard resizable window
python main.py --windowed --size 4

# 4. Launch with a specific JSON puzzle file
python main.py --windowed --puzzle puzzles/sample_3x3.json
```

## Controls

- **Left-click a face-down card**: Select a suspect and choose `INNOCENT` or `CRIMINAL`.
- **Right-click a face-down card**: Quickly submit `CRIMINAL` for that suspect.
- **Left-click a revealed card**: Highlight (or clear highlight for) all cells referenced by its clue.
- **`F11` or `Alt+Enter`**: Toggle between Borderless Windowed Fullscreen and Resizable Windowed mode.
- **`ESC`**: Deselect the current card or exit Borderless Windowed Fullscreen.

## Development & Verification

```bash
# Lint checks
ruff check .

# Strict static type checking
python -m mypy src tests main.py

# Unit tests
pytest
```