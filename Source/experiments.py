"""Experiment runner for Griductive (TV 2 deliverable).

Runs the no-guess deduction loop on the SAME puzzle set under every solver / encoding
configuration and writes one CSV row per (puzzle, configuration, repeat) with:

    primary / auxiliary variables, clause counts, SAT calls, decisions, propagations,
    backtracks, deduction steps and runtimes.

Failures (timeouts, stuck puzzles, exceptions, unimplemented components) are kept as rows
with a ``status`` value instead of being dropped, so the report can analyse them.

Configurations (cartesian product, all enabled by default):
    cardinality_mode : combinatorial | sequential   (TV 3's CNFEncoder option)
    use_pure_literal : False | True                 (TV 4's DPLLSolver option)

Every case runs in its own process so that a wall-clock timeout can be enforced without
changing ``solver.py``. Run from the ``Source/`` directory:

    python experiments.py
    python experiments.py --timeout 120 --out results/experiments.csv
    python experiments.py --sizes 3 4 --modes sequential --repeats 3
    python experiments.py --extra-dir tests/fixtures      # also run edge-case puzzles

How the numbers are counted
---------------------------
* ``sat_calls`` / ``decisions`` / ``propagations`` / ``backtracks`` / ``solver_runtime_sec``
  come from the *player-perspective* LogicAgent that drives the deduction loop
  (``LogicAgent.metrics``). ``GameEngine.submit_verdict`` re-verifies every move with its own
  verifier agent; those calls are reported separately in ``engine_verifier_sat_calls`` so
  nothing is double counted.
* ``primary_vars`` is N^2. ``aux_vars_peak`` / ``clauses_peak`` are the largest KB_t the agent
  encoded during the run, ``clauses_initial`` is the first one, and ``full_kb_*`` is the
  encoding of the complete clue set (the static size of the whole puzzle).
* ``wall_time_sec`` is the whole run, including the engine's verification.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import LogicAgent  # noqa: E402
from encoder import CNFEncoder  # noqa: E402
from engine import GameEngine  # noqa: E402
from models import HiddenStatus, VerdictResponseType  # noqa: E402
from puzzle_loader import PuzzleLoader  # noqa: E402
from solver import DPLLSolver  # noqa: E402

CSV_COLUMNS: tuple[str, ...] = (
    "puzzle_id",
    "file",
    "grid_size",
    "cardinality_mode",
    "pure_literal",
    "repeat",
    "status",
    "error",
    "primary_vars",
    "aux_vars_peak",
    "clauses_initial",
    "clauses_peak",
    "full_kb_aux_vars",
    "full_kb_clauses",
    "sat_calls",
    "decisions",
    "propagations",
    "backtracks",
    "solver_runtime_sec",
    "deduction_steps",
    "engine_verifier_sat_calls",
    "uniqueness",
    "wall_time_sec",
    "timeout_sec",
)

MODES: tuple[str, ...] = ("combinatorial", "sequential")


@dataclass(frozen=True, slots=True)
class Config:
    """One solver / encoder configuration under test."""

    cardinality_mode: str
    pure_literal: bool

    @property
    def label(self) -> str:
        return f"{self.cardinality_mode}/{'pure' if self.pure_literal else 'plain'}"


# --------------------------------------------------------------------------------------
# Single case (runs inside a child process)
# --------------------------------------------------------------------------------------
def _run_case(path_str: str, config: Config) -> dict[str, Any]:
    """Solves one puzzle with the no-guess loop and returns a result row (no timeout here)."""
    path = Path(path_str)
    puzzle = PuzzleLoader.load_from_json(path)
    row: dict[str, Any] = {
        "puzzle_id": puzzle.puzzle_id,
        "file": path.name,
        "grid_size": puzzle.grid_size,
        "cardinality_mode": config.cardinality_mode,
        "pure_literal": config.pure_literal,
        "primary_vars": puzzle.grid_size**2,
    }

    def make_agent() -> LogicAgent:
        return LogicAgent(
            encoder=CNFEncoder(cardinality_mode=config.cardinality_mode),  # type: ignore[arg-type]
            solver=DPLLSolver(use_pure_literal=config.pure_literal),
        )

    started = time.perf_counter()
    try:
        # Static size of the whole puzzle + uniqueness (separate agent, not mixed into metrics).
        all_clues = {cell.coord: cell.clue for cell in puzzle.cells}
        full_kb = CNFEncoder(
            cardinality_mode=config.cardinality_mode  # type: ignore[arg-type]
        ).encode_clue_set(puzzle.grid_size, all_clues)
        row["full_kb_aux_vars"] = full_kb.stats.auxiliary_vars
        row["full_kb_clauses"] = full_kb.stats.num_clauses
        row["uniqueness"], _ = make_agent().check_uniqueness(puzzle.grid_size, all_clues)

        # No-guess deduction loop (same order of calls as the GUI's Auto Solve).
        verifier = make_agent()
        engine = GameEngine(grid_size=puzzle.grid_size, verifier_agent=verifier)
        state = engine.load_puzzle(puzzle)
        agent = make_agent()
        agent.reset_session()
        agent.observe(state)

        aux_peak = 0
        clauses_peak = 0
        clauses_initial: int | None = None
        status = "OK"
        error = ""
        max_steps = puzzle.grid_size**2 + 1

        for _ in range(max_steps):
            if engine.is_solved:
                break
            move = agent.compute_next_auto_solve_step()
            kb = agent.last_encoded_kb
            if kb is not None:
                if clauses_initial is None:
                    clauses_initial = kb.stats.num_clauses
                aux_peak = max(aux_peak, kb.stats.auxiliary_vars)
                clauses_peak = max(clauses_peak, kb.stats.num_clauses)
            if move is None:
                status, error = "STUCK", "No forced deduction available (unsolved by logic alone)."
                break
            result = engine.submit_verdict(move.coord, move.proved_status)
            if result.response_type != VerdictResponseType.ACCEPTED:
                status = "REJECTED"
                error = f"{result.response_type} at {move.coord}: {result.message}"
                break
            state = result.public_state
            agent.observe(state)
            revealed = state.get_cell(move.coord).revealed_clue
            agent.record_accepted_move(move, revealed.text if revealed else None)
        else:
            if not engine.is_solved:
                status, error = "STUCK", "Step budget exhausted before the board was solved."

        metrics = agent.metrics
        row.update(
            status=status,
            error=error,
            aux_vars_peak=aux_peak,
            clauses_initial=clauses_initial if clauses_initial is not None else "",
            clauses_peak=clauses_peak,
            sat_calls=metrics.sat_calls,
            decisions=metrics.decisions,
            propagations=metrics.propagations,
            backtracks=metrics.backtracks,
            solver_runtime_sec=round(metrics.solver_runtime_sec, 6),
            deduction_steps=len(agent.trace),
            engine_verifier_sat_calls=verifier.metrics.sat_calls,
        )
    except NotImplementedError as exc:
        row.update(status="NOT_IMPLEMENTED", error=f"{exc}")
    except Exception as exc:  # noqa: BLE001 - every failure must end up in the CSV
        row.update(
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
        )
        row["traceback"] = traceback.format_exc(limit=3)
    row["wall_time_sec"] = round(time.perf_counter() - started, 6)
    return row


def _child_entry(conn: Connection, path_str: str, config: Config) -> None:
    """Process target: always sends exactly one dict back to the parent."""
    try:
        conn.send(_run_case(path_str, config))
    except BaseException as exc:  # noqa: BLE001
        conn.send({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


def run_case_with_timeout(path: Path, config: Config, timeout_sec: float) -> dict[str, Any]:
    """Runs one case in a fresh process; a case that exceeds the timeout is killed."""
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_entry, args=(child_conn, str(path), config), daemon=True)
    started = time.perf_counter()
    proc.start()
    child_conn.close()

    result: dict[str, Any] | None = None
    if parent_conn.poll(timeout_sec):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = {"status": "ERROR", "error": "Worker process died without a result."}
    else:
        result = {
            "status": "TIMEOUT",
            "error": f"Exceeded {timeout_sec:g}s wall-clock limit.",
            "wall_time_sec": round(time.perf_counter() - started, 6),
        }
    if proc.is_alive():
        proc.terminate()
    proc.join(timeout=5)
    parent_conn.close()

    # Fill identifying columns even when the worker never reported (timeout / crash).
    base = _identity_columns(path, config)
    for key, value in base.items():
        result.setdefault(key, value)
    return result


def _identity_columns(path: Path, config: Config) -> dict[str, Any]:
    cols: dict[str, Any] = {
        "puzzle_id": path.stem,
        "file": path.name,
        "cardinality_mode": config.cardinality_mode,
        "pure_literal": config.pure_literal,
    }
    try:
        puzzle = PuzzleLoader.load_from_json(path)
        cols.update(
            puzzle_id=puzzle.puzzle_id,
            grid_size=puzzle.grid_size,
            primary_vars=puzzle.grid_size**2,
        )
    except Exception:  # noqa: BLE001 - keep the row even for unreadable puzzle files
        pass
    return cols


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def collect_puzzles(
    puzzles_dir: Path | None, extra_dirs: list[Path], sizes: set[int] | None
) -> list[Path]:
    """The shared puzzle set: every complete JSON in puzzles/ (+ optional extra folders)."""
    paths = list(PuzzleLoader.discover_puzzles(puzzles_dir))
    for extra in extra_dirs:
        paths.extend(PuzzleLoader.discover_puzzles(extra))
    if sizes is None:
        return paths
    kept: list[Path] = []
    for path in paths:
        try:
            if PuzzleLoader.load_from_json(path).grid_size in sizes:
                kept.append(path)
        except Exception:  # noqa: BLE001 - broken files stay in so the failure is recorded
            kept.append(path)
    return kept


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Prints a compact per-configuration summary (counts of each status + median runtime)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = f"{row['cardinality_mode']}/{'pure' if row['pure_literal'] else 'plain'}"
        groups.setdefault(label, []).append(row)
    print("\n== Summary ==")
    for label, items in sorted(groups.items()):
        statuses: dict[str, int] = {}
        for item in items:
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
        ok_times = [float(i["wall_time_sec"]) for i in items if i["status"] == "OK"]
        median = f"{statistics.median(ok_times):.3f}s" if ok_times else "n/a"
        print(f"{label:<22} cases={len(items):<3} {statuses}  median_wall(OK)={median}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Griductive experiment runner (CSV output).")
    parser.add_argument(
        "--puzzles-dir", type=Path, default=None, help="Defaults to Source/puzzles."
    )
    parser.add_argument(
        "--extra-dir", type=Path, action="append", default=[],
        help="Additional folder of puzzle JSONs (e.g. edge cases). May be repeated.",
    )
    parser.add_argument("--sizes", type=int, nargs="*", default=None, help="Only these grid sizes.")
    parser.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--pure-literal", nargs="*", choices=("off", "on"), default=["off", "on"],
        help="DPLL pure-literal elimination settings to compare.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds per case.")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per case (runtime noise).")
    parser.add_argument(
        "--out", type=Path, default=ROOT_DIR / "results" / "experiments.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzles = collect_puzzles(
        args.puzzles_dir, args.extra_dir, set(args.sizes) if args.sizes else None
    )
    if not puzzles:
        sys.exit("No puzzles found. Put complete puzzle JSON files into Source/puzzles/.")

    configs = [
        Config(mode, pure == "on")
        for mode in args.modes
        for pure in args.pure_literal
    ]
    total = len(puzzles) * len(configs) * args.repeats
    print(
        f"{len(puzzles)} puzzles x {len(configs)} configs x {args.repeats} repeat(s) "
        f"= {total} runs"
    )

    rows: list[dict[str, Any]] = []
    done = 0
    for config in configs:
        for path in puzzles:
            for repeat in range(1, args.repeats + 1):
                row = run_case_with_timeout(path, config, args.timeout)
                row["repeat"] = repeat
                row["timeout_sec"] = args.timeout
                rows.append(row)
                done += 1
                print(
                    f"[{done}/{total}] {config.label:<22} {row.get('puzzle_id', path.stem):<16} "
                    f"{row['status']:<15} {row.get('wall_time_sec', '')}s",
                    flush=True,
                )
                # Write after every run so a crash or Ctrl+C never loses finished results.
                write_csv(rows, args.out)

    print_summary(rows)
    print(f"\nCSV written to {args.out}")


if __name__ == "__main__":
    main()
