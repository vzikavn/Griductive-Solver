"""Cross-check CNFEncoder against the direct SemanticEvaluator (TV 2 deliverable).

The CNF produced by ``CNFEncoder`` is compared with ``SemanticEvaluator`` on truth tables:

* Per clue: for every clue type, region shape, target status, count and cardinality mode,
  ``exists aux. CNF(primary, aux)`` must equal ``evaluate_clue(primary)``.
  Small regions are checked exhaustively; larger ones with seeded, targeted sampling
  (assignments whose hit count is k-1, k, k+1) so the boundary cases are always covered.
* Per puzzle: the hidden truth satisfies every clue and the whole CNF, uniqueness holds on
  3x3 / 4x4 levels by brute force, and random assignments agree between both sides.
* Entailment classes (CRIMINAL / INNOCENT / UNKNOWN / INCONSISTENT) on hand-built KBs.

The checker uses its own tiny DPLL (``_cnf_sat``) so these tests do not depend on TV 4's solver.
Only ``test_agent_matches_reference_entailment`` touches ``LogicAgent`` and it is skipped while
``DPLLSolver`` is still a placeholder.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from agent import LogicAgent
from encoder import CNFEncoder, EncodedKB, SemanticEvaluator
from models import (
    Clause,
    CellEntailment,
    Clue,
    ClueRegion,
    ClueType,
    Coordinate,
    HiddenStatus,
    PublicCellView,
    PublicKnowledgeState,
    RegionType,
    RevealedStatus,
)
from puzzle_loader import PuzzleLoader
from solver import DPLLSolver

CRIM = HiddenStatus.CRIMINAL
INNO = HiddenStatus.INNOCENT
MODES = ("combinatorial", "sequential")
MAX_DIRECT_CLAUSES = 40_000
PUZZLES_DIR = Path(__file__).resolve().parent.parent / "puzzles"


# --------------------------------------------------------------------------------------
# Independent CNF reference checker
# --------------------------------------------------------------------------------------
def _assign(clauses: list[set[int]], true_lits: set[int]) -> list[set[int]] | None:
    """Sets every literal in `true_lits` True in one pass. Returns None on a conflict."""
    if any(-lit in true_lits for lit in true_lits):
        return None
    false_lits = {-lit for lit in true_lits}
    out: list[set[int]] = []
    for clause in clauses:
        if not clause.isdisjoint(true_lits):
            continue
        if not clause.isdisjoint(false_lits):
            clause = clause - false_lits
            if not clause:
                return None
        out.append(clause)
    return out


def _cnf_sat(clauses: list[set[int]]) -> bool:
    """Minimal DPLL (batched unit propagation + branching) used as an independent oracle."""
    while True:
        units = {next(iter(c)) for c in clauses if len(c) == 1}
        if not units:
            break
        reduced = _assign(clauses, units)
        if reduced is None:
            return False
        clauses = reduced
    if not clauses:
        return True
    lit = next(iter(clauses[0]))
    for choice in (lit, -lit):
        reduced = _assign(clauses, {choice})
        if reduced is not None and _cnf_sat(reduced):
            return True
    return False


def _sat_under(
    clauses: tuple[Clause, ...] | list[Clause],
    var_map: dict[Coordinate, int],
    values: dict[Coordinate, bool],
) -> bool:
    """True iff some auxiliary assignment satisfies the CNF given the primary values."""
    true_lits = {var_map[c] if v else -var_map[c] for c, v in values.items()}
    reduced = _assign([set(c) for c in clauses], true_lits)
    return reduced is not None and _cnf_sat(reduced)


def _to_assignment(
    grid_size: int, values: dict[Coordinate, bool]
) -> dict[Coordinate, HiddenStatus]:
    """Completes a partial boolean assignment (missing cells -> INNOCENT)."""
    return {
        (r, c): CRIM if values.get((r, c), False) else INNO
        for r in range(grid_size)
        for c in range(grid_size)
    }


def _encode_one(
    encoder: CNFEncoder, owner: Coordinate, clue: Clue, grid_size: int
) -> tuple[list[Clause], dict[Coordinate, int]]:
    var_map = CNFEncoder.build_var_map(grid_size)
    clauses, _ = encoder.encode_single_clue(
        owner=owner,
        clue=clue,
        grid_size=grid_size,
        var_map=var_map,
        next_aux_var=grid_size * grid_size + 1,
    )
    return clauses, var_map


# --------------------------------------------------------------------------------------
# Per-clue truth-table equivalence
# --------------------------------------------------------------------------------------
EXHAUSTIVE_LIMIT = 9  # regions up to 2^9 = 512 assignments are enumerated completely

REGIONS: dict[str, tuple[int, Coordinate, ClueRegion]] = {
    "row": (4, (1, 1), ClueRegion(RegionType.ROW, index=1)),
    "row_excl": (4, (1, 1), ClueRegion(RegionType.ROW, index=1, exclude_self=True)),
    "col": (4, (0, 2), ClueRegion(RegionType.COLUMN, index=2)),
    "neighbors": (4, (2, 2), ClueRegion(RegionType.NEIGHBORS, center=(1, 1))),
    "corners": (5, (2, 2), ClueRegion(RegionType.CORNERS)),
    "boundary5": (5, (2, 2), ClueRegion(RegionType.BOUNDARY)),
    "boundary6": (6, (2, 2), ClueRegion(RegionType.BOUNDARY, exclude_self=False)),
    "explicit": (
        4,
        (0, 0),
        ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((0, 0), (1, 2), (3, 1), (2, 2))),
    ),
}


def _region_assignments(
    cells: tuple[Coordinate, ...], k: int | None, rng: random.Random
) -> list[dict[Coordinate, bool]]:
    """Exhaustive for small regions, targeted + random samples for large ones."""
    m = len(cells)
    if m <= EXHAUSTIVE_LIMIT:
        return [
            dict(zip(cells, bits, strict=True))
            for bits in itertools.product((False, True), repeat=m)
        ]
    out: list[dict[Coordinate, bool]] = []
    hit_levels = {0, m}
    if k is not None:
        hit_levels |= {h for h in (k - 1, k, k + 1) if 0 <= h <= m}
    for hits in sorted(hit_levels):
        for _ in range(8):
            chosen = set(rng.sample(cells, hits))
            out.append({c: c in chosen for c in cells})
    for _ in range(30):
        out.append({c: rng.random() < 0.5 for c in cells})
    return out


def _counts_to_test(size: int, mode: str) -> list[int]:
    """All k for small regions / sequential mode; extreme k only for big combinatorial CNF.

    Direct encoding needs C(m, k + 1) clauses, so mid-range k on a 16-20 cell region would
    produce 10^4..10^5 clauses per clue (this blow-up is exactly what the experiments measure).
    """
    if mode == "combinatorial" and size > 10:
        return sorted({k for k in (0, 1, 2, size - 2, size - 1, size) if 0 <= k <= size})
    return list(range(size + 1))


def _check_clue(
    encoder: CNFEncoder,
    grid_size: int,
    owner: Coordinate,
    clue: Clue,
    rng: random.Random,
) -> None:
    clauses, var_map = _encode_one(encoder, owner, clue, grid_size)
    cells = CNFEncoder.validate_and_resolve(clue, grid_size, owner)
    for values in _region_assignments(cells, clue.count, rng):
        expected = SemanticEvaluator.evaluate_clue(
            clue, _to_assignment(grid_size, values), grid_size, owner
        )
        actual = _sat_under(clauses, var_map, values)
        assert actual == expected, (
            f"CNF/evaluator mismatch ({encoder.cardinality_mode}) for {clue.clue_type} "
            f"{clue.target_status} count={clue.count} on {values}: "
            f"cnf={actual} evaluator={expected}"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", [CRIM, INNO])
@pytest.mark.parametrize("clue_type", [ClueType.EXACTLY, ClueType.AT_LEAST, ClueType.AT_MOST])
@pytest.mark.parametrize("region_name", list(REGIONS))
def test_cardinality_clues_match_evaluator(
    mode: str, target: HiddenStatus, clue_type: ClueType, region_name: str
) -> None:
    grid_size, owner, region = REGIONS[region_name]
    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    size = len(CNFEncoder.validate_and_resolve(Clue(ClueType.FACT, region), grid_size, owner))
    rng = random.Random(f"{mode}-{clue_type}-{region_name}-{target}")
    # k = 0 and k = |R| are the edge cases the spec calls out explicitly.
    for k in _counts_to_test(size, mode):
        clue = Clue(clue_type=clue_type, region=region, target_status=target, count=k)
        _check_clue(encoder, grid_size, owner, clue, rng)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", [CRIM, INNO])
@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("region_name", list(REGIONS))
def test_parity_clues_match_evaluator(
    mode: str, target: HiddenStatus, parity: int, region_name: str
) -> None:
    grid_size, owner, region = REGIONS[region_name]
    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    clue = Clue(ClueType.PARITY, region, target_status=target, count=parity)
    rng = random.Random(f"parity-{mode}-{region_name}-{target}-{parity}")
    _check_clue(encoder, grid_size, owner, clue, rng)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", [CRIM, INNO])
@pytest.mark.parametrize("clue_type", [ClueType.FACT, ClueType.SAME, ClueType.DIFFERENT])
@pytest.mark.parametrize("region_name", ["row", "col", "neighbors", "corners", "explicit"])
def test_relational_clues_match_evaluator(
    mode: str, target: HiddenStatus, clue_type: ClueType, region_name: str
) -> None:
    grid_size, owner, region = REGIONS[region_name]
    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    clue = Clue(clue_type, region, target_status=target)
    _check_clue(encoder, grid_size, owner, clue, random.Random(f"rel-{region_name}"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", [CRIM, INNO])
def test_two_cell_pair_clues_match_evaluator(mode: str, target: HiddenStatus) -> None:
    """The 2-cell SAME / DIFFERENT / FACT form used by the shipped levels."""
    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    pair = ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((0, 1), (2, 2)))
    for clue_type in (ClueType.SAME, ClueType.DIFFERENT):
        _check_clue(
            encoder,
            3,
            (0, 0),
            Clue(clue_type, pair, target_status=target),
            random.Random(0),
        )
    single = ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((1, 1),))
    _check_clue(
        encoder, 3, (0, 0), Clue(ClueType.FACT, single, target_status=target), random.Random(0)
    )


# --------------------------------------------------------------------------------------
# SemanticEvaluator unit tests (no CNF involved)
# --------------------------------------------------------------------------------------
def _row_clue(clue_type: ClueType, count: int | None, **kwargs: object) -> Clue:
    return Clue(
        clue_type=clue_type,
        region=ClueRegion(RegionType.ROW, index=0),
        count=count,
        **kwargs,  # type: ignore[arg-type]
    )


ROW_C_I_C = {(0, 0): CRIM, (0, 1): INNO, (0, 2): CRIM}


def test_evaluator_cardinality_semantics() -> None:
    ev = SemanticEvaluator.evaluate_clue
    assert ev(_row_clue(ClueType.EXACTLY, 2), ROW_C_I_C, 3)
    assert not ev(_row_clue(ClueType.EXACTLY, 1), ROW_C_I_C, 3)
    assert ev(_row_clue(ClueType.AT_LEAST, 2), ROW_C_I_C, 3)
    assert not ev(_row_clue(ClueType.AT_LEAST, 3), ROW_C_I_C, 3)
    assert ev(_row_clue(ClueType.AT_MOST, 2), ROW_C_I_C, 3)
    assert not ev(_row_clue(ClueType.AT_MOST, 1), ROW_C_I_C, 3)


def test_evaluator_innocent_target_counts_innocents() -> None:
    ev = SemanticEvaluator.evaluate_clue
    clue = _row_clue(ClueType.EXACTLY, 1, target_status=INNO)
    assert ev(clue, ROW_C_I_C, 3)
    assert not ev(_row_clue(ClueType.EXACTLY, 2, target_status=INNO), ROW_C_I_C, 3)


def test_evaluator_edge_counts_zero_and_full() -> None:
    ev = SemanticEvaluator.evaluate_clue
    all_inno = {(0, c): INNO for c in range(3)}
    all_crim = {(0, c): CRIM for c in range(3)}
    assert ev(_row_clue(ClueType.EXACTLY, 0), all_inno, 3)
    assert ev(_row_clue(ClueType.AT_MOST, 0), all_inno, 3)
    assert not ev(_row_clue(ClueType.AT_MOST, 0), all_crim, 3)
    assert ev(_row_clue(ClueType.EXACTLY, 3), all_crim, 3)
    assert ev(_row_clue(ClueType.AT_LEAST, 3), all_crim, 3)
    assert ev(_row_clue(ClueType.AT_LEAST, 0), all_inno, 3)


def test_evaluator_parity_semantics() -> None:
    ev = SemanticEvaluator.evaluate_clue
    assert ev(_row_clue(ClueType.PARITY, 0), ROW_C_I_C, 3)  # 2 criminals -> even
    assert not ev(_row_clue(ClueType.PARITY, 1), ROW_C_I_C, 3)
    assert ev(_row_clue(ClueType.PARITY, 1, target_status=INNO), ROW_C_I_C, 3)  # 1 innocent


def test_evaluator_fact_same_different() -> None:
    ev = SemanticEvaluator.evaluate_clue
    pair = ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((0, 0), (0, 2)))
    assert ev(Clue(ClueType.SAME, pair), ROW_C_I_C, 3)
    assert not ev(Clue(ClueType.DIFFERENT, pair), ROW_C_I_C, 3)
    mixed = ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((0, 0), (0, 1)))
    assert ev(Clue(ClueType.DIFFERENT, mixed), ROW_C_I_C, 3)
    assert not ev(Clue(ClueType.SAME, mixed), ROW_C_I_C, 3)
    single = ClueRegion(RegionType.EXPLICIT_LIST, explicit_coords=((0, 1),))
    assert ev(Clue(ClueType.FACT, single, target_status=INNO), ROW_C_I_C, 3)
    assert not ev(Clue(ClueType.FACT, single, target_status=CRIM), ROW_C_I_C, 3)
    # On a 3-cell region DIFFERENT means "not all equal".
    assert ev(_row_clue(ClueType.DIFFERENT, None), ROW_C_I_C, 3)
    assert not ev(_row_clue(ClueType.DIFFERENT, None), {(0, c): INNO for c in range(3)}, 3)


def test_evaluator_exclude_self_and_validation() -> None:
    ev = SemanticEvaluator.evaluate_clue
    clue = Clue(
        ClueType.EXACTLY,
        ClueRegion(RegionType.ROW, index=0, exclude_self=True),
        count=1,
    )
    # Owner (0, 0) is excluded, so only (0, 1) and (0, 2) are counted: 1 criminal.
    assert ev(clue, ROW_C_I_C, 3, owner_coord=(0, 0))
    assert not ev(clue, ROW_C_I_C, 3)  # no owner -> nothing excluded -> 2 criminals
    with pytest.raises(ValueError):
        ev(_row_clue(ClueType.EXACTLY, 4), ROW_C_I_C, 3)  # k out of range
    with pytest.raises(ValueError):
        ev(_row_clue(ClueType.EXACTLY, 1), {(0, 0): CRIM}, 3)  # incomplete assignment


# --------------------------------------------------------------------------------------
# Whole-puzzle equivalence on the shipped levels
# --------------------------------------------------------------------------------------
PUZZLE_PATHS = PuzzleLoader.discover_puzzles(PUZZLES_DIR)


@pytest.mark.parametrize("path", PUZZLE_PATHS, ids=[p.stem for p in PUZZLE_PATHS])
def test_puzzle_cnf_agrees_with_evaluator(path: Path) -> None:
    puzzle = PuzzleLoader.load_from_json(path)
    n = puzzle.grid_size
    clues = {cell.coord: cell.clue for cell in puzzle.cells}
    truth = {cell.coord: cell.hidden_status for cell in puzzle.cells}
    truth_bool = {coord: status == CRIM for coord, status in truth.items()}

    # (1) the hidden truth satisfies every clue under the evaluator
    assert all(
        SemanticEvaluator.evaluate_clue(c, truth, n, owner) for owner, c in clues.items()
    ), f"{path.name}: hidden truth violates a clue"

    for mode in MODES:
        encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
        kb: EncodedKB = encoder.encode_clue_set(n, clues)

        if kb.stats.num_clauses > MAX_DIRECT_CLAUSES:
            continue  # direct encoding too large for the pure-Python oracle; sequential covers it

        # (2) the hidden truth satisfies the CNF
        assert _sat_under(kb.clauses, kb.var_map, truth_bool)

        # (3) CNF == evaluator on the truth, its neighbours and random assignments
        rng = random.Random(f"{path.stem}-{mode}")
        coords = sorted(truth_bool)
        samples = [dict(truth_bool)]
        for _ in range(25):
            flipped = dict(truth_bool)
            for coord in rng.sample(coords, rng.choice([1, 1, 2])):
                flipped[coord] = not flipped[coord]
            samples.append(flipped)
        for _ in range(25):
            samples.append({coord: rng.random() < 0.5 for coord in coords})
        for values in samples:
            assignment = _to_assignment(n, values)
            expected = all(
                SemanticEvaluator.evaluate_clue(c, assignment, n, owner)
                for owner, c in clues.items()
            )
            assert _sat_under(kb.clauses, kb.var_map, values) == expected

        # (4) uniqueness: blocking the truth leaves the CNF unsatisfiable
        if n <= 4:
            blocking = {-kb.var_map[c] if truth_bool[c] else kb.var_map[c] for c in coords}
            assert not _cnf_sat([set(c) for c in kb.clauses] + [blocking]), (
                f"{path.name}: CNF admits a second model ({mode})"
            )

    # (5) brute force with the evaluator alone: the truth is the only model (3x3 / 4x4)
    if n <= 4:
        coords = sorted(truth_bool)
        models = 0
        for bits in itertools.product((False, True), repeat=len(coords)):
            assignment = {c: CRIM if b else INNO for c, b in zip(coords, bits, strict=True)}
            if all(SemanticEvaluator.evaluate_clue(c, assignment, n, o) for o, c in clues.items()):
                models += 1
                assert assignment == truth
        assert models == 1


# --------------------------------------------------------------------------------------
# Entailment classes: CRIMINAL / INNOCENT / UNKNOWN / INCONSISTENT
# --------------------------------------------------------------------------------------
def _row0(count: int, target: HiddenStatus = CRIM) -> Clue:
    return Clue(ClueType.EXACTLY, ClueRegion(RegionType.ROW, index=0), target, count=count)


def _reference_classify(kb: EncodedKB, coord: Coordinate) -> CellEntailment:
    var = kb.var_map[coord]
    base = [set(c) for c in kb.clauses]
    can_be_innocent = _cnf_sat(base + [{-var}])
    can_be_criminal = _cnf_sat(base + [{var}])
    if not can_be_innocent and not can_be_criminal:
        return CellEntailment.INCONSISTENT
    if not can_be_innocent:
        return CellEntailment.CRIMINAL
    if not can_be_criminal:
        return CellEntailment.INNOCENT
    return CellEntailment.UNKNOWN


# name -> (clues, known statuses, probe cell, expected entailment)
ENTAILMENT_CASES: dict[
    str,
    tuple[dict[Coordinate, Clue], dict[Coordinate, HiddenStatus], Coordinate, CellEntailment],
] = {
    "unknown": ({(0, 0): _row0(1)}, {(0, 0): INNO}, (0, 1), CellEntailment.UNKNOWN),
    "forced_criminal": (
        {(0, 0): _row0(1)},
        {(0, 0): INNO, (0, 1): INNO},
        (0, 2),
        CellEntailment.CRIMINAL,
    ),
    "forced_innocent": (
        {(0, 0): _row0(1)},
        {(0, 0): INNO, (0, 1): CRIM},
        (0, 2),
        CellEntailment.INNOCENT,
    ),
    "inconsistent": (
        {(0, 0): _row0(3), (1, 1): _row0(0)},
        {(0, 0): CRIM, (1, 1): INNO},
        (2, 2),
        CellEntailment.INCONSISTENT,
    ),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(ENTAILMENT_CASES))
def test_entailment_classes_reference(mode: str, case: str) -> None:
    clues, known, probe, expected = ENTAILMENT_CASES[case]
    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    kb = encoder.encode_clue_set(3, clues, known)
    assert _reference_classify(kb, probe) == expected


def _public_state(
    n: int, clues: dict[Coordinate, Clue], known: dict[Coordinate, HiddenStatus]
) -> PublicKnowledgeState:
    cells: dict[Coordinate, PublicCellView] = {}
    for r in range(n):
        for c in range(n):
            coord = (r, c)
            status = known.get(coord)
            cells[coord] = PublicCellView(
                coord=coord,
                name=f"P{r}{c}",
                profession="test",
                revealed_status=(
                    RevealedStatus.UNKNOWN if status is None else RevealedStatus(status.value)
                ),
                revealed_clue=clues.get(coord) if status is not None else None,
            )
    return PublicKnowledgeState(grid_size=n, cells=cells)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(ENTAILMENT_CASES))
def test_agent_matches_reference_entailment(mode: str, case: str) -> None:
    """Becomes active once TV 4's DPLLSolver is implemented; skipped before that."""
    clues, known, probe, expected = ENTAILMENT_CASES[case]
    agent = LogicAgent(
        encoder=CNFEncoder(cardinality_mode=mode),  # type: ignore[arg-type]
        solver=DPLLSolver(),
    )
    agent.observe(_public_state(3, clues, known))
    try:
        entailment, _ = agent.classify_cell(probe)
    except NotImplementedError:
        pytest.skip("DPLLSolver / encoder not implemented yet")
    assert entailment == expected
