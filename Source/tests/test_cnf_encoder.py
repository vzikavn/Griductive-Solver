"""Unit tests for CNFEncoder across 3x3, 4x4, 5x5 grids and all clue types."""

from collections.abc import Sequence
from itertools import product

import pytest

from encoder import CNFEncoder
from models import Clause, Clue, ClueRegion, ClueType, Coordinate, HiddenStatus, RegionType


def _exists_satisfying_extension(
    clauses: Sequence[Clause],
    total_vars: int,
    fixed_primary: dict[int, bool],
) -> bool:
    """Checks if there exists an assignment to auxiliary vars satisfying all clauses."""
    aux_vars = [v for v in range(1, total_vars + 1) if v not in fixed_primary]
    for aux_bits in product((False, True), repeat=len(aux_vars)):
        assignment = dict(fixed_primary)
        assignment.update(zip(aux_vars, aux_bits))
        if all(
            any(
                assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]
                for lit in clause
            )
            for clause in clauses
        ):
            return True
    return False


def _expected_clue_truth(
    clue_type: ClueType,
    target_status: HiddenStatus,
    count: int | None,
    statuses: tuple[HiddenStatus, ...],
) -> bool:
    matches = [s == target_status for s in statuses]
    matched = sum(matches)
    match clue_type:
        case ClueType.FACT:
            return all(matches)
        case ClueType.SAME:
            return len(set(statuses)) == 1
        case ClueType.DIFFERENT:
            return len(set(statuses)) > 1
        case ClueType.EXACTLY:
            assert count is not None
            return matched == count
        case ClueType.AT_LEAST:
            assert count is not None
            return matched >= count
        case ClueType.AT_MOST:
            assert count is not None
            return matched <= count
        case ClueType.PARITY:
            assert count is not None
            return (matched % 2) == count


@pytest.mark.parametrize("grid_size", [3, 4, 5])
def test_var_map_and_primary_stats_across_grid_sizes(grid_size: int) -> None:
    encoder = CNFEncoder()
    encoded = encoder.encode_clue_set(grid_size=grid_size, clues={})
    assert encoded.stats.primary_vars == grid_size * grid_size
    assert encoded.stats.auxiliary_vars == 0
    assert encoded.stats.num_clauses == 0
    assert len(encoded.var_map) == grid_size * grid_size


@pytest.mark.parametrize("mode", ["combinatorial", "sequential"])
@pytest.mark.parametrize(
    ("clue_type", "target_status", "count"),
    [
        (ClueType.FACT, HiddenStatus.CRIMINAL, None),
        (ClueType.FACT, HiddenStatus.INNOCENT, None),
        (ClueType.SAME, HiddenStatus.CRIMINAL, None),
        (ClueType.SAME, HiddenStatus.INNOCENT, None),
        (ClueType.DIFFERENT, HiddenStatus.CRIMINAL, None),
        (ClueType.DIFFERENT, HiddenStatus.INNOCENT, None),
        (ClueType.EXACTLY, HiddenStatus.CRIMINAL, 0),
        (ClueType.EXACTLY, HiddenStatus.CRIMINAL, 2),
        (ClueType.EXACTLY, HiddenStatus.CRIMINAL, 4),
        (ClueType.EXACTLY, HiddenStatus.INNOCENT, 3),
        (ClueType.AT_LEAST, HiddenStatus.CRIMINAL, 0),
        (ClueType.AT_LEAST, HiddenStatus.CRIMINAL, 2),
        (ClueType.AT_LEAST, HiddenStatus.INNOCENT, 4),
        (ClueType.AT_MOST, HiddenStatus.CRIMINAL, 0),
        (ClueType.AT_MOST, HiddenStatus.CRIMINAL, 1),
        (ClueType.AT_MOST, HiddenStatus.INNOCENT, 4),
        (ClueType.PARITY, HiddenStatus.CRIMINAL, 0),
        (ClueType.PARITY, HiddenStatus.CRIMINAL, 1),
        (ClueType.PARITY, HiddenStatus.INNOCENT, 1),
    ],
)
def test_exhaustive_truth_table_for_all_clue_types(
    mode: str,
    clue_type: ClueType,
    target_status: HiddenStatus,
    count: int | None,
) -> None:
    grid_size = 3
    coords: tuple[Coordinate, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
    clue = Clue(
        clue_type=clue_type,
        region=ClueRegion(region_type=RegionType.EXPLICIT_LIST, explicit_coords=coords),
        target_status=target_status,
        count=count,
        text="Truth-table verification clue",
    )

    encoder = CNFEncoder(cardinality_mode=mode)  # type: ignore[arg-type]
    encoded = encoder.encode_clue_set(grid_size=grid_size, clues={(0, 0): clue})

    for bits in product((False, True), repeat=len(coords)):
        fixed_primary = {v: False for v in encoded.var_map.values()}
        statuses: list[HiddenStatus] = []
        for coord, is_criminal in zip(coords, bits):
            fixed_primary[encoded.var_map[coord]] = is_criminal
            statuses.append(
                HiddenStatus.CRIMINAL if is_criminal else HiddenStatus.INNOCENT
            )

        expected = _expected_clue_truth(
            clue_type=clue_type,
            target_status=target_status,
            count=count,
            statuses=tuple(statuses),
        )
        actual = _exists_satisfying_extension(
            clauses=encoded.clauses,
            total_vars=encoded.stats.total_vars,
            fixed_primary=fixed_primary,
        )
        assert actual == expected


@pytest.mark.parametrize("grid_size", [3, 4, 5])
def test_extensions_corners_boundary_and_5x5_scalability(grid_size: int) -> None:
    clues: dict[Coordinate, Clue] = {
        (0, 0): Clue(
            clue_type=ClueType.EXACTLY,
            region=ClueRegion(region_type=RegionType.CORNERS),
            target_status=HiddenStatus.CRIMINAL,
            count=2,
            text="2 criminals in corners",
        ),
        (0, 1): Clue(
            clue_type=ClueType.PARITY,
            region=ClueRegion(region_type=RegionType.BOUNDARY),
            target_status=HiddenStatus.INNOCENT,
            count=1,
            text="Odd innocents on boundary",
        ),
    }

    comb_kb = CNFEncoder(cardinality_mode="combinatorial").encode_clue_set(grid_size, clues)
    seq_kb = CNFEncoder(cardinality_mode="sequential").encode_clue_set(grid_size, clues)

    boundary_len = 4 * (grid_size - 1)
    assert comb_kb.stats.primary_vars == grid_size * grid_size
    # CORNERS uses 0 aux vars in combinatorial mode; PARITY on boundary uses (boundary_len - 2)
    assert comb_kb.stats.auxiliary_vars == boundary_len - 2
    assert seq_kb.stats.auxiliary_vars > comb_kb.stats.auxiliary_vars

    # Stress check on 5x5 BOUNDARY (16 cells) with k=8
    if grid_size == 5:
        heavy_clue = {
            (0, 0): Clue(
                clue_type=ClueType.EXACTLY,
                region=ClueRegion(region_type=RegionType.BOUNDARY),
                target_status=HiddenStatus.CRIMINAL,
                count=8,
                text="8 criminals on 5x5 boundary",
            )
        }
        heavy_comb = CNFEncoder("combinatorial").encode_clue_set(5, heavy_clue)
        heavy_seq = CNFEncoder("sequential").encode_clue_set(5, heavy_clue)
        assert heavy_comb.stats.num_clauses == 22880
        assert heavy_seq.stats.num_clauses == 494