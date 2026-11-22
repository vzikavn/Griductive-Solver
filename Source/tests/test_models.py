"""Unit tests for Griductive data models and ClueRegion spatial resolution."""

from models import Clue, ClueRegion, ClueType, HiddenStatus, RegionType


def test_row_and_neighbor_regions() -> None:
    row_region = ClueRegion(region_type=RegionType.ROW, index=0)
    assert row_region.resolve_coordinates(grid_size=3) == frozenset({(0, 0), (0, 1), (0, 2)})

    nbr_region = ClueRegion(region_type=RegionType.NEIGHBORS, center=(0, 0), exclude_self=True)
    assert nbr_region.resolve_coordinates(grid_size=3, owner_coord=(0, 0)) == frozenset(
        {(0, 1), (1, 0), (1, 1)}
    )


def test_clue_validation() -> None:
    clue = Clue(
        clue_type=ClueType.EXACTLY,
        region=ClueRegion(region_type=RegionType.COLUMN, index=1),
        target_status=HiddenStatus.CRIMINAL,
        count=2,
        text="Column 2 has exactly 2 Criminals.",
    )
    assert clue.count == 2
