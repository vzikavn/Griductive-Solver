"""Core data structures and domain models for the Griductive deduction puzzle game.

This module enforces strict type safety and immutability across the boundary
between the authoritative GameEngine and the player-facing LogicAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from typing import Final, TypeAlias

# Coordinate alias representing (row_index, col_index), 0-indexed.
Coordinate: TypeAlias = tuple[int, int]

VALID_GRID_SIZES: Final[frozenset[int]] = frozenset({3, 4, 5})


class HiddenStatus(StrEnum):
    """Authoritative ground-truth identity of a character (known only to GameEngine)."""

    CRIMINAL = "CRIMINAL"
    INNOCENT = "INNOCENT"


class RevealedStatus(StrEnum):
    """Publicly known status of a character (visible to LogicAgent and GUI)."""

    UNKNOWN = "UNKNOWN"
    CRIMINAL = "CRIMINAL"
    INNOCENT = "INNOCENT"


class VerdictResponseType(StrEnum):
    """Engine evaluation outcome when a player or agent submits a verdict."""

    ACCEPTED = "ACCEPTED"  # Logically forced and matches truth; flips the card.
    NOT_PROVABLE = "NOT_PROVABLE"  # Neither Criminal nor Innocent is logically forced yet.
    CONTRADICTED = "CONTRADICTED"  # The opposite status is logically forced by public clues.
    ALREADY_REVEALED = "ALREADY_REVEALED"  # Target cell is already flipped face-up.


class RegionType(StrEnum):
    """Spatial region category referenced by a structured Clue."""

    ROW = "ROW"
    COLUMN = "COLUMN"
    NEIGHBORS = "NEIGHBORS"
    EXPLICIT_LIST = "EXPLICIT_LIST"


class ClueType(StrEnum):
    """Logical constraint classification for a character's clue."""

    FACT = "FACT"  # Direct assertion about targets (e.g., target statuses).
    SAME = "SAME"  # All referenced targets share the same status.
    DIFFERENT = "DIFFERENT"  # Referenced targets do not all share the same status.
    EXACTLY = "EXACTLY"  # Exactly K targets in the region have target_status.
    AT_LEAST = "AT_LEAST"  # At least K targets in the region have target_status.
    AT_MOST = "AT_MOST"  # At most K targets in the region have target_status.


@dataclass(frozen=True, slots=True)
class ClueRegion:
    """Structured spatial region defining which coordinates a Clue constrains.

    Attributes:
        region_type: The spatial rule category (ROW, COLUMN, NEIGHBORS, EXPLICIT_LIST).
        index: Row or column index (0-based) when region_type is ROW or COLUMN.
        center: Anchor coordinate when region_type is NEIGHBORS.
        explicit_coords: Tuple of specific coordinates when region_type is EXPLICIT_LIST.
        exclude_self: Whether to omit the clue owner's coordinate from the resolved set.
    """

    region_type: RegionType
    index: int | None = None
    center: Coordinate | None = None
    explicit_coords: tuple[Coordinate, ...] = field(default_factory=tuple)
    exclude_self: bool = False

    def resolve_coordinates(
        self,
        grid_size: int,
        owner_coord: Coordinate | None = None,
    ) -> frozenset[Coordinate]:
        """Resolves and returns all valid board coordinates belonging to this region."""
        coords: set[Coordinate] = set()

        match self.region_type:
            case RegionType.ROW:
                if self.index is None or not (0 <= self.index < grid_size):
                    raise ValueError(f"Invalid row index {self.index} for grid size {grid_size}.")
                coords = {(self.index, c) for c in range(grid_size)}

            case RegionType.COLUMN:
                if self.index is None or not (0 <= self.index < grid_size):
                    raise ValueError(
                        f"Invalid column index {self.index} for grid size {grid_size}."
                    )
                coords = {(r, self.index) for r in range(grid_size)}

            case RegionType.NEIGHBORS:
                if self.center is None:
                    raise ValueError("NEIGHBORS region requires a center coordinate.")
                cr, cc = self.center
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            coords.add((nr, nc))

            case RegionType.EXPLICIT_LIST:
                for r, c in self.explicit_coords:
                    if not (0 <= r < grid_size and 0 <= c < grid_size):
                        raise ValueError(
                            f"Explicit coordinate {(r, c)} out of bounds for size {grid_size}."
                        )
                    coords.add((r, c))

        if self.exclude_self and owner_coord is not None:
            coords.discard(owner_coord)

        return frozenset(coords)


@dataclass(frozen=True, slots=True)
class Clue:
    """Immutable logical clue attached to a cell.

    Attributes:
        clue_type: Logical operator type (FACT, SAME, DIFFERENT, EXACTLY, AT_LEAST, AT_MOST).
        region: Spatial descriptor of cells governed by this clue.
        target_status: The status being counted or asserted (CRIMINAL or INNOCENT).
        count: Cardinality threshold K for EXACTLY, AT_LEAST, or AT_MOST clues.
        text: Human-readable clue statement displayed on the card when revealed.
    """

    clue_type: ClueType
    region: ClueRegion
    target_status: HiddenStatus = HiddenStatus.CRIMINAL
    count: int | None = None
    text: str = ""

    def __post_init__(self) -> None:
        """Validates semantic consistency between clue_type and count."""
        cardinality_types = {ClueType.EXACTLY, ClueType.AT_LEAST, ClueType.AT_MOST}
        if self.clue_type in cardinality_types and self.count is None:
            raise ValueError(f"ClueType {self.clue_type} requires a non-null integer 'count'.")
        if self.clue_type not in cardinality_types and self.count is not None:
            raise ValueError(f"ClueType {self.clue_type} must not specify 'count'.")


@dataclass(slots=True)
class CharacterCell:
    """Authoritative internal cell state managed exclusively by GameEngine.

    Contains both hidden truth and public state. NEVER pass instances of
    CharacterCell directly to LogicAgent.
    """

    coord: Coordinate
    name: str
    profession: str
    hidden_status: HiddenStatus
    clue: Clue
    revealed_status: RevealedStatus = RevealedStatus.UNKNOWN

    @property
    def is_revealed(self) -> bool:
        """Returns True if the character's identity and clue have been flipped face-up."""
        return self.revealed_status != RevealedStatus.UNKNOWN

    def to_public_view(self) -> PublicCellView:
        """Projects this authoritative cell into a sanitized, immutable public snapshot.

        Strictly strips `hidden_status` and masks `clue` if the cell is still face-down.
        """
        return PublicCellView(
            coord=self.coord,
            name=self.name,
            profession=self.profession,
            revealed_status=self.revealed_status,
            revealed_clue=self.clue if self.is_revealed else None,
        )


@dataclass(frozen=True, slots=True)
class PublicCellView:
    """Sanitized, read-only projection of a cell exposed to LogicAgent and the GUI.

    Guarantees at the type and runtime level that hidden statuses and unrevealed
    clues cannot be accessed by the agent.
    """

    coord: Coordinate
    name: str
    profession: str
    revealed_status: RevealedStatus
    revealed_clue: Clue | None


@dataclass(frozen=True, slots=True)
class PublicKnowledgeState:
    """Complete read-only snapshot of all publicly observable board state.

    Passed across the architectural boundary from GameEngine to LogicAgent.
    """

    grid_size: int
    cells: dict[Coordinate, PublicCellView]
    total_criminals: int | None = None

    def get_cell(self, coord: Coordinate) -> PublicCellView:
        """Returns the public view for a given coordinate."""
        return self.cells[coord]

    def get_revealed_clues(self) -> dict[Coordinate, Clue]:
        """Returns a mapping of coordinates to revealed clues currently on the board."""
        return {
            coord: view.revealed_clue
            for coord, view in self.cells.items()
            if view.revealed_clue is not None
        }

    def get_unknown_coordinates(self) -> tuple[Coordinate, ...]:
        """Returns all coordinates whose status is still UNKNOWN."""
        return tuple(
            coord
            for coord, view in self.cells.items()
            if view.revealed_status == RevealedStatus.UNKNOWN
        )


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Structured response emitted by GameEngine after evaluating a submitted verdict."""

    coord: Coordinate
    proposed_status: HiddenStatus
    response_type: VerdictResponseType
    message: str
    public_state: PublicKnowledgeState
