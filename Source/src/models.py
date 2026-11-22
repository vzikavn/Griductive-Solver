"""Core data structures and domain models for the Griductive deduction puzzle game."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

type Coordinate = tuple[int, int]
type Literal = int
type Clause = frozenset[Literal]

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


class CellEntailment(StrEnum):
    """Logical consequence classification of an unsolved character under KB_t."""

    CRIMINAL = "CRIMINAL"
    INNOCENT = "INNOCENT"
    UNKNOWN = "UNKNOWN"
    INCONSISTENT = "INCONSISTENT"


class VerdictResponseType(StrEnum):
    """Engine evaluation outcome when a player or agent submits a verdict."""

    ACCEPTED = "ACCEPTED"
    NOT_PROVABLE = "NOT_PROVABLE"
    CONTRADICTED = "CONTRADICTED"
    ALREADY_REVEALED = "ALREADY_REVEALED"


class RegionType(StrEnum):
    """Spatial region category referenced by a structured Clue."""

    ROW = "ROW"
    COLUMN = "COLUMN"
    NEIGHBORS = "NEIGHBORS"
    EXPLICIT_LIST = "EXPLICIT_LIST"
    # Extension 1: Advanced region expressions
    CORNERS = "CORNERS"
    BOUNDARY = "BOUNDARY"


class ClueType(StrEnum):
    """Logical constraint classification for a character's clue."""

    FACT = "FACT"
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    EXACTLY = "EXACTLY"
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"
    # Extension 2: Parity constraint (count=0 for EVEN, count=1 for ODD)
    PARITY = "PARITY"


@dataclass(frozen=True, slots=True)
class ClueRegion:
    """Structured spatial region defining which coordinates a Clue constrains."""

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
                    raise ValueError(
                        f"Invalid row index {self.index} for grid size {grid_size}."
                    )
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
                if not (0 <= cr < grid_size and 0 <= cc < grid_size):
                    raise ValueError(
                        f"NEIGHBORS center {(cr, cc)} out of bounds for size {grid_size}."
                    )
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

            case RegionType.CORNERS:
                last = grid_size - 1
                coords = {(0, 0), (0, last), (last, 0), (last, last)}

            case RegionType.BOUNDARY:
                last = grid_size - 1
                coords = {
                    (r, c)
                    for r in range(grid_size)
                    for c in range(grid_size)
                    if r in (0, last) or c in (0, last)
                }

        if self.exclude_self and owner_coord is not None:
            coords.discard(owner_coord)

        return frozenset(coords)


@dataclass(frozen=True, slots=True)
class Clue:
    """Immutable logical clue attached to a cell."""

    clue_type: ClueType
    region: ClueRegion
    target_status: HiddenStatus = HiddenStatus.CRIMINAL
    count: int | None = None
    text: str = ""

    def __post_init__(self) -> None:
        """Validates semantic consistency between clue_type and count."""
        cardinality_types = {ClueType.EXACTLY, ClueType.AT_LEAST, ClueType.AT_MOST}
        if self.clue_type in cardinality_types:
            if self.count is None or self.count < 0:
                raise ValueError(
                    f"ClueType {self.clue_type} requires a non-negative integer 'count'."
                )
        elif self.clue_type == ClueType.PARITY:
            if self.count not in (0, 1):
                raise ValueError(
                    "ClueType PARITY requires 'count' to be 0 (EVEN) or 1 (ODD)."
                )
        else:
            if self.count is not None:
                raise ValueError(f"ClueType {self.clue_type} must not specify 'count'.")


@dataclass(slots=True)
class CharacterCell:
    """Authoritative internal cell state managed exclusively by GameEngine."""

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
        """Projects this authoritative cell into a sanitized, immutable public snapshot."""
        return PublicCellView(
            coord=self.coord,
            name=self.name,
            profession=self.profession,
            revealed_status=self.revealed_status,
            revealed_clue=self.clue if self.is_revealed else None,
        )


@dataclass(frozen=True, slots=True)
class PuzzleDefinition:
    """Blueprint used to initialize or reset a puzzle level in GameEngine."""

    grid_size: int
    cells: tuple[CharacterCell, ...]
    initial_revealed: frozenset[Coordinate]
    puzzle_id: str = "unnamed"
    total_criminals: int | None = None


@dataclass(frozen=True, slots=True)
class PublicCellView:
    """Sanitized, read-only projection of a cell exposed to LogicAgent and the GUI."""

    coord: Coordinate
    name: str
    profession: str
    revealed_status: RevealedStatus
    revealed_clue: Clue | None


@dataclass(frozen=True, slots=True)
class PublicKnowledgeState:
    """Complete read-only snapshot of all publicly observable board state."""

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
        """Returns all coordinates whose status is still UNKNOWN in row-major order."""
        return tuple(
            sorted(
                coord
                for coord, view in self.cells.items()
                if view.revealed_status == RevealedStatus.UNKNOWN
            )
        )


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Structured response emitted by GameEngine after evaluating a submitted verdict."""

    coord: Coordinate
    proposed_status: HiddenStatus
    response_type: VerdictResponseType
    message: str
    public_state: PublicKnowledgeState


@dataclass(frozen=True, slots=True)
class DeductionStep:
    """Single logged step in the LogicAgent's deduction trace."""

    step_number: int
    active_clue_coords: tuple[Coordinate, ...]
    sat_queries: int
    target_coord: Coordinate
    verdict: HiddenStatus
    newly_revealed_clue: str | None = None