"""Authoritative GameEngine controller for the Griductive puzzle game.

The GameEngine owns the ground-truth N x N puzzle board (hidden statuses and all clues),
validates player/agent verdict submissions, flips revealed cards, and exports sanitized
PublicKnowledgeState snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

from models import (
     VALID_GRID_SIZES,
    CharacterCell,
    Clue,
    ClueRegion,
    ClueType,
    Coordinate,
    HiddenStatus,
    PublicKnowledgeState,
    RegionType,
    RevealedStatus,
    VerdictResponseType,
    VerdictResult,
)


@dataclass(frozen=True, slots=True)
class PuzzleDefinition:
    """Blueprint used to initialize or reset a puzzle level in GameEngine."""

    grid_size: int
    cells: tuple[CharacterCell, ...]
    initial_revealed: frozenset[Coordinate]
    total_criminals: int | None = None


class GameEngine:
    """Master game controller that owns hidden truth and adjudicates verdicts.

    Strict Boundary Guarantee:
        - Holds private `_grid: dict[Coordinate, CharacterCell]` containing ground truth.
        - Never exposes `CharacterCell` references outside the class.
        - Communicates with `LogicAgent` and `GUI` exclusively through immutable
          `PublicKnowledgeState` and `VerdictResult` value objects.
    """

    def __init__(self, grid_size: int = 3) -> None:
        if grid_size not in VALID_GRID_SIZES:
            raise ValueError(
                f"Grid size must be one of {sorted(VALID_GRID_SIZES)}, got {grid_size}."
            )
        self._grid_size: int = grid_size
        self._grid: dict[Coordinate, CharacterCell] = {}
        self._initial_revealed: frozenset[Coordinate] = frozenset()
        self._total_criminals: int | None = None
        self._puzzle_blueprint: PuzzleDefinition | None = None

    @property
    def grid_size(self) -> int:
        """Returns the N dimension of the N x N puzzle grid."""
        return self._grid_size

    @property
    def is_solved(self) -> bool:
        """Returns True when every cell in the grid has been revealed."""
        return bool(self._grid) and all(cell.is_revealed for cell in self._grid.values())

    def load_puzzle(self, puzzle: PuzzleDefinition) -> PublicKnowledgeState:
        """Loads a puzzle definition into the engine and initializes the board state."""
        if puzzle.grid_size not in VALID_GRID_SIZES:
            raise ValueError(f"Unsupported grid size: {puzzle.grid_size}")

        self._puzzle_blueprint = puzzle
        self._grid_size = puzzle.grid_size
        self._initial_revealed = puzzle.initial_revealed
        self._total_criminals = puzzle.total_criminals
        self._grid.clear()

        for template_cell in puzzle.cells:
            revealed = RevealedStatus.UNKNOWN
            if template_cell.coord in puzzle.initial_revealed:
                revealed = (
                    RevealedStatus.CRIMINAL
                    if template_cell.hidden_status == HiddenStatus.CRIMINAL
                    else RevealedStatus.INNOCENT
                )
            self._grid[template_cell.coord] = CharacterCell(
                coord=template_cell.coord,
                name=template_cell.name,
                profession=template_cell.profession,
                hidden_status=template_cell.hidden_status,
                clue=template_cell.clue,
                revealed_status=revealed,
            )

        return self.get_public_knowledge_state()

    def restart_puzzle(self) -> PublicKnowledgeState:
        """Resets all cells back to their initial revealed state for the active puzzle."""
        if self._puzzle_blueprint is None:
            raise RuntimeError("Cannot restart: no puzzle is currently loaded.")
        return self.load_puzzle(self._puzzle_blueprint)

    def get_public_knowledge_state(self) -> PublicKnowledgeState:
        """Constructs and returns a sanitized, read-only public snapshot of the board.

        Hidden statuses and face-down clues are stripped via `CharacterCell.to_public_view()`.
        """
        public_cells = {
            coord: cell.to_public_view() for coord, cell in self._grid.items()
        }
        return PublicKnowledgeState(
            grid_size=self._grid_size,
            cells=public_cells,
            total_criminals=self._total_criminals,
        )

    def submit_verdict(
        self,
        coord: Coordinate,
        proposed_status: HiddenStatus,
    ) -> VerdictResult:
        """Validates a player or agent's verdict for a specific cell.

        Evaluation semantics:
            - ALREADY_REVEALED: Cell is already flipped face-up.
            - ACCEPTED: `proposed_status` is logically forced by currently revealed clues
              (and matches hidden truth). The engine flips the card face-up.
            - CONTRADICTED: The opposite of `proposed_status` is logically forced by
              currently revealed clues.
            - NOT_PROVABLE: Neither status is logically forced from public knowledge alone.
        """
        if coord not in self._grid:
            raise KeyError(f"Coordinate {coord} does not exist on the {self._grid_size}x{self._grid_size} grid.")

        cell = self._grid[coord]
        if cell.is_revealed:
            return VerdictResult(
                coord=coord,
                proposed_status=proposed_status,
                response_type=VerdictResponseType.ALREADY_REVEALED,
                message=f"{cell.name} at {coord} is already revealed as {cell.revealed_status}.",
                public_state=self.get_public_knowledge_state(),
            )

        response_type = self._verify_logical_entailment(coord, proposed_status)

        if response_type == VerdictResponseType.ACCEPTED:
            cell.revealed_status = (
                RevealedStatus.CRIMINAL
                if cell.hidden_status == HiddenStatus.CRIMINAL
                else RevealedStatus.INNOCENT
            )
            msg = (
                f"ACCEPTED: {cell.name} ({cell.profession}) is confirmed "
                f"{cell.revealed_status}! Clue revealed."
            )
        elif response_type == VerdictResponseType.CONTRADICTED:
            msg = (
                f"CONTRADICTED: Marking {cell.name} at {coord} as {proposed_status} "
                f"violates logical deductions from revealed clues."
            )
        else:
            msg = (
                f"NOT_PROVABLE: {cell.name} at {coord} cannot be proven "
                f"{proposed_status} from current public clues alone."
            )

        return VerdictResult(
            coord=coord,
            proposed_status=proposed_status,
            response_type=response_type,
            message=msg,
            public_state=self.get_public_knowledge_state(),
        )

    def _verify_logical_entailment(
        self,
        coord: Coordinate,
        proposed_status: HiddenStatus,
    ) -> VerdictResponseType:
        """Determines whether `proposed_status` at `coord` is entailed by public knowledge.

        NOTE: Per project milestone constraints, the DPLL SAT solver and CNF entailment
        checks are left as placeholders to be implemented in the AI solver phase.
        """
        raise NotImplementedError(
            "DPLL SAT solver and CNF entailment verification are not implemented yet."
        )

    @staticmethod
    def create_sample_puzzle(grid_size: int = 3) -> PuzzleDefinition:
        """Creates a sample N x N puzzle blueprint for GUI testing and layout preview."""
        if grid_size not in VALID_GRID_SIZES:
            raise ValueError(f"Sample puzzle size must be in {sorted(VALID_GRID_SIZES)}.")

        names_pool: tuple[tuple[str, str], ...] = (
            ("Alice", "Detective"),
            ("Bob", "Baker"),
            ("Charlie", "Clerk"),
            ("Diana", "Doctor"),
            ("Evan", "Engineer"),
            ("Fiona", "Florist"),
            ("George", "Guard"),
            ("Hannah", "Historian"),
            ("Ian", "Inspector"),
            ("Julia", "Judge"),
            ("Kevin", "Locksmith"),
            ("Laura", "Librarian"),
            ("Marcus", "Mechanic"),
            ("Nina", "Nurse"),
            ("Oscar", "Optician"),
            ("Paula", "Pilot"),
            ("Quinn", "Quartermaster"),
            ("Rachel", "Reporter"),
            ("Sam", "Scientist"),
            ("Tara", "Tailor"),
            ("Ulric", "Usher"),
            ("Vera", "Violinist"),
            ("Walter", "Watchmaker"),
            ("Xena", "Radiologist"),
            ("Yuri", "Yachtsman"),
        )

        cells: list[CharacterCell] = []
        idx = 0
        for r in range(grid_size):
            for c in range(grid_size):
                name, prof = names_pool[idx % len(names_pool)]
                is_crim = (r + c) % 2 == 1
                hidden = HiddenStatus.CRIMINAL if is_crim else HiddenStatus.INNOCENT

                if r == 0 and c == 0:
                    clue = Clue(
                        clue_type=ClueType.EXACTLY,
                        region=ClueRegion(region_type=RegionType.ROW, index=0),
                        target_status=HiddenStatus.CRIMINAL,
                        count=1,
                        text="Row 1 has exactly 1 Criminal.",
                    )
                elif c == r:
                    clue = Clue(
                        clue_type=ClueType.AT_LEAST,
                        region=ClueRegion(
                            region_type=RegionType.NEIGHBORS,
                            center=(r, c),
                            exclude_self=True,
                        ),
                        target_status=HiddenStatus.INNOCENT,
                        count=2,
                        text=f"At least 2 of my neighbors are Innocent.",
                    )
                elif r == grid_size - 1:
                    clue = Clue(
                        clue_type=ClueType.SAME,
                        region=ClueRegion(
                            region_type=RegionType.EXPLICIT_LIST,
                            explicit_coords=((0, c), (grid_size - 1, c)),
                        ),
                        target_status=HiddenStatus.INNOCENT,
                        text=f"Top and bottom cells in Column {c + 1} share the same status.",
                    )
                else:
                    clue = Clue(
                        clue_type=ClueType.AT_MOST,
                        region=ClueRegion(region_type=RegionType.COLUMN, index=c),
                        target_status=HiddenStatus.CRIMINAL,
                        count=2,
                        text=f"Column {c + 1} has at most 2 Criminals.",
                    )

                cells.append(
                    CharacterCell(
                        coord=(r, c),
                        name=name,
                        profession=prof,
                        hidden_status=hidden,
                        clue=clue,
                    )
                )
                idx += 1

        return PuzzleDefinition(
            grid_size=grid_size,
            cells=tuple(cells),
            initial_revealed=frozenset({(0, 0)}),
            total_criminals=sum(
                1 for cell in cells if cell.hidden_status == HiddenStatus.CRIMINAL
            ),
        )