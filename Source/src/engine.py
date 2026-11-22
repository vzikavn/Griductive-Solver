"""Authoritative GameEngine controller for the Griductive puzzle game."""

from __future__ import annotations

from agent import LogicAgent
from models import (
    VALID_GRID_SIZES,
    CellEntailment,
    CharacterCell,
    Clue,
    ClueRegion,
    ClueType,
    Coordinate,
    HiddenStatus,
    PublicKnowledgeState,
    PuzzleDefinition,
    RegionType,
    RevealedStatus,
    VerdictResponseType,
    VerdictResult,
)

__all__ = ["GameEngine", "PuzzleDefinition"]


class GameEngine:
    """Master game controller that owns hidden truth and adjudicates verdicts."""

    def __init__(
        self,
        grid_size: int = 3,
        verifier_agent: LogicAgent | None = None,
    ) -> None:
        if grid_size not in VALID_GRID_SIZES:
            raise ValueError(
                f"Grid size must be one of {sorted(VALID_GRID_SIZES)}, got {grid_size}."
            )
        self._grid_size: int = grid_size
        self._grid: dict[Coordinate, CharacterCell] = {}
        self._initial_revealed: frozenset[Coordinate] = frozenset()
        self._total_criminals: int | None = None
        self._puzzle_blueprint: PuzzleDefinition | None = None
        self._verifier: LogicAgent = (
            verifier_agent if verifier_agent is not None else LogicAgent()
        )

    @property
    def grid_size(self) -> int:
        """Returns the N dimension of the N x N puzzle grid."""
        return self._grid_size

    @property
    def puzzle_id(self) -> str:
        """Returns the identifier of the currently loaded puzzle."""
        return self._puzzle_blueprint.puzzle_id if self._puzzle_blueprint else "none"

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
        self._verifier.reset_session()

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
        """Constructs and returns a sanitized, read-only public snapshot of the board."""
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
        """Validates a player or agent's verdict for a specific cell."""
        if coord not in self._grid:
            raise KeyError(
                f"Coordinate {coord} does not exist on the {self._grid_size}x{self._grid_size} grid."
            )

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
            if cell.hidden_status != proposed_status:
                raise RuntimeError(
                    f"Puzzle inconsistency at {coord}: KB entailed {proposed_status} "
                    f"but hidden truth is {cell.hidden_status}."
                )
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
        """Checks entailment strictly via PublicKnowledgeState (no hidden-state access)."""
        public_state = self.get_public_knowledge_state()
        self._verifier.observe(public_state)
        entailment, _ = self._verifier.classify_cell(coord)

        match entailment:
            case CellEntailment.CRIMINAL:
                return (
                    VerdictResponseType.ACCEPTED
                    if proposed_status == HiddenStatus.CRIMINAL
                    else VerdictResponseType.CONTRADICTED
                )
            case CellEntailment.INNOCENT:
                return (
                    VerdictResponseType.ACCEPTED
                    if proposed_status == HiddenStatus.INNOCENT
                    else VerdictResponseType.CONTRADICTED
                )
            case CellEntailment.UNKNOWN:
                return VerdictResponseType.NOT_PROVABLE
            case CellEntailment.INCONSISTENT:
                return VerdictResponseType.CONTRADICTED

    @staticmethod
    def create_sample_puzzle(grid_size: int = 3) -> PuzzleDefinition:
        """Fallback layout preview puzzle used only when `puzzles/` has no full JSON files."""
        if grid_size not in VALID_GRID_SIZES:
            raise ValueError(f"Sample puzzle size must be in {sorted(VALID_GRID_SIZES)}.")

        names_pool: tuple[tuple[str, str], ...] = (
            ("Alice", "Detective"), ("Bob", "Baker"), ("Charlie", "Clerk"),
            ("Diana", "Doctor"), ("Evan", "Engineer"), ("Fiona", "Florist"),
            ("George", "Guard"), ("Hannah", "Historian"), ("Ian", "Inspector"),
            ("Julia", "Judge"), ("Kevin", "Locksmith"), ("Laura", "Librarian"),
            ("Marcus", "Mechanic"), ("Nina", "Nurse"), ("Oscar", "Optician"),
            ("Paula", "Pilot"), ("Quinn", "Quartermaster"), ("Rachel", "Reporter"),
            ("Sam", "Scientist"), ("Tara", "Tailor"), ("Ulric", "Usher"),
            ("Vera", "Violinist"), ("Walter", "Watchmaker"), ("Xena", "Radiologist"),
            ("Yuri", "Yachtsman"),
        )

        cells: list[CharacterCell] = []
        for idx in range(grid_size * grid_size):
            r, c = divmod(idx, grid_size)
            name, prof = names_pool[idx % len(names_pool)]
            hidden = HiddenStatus.CRIMINAL if (r + c) % 2 == 1 else HiddenStatus.INNOCENT
            row_crims = sum(1 for col in range(grid_size) if (r + col) % 2 == 1)
            clue = Clue(
                clue_type=ClueType.EXACTLY,
                region=ClueRegion(region_type=RegionType.ROW, index=r),
                target_status=HiddenStatus.CRIMINAL,
                count=row_crims,
                text=f"Row {r + 1} has exactly {row_crims} Criminal(s).",
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

        return PuzzleDefinition(
            grid_size=grid_size,
            cells=tuple(cells),
            initial_revealed=frozenset({(0, 0)}),
            puzzle_id=f"fallback_sample_{grid_size}x{grid_size}",
            total_criminals=sum(
                1 for cell in cells if cell.hidden_status == HiddenStatus.CRIMINAL
            ),
        )