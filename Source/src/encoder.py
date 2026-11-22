"""Automatic CNF encoder and direct semantic evaluator for Griductive clues."""

from __future__ import annotations

from dataclasses import dataclass

from models import (
    Clause,
    Clue,
    ClueType,
    Coordinate,
    HiddenStatus,
    PublicKnowledgeState,
    RevealedStatus,
)


@dataclass(frozen=True, slots=True)
class CNFStats:
    """Variable and clause counts generated for a CNF knowledge base."""

    primary_vars: int
    auxiliary_vars: int
    num_clauses: int

    @property
    def total_vars(self) -> int:
        """Total number of propositional variables (primary + auxiliary)."""
        return self.primary_vars + self.auxiliary_vars


@dataclass(frozen=True, slots=True)
class EncodedKB:
    """Compiled CNF representation of a knowledge base along with metadata."""

    clauses: tuple[Clause, ...]
    var_map: dict[Coordinate, int]
    stats: CNFStats


class CNFEncoder:
    """Automatically compiles structured clues and known verdicts into CNF."""

    @staticmethod
    def build_var_map(grid_size: int) -> dict[Coordinate, int]:
        """Creates a deterministic row-major mapping: (r, c) -> 1..N^2."""
        var_map: dict[Coordinate, int] = {}
        var_id = 1
        for r in range(grid_size):
            for c in range(grid_size):
                var_map[(r, c)] = var_id
                var_id += 1
        return var_map

    @staticmethod
    def validate_and_resolve(
        clue: Clue,
        grid_size: int,
        owner_coord: Coordinate | None = None,
    ) -> tuple[Coordinate, ...]:
        """Validates clue parameters and returns sorted coordinates of its region."""
        coords = tuple(
            sorted(
                clue.region.resolve_coordinates(
                    grid_size=grid_size,
                    owner_coord=owner_coord,
                )
            )
        )
        if not coords:
            raise ValueError(f"Clue '{clue.text}' resolved to an empty region.")

        if clue.clue_type in (ClueType.SAME, ClueType.DIFFERENT) and len(coords) < 2:
            raise ValueError(
                f"ClueType {clue.clue_type} requires at least 2 cells, got {len(coords)}."
            )

        if clue.clue_type in (ClueType.EXACTLY, ClueType.AT_LEAST, ClueType.AT_MOST):
            assert clue.count is not None
            if not (0 <= clue.count <= len(coords)):
                raise ValueError(
                    f"Count k={clue.count} out of range [0, {len(coords)}] for '{clue.text}'."
                )

        return coords

    def encode_knowledge_base(self, public_state: PublicKnowledgeState) -> EncodedKB:
        """Builds KB_t = CNF(revealed clues) U unit clauses for proved verdicts."""
        known_statuses: dict[Coordinate, HiddenStatus] = {}
        for coord, cell in public_state.cells.items():
            if cell.revealed_status == RevealedStatus.CRIMINAL:
                known_statuses[coord] = HiddenStatus.CRIMINAL
            elif cell.revealed_status == RevealedStatus.INNOCENT:
                known_statuses[coord] = HiddenStatus.INNOCENT

        return self.encode_clue_set(
            grid_size=public_state.grid_size,
            clues=public_state.get_revealed_clues(),
            known_statuses=known_statuses,
        )

    def encode_clue_set(
        self,
        grid_size: int,
        clues: dict[Coordinate, Clue],
        known_statuses: dict[Coordinate, HiddenStatus] | None = None,
    ) -> EncodedKB:
        """Encodes an explicit set of clues and known statuses."""
        var_map = self.build_var_map(grid_size)
        primary_count = len(var_map)
        next_aux_var = primary_count + 1
        clauses: list[Clause] = []

        if known_statuses:
            for coord in sorted(known_statuses):
                var_id = var_map[coord]
                lit = var_id if known_statuses[coord] == HiddenStatus.CRIMINAL else -var_id
                clauses.append(frozenset({lit}))

        for owner in sorted(clues):
            clue_clauses, next_aux_var = self.encode_single_clue(
                owner=owner,
                clue=clues[owner],
                grid_size=grid_size,
                var_map=var_map,
                next_aux_var=next_aux_var,
            )
            clauses.extend(clue_clauses)

        aux_count = next_aux_var - (primary_count + 1)
        stats = CNFStats(
            primary_vars=primary_count,
            auxiliary_vars=aux_count,
            num_clauses=len(clauses),
        )
        return EncodedKB(clauses=tuple(clauses), var_map=var_map, stats=stats)

    def encode_single_clue(
        self,
        owner: Coordinate,
        clue: Clue,
        grid_size: int,
        var_map: dict[Coordinate, int],
        next_aux_var: int,
    ) -> tuple[list[Clause], int]:
        """Encodes one Clue into CNF clauses and returns (clauses, updated_next_aux_var)."""
        self.validate_and_resolve(clue, grid_size, owner)
        # TODO (TV 3): Implement CNF conversion for FACT, SAME, DIFFERENT,
        # EXACTLY, AT_LEAST, AT_MOST, and PARITY.
        raise NotImplementedError("CNF clue encoding is not implemented yet.")


class SemanticEvaluator:
    """Direct semantic evaluator verifying clues on a full assignment WITHOUT CNF."""

    @staticmethod
    def evaluate_clue(
        clue: Clue,
        assignment: dict[Coordinate, HiddenStatus],
        grid_size: int,
        owner_coord: Coordinate | None = None,
    ) -> bool:
        """Returns True iff `clue` holds under the complete `assignment`."""
        CNFEncoder.validate_and_resolve(clue, grid_size, owner_coord)
        # TODO (TV 2): Directly evaluate each ClueType on `assignment`.
        raise NotImplementedError("Direct semantic evaluator is not implemented yet.")