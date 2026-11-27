"""Automatic CNF encoder and direct semantic evaluator for Griductive clues."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal as TypingLiteral

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

    def __init__(
        self,
        cardinality_mode: TypingLiteral["combinatorial", "sequential"] = "combinatorial",
    ) -> None:
        if cardinality_mode not in ("combinatorial", "sequential"):
            raise ValueError(
                f"Unsupported cardinality_mode {cardinality_mode!r}. "
                "Expected 'combinatorial' or 'sequential'."
            )
        self.cardinality_mode: TypingLiteral["combinatorial", "sequential"] = cardinality_mode

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

    @staticmethod
    def _xor2_clauses(a: int, b: int, parity: int) -> list[Clause]:
        """Encodes (a XOR b) == parity (0 for EVEN, 1 for ODD) into 2 binary clauses."""
        if parity == 1:
            return [frozenset({a, b}), frozenset({-a, -b})]
        return [frozenset({-a, b}), frozenset({a, -b})]

    @staticmethod
    def _xor3_eq_clauses(out_var: int, a: int, b: int) -> list[Clause]:
        """Encodes out_var <-> (a XOR b) into 4 ternary CNF clauses."""
        return [
            frozenset({-a, -b, -out_var}),
            frozenset({a, b, -out_var}),
            frozenset({-a, b, out_var}),
            frozenset({a, -b, out_var}),
        ]

    def _encode_parity(
        self,
        lits: tuple[int, ...],
        parity: int,
        next_aux_var: int,
    ) -> tuple[list[Clause], int]:
        """Encodes sum(lits) % 2 == parity using linear Tseitin XOR chaining."""
        m = len(lits)
        if m == 1:
            lit = lits[0] if parity == 1 else -lits[0]
            return [frozenset({lit})], next_aux_var
        if m == 2:
            return self._xor2_clauses(lits[0], lits[1], parity), next_aux_var

        clauses: list[Clause] = []
        prev_var = next_aux_var
        next_aux_var += 1
        clauses.extend(self._xor3_eq_clauses(prev_var, lits[0], lits[1]))

        for idx in range(2, m - 1):
            curr_var = next_aux_var
            next_aux_var += 1
            clauses.extend(self._xor3_eq_clauses(curr_var, prev_var, lits[idx]))
            prev_var = curr_var

        clauses.extend(self._xor2_clauses(prev_var, lits[-1], parity))
        return clauses, next_aux_var

    @staticmethod
    def _encode_at_most_combinatorial(lits: tuple[int, ...], k: int) -> list[Clause]:
        """Encodes sum(lits) <= k directly via C(m, k + 1) negative clauses (0 aux vars)."""
        m = len(lits)
        if k >= m:
            return []
        if k < 0:
            return [frozenset()]
        return [frozenset(-lit for lit in subset) for subset in combinations(lits, k + 1)]

    @staticmethod
    def _encode_at_most_sequential(
        lits: tuple[int, ...],
        k: int,
        next_aux_var: int,
    ) -> tuple[list[Clause], int]:
        """Encodes sum(lits) <= k via Sinz's O(m * k) sequential counter network."""
        m = len(lits)
        if k >= m:
            return [], next_aux_var
        if k < 0:
            return [frozenset()], next_aux_var
        if k == 0:
            return [frozenset({-lit}) for lit in lits], next_aux_var

        # Allocate auxiliary register variables s[i][j] for i in 0..m-2, j in 0..k-1
        # Meaning: s[i][j] is True iff sum(lits[0..i]) >= j + 1
        s: list[list[int]] = []
        for _ in range(m - 1):
            row: list[int] = []
            for _ in range(k):
                row.append(next_aux_var)
                next_aux_var += 1
            s.append(row)

        clauses: list[Clause] = [frozenset({-lits[0], s[0][0]})]
        for j in range(1, k):
            clauses.append(frozenset({-s[0][j]}))

        for i in range(1, m - 1):
            clauses.append(frozenset({-lits[i], s[i][0]}))
            clauses.append(frozenset({-s[i - 1][0], s[i][0]}))
            for j in range(1, k):
                clauses.append(frozenset({-lits[i], -s[i - 1][j - 1], s[i][j]}))
                clauses.append(frozenset({-s[i - 1][j], s[i][j]}))
            clauses.append(frozenset({-lits[i], -s[i - 1][k - 1]}))

        clauses.append(frozenset({-lits[m - 1], -s[m - 2][k - 1]}))
        return clauses, next_aux_var

    def _encode_at_most(
        self,
        lits: tuple[int, ...],
        k: int,
        next_aux_var: int,
    ) -> tuple[list[Clause], int]:
        """Dispatches AT_MOST(k) to the configured cardinality encoding strategy."""
        if self.cardinality_mode == "sequential":
            return self._encode_at_most_sequential(lits, k, next_aux_var)
        return self._encode_at_most_combinatorial(lits, k), next_aux_var

    def encode_single_clue(
        self,
        owner: Coordinate,
        clue: Clue,
        grid_size: int,
        var_map: dict[Coordinate, int],
        next_aux_var: int,
    ) -> tuple[list[Clause], int]:
        """Encodes one Clue into CNF clauses and returns (clauses, updated_next_aux_var)."""
        coords = self.validate_and_resolve(clue, grid_size, owner)

        # Polarity normalization: +v_c means Criminal, -v_c means Innocent
        sign = 1 if clue.target_status == HiddenStatus.CRIMINAL else -1
        lits = tuple(sign * var_map[coord] for coord in coords)
        m = len(lits)

        match clue.clue_type:
            case ClueType.FACT:
                return [frozenset({lit}) for lit in lits], next_aux_var

            case ClueType.SAME:
                clauses: list[Clause] = []
                for i in range(m - 1):
                    clauses.append(frozenset({-lits[i], lits[i + 1]}))
                    clauses.append(frozenset({lits[i], -lits[i + 1]}))
                return clauses, next_aux_var

            case ClueType.DIFFERENT:
                return [
                    frozenset(lits),
                    frozenset(-lit for lit in lits),
                ], next_aux_var

            case ClueType.AT_MOST:
                assert clue.count is not None
                return self._encode_at_most(lits, clue.count, next_aux_var)

            case ClueType.AT_LEAST:
                assert clue.count is not None
                # sum(L_i) >= k  <=>  sum(~L_i) <= m - k
                neg_lits = tuple(-lit for lit in lits)
                return self._encode_at_most(neg_lits, m - clue.count, next_aux_var)

            case ClueType.EXACTLY:
                assert clue.count is not None
                at_most_clauses, next_aux_var = self._encode_at_most(
                    lits, clue.count, next_aux_var
                )
                neg_lits = tuple(-lit for lit in lits)
                at_least_clauses, next_aux_var = self._encode_at_most(
                    neg_lits, m - clue.count, next_aux_var
                )
                return [*at_most_clauses, *at_least_clauses], next_aux_var

            case ClueType.PARITY:
                assert clue.count is not None
                return self._encode_parity(lits, clue.count, next_aux_var)


class SemanticEvaluator:
    """Direct semantic evaluator verifying clues on a full assignment WITHOUT CNF.

    Semantics (shared contract with ``CNFEncoder.encode_single_clue``). Let ``R`` be the
    region resolved by ``validate_and_resolve`` (``exclude_self`` already applied) and
    ``hits`` the number of cells in ``R`` whose status equals ``clue.target_status``:

    - FACT:       every cell of ``R`` has status ``target_status`` (``hits == |R|``).
    - SAME:       all cells of ``R`` share one status (``target_status`` is irrelevant).
    - DIFFERENT:  the cells of ``R`` do NOT all share one status (exactly-different for
                  a 2-cell region; ``target_status`` is irrelevant).
    - EXACTLY:    ``hits == count``.
    - AT_LEAST:   ``hits >= count``.
    - AT_MOST:    ``hits <= count``.
    - PARITY:     ``hits % 2 == count`` (``count`` 0 = EVEN, 1 = ODD).
    """

    @staticmethod
    def evaluate_clue(
        clue: Clue,
        assignment: dict[Coordinate, HiddenStatus],
        grid_size: int,
        owner_coord: Coordinate | None = None,
    ) -> bool:
        """Returns True iff `clue` holds under the complete `assignment`."""
        coords = CNFEncoder.validate_and_resolve(clue, grid_size, owner_coord)

        missing = [coord for coord in coords if coord not in assignment]
        if missing:
            raise ValueError(
                f"Assignment is incomplete: missing {missing} required by '{clue.text}'."
            )

        statuses = [assignment[coord] for coord in coords]
        hits = sum(1 for status in statuses if status == clue.target_status)

        match clue.clue_type:
            case ClueType.FACT:
                return hits == len(coords)
            case ClueType.SAME:
                return len(set(statuses)) == 1
            case ClueType.DIFFERENT:
                return len(set(statuses)) > 1
            case ClueType.EXACTLY:
                assert clue.count is not None
                return hits == clue.count
            case ClueType.AT_LEAST:
                assert clue.count is not None
                return hits >= clue.count
            case ClueType.AT_MOST:
                assert clue.count is not None
                return hits <= clue.count
            case ClueType.PARITY:
                assert clue.count is not None
                return hits % 2 == clue.count
            case _:
                raise ValueError(f"Unsupported ClueType: {clue.clue_type}")
