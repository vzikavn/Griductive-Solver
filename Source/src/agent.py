"""Player-perspective LogicAgent for the Griductive puzzle game."""

from __future__ import annotations

from dataclasses import dataclass

from encoder import CNFEncoder, EncodedKB
from models import (
    CellEntailment,
    Clause,
    Clue,
    Coordinate,
    DeductionStep,
    HiddenStatus,
    PublicKnowledgeState,
)
from solver import DPLLResult, DPLLSolver


@dataclass(frozen=True, slots=True)
class DeducedMove:
    """Represents a logically proven move discovered by the LogicAgent."""

    coord: Coordinate
    proved_status: HiddenStatus
    explanation: str
    sat_queries_used: int = 0


@dataclass(slots=True)
class AgentMetrics:
    """Cumulative solver statistics across a deduction session."""

    sat_calls: int = 0
    decisions: int = 0
    propagations: int = 0
    backtracks: int = 0
    solver_runtime_sec: float = 0.0

    def record(self, result: DPLLResult) -> None:
        """Accumulates metrics from a single DPLL solver invocation."""
        self.sat_calls += 1
        self.decisions += result.decisions
        self.propagations += result.propagations
        self.backtracks += result.backtracks
        self.solver_runtime_sec += result.runtime_sec


class LogicAgent:
    """Autonomous reasoning agent representing the player's logical perspective."""

    def __init__(
        self,
        encoder: CNFEncoder | None = None,
        solver: DPLLSolver | None = None,
    ) -> None:
        self._encoder: CNFEncoder = encoder if encoder is not None else CNFEncoder()
        self._solver: DPLLSolver = solver if solver is not None else DPLLSolver()
        self._knowledge_state: PublicKnowledgeState | None = None
        self._last_encoded_kb: EncodedKB | None = None
        self._metrics: AgentMetrics = AgentMetrics()
        self._trace: list[DeductionStep] = []

    @property
    def knowledge_state(self) -> PublicKnowledgeState | None:
        """Returns the latest synchronized public knowledge state."""
        return self._knowledge_state

    @property
    def last_encoded_kb(self) -> EncodedKB | None:
        """Returns the most recently compiled CNF knowledge base and stats."""
        return self._last_encoded_kb

    @property
    def metrics(self) -> AgentMetrics:
        """Returns cumulative SAT solver metrics for experiment reporting."""
        return self._metrics

    @property
    def trace(self) -> tuple[DeductionStep, ...]:
        """Returns the recorded deduction trace."""
        return tuple(self._trace)

    def reset_session(self) -> None:
        """Clears accumulated metrics and deduction trace for a new puzzle."""
        self._metrics = AgentMetrics()
        self._trace.clear()
        self._last_encoded_kb = None

    def observe(self, public_state: PublicKnowledgeState) -> None:
        """Synchronizes the agent's internal beliefs with a new public board snapshot."""
        self._knowledge_state = public_state
        self._last_encoded_kb = None

    def _get_encoded_kb(self) -> EncodedKB:
        """Returns cached EncodedKB for the current public state, encoding if needed."""
        if self._knowledge_state is None:
            raise RuntimeError("LogicAgent has not observed a PublicKnowledgeState yet.")
        if self._last_encoded_kb is None:
            self._last_encoded_kb = self._encoder.encode_knowledge_base(
                self._knowledge_state
            )
        return self._last_encoded_kb

    def encode_public_knowledge_to_cnf(self) -> list[Clause]:
        """Translates all currently revealed statuses and clues into CNF clauses."""
        return list(self._get_encoded_kb().clauses)

    def classify_cell(self, coord: Coordinate) -> tuple[CellEntailment, int]:
        """Classifies an unsolved cell via SAT refutation."""
        kb = self._get_encoded_kb()
        var_id = kb.var_map[coord]
        num_vars = kb.stats.total_vars

        res_assume_innocent = self._solver.solve(
            clauses=kb.clauses,
            num_vars=num_vars,
            assumptions=(-var_id,),
        )
        self._metrics.record(res_assume_innocent)

        res_assume_criminal = self._solver.solve(
            clauses=kb.clauses,
            num_vars=num_vars,
            assumptions=(var_id,),
        )
        self._metrics.record(res_assume_criminal)

        if not res_assume_innocent.is_sat and not res_assume_criminal.is_sat:
            return CellEntailment.INCONSISTENT, 2
        if not res_assume_innocent.is_sat:
            return CellEntailment.CRIMINAL, 2
        if not res_assume_criminal.is_sat:
            return CellEntailment.INNOCENT, 2
        return CellEntailment.UNKNOWN, 2

    def check_cell_entailment(
        self,
        coord: Coordinate,
        candidate_status: HiddenStatus,
    ) -> bool:
        """Returns True iff `candidate_status` at `coord` is strictly entailed by KB_t."""
        entailment, _ = self.classify_cell(coord)
        if candidate_status == HiddenStatus.CRIMINAL:
            return entailment == CellEntailment.CRIMINAL
        return entailment == CellEntailment.INNOCENT

    def get_hint(self) -> DeducedMove | None:
        """Finds the first logically forced character in deterministic row-major order."""
        if self._knowledge_state is None:
            return None

        total_queries = 0
        for coord in self._knowledge_state.get_unknown_coordinates():
            entailment, queries = self.classify_cell(coord)
            total_queries += queries

            if entailment == CellEntailment.CRIMINAL:
                return DeducedMove(
                    coord=coord,
                    proved_status=HiddenStatus.CRIMINAL,
                    explanation="Entailed by public clues (KB & ~C is UNSAT).",
                    sat_queries_used=total_queries,
                )
            if entailment == CellEntailment.INNOCENT:
                return DeducedMove(
                    coord=coord,
                    proved_status=HiddenStatus.INNOCENT,
                    explanation="Entailed by public clues (KB & C is UNSAT).",
                    sat_queries_used=total_queries,
                )
            if entailment == CellEntailment.INCONSISTENT:
                return None

        return None

    def compute_next_auto_solve_step(self) -> DeducedMove | None:
        """Computes the next deterministic forced move for Auto Solve (never guesses)."""
        return self.get_hint()

    def record_accepted_move(
        self,
        move: DeducedMove,
        newly_revealed_clue: str | None,
    ) -> None:
        """Appends a confirmed deduction step to the trace."""
        if self._knowledge_state is None:
            return
        active_coords = tuple(sorted(self._knowledge_state.get_revealed_clues()))
        self._trace.append(
            DeductionStep(
                step_number=len(self._trace) + 1,
                active_clue_coords=active_coords,
                sat_queries=move.sat_queries_used,
                target_coord=move.coord,
                verdict=move.proved_status,
                newly_revealed_clue=newly_revealed_clue,
            )
        )

    def check_uniqueness(
        self,
        grid_size: int,
        all_clues: dict[Coordinate, Clue],
        initial_statuses: dict[Coordinate, HiddenStatus] | None = None,
    ) -> tuple[str, dict[Coordinate, HiddenStatus] | None]:
        """Checks whether a complete clue set has UNSAT, UNIQUE, or MULTIPLE models."""
        kb = self._encoder.encode_clue_set(
            grid_size=grid_size,
            clues=all_clues,
            known_statuses=initial_statuses,
        )
        first = self._solver.solve(clauses=kb.clauses, num_vars=kb.stats.total_vars)
        self._metrics.record(first)

        if not first.is_sat or first.model is None:
            return "UNSAT", None

        first_assignment: dict[Coordinate, HiddenStatus] = {
            coord: (
                HiddenStatus.CRIMINAL if first.model[var_id] else HiddenStatus.INNOCENT
            )
            for coord, var_id in kb.var_map.items()
        }

        blocking_clause = frozenset(
            -var_id if first.model[var_id] else var_id
            for var_id in kb.var_map.values()
        )
        second = self._solver.solve(
            clauses=(*kb.clauses, blocking_clause),
            num_vars=kb.stats.total_vars,
        )
        self._metrics.record(second)

        if not second.is_sat:
            return "UNIQUE", first_assignment
        return "MULTIPLE", first_assignment