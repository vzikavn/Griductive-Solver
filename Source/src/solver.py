"""Custom DPLL SAT Solver with instrumentation metrics for Griductive."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from models import Clause, Literal

import time


@dataclass(frozen=True, slots=True)
class DPLLResult:
    """Outcome and runtime statistics of a single DPLL SAT solver invocation."""

    is_sat: bool
    model: dict[int, bool] | None
    decisions: int
    propagations: int
    backtracks: int
    runtime_sec: float


class DPLLSolver:
    """Davis-Putnam-Logemann-Loveland (DPLL) propositional SAT solver."""

    def __init__(self, use_pure_literal: bool = False) -> None:
        self.use_pure_literal: bool = use_pure_literal

    @staticmethod
    def _eval_clause(
        clause: Clause,
        assignment: dict[int, bool]
    ) -> tuple[str, Literal | None]:
        unassigned = 0
        unit = None
        for literal in clause:
            var = abs(literal)
            expected = (literal > 0)
            if var in assignment:
                if assignment[var] == expected:
                    return ("SAT", None)
            else:
                unit = literal
                unassigned += 1

        if unassigned == 0:
            return ("CONFLICT", None)
        if unassigned == 1:
            return ("UNIT", unit)
        else:
            return ("UNRESOLVED", None)

    @classmethod
    def _unit_propagate(
        cls, 
        clauses: Sequence[Clause], 
        assignment: dict[int, bool],
        counter: list[int]
    ) -> bool:
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                status, unit = cls._eval_clause(clause, assignment)
                if status == "CONFLICT":
                    return False
                if status == "UNIT":
                    assert unit is not None

                    var = abs(unit) 
                    val = (unit > 0)
                    assignment[var] = val
                    counter[1] += 1
                    changed = True
                if status == "SAT" or status == "UNRESOLVED":
                    continue
        return True

    def _dpll(
        self, 
        clauses: Sequence[Clause], 
        num_vars: int, 
        assignment: dict[int, bool], 
        counter: list[int]
    ) -> dict[int, bool] | None:
        if not self._unit_propagate(clauses, assignment, counter):
            return None

        next_var = None
        for clause in clauses:
            status, _ = self._eval_clause(clause, assignment)
            if status == "UNRESOLVED":
                for literal in clause:
                    var = abs(literal)
                    if var not in assignment:
                        if next_var is None or var < next_var:
                            next_var = var

        if next_var is None:
            for var in range(1, num_vars + 1):
                if var not in assignment:
                    assignment[var] = False
            return assignment

        counter[0] += 1
        for val in (True, False):
            next_assignment = assignment.copy()
            next_assignment[next_var] = val
            result = self._dpll(clauses, num_vars, next_assignment, counter)
            if result is not None:
                return result
            counter[2] += 1

        return None

    def solve(
        self,
        clauses: Sequence[Clause],
        num_vars: int,
        assumptions: Sequence[Literal] = (),
    ) -> DPLLResult:
        """Solves a CNF formula under optional unit literal assumptions."""

        start = time.perf_counter()
        counter = [0, 0, 0]     # [decisions, propagations, backtracks]
        comp_clauses = list(clauses)
        for assumption in assumptions:
            comp_clauses.append(frozenset({assumption}))

        if any(len(c) == 0 for c in comp_clauses):
            return DPLLResult(False, None, 0, 0, 0, time.perf_counter() - start)

        model = self._dpll(comp_clauses, num_vars, {}, counter)

        return DPLLResult(
            is_sat=model is not None,
            model=model,
            decisions=counter[0],
            propagations=counter[1],
            backtracks=counter[2],
            runtime_sec=time.perf_counter() - start
        )


        

        
        