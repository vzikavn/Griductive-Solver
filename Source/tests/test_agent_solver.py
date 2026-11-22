"""Unit tests verifying LogicAgent solver placeholders raise NotImplementedError."""

import pytest

from agent import LogicAgent
from engine import GameEngine


def test_solver_methods_raise_not_implemented() -> None:
    engine = GameEngine(grid_size=3)
    state = engine.load_puzzle(GameEngine.create_sample_puzzle(grid_size=3))
    agent = LogicAgent()
    agent.observe(state)

    with pytest.raises(NotImplementedError):
        agent.encode_public_knowledge_to_cnf()

    with pytest.raises(NotImplementedError):
        agent.get_hint()
