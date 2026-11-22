"""Unit tests verifying the strict data boundary between GameEngine and LogicAgent."""

from agent import LogicAgent
from engine import GameEngine
from models import PublicCellView, RevealedStatus


def test_public_state_hides_unrevealed_clues_and_hidden_status() -> None:
    engine = GameEngine(grid_size=3)
    public_state = engine.load_puzzle(GameEngine.create_sample_puzzle(grid_size=3))
    agent = LogicAgent()
    agent.observe(public_state)

    for cell_view in public_state.cells.values():
        assert isinstance(cell_view, PublicCellView)
        assert not hasattr(cell_view, "hidden_status")
        if cell_view.revealed_status == RevealedStatus.UNKNOWN:
            assert cell_view.revealed_clue is None
        else:
            assert cell_view.revealed_clue is not None
