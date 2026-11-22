"""JSON puzzle loader and directory discovery for Griductive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import (
    VALID_GRID_SIZES,
    CharacterCell,
    Clue,
    ClueRegion,
    ClueType,
    Coordinate,
    HiddenStatus,
    PuzzleDefinition,
    RegionType,
)


class PuzzleLoader:
    """Loads validated `PuzzleDefinition` objects from JSON files in `Source/puzzles/`."""

    @staticmethod
    def parse_coord(raw: str | list[int] | tuple[int, int], grid_size: int) -> Coordinate:
        """Parses either alphanumeric ('A1', 'C2') or 0-based [row, col] coordinates."""
        if isinstance(raw, str):
            token = raw.strip().upper()
            if len(token) < 2 or not token[0].isalpha() or not token[1:].isdigit():
                raise ValueError(f"Invalid coordinate string: {raw!r}")
            col = ord(token[0]) - ord("A")
            row = int(token[1:]) - 1
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            row, col = int(raw[0]), int(raw[1])
        else:
            raise ValueError(f"Unsupported coordinate format: {raw!r}")

        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError(
                f"Coordinate {(row, col)} out of bounds for {grid_size}x{grid_size}."
            )
        return (row, col)

    @classmethod
    def load_from_json(cls, file_path: Path | str) -> PuzzleDefinition:
        """Parses a JSON puzzle file into a `PuzzleDefinition`."""
        path = Path(file_path)
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

        grid_size = int(data["grid_size"])
        if grid_size not in VALID_GRID_SIZES:
            raise ValueError(f"Unsupported grid_size {grid_size} in {path.name}.")

        initial_revealed = frozenset(
            cls.parse_coord(item, grid_size) for item in data["initial_revealed"]
        )

        cells: list[CharacterCell] = []
        seen_coords: set[Coordinate] = set()
        for entry in data["cells"]:
            coord = cls.parse_coord(entry["coord"], grid_size)
            if coord in seen_coords:
                raise ValueError(f"Duplicate cell coordinate {coord} in {path.name}.")
            seen_coords.add(coord)

            clue_data = entry["clue"]
            reg_data = clue_data["region"]
            region_type = RegionType(reg_data["type"].upper())

            center = (
                cls.parse_coord(reg_data["center"], grid_size)
                if reg_data.get("center") is not None
                else None
            )
            explicit_coords = tuple(
                cls.parse_coord(c, grid_size)
                for c in reg_data.get("explicit_coords", ())
            )

            region = ClueRegion(
                region_type=region_type,
                index=reg_data.get("index"),
                center=center,
                explicit_coords=explicit_coords,
                exclude_self=bool(reg_data.get("exclude_self", False)),
            )
            clue = Clue(
                clue_type=ClueType(clue_data["type"].upper()),
                region=region,
                target_status=HiddenStatus(
                    clue_data.get("target_status", "CRIMINAL").upper()
                ),
                count=clue_data.get("count"),
                text=str(clue_data.get("text", "")),
            )
            cells.append(
                CharacterCell(
                    coord=coord,
                    name=str(entry["name"]),
                    profession=str(entry["profession"]),
                    hidden_status=HiddenStatus(entry["hidden_status"].upper()),
                    clue=clue,
                )
            )

        if len(cells) != grid_size * grid_size:
            raise ValueError(
                f"Expected {grid_size * grid_size} cells in {path.name}, got {len(cells)}."
            )

        total_criminals = sum(
            1 for cell in cells if cell.hidden_status == HiddenStatus.CRIMINAL
        )
        return PuzzleDefinition(
            grid_size=grid_size,
            cells=tuple(cells),
            initial_revealed=initial_revealed,
            puzzle_id=str(data.get("puzzle_id", path.stem)),
            total_criminals=total_criminals,
        )

    @staticmethod
    def discover_puzzles(puzzles_dir: Path | str | None = None) -> list[Path]:
        """Returns sorted `.json` files in `Source/puzzles/` containing a full 'cells' list."""
        search_dir = (
            Path(__file__).resolve().parent.parent / "puzzles"
            if puzzles_dir is None
            else Path(puzzles_dir)
        )
        if not search_dir.is_dir():
            return []

        valid_paths: list[Path] = []
        for path in sorted(search_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "cells" in raw:
                    valid_paths.append(path)
            except (OSError, json.JSONDecodeError):
                continue
        return valid_paths