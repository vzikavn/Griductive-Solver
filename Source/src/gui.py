"""Pygame GUI layout and interactive rendering loop for Griductive (Sunset Dark Theme).

Cleanly separates presentation and user input from `GameEngine` and `LogicAgent`.
Features:
- Borderless Windowed Fullscreen support (`pygame.NOFRAME` at desktop resolution via `F11` or `Alt+Enter`)
  plus Maximized / Resizable Windowed mode so Alt-Tab is instant and never locks the display
- Custom Character-by-Character Tracking (Letter-Spacing) Renderer for display font
  so glyphs never collide or look squished
- Sunset Gradient Background (Deep Violet -> Purple -> Magenta -> Coral Pink -> Coral -> Orange -> Warm Yellow)
- Original Griductive two-column layout that scales dynamically to any window resolution
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pygame

from agent import LogicAgent
from engine import GameEngine
from models import (
    Coordinate,
    HiddenStatus,
    PublicCellView,
    PublicKnowledgeState,
    RevealedStatus,
    VerdictResponseType,
)
from puzzle_loader import PuzzleLoader

# Column letters for alphanumeric grid labels (A1, B1, C1...)
COL_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")


@dataclass(frozen=True, slots=True)
class SunsetPalette:
    """Sunset gradient palette and high-contrast card theme."""

    # 7 requested sunset gradient stops (top to bottom)
    deep_violet: tuple[int, int, int] = (28, 12, 58)
    purple: tuple[int, int, int] = (88, 24, 122)
    magenta: tuple[int, int, int] = (186, 30, 126)
    coral_pink: tuple[int, int, int] = (232, 74, 127)
    coral: tuple[int, int, int] = (247, 118, 98)
    orange: tuple[int, int, int] = (252, 158, 66)
    warm_yellow: tuple[int, int, int] = (255, 214, 92)

    # Card & surface colors (sleek dark cards matching the original game)
    card_unknown_bg: tuple[int, int, int] = (32, 24, 44)
    card_innocent_bg: tuple[int, int, int] = (26, 62, 54)
    card_criminal_bg: tuple[int, int, int] = (78, 24, 46)

    card_border_default: tuple[int, int, int] = (110, 88, 135)
    card_border_innocent: tuple[int, int, int] = (82, 214, 154)
    card_border_criminal: tuple[int, int, int] = (255, 92, 118)
    card_border_highlight: tuple[int, int, int] = (255, 214, 92)
    card_border_selected: tuple[int, int, int] = (255, 255, 255)

    # Typography colors
    text_white: tuple[int, int, int] = (255, 255, 255)
    text_black: tuple[int, int, int] = (14, 10, 22)
    text_muted: tuple[int, int, int] = (210, 198, 226)
    text_coord: tuple[int, int, int] = (185, 170, 205)

    # Button & pill surfaces
    pill_bg: tuple[int, int, int] = (28, 18, 42)
    pill_hover: tuple[int, int, int] = (58, 34, 82)
    pill_border: tuple[int, int, int] = (232, 74, 127)

    @property
    def gradient_stops(self) -> tuple[tuple[int, int, int], ...]:
        """Returns the ordered 7-stop sunset color gradient."""
        return (
            self.deep_violet,
            self.purple,
            self.magenta,
            self.coral_pink,
            self.coral,
            self.orange,
            self.warm_yellow,
        )


PALETTE = SunsetPalette()

# Procedural avatar color presets so every suspect has a distinct portrait
AVATAR_SKIN_TONES: tuple[tuple[int, int, int], ...] = (
    (242, 198, 168),
    (216, 162, 126),
    (168, 114, 84),
    (118, 76, 54),
    (248, 214, 188),
)
AVATAR_HAIR_COLORS: tuple[tuple[int, int, int], ...] = (
    (42, 32, 30),
    (224, 178, 84),
    (164, 78, 44),
    (82, 54, 42),
    (28, 26, 36),
)
AVATAR_SHIRT_COLORS: tuple[tuple[int, int, int], ...] = (
    (232, 74, 127),
    (66, 153, 225),
    (252, 158, 66),
    (159, 122, 234),
    (72, 187, 120),
    (237, 100, 166),
)


@dataclass(slots=True)
class UIButton:
    """Clickable pill button in the UI."""

    label: str
    rect: pygame.Rect
    callback: Callable[[], None]
    accent_color: tuple[int, int, int] = PALETTE.pill_border


class GriductiveGUI:
    """Interactive Pygame frontend for Griductive with Borderless Windowed Fullscreen support."""

    DEFAULT_WIDTH: int = 1280
    DEFAULT_HEIGHT: int = 840
    MIN_WIDTH: int = 960
    MIN_HEIGHT: int = 680
    FPS: int = 60


    AUTO_SOLVE_INTERVAL_MS: int = 350

    def __init__(
        self,
        engine: GameEngine,
        agent: LogicAgent,
        puzzle_paths: list[Path] | None = None,
        start_windowed_fullscreen: bool = True,
    ) -> None:
        self._engine: GameEngine = engine
        self._agent: LogicAgent = agent

        self._public_state: PublicKnowledgeState = self._engine.get_public_knowledge_state()
        self._agent.observe(self._public_state)

        self._selected_coord: Coordinate | None = None
        self._active_clue_owner: Coordinate | None = None
        self._highlighted_coords: frozenset[Coordinate] = frozenset()

        self._status_message: str = (
            "Tap a suspect to call them criminal or innocent. "
            "Press F11 to toggle Windowed Fullscreen."
        )
        self._status_color: tuple[int, int, int] = PALETTE.warm_yellow

        # Windowed Fullscreen (Borderless) vs Standard Resizable Window state
        self._win_w: int = self.DEFAULT_WIDTH
        self._win_h: int = self.DEFAULT_HEIGHT
        self._desktop_size: tuple[int, int] = (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self._windowed_size: tuple[int, int] = (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self._is_windowed_fullscreen: bool = start_windowed_fullscreen

        # Computed layout regions (updated dynamically on resize)
        self._left_panel_x: int = 56
        self._left_panel_w: int = 400
        self._grid_area_rect: pygame.Rect = pygame.Rect(490, 28, 680, 710)
        self._bottom_pills_y: int = 760

        self._cell_rects: dict[Coordinate, pygame.Rect] = {}
        self._control_pills: list[UIButton] = []
        self._window_mode_btn: UIButton | None = None
        self._quit_btn: UIButton | None = None
        self._running: bool = False

        # Pygame surfaces & fonts
        self._screen: pygame.Surface | None = None
        self._bg_surface: pygame.Surface | None = None
        self._font_logo: pygame.font.Font | None = None
        self._font_section: pygame.font.Font | None = None
        self._font_card_name: pygame.font.Font | None = None
        self._font_card_name_small: pygame.font.Font | None = None
        self._font_card_prof: pygame.font.Font | None = None
        self._font_coord: pygame.font.Font | None = None
        self._font_clue: pygame.font.Font | None = None
        self._font_body: pygame.font.Font | None = None
        self._font_pill: pygame.font.Font | None = None

        # Puzzle loader & continuous Auto Solve state (4 dòng mới thêm)
        self._puzzle_paths: list[Path] = (
            list(puzzle_paths)
            if puzzle_paths is not None
            else PuzzleLoader.discover_puzzles()
        )
        self._current_puzzle_idx: int = 0
        self._auto_solving: bool = False
        self._last_auto_step_ms: int = 0

    def _load_display_font(self, size: int) -> pygame.font.Font:
        """Loads Pricedown Black (.ttf/.otf) from disk or system fonts without artificial bold fattening."""
        search_dirs = [
            Path.cwd(),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent / "assets" / "fonts",
            Path(__file__).resolve().parent.parent / "assets" / "fonts",
        ]
        candidate_names = (
            "pricedown bl.otf",
            "pricedown bl.ttf",
            "Pricedown Black.ttf",
            "Pricedown Black.otf",
            "pricedown.ttf",
            "pricedown.otf",
        )
        for folder in search_dirs:
            for fname in candidate_names:
                fpath = folder / fname
                if fpath.is_file():
                    return pygame.font.Font(str(fpath), size)

        for sys_name in ("pricedownblack", "pricedown", "impact", "arialblack"):
            matched = pygame.font.match_font(sys_name)
            if matched:
                return pygame.font.Font(matched, size)

        return pygame.font.SysFont("dejavusans", size, bold=True)

    def _detect_desktop_resolution(self) -> tuple[int, int]:
        """Detects the primary monitor's native desktop resolution before creating the window."""
        desktop_sizes = pygame.display.get_desktop_sizes()
        if desktop_sizes:
            return desktop_sizes[0]
        info = pygame.display.Info()
        if info.current_w > 0 and info.current_h > 0:
            return (info.current_w, info.current_h)
        return (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

    def _init_pygame(self) -> None:
        """Initializes Pygame in Borderless Windowed Fullscreen (or Resizable Window) mode."""
        pygame.init()
        pygame.display.set_caption(
            "Griductive — Logical Deduction Solver (F11: Windowed Fullscreen)"
        )

        self._desktop_size = self._detect_desktop_resolution()
        # Default fallback windowed size is 85% of desktop resolution
        self._windowed_size = (
            max(self.MIN_WIDTH, int(self._desktop_size[0] * 0.85)),
            max(self.MIN_HEIGHT, int(self._desktop_size[1] * 0.85)),
        )

        # Load Pricedown Black for headers and clean geometric sans for small UI/button labels
        self._font_logo = self._load_display_font(52)
        self._font_section = self._load_display_font(22)
        self._font_card_name = self._load_display_font(21)
        self._font_card_name_small = self._load_display_font(17)
        self._font_pill = pygame.font.SysFont("DejaVu Sans", 14, bold=True)

        self._font_card_prof = pygame.font.SysFont("DejaVu Sans Mono,DejaVu Sans", 12, bold=True)
        self._font_coord = pygame.font.SysFont("DejaVu Sans Mono,DejaVu Sans", 12, bold=True)
        self._font_clue = pygame.font.SysFont("DejaVu Sans", 13, bold=False)
        self._font_body = pygame.font.SysFont("DejaVu Sans", 14, bold=False)

        self._apply_window_mode()

    def _apply_window_mode(self) -> None:
        """Applies either Borderless Windowed Fullscreen (NOFRAME at desktop size) or Resizable Window."""
        if self._is_windowed_fullscreen:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
            dw, dh = self._desktop_size
            self._screen = pygame.display.set_mode((dw, dh), pygame.NOFRAME)
            self._on_window_resized(dw, dh)
        else:
            os.environ["SDL_VIDEO_CENTERED"] = "1"
            ww, wh = self._windowed_size
            self._screen = pygame.display.set_mode((ww, wh), pygame.RESIZABLE)
            self._on_window_resized(ww, wh)

    def toggle_windowed_fullscreen(self) -> None:
        """Toggles between Borderless Windowed Fullscreen and standard Resizable Window."""
        if not self._is_windowed_fullscreen:
            # Save current resizable window dimensions before expanding to borderless fullscreen
            self._windowed_size = (self._win_w, self._win_h)
        self._is_windowed_fullscreen = not self._is_windowed_fullscreen
        self._apply_window_mode()

    def _on_window_resized(self, width: int, height: int) -> None:
        """Recalculates all responsive panel coordinates, gradient surface, and grid cards."""
        self._win_w = max(self.MIN_WIDTH, width)
        self._win_h = max(self.MIN_HEIGHT, height)

        max_content_w = min(self._win_w - 80, 1540)
        content_left = (self._win_w - max_content_w) // 2

        self._left_panel_x = content_left
        self._left_panel_w = min(430, int(max_content_w * 0.34))

        grid_left = self._left_panel_x + self._left_panel_w + 44
        avail_grid_w = max_content_w - self._left_panel_w - 44
        avail_grid_h = self._win_h - 118

        grid_w = min(avail_grid_w, int(avail_grid_h * 1.05))
        grid_h = min(avail_grid_h, int(grid_w * 1.04))
        grid_x = grid_left + (avail_grid_w - grid_w) // 2
        grid_y = max(28, (self._win_h - 76 - grid_h) // 2)

        self._grid_area_rect = pygame.Rect(grid_x, grid_y, grid_w, grid_h)
        self._bottom_pills_y = self._grid_area_rect.bottom + 18

        self._bg_surface = self._create_sunset_gradient_surface(self._win_w, self._win_h)
        self._build_control_pills()
        self._recompute_grid_layout()

    def _create_sunset_gradient_surface(self, width: int, height: int) -> pygame.Surface:
        """Pre-renders the 7-color sunset gradient at the current window resolution."""
        surf = pygame.Surface((width, height))
        stops = PALETTE.gradient_stops
        num_segments = len(stops) - 1

        for y in range(height):
            t = y / max(1, height - 1)
            seg_float = t * num_segments
            seg_idx = min(int(seg_float), num_segments - 1)
            local_t = seg_float - seg_idx

            c1 = stops[seg_idx]
            c2 = stops[seg_idx + 1]
            r = int(c1[0] + (c2[0] - c1[0]) * local_t)
            g = int(c1[1] + (c2[1] - c1[1]) * local_t)
            b = int(c1[2] + (c2[2] - c1[2]) * local_t)
            pygame.draw.line(surf, (r, g, b), (0, y), (width, y))

        veil = pygame.Surface((width, height), pygame.SRCALPHA)
        veil.fill((18, 8, 34, 85))
        surf.blit(veil, (0, 0))
        return surf

    def _build_control_pills(self) -> None:
        """Builds the bottom control pills and top window-mode / quit buttons."""
        labels_and_callbacks: list[tuple[str, Callable[[], None], tuple[int, int, int]]] = [
            ("Load (3x3 / 4x4 / 5x5)", self._on_load_clicked, PALETTE.coral_pink),
            ("Restart", self._on_restart_clicked, PALETTE.coral),
            ("Hint", self._on_hint_clicked, PALETTE.warm_yellow),
            ("Auto Solve", self._on_auto_solve_clicked, PALETTE.orange),
        ]

        pill_h = 38
        gap = 12
        pill_widths = [190, 125, 115, 140]
        total_w = sum(pill_widths) + (len(pill_widths) - 1) * gap
        start_x = self._grid_area_rect.centerx - total_w // 2

        self._control_pills = []
        curr_x = start_x
        for (label, cb, accent), pw in zip(labels_and_callbacks, pill_widths):
            rect = pygame.Rect(curr_x, self._bottom_pills_y, pw, pill_h)
            self._control_pills.append(
                UIButton(label=label, rect=rect, callback=cb, accent_color=accent)
            )
            curr_x += pw + gap

        mode_label = (
            "Windowed Mode (F11)"
            if self._is_windowed_fullscreen
            else "Windowed Fullscreen (F11)"
        )
        self._window_mode_btn = UIButton(
            label=mode_label,
            rect=pygame.Rect(self._left_panel_x, 28, 214, 34),
            callback=self.toggle_windowed_fullscreen,
            accent_color=PALETTE.warm_yellow,
        )

        self._quit_btn = UIButton(
            label="Quit (ESC)",
            rect=pygame.Rect(self._left_panel_x + 226, 28, 104, 34),
            callback=self._quit_game,
            accent_color=PALETTE.coral_pink,
        )

    def _quit_game(self) -> None:
        """Cleanly stops the GUI loop."""
        self._running = False

    def _recompute_grid_layout(self) -> None:
        """Calculates clean card rectangles for the active N x N grid."""
        self._cell_rects.clear()
        n = self._public_state.grid_size
        gap = 12
        avail_w = self._grid_area_rect.width - gap * (n - 1)
        avail_h = self._grid_area_rect.height - gap * (n - 1)
        card_w = avail_w // n
        card_h = avail_h // n

        for r in range(n):
            for c in range(n):
                x = self._grid_area_rect.x + c * (card_w + gap)
                y = self._grid_area_rect.y + r * (card_h + gap)
                self._cell_rects[(r, c)] = pygame.Rect(x, y, card_w, card_h)

    @staticmethod
    def _format_coord_label(coord: Coordinate) -> str:
        """Converts 0-based (row, col) into original Griductive format ('A1', 'B1', 'C4')."""
        r, c = coord
        col_letter = COL_LETTERS[c] if c < len(COL_LETTERS) else f"C{c + 1}"
        return f"{col_letter}{r + 1}"

    def _get_verdict_popup_rects(self, card_rect: pygame.Rect) -> tuple[pygame.Rect, pygame.Rect]:
        """Returns the (Innocent_Button_Rect, Criminal_Button_Rect) inside a selected card."""
        pad = 8
        btn_h = 28
        btn_w = (card_rect.width - pad * 3) // 2
        y = card_rect.bottom - btn_h - 8
        inn_rect = pygame.Rect(card_rect.x + pad, y, btn_w, btn_h)
        crim_rect = pygame.Rect(card_rect.x + pad * 2 + btn_w, y, btn_w, btn_h)
        return inn_rect, crim_rect

    def run(self) -> None:
        """Starts the main synchronous Pygame event and rendering loop."""
        self._init_pygame()
        clock = pygame.time.Clock()
        self._running = True

        while self._running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11 or (
                        event.key == pygame.K_RETURN and (event.mod & pygame.KMOD_ALT)
                    ):
                        self.toggle_windowed_fullscreen()
                    elif event.key == pygame.K_ESCAPE:
                        if self._selected_coord is not None:
                            self._selected_coord = None
                        elif self._is_windowed_fullscreen:
                            self.toggle_windowed_fullscreen()
                elif event.type == pygame.VIDEORESIZE:
                    if not self._is_windowed_fullscreen:
                        self._on_window_resized(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos, event.button)

            if self._auto_solving:
                now_ms = pygame.time.get_ticks()
                if now_ms - self._last_auto_step_ms >= self.AUTO_SOLVE_INTERVAL_MS:
                    self._last_auto_step_ms = now_ms
                    self._execute_single_auto_solve_step()

            self._render_frame(mouse_pos)
            pygame.display.flip()
            clock.tick(self.FPS)

        pygame.quit()

    def _handle_mouse_click(self, pos: tuple[int, int], button: int) -> None:
        """Routes mouse clicks across window buttons, action pills, verdict popups, and cards."""
        if button == 1:
            if self._window_mode_btn is not None and self._window_mode_btn.rect.collidepoint(pos):
                self._window_mode_btn.callback()
                return

            if self._quit_btn is not None and self._quit_btn.rect.collidepoint(pos):
                self._quit_btn.callback()
                return

            for pill in self._control_pills:
                if pill.rect.collidepoint(pos):
                    pill.callback()
                    return

            if self._selected_coord is not None and self._selected_coord in self._cell_rects:
                sel_rect = self._cell_rects[self._selected_coord]
                inn_rect, crim_rect = self._get_verdict_popup_rects(sel_rect)
                if inn_rect.collidepoint(pos):
                    self._submit_selected_verdict(HiddenStatus.INNOCENT)
                    return
                if crim_rect.collidepoint(pos):
                    self._submit_selected_verdict(HiddenStatus.CRIMINAL)
                    return

        for coord, rect in self._cell_rects.items():
            if rect.collidepoint(pos):
                self._on_cell_clicked(coord, button)
                return

        self._selected_coord = None

    def _on_cell_clicked(self, coord: Coordinate, mouse_button: int) -> None:
        """Handles clicking a suspect card on the grid."""
        self._auto_solving = False 
        cell_view = self._public_state.get_cell(coord)
        label = self._format_coord_label(coord)

        if cell_view.revealed_status != RevealedStatus.UNKNOWN:
            self._selected_coord = None
            if self._active_clue_owner == coord:
                self._active_clue_owner = None
                self._highlighted_coords = frozenset()
                self._set_status(f"Cleared clue region highlight for {cell_view.name} ({label}).")
            elif cell_view.revealed_clue is not None:
                self._active_clue_owner = coord
                self._highlighted_coords = cell_view.revealed_clue.region.resolve_coordinates(
                    grid_size=self._public_state.grid_size,
                    owner_coord=coord,
                )
                self._set_status(
                    f"{cell_view.name} ({label}): \"{cell_view.revealed_clue.text}\"",
                    PALETTE.warm_yellow,
                )
            return

        if mouse_button == 3:
            self._selected_coord = coord
            self._submit_selected_verdict(HiddenStatus.CRIMINAL)
        else:
            if self._selected_coord == coord:
                self._selected_coord = None
            else:
                self._selected_coord = coord
                self._set_status(
                    f"Calling {cell_view.name} ({label}) — choose INNOCENT or CRIMINAL on the card.",
                    PALETTE.text_white,
                )

    def _submit_selected_verdict(self, proposed_status: HiddenStatus) -> None:
        """Submits a verdict for the selected suspect to GameEngine."""
        if self._selected_coord is None:
            return

        target = self._selected_coord
        try:
            result = self._engine.submit_verdict(target, proposed_status)
        except NotImplementedError as exc:
            self._set_status(f"NOT_PROVABLE (Solver placeholder: {exc})", PALETTE.warm_yellow)
            return

        self._public_state = result.public_state
        self._agent.observe(self._public_state)

        match result.response_type:
            case VerdictResponseType.ACCEPTED:
                color = PALETTE.card_border_innocent
                self._selected_coord = None
            case VerdictResponseType.NOT_PROVABLE:
                color = PALETTE.warm_yellow
            case VerdictResponseType.CONTRADICTED:
                color = PALETTE.card_border_criminal
            case VerdictResponseType.ALREADY_REVEALED:
                color = PALETTE.text_muted

        self._set_status(f"[{result.response_type}] {result.message}", color)

    def _on_load_clicked(self) -> None:
        """Cycles through JSON puzzles in `puzzles/`; falls back to sample puzzles."""
        self._auto_solving = False
        if self._puzzle_paths:
            self._current_puzzle_idx = (
                self._current_puzzle_idx + 1
            ) % len(self._puzzle_paths)
            next_path = self._puzzle_paths[self._current_puzzle_idx]
            puzzle = PuzzleLoader.load_from_json(next_path)
            self._public_state = self._engine.load_puzzle(puzzle)
            status_msg = (
                f"Loaded '{puzzle.puzzle_id}' ({puzzle.grid_size}x{puzzle.grid_size}) "
                f"from {next_path.name}."
            )
        else:
            next_size = {3: 4, 4: 5, 5: 3}.get(self._public_state.grid_size, 3)
            puzzle = GameEngine.create_sample_puzzle(grid_size=next_size)
            self._public_state = self._engine.load_puzzle(puzzle)
            status_msg = (
                f"No complete JSON in puzzles/. Loaded fallback {next_size}x{next_size}."
            )

        self._agent.reset_session()
        self._agent.observe(self._public_state)
        self._selected_coord = None
        self._active_clue_owner = None
        self._highlighted_coords = frozenset()
        self._recompute_grid_layout()
        self._set_status(status_msg, PALETTE.warm_yellow)

    def _on_restart_clicked(self) -> None:
        """Resets the current puzzle to its starting state."""
        self._auto_solving = False
        self._public_state = self._engine.restart_puzzle()
        self._agent.reset_session()
        self._agent.observe(self._public_state)
        self._selected_coord = None
        self._active_clue_owner = None
        self._highlighted_coords = frozenset()
        self._set_status("Scenario restarted.", PALETTE.text_white)

    def _on_hint_clicked(self) -> None:
        """Queries LogicAgent for a hint."""
        try:
            move = self._agent.get_hint()
        except NotImplementedError as exc:
            self._set_status(f"Hint placeholder: {exc}", PALETTE.warm_yellow)
            return

        if move is None:
            self._set_status("No forced move found by LogicAgent.", PALETTE.warm_yellow)
        else:
            self._selected_coord = move.coord
            label = self._format_coord_label(move.coord)
            self._set_status(
                f"HINT: {label} is {move.proved_status} ({move.explanation})",
                PALETTE.card_border_innocent,
            )

    def _on_auto_solve_clicked(self) -> None:
        """Toggles continuous timer-based Auto Solve ON or OFF."""
        if self._engine.is_solved:
            self._auto_solving = False
            self._set_status(
                "Puzzle is already completely solved!", PALETTE.card_border_innocent
            )
            return

        self._auto_solving = not self._auto_solving
        if self._auto_solving:
            self._last_auto_step_ms = 0
            self._set_status(
                "Auto Solve started... (Click again to pause)", PALETTE.orange
            )
        else:
            self._set_status("Auto Solve paused.", PALETTE.warm_yellow)

    def _execute_single_auto_solve_step(self) -> None:
        """Executes one deterministic deduction step for the Auto Solve timer loop."""
        if self._engine.is_solved:
            self._auto_solving = False
            self._set_status(
                "Auto Solve complete! All characters proved.",
                PALETTE.card_border_innocent,
            )
            return

        try:
            move = self._agent.compute_next_auto_solve_step()
        except NotImplementedError as exc:
            self._auto_solving = False
            self._set_status(f"Auto Solve stopped ({exc})", PALETTE.warm_yellow)
            return

        if move is None:
            self._auto_solving = False
            self._set_status(
                "Auto Solve stopped: No forced deductions available.",
                PALETTE.warm_yellow,
            )
            return

        self._selected_coord = move.coord
        self._submit_selected_verdict(move.proved_status)

        revealed_cell = self._public_state.get_cell(move.coord)
        clue_text = (
            revealed_cell.revealed_clue.text if revealed_cell.revealed_clue else None
        )
        self._agent.record_accepted_move(move, clue_text)

        if self._engine.is_solved:
            self._auto_solving = False
            self._set_status(
                "Auto Solve complete! All characters proved.",
                PALETTE.card_border_innocent,
            )

    def _set_status(
        self,
        message: str,
        color: tuple[int, int, int] = PALETTE.text_white,
    ) -> None:
        """Updates the feedback message shown in the status bar."""
        self._status_message = message
        self._status_color = color

    def _render_spaced_text(
        self,
        text: str,
        font: pygame.font.Font,
        pos: tuple[int, int],
        fg_color: tuple[int, int, int] = PALETTE.text_white,
        shadow_color: tuple[int, int, int] = PALETTE.text_black,
        shadow_offset: int = 2,
        letter_spacing: int = 3,
        center: bool = False,
    ) -> pygame.Rect:
        """Renders display text character-by-character with explicit letter-spacing (tracking)."""
        assert self._screen is not None
        if not text:
            return pygame.Rect(pos[0], pos[1], 0, 0)

        fg_glyphs = [font.render(ch, True, fg_color) for ch in text]
        sh_glyphs = [font.render(ch, True, shadow_color) for ch in text]

        total_w = (
            sum(g.get_width() for g in fg_glyphs)
            + max(0, len(fg_glyphs) - 1) * letter_spacing
        )
        max_h = max(g.get_height() for g in fg_glyphs)

        if center:
            start_x = pos[0] - total_w // 2
            start_y = pos[1] - max_h // 2
        else:
            start_x, start_y = pos

        curr_x = start_x
        for sh_surf, fg_surf in zip(sh_glyphs, fg_glyphs):
            self._screen.blit(sh_surf, (curr_x + shadow_offset, start_y + shadow_offset))
            curr_x += fg_surf.get_width() + letter_spacing

        curr_x = start_x
        for fg_surf in fg_glyphs:
            self._screen.blit(fg_surf, (curr_x, start_y))
            curr_x += fg_surf.get_width() + letter_spacing

        return pygame.Rect(start_x, start_y, total_w, max_h)

    def _render_frame(self, mouse_pos: tuple[int, int]) -> None:
        """Renders the full sunset gradient background, left info column, N x N grid, and bottom pills."""
        assert self._screen is not None
        assert self._bg_surface is not None

        self._screen.blit(self._bg_surface, (0, 0))
        self._render_left_info_panel(mouse_pos)

        for coord, rect in self._cell_rects.items():
            cell_view = self._public_state.get_cell(coord)
            self._render_suspect_card(cell_view, rect, mouse_pos)

        self._render_bottom_pills(mouse_pos)

    def _render_left_info_panel(self, mouse_pos: tuple[int, int]) -> None:
        """Renders the left branding, window mode toggles, instructions, and status log."""
        assert self._screen is not None
        assert self._font_logo is not None
        assert self._font_section is not None
        assert self._font_body is not None
        assert self._font_coord is not None
        assert self._font_card_name_small is not None
        assert self._font_clue is not None

        x = self._left_panel_x
        y = max(96, (self._win_h - 560) // 2)

        # Windowed Fullscreen / Windowed Mode button + Quit button
        for top_btn in (self._window_mode_btn, self._quit_btn):
            if top_btn is not None:
                is_hov = top_btn.rect.collidepoint(mouse_pos)
                bg = PALETTE.pill_hover if is_hov else PALETTE.pill_bg
                pygame.draw.rect(self._screen, bg, top_btn.rect, border_radius=17)
                pygame.draw.rect(
                    self._screen,
                    top_btn.accent_color,
                    top_btn.rect,
                    width=1,
                    border_radius=17,
                )
                btn_surf = self._font_coord.render(top_btn.label, True, top_btn.accent_color)
                self._screen.blit(btn_surf, btn_surf.get_rect(center=top_btn.rect.center))

        # Detective Silhouette Icon + "Griductive" Title with generous letter-spacing
        self._draw_detective_icon(x + 22, y + 30)
        self._render_spaced_text(
            "Griductive",
            self._font_logo,
            (x + 56, y),
            fg_color=PALETTE.text_white,
            shadow_color=PALETTE.text_black,
            shadow_offset=3,
            letter_spacing=4,
        )

        # LOGIC • GRID pill + LEVEL dots
        badge_rect = pygame.Rect(x, y + 76, 128, 32)
        pygame.draw.rect(self._screen, PALETTE.pill_bg, badge_rect, border_radius=16)
        pygame.draw.rect(self._screen, PALETTE.warm_yellow, badge_rect, width=1, border_radius=16)
        badge_lbl = self._font_coord.render("LOGIC  •  GRID", True, PALETTE.warm_yellow)
        self._screen.blit(badge_lbl, badge_lbl.get_rect(center=badge_rect.center))

        lvl_lbl = self._font_coord.render("LEVEL", True, PALETTE.text_white)
        self._screen.blit(lvl_lbl, (x + 148, y + 84))
        dot_colors = (
            PALETTE.warm_yellow,
            PALETTE.orange,
            PALETTE.coral,
            PALETTE.coral_pink,
            PALETTE.magenta,
        )
        for i, dcol in enumerate(dot_colors):
            pygame.draw.circle(self._screen, dcol, (x + 206 + i * 16, y + 92), 5)

        # Subtitle
        sub_surf = self._font_body.render("A daily grid-deduction puzzle.", True, PALETTE.text_white)
        self._screen.blit(sub_surf, (x, y + 128))

        # Divider line
        pygame.draw.line(
            self._screen,
            PALETTE.warm_yellow,
            (x, y + 164),
            (x + 80, y + 164),
            width=2,
        )

        # HOW TO PLAY section
        self._render_spaced_text(
            "HOW TO PLAY",
            self._font_section,
            (x, y + 182),
            fg_color=PALETTE.warm_yellow,
            shadow_offset=2,
            letter_spacing=3,
        )

        rules = [
            "Every person is criminal or innocent, and every clue is true.",
            "• Tap a suspect to call them criminal or innocent.",
            "• A right call flips the card and reveals its clue.",
            "• Only call what the clues prove.",
            "• Click any revealed clue to spotlight its region.",
        ]
        ry = y + 222
        for idx, line in enumerate(rules):
            color = PALETTE.text_white if idx == 0 else PALETTE.text_muted
            self._render_wrapped_text(
                text=line,
                font=self._font_body,
                color=color,
                x=x,
                y=ry,
                max_width=self._left_panel_w - 16,
                shadow=True,
            )
            ry += 40 if idx == 0 else 32

        # Live Feedback Status Box
        status_box = pygame.Rect(x, y + 414, self._left_panel_w - 10, 102)
        pygame.draw.rect(self._screen, PALETTE.pill_bg, status_box, border_radius=12)
        pygame.draw.rect(self._screen, self._status_color, status_box, width=2, border_radius=12)

        self._render_spaced_text(
            "VERDICT LOG",
            self._font_card_name_small,
            (status_box.x + 14, status_box.y + 10),
            fg_color=self._status_color,
            shadow_offset=1,
            letter_spacing=2,
        )
        self._render_wrapped_text(
            text=self._status_message,
            font=self._font_clue,
            color=PALETTE.text_white,
            x=status_box.x + 14,
            y=status_box.y + 38,
            max_width=status_box.width - 28,
        )

    def _draw_detective_icon(self, cx: int, cy: int) -> None:
        """Draws a sleek fedora detective silhouette icon beside the Griductive title."""
        assert self._screen is not None
        pygame.draw.polygon(
            self._screen,
            PALETTE.text_black,
            [(cx - 18, cy + 18), (cx + 18, cy + 18), (cx + 12, cy + 4), (cx - 12, cy + 4)],
        )
        pygame.draw.circle(self._screen, PALETTE.text_black, (cx, cy - 4), 8)
        pygame.draw.ellipse(self._screen, PALETTE.text_black, (cx - 16, cy - 14, 32, 7))
        pygame.draw.polygon(
            self._screen,
            PALETTE.text_black,
            [(cx - 10, cy - 12), (cx + 10, cy - 12), (cx + 7, cy - 22), (cx - 7, cy - 22)],
        )
        pygame.draw.line(
            self._screen, PALETTE.coral_pink, (cx - 8, cy - 13), (cx + 8, cy - 13), width=2
        )

    def _draw_suspect_avatar(
        self,
        coord: Coordinate,
        center_x: int,
        center_y: int,
        scale: float = 1.0,
    ) -> None:
        """Draws a clean stylized flat-vector character avatar matching the original game's portraits."""
        assert self._screen is not None
        seed = coord[0] * 7 + coord[1] * 13
        skin = AVATAR_SKIN_TONES[seed % len(AVATAR_SKIN_TONES)]
        hair = AVATAR_HAIR_COLORS[(seed * 3) % len(AVATAR_HAIR_COLORS)]
        shirt = AVATAR_SHIRT_COLORS[(seed * 5) % len(AVATAR_SHIRT_COLORS)]

        head_r = max(6, int(15 * scale))
        shoulder_w = max(14, int(34 * scale))
        shoulder_h = max(8, int(16 * scale))

        shoulder_rect = pygame.Rect(
            center_x - shoulder_w // 2,
            center_y + int(6 * scale),
            shoulder_w,
            shoulder_h,
        )
        pygame.draw.rect(
            self._screen,
            shirt,
            shoulder_rect,
            border_top_left_radius=int(10 * scale),
            border_top_right_radius=int(10 * scale),
        )

        pygame.draw.circle(
            self._screen,
            hair,
            (center_x, center_y - int(7 * scale)),
            head_r + max(1, int(2 * scale)),
        )
        pygame.draw.circle(
            self._screen,
            skin,
            (center_x, center_y - int(4 * scale)),
            head_r,
        )

        arc_rect = pygame.Rect(
            center_x - head_r,
            center_y - int(4 * scale) - head_r,
            head_r * 2,
            head_r,
        )
        pygame.draw.rect(
            self._screen,
            hair,
            arc_rect,
            border_top_left_radius=head_r,
            border_top_right_radius=head_r,
        )

        eye_offset_x = max(2, int(5 * scale))
        eye_y = center_y - int(4 * scale)
        eye_r = max(1, int(2 * scale))
        pygame.draw.circle(
            self._screen, PALETTE.text_black, (center_x - eye_offset_x, eye_y), eye_r
        )
        pygame.draw.circle(
            self._screen, PALETTE.text_black, (center_x + eye_offset_x, eye_y), eye_r
        )

    def _render_suspect_card(
        self,
        cell: PublicCellView,
        rect: pygame.Rect,
        mouse_pos: tuple[int, int],
    ) -> None:
        """Renders a grid card with crisp character spacing and original Griductive simplicity."""
        assert self._screen is not None
        assert self._font_coord is not None
        assert self._font_card_name is not None
        assert self._font_card_name_small is not None
        assert self._font_card_prof is not None
        assert self._font_clue is not None

        match cell.revealed_status:
            case RevealedStatus.UNKNOWN:
                bg_color = PALETTE.card_unknown_bg
                border_color = PALETTE.card_border_default
                border_w = 1
            case RevealedStatus.INNOCENT:
                bg_color = PALETTE.card_innocent_bg
                border_color = PALETTE.card_border_innocent
                border_w = 2
            case RevealedStatus.CRIMINAL:
                bg_color = PALETTE.card_criminal_bg
                border_color = PALETTE.card_border_criminal
                border_w = 2

        if cell.coord in self._highlighted_coords:
            border_color = PALETTE.card_border_highlight
            border_w = 3
        elif cell.coord == self._selected_coord:
            border_color = PALETTE.card_border_selected
            border_w = 3

        pygame.draw.rect(self._screen, bg_color, rect, border_radius=12)
        pygame.draw.rect(self._screen, border_color, rect, width=border_w, border_radius=12)

        coord_label = self._format_coord_label(cell.coord)
        coord_surf = self._font_coord.render(coord_label, True, PALETTE.text_coord)
        self._screen.blit(coord_surf, (rect.x + 10, rect.y + 8))

        if cell.revealed_status == RevealedStatus.UNKNOWN:
            avatar_cy = rect.y + int(rect.height * 0.38)
            self._draw_suspect_avatar(cell.coord, rect.centerx, avatar_cy, scale=1.0)

            name_y = rect.y + int(rect.height * 0.67)
            self._render_spaced_text(
                cell.name,
                self._font_card_name,
                (rect.centerx, name_y),
                fg_color=PALETTE.text_white,
                shadow_color=PALETTE.text_black,
                shadow_offset=2,
                letter_spacing=3,
                center=True,
            )

            prof_surf = self._font_card_prof.render(cell.profession, True, PALETTE.text_muted)
            self._screen.blit(
                prof_surf,
                prof_surf.get_rect(center=(rect.centerx, name_y + 23)),
            )

            if cell.coord == self._selected_coord:
                inn_rect, crim_rect = self._get_verdict_popup_rects(rect)
                for b_rect, lbl, b_col in (
                    (inn_rect, "INNOCENT", PALETTE.card_border_innocent),
                    (crim_rect, "CRIMINAL", PALETTE.card_border_criminal),
                ):
                    is_hov = b_rect.collidepoint(mouse_pos)
                    fill = b_col if is_hov else PALETTE.text_black
                    txt_col = PALETTE.text_black if is_hov else b_col
                    pygame.draw.rect(self._screen, fill, b_rect, border_radius=6)
                    pygame.draw.rect(self._screen, b_col, b_rect, width=1, border_radius=6)
                    b_surf = self._font_coord.render(lbl, True, txt_col)
                    self._screen.blit(b_surf, b_surf.get_rect(center=b_rect.center))

        else:
            mini_avatar_x = rect.centerx - 32
            mini_avatar_y = rect.y + 24
            self._draw_suspect_avatar(cell.coord, mini_avatar_x, mini_avatar_y, scale=0.55)

            self._render_spaced_text(
                cell.name,
                self._font_card_name_small,
                (mini_avatar_x + 18, rect.y + 8),
                fg_color=PALETTE.text_white,
                shadow_offset=1,
                letter_spacing=2,
            )
            prof_surf = self._font_card_prof.render(cell.profession, True, PALETTE.text_muted)
            self._screen.blit(prof_surf, (mini_avatar_x + 18, rect.y + 28))

            if cell.revealed_clue is not None:
                self._render_wrapped_text(
                    text=cell.revealed_clue.text,
                    font=self._font_clue,
                    color=PALETTE.text_white,
                    x=rect.x + 10,
                    y=rect.y + int(rect.height * 0.46),
                    max_width=rect.width - 20,
                    center_horizontal=True,
                    card_centerx=rect.centerx,
                )

    def _render_bottom_pills(self, mouse_pos: tuple[int, int]) -> None:
        """Renders the rounded bottom pill buttons with clean spacing."""
        assert self._screen is not None
        assert self._font_pill is not None

        for pill in self._control_pills:
            is_hovered = pill.rect.collidepoint(mouse_pos)
            bg = PALETTE.pill_hover if is_hovered else PALETTE.pill_bg
            pygame.draw.rect(self._screen, bg, pill.rect, border_radius=19)
            pygame.draw.rect(
                self._screen, pill.accent_color, pill.rect, width=2, border_radius=19
            )
            self._render_spaced_text(
                pill.label,
                self._font_pill,
                pill.rect.center,
                fg_color=PALETTE.text_white,
                shadow_color=PALETTE.text_black,
                shadow_offset=1,
                letter_spacing=1,
                center=True,
            )

    def _render_wrapped_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        max_width: int,
        center_horizontal: bool = False,
        card_centerx: int = 0,
        shadow: bool = False,
    ) -> None:
        """Word-wraps and renders multi-line text with optional centering and drop shadow."""
        assert self._screen is not None
        words = text.split(" ")
        lines: list[str] = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        line_height = font.get_linesize()
        for i, line in enumerate(lines):
            line_y = y + i * line_height
            if shadow:
                sh_surf = font.render(line, True, PALETTE.text_black)
                if center_horizontal:
                    self._screen.blit(
                        sh_surf, sh_surf.get_rect(center=(card_centerx + 1, line_y + 1))
                    )
                else:
                    self._screen.blit(sh_surf, (x + 1, line_y + 1))

            surf = font.render(line, True, color)
            if center_horizontal:
                self._screen.blit(surf, surf.get_rect(center=(card_centerx, line_y)))
            else:
                self._screen.blit(surf, (x, line_y))