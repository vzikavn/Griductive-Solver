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

import math
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
    icon: str | None = None


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
        start_windowed_fullscreen: bool = False,
    ) -> None:
        self._engine: GameEngine = engine
        self._agent: LogicAgent = agent

        self._public_state: PublicKnowledgeState = self._engine.get_public_knowledge_state()
        self._agent.observe(self._public_state)

        self._selected_coord: Coordinate | None = None
        self._active_clue_owner: Coordinate | None = None
        self._highlighted_coords: frozenset[Coordinate] = frozenset()

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

        # Puzzle loader & continuous Auto Solve state
        self._puzzle_paths: list[Path] = (
            list(puzzle_paths)
            if puzzle_paths is not None
            else PuzzleLoader.discover_puzzles()
        )
        self._current_puzzle_idx: int = 0
        self._auto_solving: bool = False
        self._last_auto_step_ms: int = 0
        
        # Persistent Verdict Log History (Max 3 lines)
        self._status_log: list[tuple[str, tuple[int, int, int]]] = [
            ("Welcome to Griductive. Logic engine initialized.", PALETTE.text_muted)
        ]
        
        # Transient UI State (Renders at the bottom of the box)
        self._transient_message: str = ""
        self._transient_color: tuple[int, int, int] = PALETTE.warm_yellow
        
        # Animation & Polish
        self._pulse_timers: dict[Coordinate, int] = {}
        
        # Architecture: Data-Driven Level Select
        self._show_level_select: bool = False
        self._level_buttons: list[UIButton] = []

        # Settings
        self._settings_btn: UIButton | None = None
        self._show_settings: bool = False
        self._settings_menu_btns: list[UIButton] = []

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
        self._windowed_size = (
            max(self.MIN_WIDTH, int(self._desktop_size[0] * 0.85)),
            max(self.MIN_HEIGHT, int(self._desktop_size[1] * 0.85)),
        )

        # Load fonts
        self._font_logo = self._load_display_font(52)
        self._font_section = self._load_display_font(22)
        self._font_card_name = self._load_display_font(21)
        self._font_card_name_small = self._load_display_font(17)
        self._font_pill = pygame.font.SysFont("DejaVu Sans", 14, bold=True)

        self._font_card_prof = pygame.font.SysFont("DejaVu Sans Mono,DejaVu Sans", 12, bold=True)
        self._font_coord = pygame.font.SysFont("DejaVu Sans Mono,DejaVu Sans", 12, bold=True)
        self._font_clue = pygame.font.SysFont("DejaVu Sans", 15, bold=False)
        self._font_body = pygame.font.SysFont("DejaVu Sans", 16, bold=False)

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

    def _draw_icon(self, icon_name: str, cx: int, cy: int, color: tuple[int, int, int]) -> None:
        """Draws simple, resolution-independent vector icons for UI buttons."""
        assert self._screen is not None
        if icon_name == "hamburger":
            for i in [-5, 0, 5]:
                pygame.draw.line(self._screen, color, (cx - 8, cy + i), (cx + 8, cy + i), 2)
        elif icon_name == "level":
            for i in [-4, 0, 4]:
                pygame.draw.line(self._screen, color, (cx-6, cy+i), (cx+6, cy+i), 2)
        elif icon_name == "restart":
            pygame.draw.arc(self._screen, color, (cx-7, cy-7, 14, 14), 0.5, 5.0, 2)
            pygame.draw.polygon(self._screen, color, [(cx+3, cy-7), (cx+7, cy-11), (cx+7, cy-3)])
        elif icon_name == "hint":
            pygame.draw.circle(self._screen, color, (cx, cy-3), 5, 2)
            pygame.draw.line(self._screen, color, (cx-3, cy+4), (cx+3, cy+4), 2)
            pygame.draw.line(self._screen, color, (cx-2, cy+7), (cx+2, cy+7), 2)
        elif icon_name == "auto":
            pygame.draw.polygon(self._screen, color, [(cx-4, cy-6), (cx-4, cy+6), (cx+6, cy)])

    def _build_control_pills(self) -> None:
        """Builds the bottom control pills with icons and the top-right Settings hamburger."""
        labels_and_callbacks = [
            ("Choose level", self._on_load_clicked, PALETTE.coral_pink, "level"),
            ("Restart", self._on_restart_clicked, PALETTE.coral, "restart"),
            ("Hint", self._on_hint_clicked, PALETTE.warm_yellow, "hint"),
            ("Auto Solve", self._on_auto_solve_clicked, PALETTE.orange, "auto"),
        ]

        pill_h = 38
        gap = 12
        pill_widths = [180, 135, 125, 150]
        total_w = sum(pill_widths) + (len(pill_widths) - 1) * gap
        start_x = self._grid_area_rect.centerx - total_w // 2

        self._control_pills = []
        curr_x = start_x
        for (label, cb, accent, icon), pw in zip(labels_and_callbacks, pill_widths):
            rect = pygame.Rect(curr_x, self._bottom_pills_y, pw, pill_h)
            self._control_pills.append(
                UIButton(label=label, rect=rect, callback=cb, accent_color=accent, icon=icon)
            )
            curr_x += pw + gap

        # The new top-right Settings Hamburger
        self._settings_btn = UIButton(
            label="",
            rect=pygame.Rect(self._win_w - 60, 24, 40, 40),
            callback=self._on_settings_clicked,
            accent_color=PALETTE.text_muted,
            icon="hamburger"
        )

    def _quit_game(self) -> None:
        """Cleanly stops the GUI loop."""
        self._running = False

    def _recompute_grid_layout(self) -> None:
        """Calculates clean card rectangles scaling dynamically based on grid size."""
        self._cell_rects.clear()
        n = self._public_state.grid_size
        gap = max(6, 24 - (n * 4)) 
        
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
                        elif getattr(self, '_show_settings', False):
                            self._show_settings = False
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
        """Routes mouse clicks across window overlays, action pills, verdict popups, and cards."""
        if getattr(self, '_show_settings', False):
            if button == 1:
                for btn in self._settings_menu_btns:
                    if btn.rect.collidepoint(pos):
                        btn.callback()
                        return
                self._show_settings = False
            return
            
        if self._show_level_select:
            if button == 1:
                for btn in self._level_buttons:
                    if btn.rect.collidepoint(pos):
                        btn.callback()
                        return
                self._show_level_select = False
            return
            
        if button == 1:
            if hasattr(self, '_settings_btn') and self._settings_btn and self._settings_btn.rect.collidepoint(pos):
                self._settings_btn.callback()
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
                self._set_status("", log=False)
            elif cell_view.revealed_clue is not None:
                self._active_clue_owner = coord
                self._highlighted_coords = cell_view.revealed_clue.region.resolve_coordinates(
                    grid_size=self._public_state.grid_size,
                    owner_coord=coord,
                )
                self._set_status(
                    f"{cell_view.name} ({label}): \"{cell_view.revealed_clue.text}\"",
                    PALETTE.warm_yellow,
                    log=False
                )
            return

        if mouse_button == 3:
            self._selected_coord = coord
            self._submit_selected_verdict(HiddenStatus.CRIMINAL)
        else:
            if self._selected_coord == coord:
                self._selected_coord = None
                self._set_status("", log=False)
            else:
                self._selected_coord = coord
                self._set_status("", log=False)

    def _submit_selected_verdict(self, proposed_status: HiddenStatus) -> None:
        """Submits a verdict for the selected suspect to GameEngine."""
        if self._selected_coord is None:
            return

        target = self._selected_coord
        try:
            result = self._engine.submit_verdict(target, proposed_status)
        except NotImplementedError as exc:
            self._set_status(f"NOT_PROVABLE (Solver placeholder: {exc})", PALETTE.warm_yellow, log=True)
            return

        self._public_state = result.public_state
        self._agent.observe(self._public_state)

        match result.response_type:
            case VerdictResponseType.ACCEPTED:
                color = PALETTE.card_border_innocent
                self._pulse_timers[target] = pygame.time.get_ticks()
                self._selected_coord = None
            case VerdictResponseType.NOT_PROVABLE:
                color = PALETTE.warm_yellow
            case VerdictResponseType.CONTRADICTED:
                color = PALETTE.card_border_criminal
            case VerdictResponseType.ALREADY_REVEALED:
                color = PALETTE.text_muted

        self._set_status(result.message, color, log=True)

    def _on_load_clicked(self) -> None:
        """Toggles the data-driven level selection overlay."""
        self._auto_solving = False
        self._show_level_select = not self._show_level_select
        
        if self._show_level_select:
            self._build_level_buttons()
            
    def _build_level_buttons(self) -> None:
        """Dynamically generates buttons based on discovered JSON puzzle files."""
        self._level_buttons.clear()
        modal_w, modal_h = 600, 500
        start_x = (self._win_w - modal_w)//2 + 50
        start_y = (self._win_h - modal_h)//2 + 100
        
        btn_w, btn_h = 240, 40
        gap = 20
        
        def make_callback(path: Path | None, size: int | None = None) -> Callable[[], None]:
            def callback():
                if path:
                    puzzle = PuzzleLoader.load_from_json(path)
                    msg = f"Loaded '{puzzle.puzzle_id}' from {path.name}."
                else:
                    puzzle = GameEngine.create_sample_puzzle(grid_size=size)
                    msg = f"Loaded Fallback {size}x{size}."
                
                self._public_state = self._engine.load_puzzle(puzzle)
                self._agent.reset_session()
                self._agent.observe(self._public_state)
                self._recompute_grid_layout()
                self._set_status(msg, PALETTE.card_border_innocent, log=True)
                self._show_level_select = False
            return callback

        col, row = 0, 0
        for path in self._puzzle_paths:
            rect = pygame.Rect(start_x + col*(btn_w+gap), start_y + row*(btn_h+gap), btn_w, btn_h)
            self._level_buttons.append(UIButton(
                label=f"Load {path.stem}", rect=rect, callback=make_callback(path), accent_color=PALETTE.coral_pink
            ))
            row += 1
            if row > 6:
                row = 0
                col += 1
                
        if not self._puzzle_paths:
            for i, size in enumerate([3, 4, 5, 6]):
                rect = pygame.Rect(start_x, start_y + i*(btn_h+gap), btn_w, btn_h)
                self._level_buttons.append(UIButton(
                    label=f"Fallback {size}x{size}", rect=rect, callback=make_callback(None, size), accent_color=PALETTE.warm_yellow
                ))

    def _on_restart_clicked(self) -> None:
        """Resets the current puzzle to its starting state."""
        self._auto_solving = False
        self._public_state = self._engine.restart_puzzle()
        self._agent.reset_session()
        self._agent.observe(self._public_state)
        self._selected_coord = None
        self._active_clue_owner = None
        self._highlighted_coords = frozenset()
        self._set_status("Scenario restarted.", PALETTE.text_white, log=True)

    def _on_hint_clicked(self) -> None:
        """Queries LogicAgent for a hint without revealing the deduced solution."""
        try:
            move = self._agent.get_hint()
        except NotImplementedError as exc:
            self._set_status(f"Hint placeholder: {exc}", PALETTE.warm_yellow, log=True)
            return

        if move is None:
            self._set_status("No forced move found by LogicAgent.", PALETTE.warm_yellow, log=True)
        else:
            self._selected_coord = move.coord
            label = self._format_coord_label(move.coord)
            
            self._set_status(
                f"HINT: A verdict can be logically deduced for suspect {label}.",
                PALETTE.warm_yellow,
                log=True
            )

    def _on_auto_solve_clicked(self) -> None:
        """Toggles continuous timer-based Auto Solve ON or OFF."""
        if self._engine.is_solved:
            self._auto_solving = False
            self._set_status(
                "Puzzle is already completely solved!", PALETTE.card_border_innocent, log=True
            )
            return

        self._auto_solving = not self._auto_solving
        if self._auto_solving:
            self._last_auto_step_ms = 0
            self._set_status(
                "Auto Solve started... (Click again to pause)", PALETTE.orange, log=True
            )
        else:
            self._set_status("Auto Solve paused.", PALETTE.warm_yellow, log=True)

    def _execute_single_auto_solve_step(self) -> None:
        """Executes one deterministic deduction step for the Auto Solve timer loop."""
        if self._engine.is_solved:
            self._auto_solving = False
            self._set_status(
                "Auto Solve complete! All characters proved.",
                PALETTE.card_border_innocent,
                log=True
            )
            return

        try:
            move = self._agent.compute_next_auto_solve_step()
        except NotImplementedError as exc:
            self._auto_solving = False
            self._set_status(f"Auto Solve stopped ({exc})", PALETTE.warm_yellow, log=True)
            return

        if move is None:
            self._auto_solving = False
            self._set_status(
                "Auto Solve stopped: No forced deductions available.",
                PALETTE.warm_yellow,
                log=True
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
                log=True
            )

    def _set_status(
        self,
        message: str,
        color: tuple[int, int, int] = PALETTE.text_white,
        log: bool = True
    ) -> None:
        """Updates feedback. If log=True, records it in the history."""
        if log:
            self._status_log.append((message, color))
            if len(self._status_log) > 3:
                self._status_log.pop(0)
            self._transient_message = ""
        else:
            self._transient_message = message
            self._transient_color = color

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
        """Renders the full UI, now explicitly layered with Settings overlays."""
        assert self._screen is not None
        assert self._bg_surface is not None

        self._screen.blit(self._bg_surface, (0, 0))
        self._render_left_info_panel(mouse_pos)

        for coord, rect in self._cell_rects.items():
            cell_view = self._public_state.get_cell(coord)
            self._render_suspect_card(cell_view, rect, mouse_pos)

        self._render_bottom_pills(mouse_pos)
        
        # Render the top-right Settings Hamburger
        if hasattr(self, '_settings_btn') and self._settings_btn:
            is_hovered = self._settings_btn.rect.collidepoint(mouse_pos)
            bg = PALETTE.pill_hover if is_hovered else PALETTE.pill_bg
            pygame.draw.rect(self._screen, bg, self._settings_btn.rect, border_radius=20)
            pygame.draw.rect(self._screen, self._settings_btn.accent_color, self._settings_btn.rect, width=2, border_radius=20)
            self._draw_icon("hamburger", self._settings_btn.rect.centerx, self._settings_btn.rect.centery, PALETTE.text_white)
            
        if self._show_level_select:
            self._render_level_select_overlay(mouse_pos)
            
        if getattr(self, '_show_settings', False):
            self._render_settings_overlay(mouse_pos)

    def _render_left_info_panel(self, mouse_pos: tuple[int, int]) -> None:
        """Renders the left branding and status log with increased readability."""
        assert self._screen is not None
        x = self._left_panel_x
        y = max(96, (self._win_h - 500) // 2)

        # Detective Silhouette Icon + "Griductive" Title
        self._draw_detective_icon(x + 22, y + 30)
        self._render_spaced_text(
            "Griductive", self._font_logo, (x + 56, y),
            fg_color=PALETTE.text_white, shadow_color=PALETTE.magenta, shadow_offset=3, letter_spacing=4,
        )

        lvl_lbl = self._font_coord.render("LEVEL", True, PALETTE.text_white)
        self._screen.blit(lvl_lbl, (x, y + 76))
        
        difficulty_colors = {3: (82, 214, 154), 4: (255, 214, 92), 5: (252, 158, 66), 6: (178, 34, 34)}
        level_color = difficulty_colors.get(self._public_state.grid_size, PALETTE.text_white)
        level_name = self._engine.puzzle_id.replace("_", " ").upper()
        if len(level_name) > 18:
            level_name = level_name[:15] + "..."
            
        lvl_name_surf = self._font_coord.render(level_name, True, level_color)
        self._screen.blit(lvl_name_surf, (x + 58, y + 76))

        pygame.draw.line(self._screen, PALETTE.warm_yellow, (x, y + 104), (x + 80, y + 104), width=2)

        # HOW TO PLAY panel with increased vertical line spacing
        how_to_rect = pygame.Rect(x, y + 112, self._left_panel_w - 10, 240)
        panel_surf = pygame.Surface((how_to_rect.width, how_to_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (*PALETTE.card_unknown_bg, 210), panel_surf.get_rect(), border_radius=12)
        self._screen.blit(panel_surf, how_to_rect.topleft)

        self._render_spaced_text("HOW TO PLAY", self._font_section, (x + 16, y + 124), fg_color=PALETTE.warm_yellow, shadow_offset=2, letter_spacing=3)

        rules = [
            "Every person is criminal or innocent, and every clue is true.",
            "• Tap a suspect to call them criminal or innocent.",
            "• A right call flips the card and reveals its clue.",
            "• Only call what the clues prove.",
            "• Click any revealed clue to spotlight its region.",
        ]
        ry = y + 164
        for idx, line in enumerate(rules):
            color = PALETTE.text_white if idx == 0 else PALETTE.text_muted
            self._render_wrapped_text(line, self._font_body, color, x + 16, ry, self._left_panel_w - 42, shadow=False)
            ry += 48 if idx == 0 else 38

        # VERDICT LOG with increased vertical line spacing
        status_box = pygame.Rect(x, y + 368, self._left_panel_w - 10, 114)
        pygame.draw.rect(self._screen, PALETTE.pill_bg, status_box, border_radius=12)
        pygame.draw.rect(self._screen, PALETTE.card_border_default, status_box, width=2, border_radius=12)

        self._render_spaced_text("VERDICT LOG", self._font_card_name_small, (status_box.x + 14, status_box.y + 10), fg_color=PALETTE.text_white, shadow_offset=1, letter_spacing=2)
        
        log_y = status_box.y + 34
        for msg, color in self._status_log:
            surf = self._font_clue.render(msg, True, color)
            self._screen.blit(surf, (status_box.x + 14, log_y))
            log_y += 22
            
        if self._transient_message:
            pygame.draw.line(self._screen, PALETTE.card_unknown_bg, (status_box.x + 14, status_box.bottom - 28), (status_box.right - 14, status_box.bottom - 28), width=2)
            transient_surf = self._font_clue.render(self._transient_message, True, self._transient_color)
            self._screen.blit(transient_surf, (status_box.x + 14, status_box.bottom - 24))

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
        assert self._screen is not None

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
            # NEW: Draw a coral pink spotlight background plate behind the card 
            # to visually group the region required by the rubric.
            highlight_rect = rect.inflate(12, 12)
            pygame.draw.rect(self._screen, PALETTE.coral_pink, highlight_rect, border_radius=16)

            border_color = PALETTE.card_border_highlight
            border_w = 3
        elif cell.coord == self._selected_coord:
            border_color = PALETTE.card_border_selected
            border_w = 3

        # Render the card surface on top of the spotlight
        pygame.draw.rect(self._screen, bg_color, rect, border_radius=12)
        pygame.draw.rect(self._screen, border_color, rect, width=border_w, border_radius=12)

        coord_label = self._format_coord_label(cell.coord)
        coord_surf = self._font_coord.render(coord_label, True, PALETTE.text_coord)
        self._screen.blit(coord_surf, (rect.x + 10, rect.y + 8))

        if cell.revealed_status == RevealedStatus.UNKNOWN:
            is_sel = (cell.coord == self._selected_coord)
            
            # Shift avatar and name upwards if the card is selected to make room for verdict buttons
            avatar_cy = rect.y + int(rect.height * (0.30 if is_sel else 0.38))
            self._draw_suspect_avatar(cell.coord, rect.centerx, avatar_cy, scale=1.0)

            name_y = rect.y + int(rect.height * (0.58 if is_sel else 0.67))
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

            # Hide profession text completely if selected to prevent any overlap
            if not is_sel:
                prof_surf = self._font_card_prof.render(cell.profession, True, PALETTE.text_muted)
                self._screen.blit(
                    prof_surf,
                    prof_surf.get_rect(center=(rect.centerx, name_y + 23)),
                )

            if is_sel:
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
            # CARD FLIP: Hide portrait and name, only show centered clue text
            if cell.revealed_clue is not None:
                text_y = rect.y + int(rect.height * 0.35)
                self._render_wrapped_text(
                    text=cell.revealed_clue.text,
                    font=self._font_clue,
                    color=PALETTE.text_white,
                    x=rect.x + 10,
                    y=text_y,
                    max_width=rect.width - 20,
                    center_horizontal=True,
                    card_centerx=rect.centerx,
                    shadow=True,
                )

        # Render Expanding Pulse Effect for Newly Accepted Moves
        if cell.coord in self._pulse_timers:
            elapsed = pygame.time.get_ticks() - self._pulse_timers[cell.coord]
            duration_ms = 450
            if elapsed < duration_ms:
                progress = elapsed / duration_ms
                expansion = int(35 * progress)
                pulse_rect = rect.inflate(expansion, expansion)
                alpha = int(255 * (1.0 - progress))
                
                pulse_surf = pygame.Surface(pulse_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(pulse_surf, (*border_color[:3], alpha), pulse_surf.get_rect(), width=3, border_radius=12)
                self._screen.blit(pulse_surf, pulse_rect.topleft)

    def _render_bottom_pills(self, mouse_pos: tuple[int, int]) -> None:
        """Renders the rounded bottom pill buttons alongside their respective icons."""
        assert self._screen is not None
        assert self._font_pill is not None

        for pill in self._control_pills:
            is_hovered = pill.rect.collidepoint(mouse_pos)
            bg = pill.accent_color if is_hovered else PALETTE.pill_bg
            text_col = PALETTE.text_black if is_hovered else PALETTE.text_white
            shadow_col = pill.accent_color if is_hovered else PALETTE.text_black

            pygame.draw.rect(self._screen, bg, pill.rect, border_radius=19)
            if not is_hovered:
                pygame.draw.rect(self._screen, pill.accent_color, pill.rect, width=2, border_radius=19)

            if pill.icon:
                # Estimate total text width dynamically and draw icon perfectly side-by-side
                est_text_w = self._font_pill.size(pill.label)[0] + len(pill.label) * 1
                text_cx = pill.rect.centerx + 12
                icon_cx = text_cx - (est_text_w // 2) - 18
                self._draw_icon(pill.icon, icon_cx, pill.rect.centery, text_col)
            else:
                text_cx = pill.rect.centerx

            self._render_spaced_text(
                pill.label, self._font_pill, (text_cx, pill.rect.centery),
                fg_color=text_col, shadow_color=shadow_col,
                shadow_offset=1, letter_spacing=1, center=True,
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

    def _on_settings_clicked(self) -> None:
        """Toggles the Settings overlay in the center of the screen."""
        self._show_settings = not self._show_settings
        if self._show_settings:
            modal_w, modal_h = 320, 220
            start_x = (self._win_w - modal_w)//2
            start_y = (self._win_h - modal_h)//2
            
            mode_label = "Windowed Mode" if self._is_windowed_fullscreen else "Fullscreen Mode"
            self._settings_menu_btns = [
                UIButton(mode_label, pygame.Rect(start_x + 40, start_y + 80, 240, 40), self._toggle_and_close_settings, PALETTE.warm_yellow),
                UIButton("Quit Game", pygame.Rect(start_x + 40, start_y + 140, 240, 40), self._quit_game, PALETTE.coral_pink)
            ]

    def _toggle_and_close_settings(self) -> None:
        self.toggle_windowed_fullscreen()
        self._show_settings = False

    def _render_settings_overlay(self, mouse_pos: tuple[int, int]) -> None:
        """Draws a darkened background and the Settings modal."""
        assert self._screen is not None
        overlay = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
        overlay.fill((14, 10, 22, 180))
        self._screen.blit(overlay, (0, 0))
        
        modal_w, modal_h = 320, 220
        modal_rect = pygame.Rect((self._win_w - modal_w)//2, (self._win_h - modal_h)//2, modal_w, modal_h)
        pygame.draw.rect(self._screen, PALETTE.pill_bg, modal_rect, border_radius=16)
        pygame.draw.rect(self._screen, PALETTE.text_muted, modal_rect, width=2, border_radius=16)
        
        self._render_spaced_text("MENU", self._font_section, (modal_rect.centerx, modal_rect.y + 35), fg_color=PALETTE.text_white, center=True)
        
        for btn in self._settings_menu_btns:
            is_hovered = btn.rect.collidepoint(mouse_pos)
            bg = btn.accent_color if is_hovered else PALETTE.card_unknown_bg
            txt_col = PALETTE.text_black if is_hovered else PALETTE.text_white
            pygame.draw.rect(self._screen, bg, btn.rect, border_radius=8)
            if not is_hovered:
                pygame.draw.rect(self._screen, btn.accent_color, btn.rect, width=1, border_radius=8)
            btn_surf = self._font_coord.render(btn.label, True, txt_col)
            self._screen.blit(btn_surf, btn_surf.get_rect(center=btn.rect.center))

    def _render_level_select_overlay(self, mouse_pos: tuple[int, int]) -> None:
        """Renders a data-driven level selection modal over the game."""
        assert self._screen is not None
        
        overlay = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
        overlay.fill((14, 10, 22, 210))
        self._screen.blit(overlay, (0, 0))
        
        modal_w, modal_h = 600, 500
        modal_rect = pygame.Rect((self._win_w - modal_w)//2, (self._win_h - modal_h)//2, modal_w, modal_h)
        pygame.draw.rect(self._screen, PALETTE.pill_bg, modal_rect, border_radius=16)
        pygame.draw.rect(self._screen, PALETTE.coral_pink, modal_rect, width=2, border_radius=16)
        
        self._render_spaced_text(
            "SELECT PUZZLE", 
            self._font_logo, 
            (modal_rect.centerx, modal_rect.y + 50), 
            fg_color=PALETTE.text_white, 
            shadow_color=PALETTE.magenta, 
            center=True
        )

        for btn in self._level_buttons:
            is_hovered = btn.rect.collidepoint(mouse_pos)
            bg = btn.accent_color if is_hovered else PALETTE.card_unknown_bg
            txt_col = PALETTE.text_black if is_hovered else PALETTE.text_white
            
            pygame.draw.rect(self._screen, bg, btn.rect, border_radius=8)
            if not is_hovered:
                pygame.draw.rect(self._screen, btn.accent_color, btn.rect, width=1, border_radius=8)
                
            btn_surf = self._font_coord.render(btn.label, True, txt_col)
            self._screen.blit(btn_surf, btn_surf.get_rect(center=btn.rect.center))
