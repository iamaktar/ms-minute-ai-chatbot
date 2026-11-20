"""Animated talking avatar widget for Ms. Minute."""

from __future__ import annotations

import math
import random
import tkinter as tk
from typing import Literal

import customtkinter as ctk

AvatarState = Literal["idle", "listening", "thinking", "speaking"]


class TalkingAvatar(ctk.CTkFrame):
    """Canvas-based reactive avatar that animates for idle, listening, thinking, and speaking."""

    def __init__(self, master: ctk.CTkBaseClass, size: int = 136) -> None:
        super().__init__(master, fg_color="transparent")
        self._size = size
        self._state: AvatarState = "idle"
        self._phase: float = 0.0
        self._running: bool = True

        self._canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            bg="#212121",
            highlightthickness=0,
        )
        self._canvas.pack(anchor="center")

        self._state_label = ctk.CTkLabel(
            self,
            text="Idle",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9CA3AF",
        )
        self._state_label.pack(pady=(4, 0))

        self._animate()

    def set_state(self, state: AvatarState) -> None:
        """Change the avatar's visual animation state."""
        self._state = state
        labels: dict[AvatarState, tuple[str, str]] = {
            "idle": ("Idle", "#9CA3AF"),
            "listening": ("Listening...", "#FBBF24"),
            "thinking": ("Thinking...", "#60A5FA"),
            "speaking": ("Speaking...", "#C084FC"),
        }
        text, color = labels[state]
        self._state_label.configure(text=text, text_color=color)

    def stop(self) -> None:
        """Stop the animation loop before window destruction."""
        self._running = False

    def _animate(self) -> None:
        """Redraw the avatar frame at ~30 FPS."""
        if not self._running:
            return

        self._phase += 0.16
        c = self._canvas
        c.delete("all")

        cx = self._size / 2
        cy = self._size / 2

        if self._state == "idle":
            pulse = math.sin(self._phase * 0.6) * 4
            outer_r = 46 + pulse
            inner_r = 32 + pulse * 0.5
            c.create_oval(
                cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r,
                outline="#1E3A8A", width=3,
            )
            c.create_oval(
                cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
                fill="#1D4ED8", outline="#60A5FA", width=2,
            )
            c.create_text(
                cx, cy, text="MM", fill="#FFFFFF", font=("Segoe UI", 16, "bold")
            )

        elif self._state == "listening":
            ring1 = 36 + (self._phase * 8) % 24
            ring2 = 36 + ((self._phase * 8) + 12) % 24
            c.create_oval(
                cx - ring1, cy - ring1, cx + ring1, cy + ring1,
                outline="#D97706", width=2,
            )
            c.create_oval(
                cx - ring2, cy - ring2, cx + ring2, cy + ring2,
                outline="#FBBF24", width=2,
            )
            core = 30 + math.sin(self._phase * 1.5) * 3
            c.create_oval(
                cx - core, cy - core, cx + core, cy + core,
                fill="#B45309", outline="#FDE68A", width=2,
            )
            c.create_text(
                cx, cy, text="MIC", fill="#FFFFFF", font=("Segoe UI", 12, "bold")
            )

        elif self._state == "thinking":
            orbit_r = 48
            start_deg = (self._phase * 45) % 360
            c.create_oval(
                cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r,
                outline="#1E293B", width=3,
            )
            c.create_arc(
                cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r,
                start=start_deg, extent=110, style="arc",
                outline="#38BDF8", width=4,
            )
            c.create_arc(
                cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r,
                start=(start_deg + 180) % 360, extent=70, style="arc",
                outline="#818CF8", width=4,
            )
            core = 30
            c.create_oval(
                cx - core, cy - core, cx + core, cy + core,
                fill="#0F172A", outline="#38BDF8", width=2,
            )
            c.create_text(
                cx, cy, text="...", fill="#38BDF8", font=("Segoe UI", 18, "bold")
            )

        elif self._state == "speaking":
            glow = 50 + math.sin(self._phase * 2.2) * 5
            c.create_oval(
                cx - glow, cy - glow, cx + glow, cy + glow,
                outline="#7E22CE", width=3,
            )
            core = 36
            c.create_oval(
                cx - core, cy - core, cx + core, cy + core,
                fill="#581C87", outline="#C084FC", width=2,
            )
            # Animated voice equalizer bars inside the orb
            bar_count = 5
            bar_width = 5
            gap = 4
            total_w = bar_count * bar_width + (bar_count - 1) * gap
            start_x = cx - total_w / 2
            for i in range(bar_count):
                wave = abs(math.sin(self._phase * 2.5 + i * 0.9))
                jitter = random.uniform(0.75, 1.15)
                bar_h = max(8, min(36, int((10 + wave * 24) * jitter)))
                x0 = start_x + i * (bar_width + gap)
                y0 = cy - bar_h / 2
                x1 = x0 + bar_width
                y1 = cy + bar_h / 2
                c.create_rectangle(x0, y0, x1, y1, fill="#E9D5FF", outline="")

        self.after(33, self._animate)