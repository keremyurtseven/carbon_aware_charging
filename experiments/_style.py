"""Shared figure style for the experiment scripts: one palette, one look.

Colour meaning is the same in every figure (Okabe-Ito palette):
  XY mixer = vermilion, X / penalty mixer = blue, exact / classical optimum = bluish green,
  baselines = black / grey, windy nights = orange, calm nights = sky blue.
Markers, line styles and hatches carry the same information, so no figure relies on colour alone.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

XY = "#D55E00"        # vermilion: XY mixer
X = "#0072B2"         # blue: X / penalty mixer
EXACT = "#009E73"     # bluish green: exact / classical optimum
BLACK = "#000000"
GREY = "#6B6B6B"      # baselines
LIGHT_GREY = "#B8B8B8"
WINDY = "#E69F00"     # orange
CALM = "#56B4E9"      # sky blue
PURPLE = "#CC79A7"

FIG_WIDTH = 12.0      # inches; every figure is about this wide
DPI = 200


def apply() -> None:
    """Set the global matplotlib style (fonts >= 11 pt, no top/right spines, light grid)."""
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "savefig.dpi": DPI,
    })


def panel_title(ax, text: str) -> None:
    """Short left-aligned panel title."""
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", pad=8)


def heading(fig, title: str, caption: str | None = None, caption_width: int | None = None) -> float:
    """Draw a one-line title and an optional small caption at the top of ``fig``.

    Returns the figure fraction (from the bottom) below which the panels should be laid out,
    i.e. pass it as the ``top`` of ``tight_layout(rect=(0, bottom, 1, top))``.
    """
    w, h = fig.get_size_inches()
    fig.text(0.01, 1 - 0.12 / h, title, fontsize=15, fontweight="bold", ha="left", va="top")
    used = 0.12 + 0.30
    if caption:
        width = caption_width or int(w * 10.5)
        text = textwrap.fill(caption, width=width)
        fig.text(0.01, 1 - used / h, text, fontsize=11, color="#333333", ha="left", va="top", linespacing=1.3)
        used += 0.2 * (text.count("\n") + 1) + 0.05
    return 1 - (used + 0.12) / h


MAX_PX = 2000  # Qollab shows images up to 2048 x 2048 pixels


def cap_png(path: Path, max_px: int = MAX_PX) -> None:
    """Downscale a PNG in place so that neither side exceeds ``max_px`` pixels."""
    from PIL import Image

    with Image.open(path) as im:
        if max(im.size) <= max_px:
            return
        im = im.copy()
    im.thumbnail((max_px, max_px), Image.LANCZOS)
    im.save(path, optimize=True)


def save(fig, base: Path) -> None:
    """Write ``base``.png (at most 2000 px per side) and ``base``.svg."""
    base = Path(base)
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    cap_png(base.with_suffix(".png"))
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
