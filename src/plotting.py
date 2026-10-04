"""
Common figure settings and saving for all scripts in parts/. Every figure is
saved at the width it is printed at in the two-column report, so that the font
sizes set here are the font sizes in the report.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 4 - Substantial
Role: Extended our original save_fig, which only saved the current figure.
    Claude wrote the font sizes in rcParams, the resizing of each figure to
    the column width (3.4 in) or page width (7 in) depending on its original
    size, the wrapping of long titles and y-labels in one-column figures, and
    _legends_below_axes, which moves every legend below its axes with as many
    columns as fit, so that no legend covers the data.

Verification:
    Reviewed by the project authors. All scripts in parts/ were rerun, the
    printed numbers were compared with the previous files in outputs/ and
    found identical, and all 30 figures were inspected.
"""
from pathlib import Path
from textwrap import fill
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Widths in the two-column revtex report, in inches: one column and the full page.
COLUMN_WIDTH, PAGE_WIDTH = 3.4, 7.0

# Font sizes in points as they appear in the report, since save_fig saves every
# figure at the width it is printed at.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
})


def save_fig(name):
    """Save the current figure at the width it gets in the report.

    Figures made 11 inches or wider (all multi-panel figures) are printed
    across the page, the others in one column. The aspect ratio is kept, and
    titles and axis labels too long for a one-column figure are wrapped onto
    two lines. Legends are placed below the axes so that they never cover the
    curves.
    """
    FIG_DIR.mkdir(exist_ok=True)
    fig = plt.gcf()
    width, height = fig.get_size_inches()
    target = PAGE_WIDTH if width >= 11 else COLUMN_WIDTH
    fig.set_size_inches(target, height * target / width)
    if target == COLUMN_WIDTH:
        for ax in fig.axes:
            title, ylabel = ax.get_title(), ax.get_ylabel()
            if "$" not in title:                   # never break inside mathtext
                ax.set_title(fill(title, 40))
            if "$" not in ylabel:
                ax.set_ylabel(fill(ylabel, 26))
    fig.tight_layout()
    _legends_below_axes(fig)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _legends_below_axes(fig):
    """Move every legend out of its axes to just below the x-axis label, with as
    many columns as fit in the width of the axes, so that it never covers data."""
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        bottom = ax.get_tightbbox(renderer).transformed(ax.transAxes.inverted()).y0
        ax_width = ax.get_window_extent(renderer).width
        for ncol in range(len(labels), 0, -1):    # as many columns as fit
            legend = ax.legend(handles, labels, loc="upper center", ncol=ncol,
                               bbox_to_anchor=(0.5, bottom - 0.03), frameon=False,
                               columnspacing=1.0, handlelength=1.6)
            if legend.get_window_extent(renderer).width <= 1.1 * ax_width:
                break
