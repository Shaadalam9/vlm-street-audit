"""Saving Plotly figures. Every figure in this project is made with Plotly.

Figures go to `outputs/figures/<name>.html` (interactive). A static PNG/PDF/SVG
copy is written too when `formats` asks for it; that needs Kaleido and Chrome
(`uv add kaleido`, then `uv run plotly_get_chrome`).
"""

from collections.abc import Iterable
from pathlib import Path

import plotly.graph_objects as go

from .config import PROJECT_ROOT

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TEMPLATE = "plotly_white"
FONT = dict(family="Arial", size=14)


def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=TEMPLATE, font=FONT, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def save_figure(fig: go.Figure, name: str, formats: Iterable[str] = (), directory: Path = FIGURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    html = directory / f"{name}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    for extension in formats:
        fig.write_image(directory / f"{name}.{extension}", scale=2)
    return html
