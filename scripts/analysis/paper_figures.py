"""The five figures of the paper, made to be understood at a glance.

Reads the published answers and analysis tables (outputs/vlm/) and, for Figures 1 and 3, seven frames kept in
docs/figures/paper/frames/. Writes docs/figures/paper/fig*.png and .html.

Run (after analyse_audit.py):  uv run python scripts/analysis/paper_figures.py
"""

import base64
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from vlm_street_audit.config import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "docs" / "figures" / "paper"
TABLES = ROOT / "outputs" / "vlm" / "analysis_audit"
# The seven frames shown in Figures 1 and 3, copied from data/snippets/ (not in git) so the figures can be redrawn
# from a fresh clone.
FRAMES = OUT / "frames"
FONT = dict(family="Arial", size=15, color="#1f1f1f")
INK_MUTED = "#5f5e5a"
MODEL_NAMES = {
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma4-31b": "Gemma 4 31B",
    "molmo2-8b": "Molmo2-8B",
    "internvl3.5-8b": "InternVL3.5-8B",
    "ministral3-14b": "Ministral 3 14B",
    "kanana-1.5-v-3b": "Kanana-1.5-V 3B",
}
MODEL_COLOURS = {
    "qwen3.8-27b": "#2a78d6",
    "gemma4-31b": "#eb6834",
    "molmo2-8b": "#1baf7a",
    "internvl3.5-8b": "#eda100",
    "ministral3-14b": "#e87ba4",
    "kanana-1.5-v-3b": "#008300",
}
N_MODELS = {5: "five", 6: "six", 7: "seven"}[len(MODEL_NAMES)]
CONTINENT_COLOURS = {
    "Asia": "#2a78d6",
    "Europe": "#1baf7a",
    "North America": "#eda100",
    "Africa": "#eb6834",
    "South America": "#8f4bd6",
    "Oceania": "#5f5e5a",
}
WEST, REST = "Europe & North America", "Rest of the world"


def save(fig: go.Figure, name: str, width: int, height: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        font=FONT, template="plotly_white", width=width, height=height, paper_bgcolor="white", plot_bgcolor="white"
    )
    fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn", div_id=name)  # fixed id, same file
    fig.write_image(OUT / f"{name}.png", scale=2)
    print("wrote", name)


# ---- Figure 1: what pedestrians lack, city by city, with six of the streets ---------------------------------
# (clip, auditor, need, side, panel title, tag), in reading order: north row then south row, west to east. Each was
# checked against its frame (2026-11-10): the side picks the entry whose point lies on the problem itself, e.g. the
# grass verge in Spring Hill rather than a driveway, and the stalls and motorbikes in Valledupar rather than a taxi in
# the road. In Kazan the sign, tactile paving and kerb gap mark a crossing point whose road markings are missing.
EXAMPLES = [
    (
        "8wdxcC3VkQY_00160000_010000",
        "gemma4-31b",
        "no_footway",
        "right_side",
        "Spring Hill, United States",
        "No footway",
    ),
    (
        "cKRt8KHugxA_02030000_010000",
        "qwen3.8-27b",
        "no_crossing",
        "left_side",
        "Kazan, Russia",
        "Sign, but no markings",
    ),
    (
        "MMSSZo4HdOo_02862000_010000",
        "qwen3.8-27b",
        "blocked_footway",
        "left_side",
        "Peshawar, Pakistan",
        "Path blocked",
    ),
    (
        "m2BHJJ2x9oo_00986000_010000",
        "qwen3.8-27b",
        "blocked_footway",
        "right_side",
        "Valledupar, Colombia",
        "Path blocked",
    ),
    ("Q3WvZW9zBk8_00851000_010000", "qwen3.8-27b", "no_crossing", "road", "Kinshasa, DR Congo", "Crossing needed"),
    (
        "wMu6Va5PhGY_00171000_010000",
        "qwen3.8-27b",
        "no_footway",
        "left_side",
        "Sydney, Australia",
        "“No footway” (wrong)",
    ),
]
GAP_COLOURS = {
    "Crossing and footway": "#8f2d56",
    "Crossing": "#eb6834",
    "Footway": "#2a78d6",
    "Lighting or other": "#eda100",
    "No agreed need": "#dfe4ea",  # pale and neutral, so the gaps stand out on the dark land
}
WORLD_W, WORLD_H = 2400, 1960
WORLD_LON, WORLD_LAT = (-180.0, 180.0), (-58.0, 84.0)  # the whole world but Antarctica
# The map's colours: continents dark enough to read as land, a pale ocean, and every dot ringed in white so it stands
# out on either.
MAP = dict(land="#4b5866", ocean="#eef2f6", border="#6b7785", coast="#4b5866", line="#1f1f1f")
ROW_H, ROW_GAP, THUMB_W = 0.245, 0.03, 0.31  # shares of the figure: frame rows, gap to the map, frame width
WORLD_GEO = dict(
    projection_type="equirectangular",
    lonaxis_range=list(WORLD_LON),
    lataxis_range=list(WORLD_LAT),
    showframe=False,
    domain=dict(x=[0.0, 1.0], y=[ROW_H + ROW_GAP, 1 - ROW_H - ROW_GAP]),
)


def frame_file(record: dict, number: int) -> Path:
    """The answer's frame `number` (1-based): the copy in FRAMES, made from data/snippets/ the first time."""
    source = ROOT / record["frames"][number - 1]
    copy = FRAMES / f"{record['clip_id']}_{source.name}"
    if not copy.exists():
        FRAMES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
    return copy


def need_entry(clip: str, auditor: str, need: str, side: str) -> tuple[dict, dict]:
    record = json.loads((ROOT / "outputs/vlm/audit/cities" / auditor / f"{clip}.json").read_text())
    entry = next(d for d in record["answer"]["needed"] if d["need"] == need and d["where"] == side)
    return record, entry


def city_gaps(city: pd.DataFrame) -> pd.DataFrame:
    """Each city's gap for the map: crossing, footway, both, something else, or no agreed need."""

    def gap(r) -> str:
        crossing, footway = r.need_no_crossing > 0, r.need_no_footway > 0
        other = any(
            r[c] > 0 for c in city.columns if c.startswith("need_") and c not in ("need_no_crossing", "need_no_footway")
        )
        if crossing and footway:
            return "Crossing and footway"
        return "Crossing" if crossing else "Footway" if footway else "Lighting or other" if other else "No agreed need"

    return city.assign(gap=city.apply(gap, axis=1))


def geo_to_paper():
    """Where Plotly draws a longitude and latitude on Figure 1, in paper coordinates. Measured on a render with two
    markers, since Plotly fits the map into its domain; the projection is equirectangular, so the rest is linear."""
    probe = go.Figure()
    points = [(WORLD_LON[0] + 20, WORLD_LAT[0] + 10), (WORLD_LON[1] - 20, WORLD_LAT[1] - 10)]
    for lon, lat in points:
        probe.add_trace(go.Scattergeo(lon=[lon], lat=[lat], mode="markers", marker=dict(size=10, color="#ff00ff")))
    probe.update_geos(**WORLD_GEO, showland=False, showcountries=False, showcoastlines=False)
    probe.update_layout(width=WORLD_W, height=WORLD_H, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    image = Image.open(io.BytesIO(probe.to_image(format="png", scale=1))).convert("RGB")
    pixels = np.asarray(image).astype(int)
    ys, xs = np.nonzero((pixels[..., 0] > 230) & (pixels[..., 1] < 40) & (pixels[..., 2] > 230))
    west = xs < WORLD_W / 2
    (x0, y0), (x1, y1) = (xs[west].mean(), ys[west].mean()), (xs[~west].mean(), ys[~west].mean())
    (lon0, lat0), (lon1, lat1) = points

    def convert(lon: float, lat: float) -> tuple[float, float]:
        px = x0 + (lon - lon0) / (lon1 - lon0) * (x1 - x0)
        py = y0 + (lat - lat0) / (lat1 - lat0) * (y1 - y0)
        return px / WORLD_W, 1 - py / WORLD_H

    return convert


def figure_world(city: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    city = city_gaps(city)
    to_paper = geo_to_paper()
    fig = go.Figure()
    for label, colour in GAP_COLOURS.items():
        part = city[city.gap == label]
        fig.add_trace(
            go.Scattergeo(
                lat=part.lat,
                lon=part.lon,
                mode="markers",
                name=f"{label} ({len(part)})",
                text=part.locality + ", " + part.country,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=18, color=colour, line=dict(width=2.5, color="white")),
            )
        )
    fig.update_geos(
        **WORLD_GEO,
        showland=True,
        landcolor=MAP["land"],
        showcountries=True,
        countrycolor=MAP["border"],
        coastlinecolor=MAP["coast"],
        showocean=True,
        oceancolor=MAP["ocean"],
        showlakes=True,
        lakecolor=MAP["ocean"],
    )

    def ring(x: float, y: float, r: float, colour: str, width: float) -> None:
        dy = r * WORLD_W / WORLD_H  # round on the page, not in paper units
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            x0=x - r,
            x1=x + r,
            y0=y - dy,
            y1=y + dy,
            line=dict(color=colour, width=width),
        )

    used = []
    img_h = ROW_H - 0.035  # the rest of the row is the panel title
    for n, (clip, auditor, need, side, title, tag) in enumerate(EXAMPLES):
        record, entry = need_entry(clip, auditor, need, side)
        frame = frame_file(record, entry["frame"])
        top = n < 3
        x0, y0 = 0.015 + (n % 3) * (THUMB_W + 0.0225), (1 - ROW_H if top else 0.0)
        fig.add_layout_image(
            source="data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode(),
            xref="paper",
            yref="paper",
            x=x0,
            y=y0 + img_h,
            sizex=THUMB_W,
            sizey=img_h,
            sizing="stretch",
            xanchor="left",
            yanchor="top",
            layer="above",
        )
        colour = INK_MUTED if "wrong" in tag else "#c62828"
        px, py = x0 + entry["x"] / 1000 * THUMB_W, y0 + img_h - entry["y"] / 1000 * img_h
        ring(px, py, 0.013, "white", 8)
        ring(px, py, 0.013, colour, 4.5)
        side = "center" if 250 < entry["x"] < 750 else ("left" if entry["x"] < 500 else "right")
        fig.add_annotation(
            x=px,
            y=py + (0.034 if entry["y"] > 300 else -0.034),
            xref="paper",
            yref="paper",
            text=f"<b>{tag}</b>",
            showarrow=False,
            font=dict(size=26, color="white"),
            bgcolor=colour,
            borderpad=7,
            xanchor=side,
        )
        fig.add_annotation(
            x=x0 + THUMB_W,
            y=y0,
            xref="paper",
            yref="paper",
            text=MODEL_NAMES[auditor],
            showarrow=False,
            font=dict(size=18, color="white"),
            bgcolor="rgba(0,0,0,0.6)",
            xanchor="right",
            yanchor="bottom",
            borderpad=4,
        )
        fig.add_annotation(
            x=x0,
            y=y0 + img_h + 0.003,
            xref="paper",
            yref="paper",
            text=f"<b>{'abcdef'[n]}</b>  {title}",
            showarrow=False,
            font=dict(size=27),
            xanchor="left",
            yanchor="bottom",
        )
        # a line from the frame to its city on the map
        c = city[city.locality == title.split(",")[0]].iloc[0]
        cx, cy = to_paper(c.lon, c.lat)
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=x0 + THUMB_W / 2,
            y0=y0 if top else ROW_H,
            x1=cx,
            y1=cy,
            line=dict(color="white", width=6),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=x0 + THUMB_W / 2,
            y0=y0 if top else ROW_H,
            x1=cx,
            y1=cy,
            line=dict(color=MAP["line"], width=2.2),
        )  # a dark line on a white halo, visible on land and sea alike
        ring(cx, cy, 0.0095, "white", 7)
        ring(cx, cy, 0.0095, MAP["line"], 3.5)
        used.append({"panel": "abcdef"[n], "city": title, "model": MODEL_NAMES[auditor], **entry})
    # The legend, drawn by hand: Plotly's own draws the dots smaller than on the map.
    lx, ly, step = 0.02, ROW_H + ROW_GAP + 0.21, 0.031
    fig.add_annotation(
        x=lx - 0.006,
        y=ly + step,
        xref="paper",
        yref="paper",
        text="<b>What pedestrians lack</b>",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=24),
    )
    for k, (label, colour) in enumerate(GAP_COLOURS.items()):
        y = ly - k * step
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            x0=lx - 0.0045,
            x1=lx + 0.0045,
            y0=y - 0.0045 * WORLD_W / WORLD_H,
            y1=y + 0.0045 * WORLD_W / WORLD_H,
            fillcolor=colour,
            line=dict(color=MAP["land"], width=1.5),
        )
        fig.add_annotation(
            x=lx + 0.012,
            y=y,
            xref="paper",
            yref="paper",
            text=f"{label} ({(city.gap == label).sum()})",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=23),
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        font=FONT,
        template="plotly_white",
        width=WORLD_W,
        height=WORLD_H,
        paper_bgcolor="white",
    )
    # Drawn at full size (the frames need the pixels), so no further scaling.
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / "fig1_world.html", include_plotlyjs="cdn", div_id="fig1_world")
    fig.write_image(OUT / "fig1_world.png", scale=1)
    print("wrote fig1_world")
    return city.gap.value_counts(), used


# ---- Figure 2: the gaps by continent, and along the road-death gradient -------------------------------------
def figure_gaps(by_continent: pd.DataFrame, city: pd.DataFrame) -> float:
    order = ["Africa", "Asia", "South America", "Europe", "North America"]
    counts = by_continent[by_continent.key == "need_no_crossing"].set_index("continent").cities
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.1,
        subplot_titles=[
            "<b>a</b>  Share of cities lacking each, by continent",
            "<b>b</b>  More road deaths, more gaps",
        ],
    )
    labels = [f"{c} ({counts[c]})" for c in order]
    for key, name in (("need_no_crossing", "A crossing"), ("need_no_footway", "A footway")):
        part = by_continent[by_continent.key == key].set_index("continent").reindex(order)
        colour = GAP_COLOURS["Crossing" if key == "need_no_crossing" else "Footway"]
        fig.add_trace(
            go.Bar(
                y=labels,
                x=part.share,
                orientation="h",
                name=name,
                marker=dict(color=colour, cornerradius=4),
                text=[f"<b>{v:.0%}</b>" for v in part.share],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(size=16, color="white"),
                hovertemplate="%{y}: %{x:.0%}<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(range=[0, 0.6], tickformat=".0%", dtick=0.2, gridcolor="#ecebe8", zeroline=False, row=1, col=1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=16), row=1, col=1)

    data = city.dropna(subset=["traffic_mortality", "needs"])
    rho = data.traffic_mortality.rank().corr(data.needs.rank())
    fig.add_trace(
        go.Scatter(
            x=data.traffic_mortality,
            y=data.needs,
            mode="markers",
            showlegend=False,
            text=data.locality,
            hovertemplate="%{text}: %{x} deaths, %{y} missing<extra></extra>",
            marker=dict(size=12, color="#8f2d56", opacity=0.45, line=dict(width=1, color="white")),
        ),
        row=1,
        col=2,
    )
    # the gradient: mean missing items per clip in each quarter of the cities by road deaths, with its range
    quarters = pd.qcut(data.traffic_mortality, 4)
    binned = data.groupby(quarters, observed=True).agg(x=("traffic_mortality", "median"), y=("needs", "mean"))
    fig.add_trace(
        go.Scatter(
            x=binned.x,
            y=binned.y,
            mode="lines+markers",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="#1f1f1f", width=3),
            marker=dict(size=13, color="#1f1f1f", line=dict(width=2, color="white")),
        ),
        row=1,
        col=2,
    )
    fig.add_annotation(
        x=binned.x.iloc[-1],
        y=binned.y.iloc[-1],
        xref="x2",
        yref="y2",
        text="mean of each<br>quarter of cities",
        showarrow=True,
        arrowhead=0,
        ax=-10,
        ay=-62,
        font=dict(size=14, color="#1f1f1f"),
        arrowcolor="#1f1f1f",
        xanchor="center",
    )
    for name, ax, ay in (
        ("Kingston", 46, -10),
        ("Peshawar", -58, -8),
        ("Kinshasa", 0, -26),
        ("Benin City", -10, -26),
        ("Nijmegen", 34, -28),
    ):
        r = data[data.locality == name].iloc[0]
        fig.add_annotation(
            x=r.traffic_mortality,
            y=r.needs,
            xref="x2",
            yref="y2",
            text=name,
            showarrow=True,
            arrowhead=0,
            ax=ax,
            ay=ay,
            font=dict(size=14, color=INK_MUTED),
            arrowcolor="#9a9993",
        )
    fig.update_xaxes(
        title="Road deaths per 100,000 people (country)", gridcolor="#ecebe8", zeroline=False, row=1, col=2
    )
    fig.update_yaxes(
        title="Missing items per street clip", gridcolor="#ecebe8", zeroline=False, range=[-0.2, 3.4], row=1, col=2
    )
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=17), xanchor="left", x=0)
    fig.layout.annotations[1].update(x=0.56, text=f"<b>b</b>  More road deaths, more gaps (Spearman’s ρ = {rho:.2f})")
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.25,
        bargroupgap=0.08,
        legend=dict(orientation="h", x=0.0, y=-0.12, font=dict(size=15)),
    )
    save(fig, "fig2_gaps", 1200, 440)
    return rho


# ---- Figure 3: one street, six auditors ---------------------------------------------------------------------
STREET_CLIP, STREET_FRAME = "8dWgC409JbI_00685000_010000", 3  # Kingston, Jamaica: the models' most divided street
FOOTWAY_CELLS = {  # sidewalks in teal (wider is darker), informal paths in amber, none in red, not visible in grey
    "sidewalk_3m_or_more": ("Sidewalk ≥ 3 m", "#0b6e69"),
    "sidewalk_1_to_3m": ("Sidewalk 1–3 m", "#1a9e8f"),
    "sidewalk_under_1m": ("Sidewalk < 1 m", "#7ccbbd"),
    "informal_path_1m_or_more": ("Path ≥ 1 m", "#eda100"),
    "informal_path_under_1m": ("Path < 1 m", "#f5c862"),
    "physical_barrier": ("Behind barrier", "#2a78d6"),
    "none": ("None", "#c62828"),
    "not_visible": ("Not visible", "#b4b2a9"),
}
SAME_CELL = "#eceae4"
STREET_COLUMNS = [
    ("People in<br>the road", lambda a: ("Yes" if a["scene"]["pedestrians_in_carriageway"] else "No", SAME_CELL)),
    ("Reason<br>to cross", lambda a: ("Yes" if a["crossing"]["crossing_demand"] else "No", SAME_CELL)),
    ("Crossing", lambda a: (a["crossing"]["crossing_facility"].replace("_", " ").capitalize(), SAME_CELL)),
    ("Left<br>footway", lambda a: FOOTWAY_CELLS[a["left_side"]["footway"]]),
    ("Right<br>footway", lambda a: FOOTWAY_CELLS[a["right_side"]["footway"]]),
]


def figure_one_street() -> None:
    width, height = 2400, 800
    answers = {
        m: json.loads((ROOT / "outputs/vlm/audit/cities" / m / f"{STREET_CLIP}.json").read_text())["answer"]
        for m in MODEL_NAMES
    }
    record = json.loads((ROOT / "outputs/vlm/audit/cities/qwen3.8-27b" / f"{STREET_CLIP}.json").read_text())
    frame = frame_file(record, STREET_FRAME)
    fig = go.Figure()
    img_w = 0.47
    img_h = img_w * width / height * 9 / 16
    fig.add_layout_image(
        source="data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode(),
        xref="paper",
        yref="paper",
        x=0.0,
        y=0.47 + img_h / 2,
        sizex=img_w,
        sizey=img_h,
        sizing="stretch",
        xanchor="left",
        yanchor="top",
        layer="below",
    )
    fig.add_annotation(
        x=0.0,
        y=0.47 + img_h / 2 + 0.012,
        xref="paper",
        yref="paper",
        text="<b>Kingston, Jamaica</b>",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=30),
    )
    gx0, gx1, name_w = 0.5, 0.995, 0.115
    col_w = (gx1 - gx0 - name_w) / len(STREET_COLUMNS)
    top, row_h = 0.83, 0.12
    for j, (header, cell) in enumerate(STREET_COLUMNS):
        cx = gx0 + name_w + (j + 0.5) * col_w
        distinct = len({cell(a)[0] for a in answers.values()})
        verdict = "✓ all agree" if distinct == 1 else f"✗ {distinct} answers"
        fig.add_annotation(
            x=cx,
            y=top + 0.02,
            xref="paper",
            yref="paper",
            text=f"<b>{header}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=24),
        )
        fig.add_annotation(
            x=cx,
            y=top - len(answers) * row_h - 0.02,
            xref="paper",
            yref="paper",
            text=f"<b>{verdict}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=24, color="#1a7f37" if distinct == 1 else "#c62828"),
        )
    for i, (model, name) in enumerate(MODEL_NAMES.items()):
        y1 = top - i * row_h
        ym = y1 - row_h / 2
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            x0=gx0,
            x1=gx0 + 0.01,
            y0=ym - 0.005 * width / height,
            y1=ym + 0.005 * width / height,
            fillcolor=MODEL_COLOURS[model],
            line_width=0,
        )
        fig.add_annotation(
            x=gx0 + 0.016,
            y=ym,
            xref="paper",
            yref="paper",
            text=name,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=21),
        )
        for j, (_, cell) in enumerate(STREET_COLUMNS):
            text, colour = cell(answers[model])
            x0 = gx0 + name_w + j * col_w
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0 + 0.002,
                x1=x0 + col_w - 0.002,
                y0=y1 - row_h + 0.006,
                y1=y1 - 0.006,
                fillcolor=colour,
                line_width=0,
            )
            dark = colour in ("#0b6e69", "#1a9e8f", "#c62828", "#2a78d6")
            fig.add_annotation(
                x=x0 + col_w / 2,
                y=ym,
                xref="paper",
                yref="paper",
                text=text,
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=20, color="white" if dark else "#1f1f1f"),
            )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        font=FONT, template="plotly_white", width=width, height=height, paper_bgcolor="white", plot_bgcolor="white"
    )
    fig.write_html(OUT / "fig3_one_street.html", include_plotlyjs="cdn", div_id="fig3_one_street")
    fig.write_image(OUT / "fig3_one_street.png", scale=1)
    print("wrote fig3_one_street")


# ---- Figure 4: the models agree less outside Europe and North America ---------------------------------
ITEM_LABELS = {
    "crossing_quality": "Crossing quality",
    "crossing_facility": "Crossing type",
    "crossing_demand": "Reason to cross",
    "street_lighting": "Street lighting",
    "area_type": "Urban or rural",
    "transit_stops": "Transit stops",
    "land_use": "Land use",
    "bike_facility": "Bike facility",
    "number_of_lanes": "Number of lanes",
    "pedestrian_fencing": "Pedestrian fencing",
    "median": "Median",
    "obstruction": "Footway blocked",
    "trip_hazards": "Trip hazards",
    "footway": "Footway type",
    "kerb_ramps": "Kerb ramps",
}


def figure_agreement(agreement: pd.DataFrame) -> None:
    wide = agreement.pivot(index="item", columns="region", values="fleiss_kappa").reindex(list(ITEM_LABELS))
    wide = wide.dropna().sort_values(WEST)
    fig = go.Figure()
    for item, r in wide.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r[WEST], r[REST]],
                y=[ITEM_LABELS[item]] * 2,
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color="#c3c2b7", width=4),
            )
        )
    for region, colour, symbol in ((WEST, "#2a78d6", "circle"), (REST, "#eb6834", "diamond")):
        fig.add_trace(
            go.Scatter(
                x=wide[region],
                y=[ITEM_LABELS[i] for i in wide.index],
                mode="markers",
                name=region,
                marker=dict(size=15, color=colour, symbol=symbol, line=dict(width=1.2, color="white")),
            )
        )
    for x, text in ((0.1, "the models disagree"), (0.7, "the models agree")):
        fig.add_annotation(
            x=x,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"<i>{text}</i>",
            showarrow=False,
            font=dict(size=14, color=INK_MUTED),
        )
    fig.add_vrect(x0=-0.1, x1=0.2, fillcolor="#f3f2ee", line_width=0, layer="below")
    fig.update_xaxes(
        title=f"Agreement between the {N_MODELS} models (Fleiss’ κ)",
        range=[-0.1, 0.8],
        dtick=0.2,
        gridcolor="#ecebe8",
        zeroline=True,
        zerolinecolor="#9a9993",
    )
    fig.update_yaxes(tickfont=dict(size=15))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", font=dict(size=15)),
    )
    save(fig, "fig4_agreement", 1000, 560)


# ---- Figure 5: which models make good auditors ------------------------------------------------------------
def figure_models(quality: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    visible = quality[quality.criterion == "deficit_visible"].groupby("auditor").share.mean()
    backed = (
        consistency.assign(w=consistency.answers * consistency.listed_backed)
        .groupby("model")
        .apply(lambda d: d.w.sum() / d.answers.sum(), include_groups=False)
    )
    per_clip = (
        consistency.assign(w=consistency.answers * consistency.needs_per_clip)
        .groupby("model")
        .apply(lambda d: d.w.sum() / d.answers.sum(), include_groups=False)
    )
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.5, x1=1.02, y0=0.5, y1=1.06, fillcolor="#eef5ec", line_width=0, layer="below")
    fig.add_annotation(
        x=1.0,
        y=1.03,
        text="<i>sound auditors</i>",
        showarrow=False,
        xanchor="right",
        font=dict(size=14, color="#3d6b35"),
    )
    rows = []
    for model, name in MODEL_NAMES.items():
        if model not in visible.index:  # lists no needs (Kanana), so nothing was judged
            continue
        fig.add_trace(
            go.Scatter(
                x=[visible[model]],
                y=[backed[model]],
                mode="markers+text",
                name=name,
                text=[f"<b>{name}</b>"],
                textposition="bottom center" if model in ("gemma4-31b", "qwen3.8-27b", "molmo2-8b") else "top center",
                textfont=dict(size=16, color=MODEL_COLOURS[model]),
                marker=dict(size=26, color=MODEL_COLOURS[model], line=dict(width=1.5, color="white")),
                showlegend=False,
            )
        )
        rows.append({"model": name, "visible": visible[model], "backed": backed[model], "per_clip": per_clip[model]})
    fig.update_xaxes(
        title="Listed needs the judges see in the frames",
        range=[0, 1.02],
        tickformat=".0%",
        dtick=0.25,
        gridcolor="#ecebe8",
        zeroline=False,
    )
    fig.update_yaxes(
        title="Listed needs backed by the model’s own audit",
        range=[0, 1.06],
        tickformat=".0%",
        dtick=0.25,
        gridcolor="#ecebe8",
        zeroline=False,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    save(fig, "fig5_models", 800, 520)
    return pd.DataFrame(rows)


def main() -> int:
    city = pd.read_csv(TABLES / "cities_infrastructure.csv")
    gaps, used = figure_world(city)
    print(json.dumps(used, indent=1, ensure_ascii=False))
    print(gaps)
    print("rho", figure_gaps(pd.read_csv(TABLES / "by_continent.csv"), city))
    figure_one_street()
    figure_agreement(pd.read_csv(TABLES / "agreement_by_region.csv"))
    print(figure_models(pd.read_csv(TABLES / "suggestion_quality.csv"), pd.read_csv(TABLES / "needs_consistency.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
