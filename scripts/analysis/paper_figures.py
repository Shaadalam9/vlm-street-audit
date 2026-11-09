"""The six figures of the paper, made to be understood at a glance.

Reads the published answers and analysis tables (outputs/vlm/) and, for Figure 1, the frames of six city
clips (data/clip_sets/, built by build_clip_sets.py). Writes docs/figures/paper/fig*.png and .html.

Run (after analyse_audit.py):  uv run python scripts/analysis/paper_figures.py
"""

import base64
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vlm_street_audit.config import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "docs" / "figures" / "paper"
TABLES = ROOT / "outputs" / "vlm" / "analysis_audit"
FONT = dict(family="Arial", size=15, color="#1f1f1f")
INK_MUTED = "#5f5e5a"
MODEL_NAMES = {
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma4-31b": "Gemma 4 31B",
    "molmo2-8b": "Molmo2-8B",
    "internvl3.5-8b": "InternVL3.5-8B",
    "ministral3-14b": "Ministral 3 14B",
}
MODEL_COLOURS = {
    "qwen3.8-27b": "#2a78d6",
    "gemma4-31b": "#eb6834",
    "molmo2-8b": "#1baf7a",
    "internvl3.5-8b": "#eda100",
    "ministral3-14b": "#e87ba4",
}
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
    fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(OUT / f"{name}.png", scale=2)
    print("wrote", name)


# ---- Figure 1: what the models see, on six continents ----------------------------------------------------
EXAMPLES = [  # (clip, auditor, index of the need in its answer, panel title, tag)
    ("Q3WvZW9zBk8_00851000_010000", "qwen3.8-27b", "no_crossing", "Kinshasa, DR Congo", "Crossing needed"),
    ("MMSSZo4HdOo_02862000_010000", "qwen3.8-27b", "blocked_footway", "Peshawar, Pakistan", "Path blocked"),
    ("cKRt8KHugxA_02030000_010000", "qwen3.8-27b", "no_crossing", "Kazan, Russia", "Sign, but no crossing"),
    ("8wdxcC3VkQY_00160000_010000", "gemma4-31b", "no_footway", "Spring Hill, United States", "No footway"),
    ("m2BHJJ2x9oo_00986000_010000", "qwen3.8-27b", "blocked_footway", "Valledupar, Colombia", "Path blocked"),
    ("wMu6Va5PhGY_00171000_010000", "qwen3.8-27b", "no_footway", "Sydney, Australia", "“No footway” (wrong)"),
]


def need_entry(clip: str, auditor: str, need: str) -> tuple[dict, dict]:
    record = json.loads((ROOT / "outputs/vlm/audit/cities" / auditor / f"{clip}.json").read_text())
    entry = next(d for d in record["answer"]["needed"] if d["need"] == need)
    return record, entry


def figure_examples() -> list[dict]:
    fig = make_subplots(
        rows=2,
        cols=3,
        horizontal_spacing=0.012,
        vertical_spacing=0.07,
        subplot_titles=[f"<b>{'abcdef'[i]}</b>  {e[3]}" for i, e in enumerate(EXAMPLES)],
    )
    used = []
    for i, (clip, auditor, need, title, tag) in enumerate(EXAMPLES):
        row, col = divmod(i, 3)
        record, entry = need_entry(clip, auditor, need)
        frame = ROOT / record["frames"][entry["frame"] - 1]
        data = base64.b64encode(frame.read_bytes()).decode()
        fig.add_trace(go.Image(source=f"data:image/jpeg;base64,{data}", hoverinfo="skip"), row=row + 1, col=col + 1)
        x, y = entry["x"] / 1000 * 1280, entry["y"] / 1000 * 720
        wrong = "wrong" in tag
        colour = "#c62828" if not wrong else "#5f5e5a"
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(width=5, color="white")),
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(size=30, color="rgba(0,0,0,0)", line=dict(width=3.5, color=colour)),
            ),
            row=row + 1,
            col=col + 1,
        )
        axis = "" if i == 0 else str(i + 1)
        fig.add_annotation(
            x=x,
            y=y - 70 if y > 160 else y + 70,
            xref=f"x{axis}",
            yref=f"y{axis}",
            text=f"<b>{tag}</b>",
            showarrow=False,
            font=dict(size=17, color="white"),
            bgcolor=colour,
            borderpad=5,
            xanchor="center" if 220 < x < 1060 else ("left" if x <= 220 else "right"),
        )
        fig.add_annotation(
            x=1270,
            y=712,
            xref=f"x{axis}",
            yref=f"y{axis}",
            text=MODEL_NAMES[auditor],
            showarrow=False,
            font=dict(size=13, color="white"),
            bgcolor="rgba(0,0,0,0.55)",
            xanchor="right",
            yanchor="bottom",
            borderpad=3,
        )
        used.append({"panel": "abcdef"[i], "city": title, "model": MODEL_NAMES[auditor], **entry})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=17))
    fig.update_layout(margin=dict(l=4, r=4, t=34, b=4))
    save(fig, "fig1_examples", 1200, 500)
    return used


# ---- Figure 2: what each city lacks, on a map -------------------------------------------------------------
GAP_COLOURS = {
    "Crossing and footway": "#8f2d56",
    "Crossing": "#eb6834",
    "Footway": "#2a78d6",
    "Lighting or other": "#eda100",
    "No agreed need": "#c3c2b7",
}


def figure_map(city: pd.DataFrame) -> pd.Series:
    def gap(r) -> str:
        crossing, footway = r.need_no_crossing > 0, r.need_no_footway > 0
        other = any(
            r[c] > 0 for c in city.columns if c.startswith("need_") and c not in ("need_no_crossing", "need_no_footway")
        )
        if crossing and footway:
            return "Crossing and footway"
        return "Crossing" if crossing else "Footway" if footway else "Lighting or other" if other else "No agreed need"

    city = city.assign(gap=city.apply(gap, axis=1))
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
                marker=dict(size=13, color=colour, line=dict(width=1.2, color="white")),
            )
        )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#ecebe8",
        showcountries=True,
        countrycolor="#ffffff",
        coastlinecolor="#c3c2b7",
        showocean=False,
        showframe=False,
        lataxis_range=[-50, 72],
        lonaxis_range=[-130, 160],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            y=-0.02,
            x=0.5,
            xanchor="center",
            font=dict(size=15),
            title=dict(text="What pedestrians lack:  ", side="left"),
        ),
    )
    save(fig, "fig2_map", 1200, 560)
    return city.gap.value_counts()


# ---- Figure 3: the three big gaps, by continent ---------------------------------------------------------
def figure_continents(by_continent: pd.DataFrame) -> None:
    needs = [
        ("need_no_crossing", "A crossing where people cross"),
        ("need_no_footway", "A footway"),
        ("need_no_street_lighting", "Street lighting"),
    ]
    order = ["Africa", "Asia", "South America", "Europe", "North America"]
    counts = by_continent[by_continent.key == "need_no_crossing"].set_index("continent").cities
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[f"<b>{t}</b>" for _, t in needs]
    )
    for col, (key, _) in enumerate(needs, start=1):
        part = by_continent[by_continent.key == key].set_index("continent").reindex(order)
        fig.add_trace(
            go.Bar(
                y=[f"{c} ({counts[c]})" for c in order],
                x=part.share,
                orientation="h",
                showlegend=False,
                marker_color=[CONTINENT_COLOURS[c] for c in order],
                text=[f"{v:.0%}" for v in part.share],
                textposition="outside",
                textfont=dict(size=17),
                cliponaxis=False,
            ),
            row=1,
            col=col,
        )
    # Intervals are in the text and tables; Africa (7 cities) and South America (5) are wide.
    fig.update_xaxes(range=[0, 0.8], tickformat=".0%", tickvals=[0, 0.25, 0.5], gridcolor="#ecebe8", zeroline=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=16))
    fig.update_annotations(font=dict(size=17))
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10), bargap=0.3)
    save(fig, "fig3_continents", 1200, 380)


# ---- Figure 4: more road deaths, more missing infrastructure -------------------------------------------
def figure_deaths(city: pd.DataFrame) -> float:
    data = city.dropna(subset=["traffic_mortality", "needs"])
    rho = data.traffic_mortality.rank().corr(data.needs.rank())
    fig = go.Figure()
    for continent, colour in CONTINENT_COLOURS.items():
        part = data[data.continent == continent]
        fig.add_trace(
            go.Scatter(
                x=part.traffic_mortality,
                y=part.needs,
                mode="markers",
                name=continent,
                text=part.locality,
                hovertemplate="%{text}: %{x}, %{y}<extra></extra>",
                marker=dict(size=13, color=colour, opacity=0.85, line=dict(width=1, color="white")),
            )
        )
    slope, intercept = np.polyfit(data.traffic_mortality, data.needs, 1)
    xs = np.array([data.traffic_mortality.min(), data.traffic_mortality.max()])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=slope * xs + intercept,
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="#1f1f1f", width=2, dash="dash"),
        )
    )
    for name in ["Kinshasa", "Kingston", "Peshawar", "Benin City", "Bamberg", "Nijmegen", "Spring Hill"]:
        row = data[data.locality == name]
        if not row.empty:
            r = row.iloc[0]
            ax, ay = (34, 22) if name == "Spring Hill" else (0, -26)
            fig.add_annotation(
                x=r.traffic_mortality,
                y=r.needs,
                text=name,
                showarrow=True,
                arrowhead=0,
                ax=ax,
                ay=ay,
                font=dict(size=14, color=INK_MUTED),
                arrowcolor="#9a9993",
            )
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        text=f"Spearman’s ρ = {rho:.2f} (75 cities)",
        font=dict(size=16),
    )
    fig.update_xaxes(title="Road deaths per 100,000 people (country)", gridcolor="#ecebe8", zeroline=False)
    fig.update_yaxes(title="Missing items per street clip", gridcolor="#ecebe8", zeroline=False, range=[-0.2, 3.85])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08, x=0, font=dict(size=15))
    )
    save(fig, "fig4_road_deaths", 1100, 520)
    return rho


# ---- Figure 5: the models agree less outside Europe and North America ---------------------------------
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
        title="Agreement between the five models (Fleiss’ κ)",
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
    save(fig, "fig5_agreement", 1000, 560)


# ---- Figure 6: which models make good auditors ------------------------------------------------------------
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
    save(fig, "fig6_models", 800, 520)
    return pd.DataFrame(rows)


def main() -> int:
    city = pd.read_csv(TABLES / "cities_infrastructure.csv")
    used = figure_examples()
    print(json.dumps(used, indent=1, ensure_ascii=False))
    print(figure_map(city))
    figure_continents(pd.read_csv(TABLES / "by_continent.csv"))
    print("rho", figure_deaths(city))
    figure_agreement(pd.read_csv(TABLES / "agreement_by_region.csv"))
    print(figure_models(pd.read_csv(TABLES / "suggestion_quality.csv"), pd.read_csv(TABLES / "needs_consistency.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
