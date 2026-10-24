"""World map of the drawn cities against the eligible frame.

Run:  uv run python scripts/plot_city_sample.py
"""

import pandas as pd
import plotly.graph_objects as go

from vlm_street_audit.config import PROJECT_ROOT
from vlm_street_audit.figures import save_figure, style

SAMPLE_DIR = PROJECT_ROOT / "data" / "city_sample"


def main() -> None:
    frame = pd.read_csv(SAMPLE_DIR / "eligible_frame_with_pi.csv")
    sample = pd.read_csv(SAMPLE_DIR / "design_manifest.csv")
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=frame["lat"], lon=frame["lon"], mode="markers", name=f"Eligible cities ({len(frame)})",
        marker=dict(size=4, color="lightgrey"), hoverinfo="text",
        text=frame["locality"] + ", " + frame["country"],
    ))
    fig.add_trace(go.Scattergeo(
        lat=sample["lat"], lon=sample["lon"], mode="markers", name=f"Drawn cities ({len(sample)})",
        marker=dict(size=6 + 14 * sample["pi"], color="firebrick", line=dict(width=0.5, color="white")),
        hoverinfo="text",
        text=sample["locality"] + ", " + sample["country"] + "<br>inclusion probability "
        + sample["pi"].round(3).astype(str) + "<br>design weight " + sample["design_weight"].round(1).astype(str),
    ))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="white",
                    landcolor="whitesmoke", showocean=False)
    fig.update_layout(title=f"City sample: {len(sample)} cities in {sample['country'].nunique()} countries",
                      legend=dict(orientation="h", y=-0.05))
    print(save_figure(style(fig), "city_sample_map"))


if __name__ == "__main__":
    main()
