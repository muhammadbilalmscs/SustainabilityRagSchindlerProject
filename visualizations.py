"""
visualizations.py - Phase 7: turn verified figures into tables, KPIs and Plotly charts.

Responsibility
    figures_to_dataframe(): list of figure dicts -> tidy pandas DataFrame
    group_by_unit():        split figures so each group has ONE unit
    make_chart():           one unit group -> a Plotly bar chart, or None

Why Plotly?
    Plotly charts are interactive (hover shows exact values and pages, zoom,
    download as PNG) and Streamlit shows them with st.plotly_chart().

Honesty rules for charts (from the project brief)
    - Only verified figures arrive here (rag_pipeline.verify_figures).
    - Never mix units: MWh and t CO2e never share an axis. Each unit gets its
      own chart.
    - A chart needs at least 2 values; a single value is shown as a KPI instead.
    - Years stay categories ("2020", "2024", "2025"), not a continuous time axis,
      because the report skips 2021-2023 and a line would invent a trend between them.
"""

import pandas as pd
import plotly.express as px


def figures_to_dataframe(figures):
    """
    Input:  list of dicts with label, year, value, unit, page
    Output: DataFrame with columns Label, Year, Value, Unit, Page
            (duplicates removed, sorted by label and year)
    """
    if not figures:
        return pd.DataFrame(columns=["Label", "Year", "Value", "Unit", "Page"])

    df = pd.DataFrame(figures)
    df = df.rename(columns={"label": "Label", "year": "Year", "value": "Value",
                            "unit": "Unit", "page": "Page"})
    df = df[["Label", "Year", "Value", "Unit", "Page"]]
    df["Year"] = df["Year"].astype(str)
    df = df.drop_duplicates(subset=["Label", "Year", "Unit"])
    return df.sort_values(["Label", "Year"]).reset_index(drop=True)


def unit_key(unit):
    """
    A simple key so that small spelling differences count as the same unit:
    "t CO₂e", "t CO2e" and "tCO2e" -> "tco2e". Different units stay different.
    """
    return unit.replace("₂", "2").replace(" ", "").lower()


def group_by_unit(df):
    """
    Split a figures DataFrame into one DataFrame per unit.

    Input:  DataFrame from figures_to_dataframe
    Output: dict {display unit: DataFrame}, e.g. {"MWh": df1, "t CO₂e": df2}
    """
    groups = {}
    if df.empty:
        return groups
    keys = df["Unit"].apply(unit_key)
    for key in keys.unique():
        group = df[keys == key]
        display_unit = group["Unit"].iloc[0]  # show the unit as the report wrote it
        groups[display_unit] = group.reset_index(drop=True)
    return groups


def make_chart(df, unit):
    """
    Build a bar chart for figures that all share one unit.

    Input:  df (one unit group), unit (str, for the axis title)
    Output: plotly Figure, or None when a chart would not be meaningful (< 2 values)

    - Several years  -> grouped bars: x = year, one colour per label
    - One year only  -> horizontal bars: one bar per label (easy to read long labels)
    """
    if len(df) < 2:
        return None

    years = df["Year"].unique()
    hover = {"Page": True, "Unit": True, "Value": ":,.2f"}

    if len(years) > 1:
        fig = px.bar(
            df, x="Year", y="Value", color="Label", barmode="group",
            text_auto=",.0f", hover_data=hover,
            category_orders={"Year": sorted(years)},
        )
        fig.update_xaxes(type="category", title="Reporting year")
    else:
        fig = px.bar(
            df.sort_values("Value"), x="Value", y="Label", orientation="h",
            text_auto=",.0f", hover_data=hover,
        )
        fig.update_yaxes(title="")

    pages = ", ".join(str(page) for page in sorted(df["Page"].unique()))
    fig.update_layout(
        title=f"{unit}  (source: page {pages})",
        yaxis_title=unit if len(years) > 1 else None,
        xaxis_title=unit if len(years) == 1 else "Reporting year",
        legend_title_text="",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Quick manual test:  python visualizations.py   (no PDF or API key needed)
# Opens the chart in your browser.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_figures = [
        {"label": "Total scope 1", "year": "2025", "value": 125756, "unit": "t CO₂e", "page": 37},
        {"label": "Total scope 1", "year": "2024", "value": 131225, "unit": "t CO₂e", "page": 37},
        {"label": "Total scope 1", "year": "2020", "value": 129680, "unit": "t CO₂e", "page": 37},
        {"label": "Purchased electricity (location-based)", "year": "2025", "value": 51766, "unit": "t CO2e", "page": 37},
        {"label": "Purchased electricity (location-based)", "year": "2024", "value": 53079, "unit": "t CO2e", "page": 37},
        {"label": "Natural gas", "year": "2025", "value": 65671, "unit": "MWh", "page": 36},
    ]
    table = figures_to_dataframe(sample_figures)
    print(table, "\n")
    for unit, group in group_by_unit(table).items():
        chart = make_chart(group, unit)
        print(f"{unit!r}: {len(group)} values -> {'chart' if chart else 'KPI only (1 value)'}")
        if chart:
            chart.show()
