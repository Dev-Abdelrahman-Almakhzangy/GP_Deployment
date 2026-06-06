"""Presentation helpers — custom CSS + Altair horizontal charts.

Keeps app.py focused on layout/logic while styling lives here.
"""
import altair as alt
import pandas as pd
import streamlit as st

# Palette (cohesive with the thesis navy + a "health" green accent)
GREEN = "#3FAE6A"
RED = "#D9655B"
MUTED = "#7C8AA0"
TEXT = "#E6E9EE"

_CSS = """
<style>
/* hide default chrome for a cleaner look */
#MainMenu, footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

/* layout */
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1080px;}
html, body, [class*="css"] {font-family: Arial, "Helvetica Neue", sans-serif;}

/* app header band */
.app-header {
  background: linear-gradient(90deg, #1F2D4A 0%, #2E4057 100%);
  padding: 22px 26px; border-radius: 14px; margin-bottom: 22px;
}
.app-header h1 {color: #fff; font-size: 1.6rem; margin: 0;}
.app-header p {color: #c4ccd8; margin: 4px 0 0; font-size: .95rem;}

/* tabs: larger, spaced */
button[data-baseweb="tab"] {font-size: 1rem; padding: 12px 20px;}
[data-baseweb="tab-list"] {gap: 6px;}

/* metric -> card */
[data-testid="stMetric"] {
  background: #1A2436; border: 1px solid #2E4057;
  padding: 16px 20px; border-radius: 12px;
}

/* bordered containers rounded */
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}

/* buttons */
.stButton > button {border-radius: 10px; padding: .5rem 1.4rem; font-weight: 600;}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="app-header"><h1>{title}</h1>'
        f'{f"<p>{subtitle}</p>" if subtitle else ""}</div>',
        unsafe_allow_html=True,
    )


def prob_bar(ranked, top_color=GREEN):
    """Horizontal probability bars; the top class is highlighted."""
    df = pd.DataFrame(ranked, columns=["label", "prob"])
    df["pct"] = (df["prob"] * 100).round(1)
    top = df["label"].iloc[0]
    df["top"] = df["label"] == top

    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=13)),
        x=alt.X("prob:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", grid=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=22).encode(
        color=alt.condition(alt.datum.top, alt.value(top_color), alt.value(MUTED)),
        tooltip=[alt.Tooltip("label:N", title="class"),
                 alt.Tooltip("pct:Q", title="probability %")],
    )
    labels = base.mark_text(align="left", dx=5, color=TEXT, fontSize=12).encode(
        text=alt.Text("pct:Q", format=".1f")
    )
    return (bars + labels).properties(height=max(120, len(df) * 34))


def shap_bar(names, values):
    """Diverging horizontal contribution bars, sorted by magnitude."""
    df = pd.DataFrame({"feature": names, "contribution": values})
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    order = df["feature"].tolist()

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
        y=alt.Y("feature:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=13)),
        x=alt.X("contribution:Q", title="Contribution to prediction",
                axis=alt.Axis(grid=False)),
        color=alt.condition(alt.datum.contribution > 0,
                            alt.value(GREEN), alt.value(RED)),
        tooltip=[alt.Tooltip("feature:N"),
                 alt.Tooltip("contribution:Q", format=".3f")],
    ).properties(height=max(120, len(df) * 30))
    return chart


def _fmt_val(v):
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"


def waterfall(base, names, values, pred):
    """SHAP waterfall: builds the prediction from baseline, one feature at a time.

    Each bar floats from the running total to the new total; green steps up,
    red steps down. Reads left-to-right as 'how we got from average to this'.
    """
    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    rows = [{"step": "Baseline", "start": 0.0, "end": base,
             "kind": "base", "label": f"{base:.2f}", "rank": 0}]
    running = base
    for rank, i in enumerate(order, start=1):
        start = running
        running += values[i]
        rows.append({"step": names[i].replace("_", " "),
                     "start": start, "end": running,
                     "kind": "pos" if values[i] >= 0 else "neg",
                     "label": f"{values[i]:+.2f}", "rank": rank})
    rows.append({"step": "Predicted", "start": 0.0, "end": running,
                 "kind": "total", "label": f"{running:.2f}", "rank": len(order) + 1})
    df = pd.DataFrame(rows)
    df["label_x"] = df[["start", "end"]].max(axis=1)

    y = alt.Y("step:N", sort=alt.SortField("rank"), title=None,
              axis=alt.Axis(labelLimit=240, labelFontSize=13))
    color = alt.Color("kind:N", legend=None, scale=alt.Scale(
        domain=["base", "pos", "neg", "total"],
        range=[MUTED, GREEN, RED, "#3E7CB1"]))

    bars = alt.Chart(df).mark_bar(cornerRadius=3, height=24).encode(
        y=y, x=alt.X("start:Q", title="Yield", axis=alt.Axis(grid=False)),
        x2="end:Q", color=color,
        tooltip=[alt.Tooltip("step:N"), alt.Tooltip("label:N", title="effect")],
    )
    text = alt.Chart(df).mark_text(align="left", dx=5, color=TEXT, fontSize=12).encode(
        y=y, x="label_x:Q", text="label:N",
    )
    return (bars + text).properties(height=max(160, len(df) * 38))


def importance_bar(names, values):
    """Global feature importance (all positive) as horizontal bars."""
    df = pd.DataFrame({"feature": [n.replace("_", " ") for n in names],
                       "importance": values})
    df = df.sort_values("importance", ascending=False)
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20, color=GREEN).encode(
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)),
        x=alt.X("importance:Q", title="Importance (gain)", axis=alt.Axis(grid=False)),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".0f")],
    ).properties(height=max(120, len(df) * 30))
