"""Wheat Health & Yield — Streamlit demo serving all four thesis models.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import ui
from inference import disease, rust, yield_pred, egypt

st.set_page_config(page_title="Wheat Health & Yield", page_icon="🌾", layout="wide")
ui.inject_css()
ui.header("🌾 Wheat Health & Yield Intelligence",
          "Disease · rust severity · yield · national forecast — four models, one interface")


# --- Cached loaders: models load once, survive reruns ----------------------
@st.cache_resource
def _disease():
    return disease.load_session()

@st.cache_resource
def _rust():
    return rust.load_session()

@st.cache_resource
def _yield():
    b = yield_pred.load_booster()
    return b, yield_pred.feature_names(b)

@st.cache_resource
def _egypt():
    return egypt.load_model(), egypt.load_history()


def _classify_tab(session, key, top_label="Prediction"):
    """Shared layout for the two image classifiers."""
    up = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], key=key)
    if not up:
        st.info("Upload an image to run the model.")
        return None
    img = Image.open(up)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.image(img, caption="Input", use_container_width=True)
    return img, right


tab1, tab2, tab3, tab4 = st.tabs(
    ["🍂  Disease", "🦠  Rust severity", "🌱  Yield", "📈  Egypt forecast"]
)

# --- Project 1: disease classification -------------------------------------
with tab1:
    st.subheader("Wheat disease classification")
    st.caption("HybridViT · ConvNeXt-Tiny backbone + transformer fusion · 15 classes")
    res = _classify_tab(_disease(), "dis")
    if res:
        img, right = res
        ranked = disease.predict(_disease(), img)
        with right:
            st.metric("Top prediction", ranked[0][0], f"{ranked[0][1]:.1%} confidence")
            st.altair_chart(ui.prob_bar(ranked), use_container_width=True)
            cam = disease.grad_cam_overlay(img)
            if cam is not None:
                st.image(cam, caption="Grad-CAM (stage-3)", use_container_width=True)

# --- Project 2: rust severity ----------------------------------------------
with tab2:
    st.subheader("Yellow rust severity")
    st.caption("EfficientNet-B0")
    res = _classify_tab(_rust(), "rust")
    if res:
        img, right = res
        ranked = rust.predict(_rust(), img)
        with right:
            st.metric("Estimated severity", ranked[0][0], f"{ranked[0][1]:.1%} confidence")
            st.altair_chart(ui.prob_bar(ranked), use_container_width=True)

# --- Project 3: yield prediction -------------------------------------------
with tab3:
    st.subheader("Wheat yield prediction")
    st.caption("LightGBM · gradient-boosted trees on agronomic features")
    booster, names = _yield()
    spec = yield_pred.FEATURE_SPEC

    with st.container(border=True):
        st.markdown("**Inputs**")
        cols = st.columns(2)
        feats = {}
        for i, n in enumerate(names):
            col = cols[i % 2]
            label = n.replace("_", " ")
            s = spec.get(n)
            if s and s["kind"] == "toggle":
                feats[n] = 1.0 if col.toggle(label, value=s.get("default", False),
                                             key=f"y_{n}") else 0.0
            elif s and s["kind"] == "number":
                feats[n] = float(col.number_input(
                    label, value=float(s.get("default", 0.0)),
                    step=float(s.get("step", 1.0)), key=f"y_{n}"))
            else:
                feats[n] = col.number_input(label, value=0.0, format="%.4f", key=f"y_{n}")
        run = st.button("Predict yield", type="primary")

    if run:
        pred = yield_pred.predict(booster, feats)
        col, _ = st.columns([1, 2])
        col.metric("Predicted yield", f"{pred:.3f}")

# --- Project 4: Egypt national forecast ------------------------------------
with tab4:
    st.subheader("Egypt national yield forecast")
    st.caption("Linear Regression · recursive lag-based forecast on FAO data")
    model, history = _egypt()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption(f"Actual series: {int(history['Year'].min())}–{int(history['Year'].max())}")
        end = st.slider("Forecast through year", 2025, 2035, 2027)
    run = st.button("Run forecast", type="primary")

    if run:
        years = list(range(2025, end + 1))      # matches the notebook (starts 2025)
        fc = egypt.recursive_forecast(model, years, history)
        fdf = pd.DataFrame(fc, columns=["Year", "Yield"])
        chart = pd.concat([
            history.assign(series="actual"),
            fdf.assign(series="forecast"),
        ])
        import altair as alt
        line = alt.Chart(chart).mark_line(strokeWidth=2.5).encode(
            x=alt.X("Year:Q", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("Yield:Q", title="Yield", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["actual", "forecast"],
                                            range=[ui.MUTED, ui.GREEN])),
        ).properties(height=380)
        st.altair_chart(line, use_container_width=True)
        st.dataframe(
            fdf.set_index("Year").style.format({"Yield": "{:.3f}"}),
            use_container_width=True,
        )
