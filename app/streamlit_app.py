"""
Energy Consumption Forecast — Streamlit Demo
=============================================
Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import mae, rmse, mape

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TARGET = "total load actual"


# ---------------------------------------------------------------------------
# Load data and model
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(DATA_DIR, "features_dataset.csv"), parse_dates=["time"]
    )


@st.cache_resource
def load_model():
    with open(os.path.join(DATA_DIR, "best_xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(DATA_DIR, "feature_columns.json"), "r") as f:
        features = json.load(f)
    return model, features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Energy Consumption Forecast")
    st.markdown("""
                24-hour-ahead forecasting for Spain's electricity grid. Select a date range to see how the 
                model performs against actual consumption.

                **Objective:** Predict hourly energy consumption 24 hours ahead using only information available 
                at prediction time: historical demand patterns, time features, and day-ahead generation forecasts.

                **Data:** [Hourly Energy Demand, Generation, Prices and Weather]
                (https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). 
                4 years of hourly data from Spain (2015-2018) covering consumption, generation by source, electricity 
                prices, and weather across 5 cities.

                **Approach:** Engineered lag features (24h, 48h, 168h), rolling statistics, 
                and time features. Compared XGBoost, LightGBM, and Random Forest against naive baselines and the 
                TSO's own forecast. Best model achieves 1.48% MAPE, an 82% improvement over the naive baseline.
    """)

    try:
        df = load_data()
        model, feature_cols = load_model()
    except FileNotFoundError as e:
        st.error(
            f"**Data or model not found.** Run notebooks 02 and 03 first "
            f"to generate the required files.\n\n`{e}`"
        )
        st.stop()

    # Predictions
    df_valid = df.dropna(subset=feature_cols + [TARGET]).copy()
    df_valid["prediction"] = model.predict(df_valid[feature_cols])
    df_valid["residual"] = df_valid[TARGET] - df_valid["prediction"]
    df_valid["abs_error"] = df_valid["residual"].abs()

    # --- Sidebar ---
    st.sidebar.header("Settings")

    min_date = df_valid["time"].min().date()
    max_date = df_valid["time"].max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(max_date - pd.Timedelta(days=14), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range

    mask = (df_valid["time"].dt.date >= start) & (df_valid["time"].dt.date <= end)
    view = df_valid[mask]

    if len(view) == 0:
        st.warning("No data in selected range.")
        st.stop()

    st.sidebar.markdown("""
                        **Note:** Oct 2018 - Dec 2018 is the held-out test period. Predictions on earlier dates are on training 
                        data and will appear more accurate.
    """)

    # --- Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae(view[TARGET], view['prediction']):,.0f} MW")
    col2.metric("RMSE", f"{rmse(view[TARGET], view['prediction']):,.0f} MW")
    col3.metric("MAPE", f"{mape(view[TARGET], view['prediction']):.1f}%")
    col4.metric("Hours shown", f"{len(view):,}")

    # --- Forecast chart ---
    st.subheader("Actual vs Predicted")
    st.markdown("""
                Predictions from XGBoost trained with time features, lag/rolling statistics, and the TSO's day-ahead solar and 
                wind generation forecasts. This feature set outperformed raw weather variables in ablation testing 
                (1.48% vs 1.65% MAPE), likely because the generation forecasts are pre-processed weather signals that 
                are more directly relevant to grid conditions. The model uses only information genuinely available 24 
                hours before prediction time.
    """)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["time"],
            y=view[TARGET],
            name="Actual",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["time"],
            y=view["prediction"],
            name="Predicted",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Error breakdown tabs ---
    st.subheader("Error Breakdown")
    st.subheader("Error Breakdown")
    st.markdown("""
                Model errors are not uniform. Morning ramp-up hours and evenings are hardest to predict 
                due to variable demand timing. Mid-week days are easiest, while weekends and 
                Mondays show higher errors from less predictable activity patterns. The residual distribution 
                is centered at zero with no systematic bias, though a few large errors occur 
                on holidays where the model expects normal demand.
    """)
    tab1, tab2, tab3 = st.tabs(["By Hour", "By Day of Week", "Residuals"])

    with tab1:
        hourly = view.groupby("hour")["abs_error"].mean().reset_index()
        fig_h = px.bar(
            hourly,
            x="hour",
            y="abs_error",
            labels={"hour": "Hour of day", "abs_error": "MAE (MW)"},
            color_discrete_sequence=["#ff7f0e"],
        )
        fig_h.update_layout(height=350)
        st.plotly_chart(fig_h, use_container_width=True)

    with tab2:
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        daily = view.groupby("dayofweek")["abs_error"].mean().reset_index()
        daily["day"] = daily["dayofweek"].map(day_map)
        fig_d = px.bar(
            daily,
            x="day",
            y="abs_error",
            labels={"day": "Day", "abs_error": "MAE (MW)"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_d.update_layout(height=350)
        st.plotly_chart(fig_d, use_container_width=True)

    with tab3:
        fig_r = px.histogram(
            view,
            x="residual",
            nbins=60,
            labels={"residual": "Residual (MW)"},
            color_discrete_sequence=["#2ca02c"],
        )
        fig_r.add_vline(x=0, line_dash="dash", line_color="red")
        fig_r.update_layout(height=350)
        st.plotly_chart(fig_r, use_container_width=True)

    # --- Feature importance ---
    st.subheader("Top 15 Feature Importances")
    st.markdown("""
                Hour of day dominates, followed by rolling mean statistics that capture recent 
                demand trends. The solar generation forecast ranks #7, confirming it adds signal 
                beyond what time and lag features provide. Longer-term lags (168h, 48h) rank lower 
                because the rolling means already summarise the same historical patterns more smoothly.
    """)

    importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=True)
        .tail(15)
    )

    fig_imp = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance (gain)", "feature": ""},
        color_discrete_sequence=["#9467bd"],
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

    # --- Model Comparison ---
    st.subheader("Model Comparison")
    st.markdown("""
                **Baselines** set the floor. Naive uses the consumption value from 24 hours ago. 
                Seasonal naive uses the value from exactly one week ago (same hour, same day of week). 
                7-day rolling mean averages the past week. These are simple heuristics that any ML model should beat.

                **Random Forest** is a bagging ensemble that builds many independent decision trees and 
                averages their predictions. It improves on baselines but can't learn sequential error patterns 
                the way boosting methods can.

                **XGBoost (default and tuned)** and **LightGBM** are gradient boosting models that build trees 
                sequentially, with each tree correcting the errors of the previous one. XGBoost tuned slightly 
                outperforms default after grid search over depth, learning rate, and number of trees. LightGBM 
                produces similar results, confirming that performance is driven by feature engineering rather than 
                model choice.

                **XGBoost (gen forecasts)** is the same tuned XGBoost but trained with day-ahead solar and wind 
                generation forecasts instead of raw weather features. This produced the best ML result (1.48% MAPE), 
                as the generation forecasts act as pre-processed weather signals more directly relevant to grid 
                conditions.

                **TSO Forecast** is the benchmark from Red Electrica, Spain's grid operator, using proprietary 
                models and real-time data. It represents the practical ceiling for this problem given publicly 
                available data.
    """)

    try:
        comparison = pd.read_csv(
            os.path.join(DATA_DIR, "model_comparison.csv"), index_col=0
        )

        if "model" in comparison.columns:
            comparison = comparison.drop(columns=["model"])

        tab_mae, tab_rmse, tab_mape = st.tabs(["MAE", "RMSE", "MAPE"])

        with tab_mae:
            fig = px.bar(
                comparison.reset_index(),
                x=comparison.index,
                y="MAE",
                labels={"index": "Model", "MAE": "MAE (MW)"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with tab_rmse:
            fig = px.bar(
                comparison.reset_index(),
                x=comparison.index,
                y="RMSE",
                labels={"index": "Model", "RMSE": "RMSE (MW)"},
                color_discrete_sequence=["#ff7f0e"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with tab_mape:
            fig = px.bar(
                comparison.reset_index(),
                x=comparison.index,
                y="MAPE",
                labels={"index": "Model", "MAPE": "MAPE (%)"},
                color_discrete_sequence=["#2ca02c"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Raw table
        st.dataframe(
            comparison.style.format("{:.2f}").apply(
                lambda x: [
                    "background-color: #FFD700"
                    if i == "XGBoost (gen forecasts)"
                    else ""
                    for i in x.index
                ],
                axis=0,
            )
        )

    except FileNotFoundError:
        st.warning("model_comparison.csv not found. Run notebook 03 first.")

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Data: Hourly Energy Demand, Generation, Prices and Weather (Kaggle). "
        "Model: XGBoost with walk-forward validation."
    )


if __name__ == "__main__":
    main()
