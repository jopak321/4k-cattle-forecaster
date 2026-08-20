import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import urllib.parse
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="4K Cattle Forecaster", layout="wide")
st.title("🥩 4K Cattle Market Forecaster")
st.markdown("Machine Learning predictions and strategic market analysis.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Data & AI Settings")
api_key = st.sidebar.text_input("USDA API Key:", type="password")

# --- Names mapped to IMPS codes ---
cut_options = {
    "90% Trimmings (90)": "90",
    "81% Trimmings (81)": "81",
    "Striploin (180)": "180",
    "Ribeye Hvy (112A)": "112A",
    "Top Butt CC (184B)": "184B",
    "Chuck Roll Neck Off (116A)": "116A",
    "Tri Tip Peeled (185D)": "185D",
    "Tenderloin (189A)": "189A"
}
selected_cut_name = st.sidebar.selectbox("Primary Target Commodity:", list(cut_options.keys()))
target_code = cut_options[selected_cut_name]
forecast_weeks = st.sidebar.slider("Weeks to Forecast", 4, 26, 12)

st.sidebar.markdown("---")
st.sidebar.header("Cost Assumptions")
freight_cost = st.sidebar.number_input("Freight to Storage ($/lb)", value=0.050, format="%.3f", step=0.010)

storage_cost_per_week = 0.015
st.sidebar.info(f"🔒 Storage Cost is hardcoded at **${storage_cost_per_week:.2f} /lb/week**")

# --- Buyer Target Price ---
st.sidebar.markdown("---")
st.sidebar.header("Buyer Strategy Tools")
target_promo_price = st.sidebar.number_input(
    "Promo Target Price ($/lb)", value=0.00, step=0.25, format="%.2f",
    help="Set to 0 to disable. If set, draws a target execution line on the forecast chart."
)

# ==========================================
# DATA FETCHING
# ==========================================
@st.cache_data(ttl=86400, show_spinner="Fetching fresh USDA data across all sections...")
def get_beef_data(key, item_code):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4 * 365)
    date_range = urllib.parse.quote(f"{start_date.strftime('%m/%d/%Y')}:{end_date.strftime('%m/%d/%Y')}")

    df_list = []

    if item_code in ["81", "90"]:
        url = f"https://mpr.datamart.ams.usda.gov/services/v1.1/reports/2462/National?q=report_date={date_range}"
        urls_to_check = [(url, 'price_range_avg')]
    else:
        base_url = "https://mpr.datamart.ams.usda.gov/services/v1.1/reports/2461/"
        sections = ["Choice%20Cuts", "Select%20Cuts", "Choice%2FSelect%20Overlaps"]
        urls_to_check = [(base_url + s + f"?q=report_date={date_range}", 'weighted_average') for s in sections]

    for fetch_url, col in urls_to_check:
        resp = requests.get(fetch_url, auth=HTTPBasicAuth(key, ''))
        if resp.status_code == 200 and 'results' in resp.json():
            temp_df = pd.DataFrame(resp.json()['results'])
            if not temp_df.empty:
                desc_col = 'item_desc' if 'item_desc' in temp_df.columns else 'item_description'
                if desc_col not in temp_df.columns and len(temp_df.columns) > 2:
                    desc_col = temp_df.columns[2]

                if desc_col in temp_df.columns:
                    matched = temp_df[temp_df[desc_col].astype(str).str.contains(
                        item_code, case=False, na=False, regex=True)].copy()
                    if not matched.empty:
                        matched['target_price_col'] = matched[col]
                        df_list.append(matched)

    if not df_list:
        return pd.DataFrame(), ""

    df = pd.concat(df_list, ignore_index=True)
    price_col = 'target_price_col'

    df[price_col] = df[price_col].astype(str).str.replace(',', '', regex=False)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce') / 100.0
    df = df[df[price_col] > 0]

    if df.empty:
        return df, price_col

    df['report_date'] = pd.to_datetime(df['report_date'])
    df = df.groupby('report_date')[price_col].mean().reset_index()
    df = df.sort_values(by='report_date', ascending=True)

    return df, price_col


# ==========================================
# MODEL TRAINING
# FIX: cache_resource (not cache_data) — a fitted Prophet model holds a
# compiled Stan backend that does not survive pickling. Also takes an explicit
# `fingerprint` so the cache actually invalidates when new USDA rows land
# (the underscore on _df excludes it from the cache key).
# ==========================================
@st.cache_resource(ttl=86400, show_spinner="Training AI...")
def run_prophet(_df, price_col, weeks_out, item_code, fingerprint):
    df_prophet = pd.DataFrame({'ds': _df['report_date'], 'y': _df[price_col]}).dropna()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=weeks_out, freq='W-FRI')
    return model, model.predict(future)


# ==========================================
# BACKTEST
# FIX: rolling-origin cross-validation scored against held-out data, plus a
# naive carry-forward baseline. In-sample R² measured fit to data the model
# already saw and was effectively always "reliable".
# ==========================================
@st.cache_data(ttl=86400, show_spinner="Backtesting against held-out history...")
def backtest_model(_model, _df, price_col, weeks_out, item_code, fingerprint):
    horizon_days = weeks_out * 7
    span_days = (_df['report_date'].max() - _df['report_date'].min()).days
    initial_days = span_days - (horizon_days * 3)

    if initial_days < horizon_days * 2:
        return None  # not enough history for this horizon

    df_cv = cross_validation(
        _model,
        initial=f"{initial_days} days",
        period=f"{max(horizon_days // 2, 7)} days",
        horizon=f"{horizon_days} days",
        disable_tqdm=True,
    )

    # Naive baseline: carry the last observed price forward from each cutoff
    obs = _df.rename(columns={'report_date': 'ds', price_col: 'y'})[['ds', 'y']]
    naive_map = {
        c: obs.loc[obs['ds'] <= c, 'y'].iloc[-1]
        for c in df_cv['cutoff'].unique()
    }
    df_cv['naive'] = df_cv['cutoff'].map(naive_map)

    mae_model = mean_absolute_error(df_cv['y'], df_cv['yhat'])
    mae_naive = mean_absolute_error(df_cv['y'], df_cv['naive'])

    return {
        'r2': r2_score(df_cv['y'], df_cv['yhat']),
        'mae': mae_model,
        'mae_naive': mae_naive,
        'skill': 1 - (mae_model / mae_naive) if mae_naive > 0 else 0.0,
        'mape': float(np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])) * 100),
        'n_folds': df_cv['cutoff'].nunique(),
    }


# ==========================================
# PLOTTING
# FIX: replaces prophet.plot.plot_plotly, which runs `assert m.history` and
# raises ValueError because pandas refuses to evaluate a DataFrame's truthiness.
# ==========================================
def plot_forecast(df_hist, price_col, forecast, cut_name):
    fig = go.Figure()

    # Confidence band (upper first, then lower with fill='tonexty')
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(width=0),
        hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(0,114,178,0.20)', name='Confidence Range'))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', line=dict(color='#0072B2', width=2), name='Forecast'))

    # Actual observed prices
    fig.add_trace(go.Scatter(
        x=df_hist['report_date'], y=df_hist[price_col],
        mode='markers', marker=dict(color='black', size=4), name='Actual'))

    fig.update_layout(
        title=f"{cut_name} Projection",
        xaxis_title="Date", yaxis_title="Price ($/lb)",
        hovermode="x unified", height=600)
    return fig


# ==========================================
# MAIN APP EXECUTION
# ==========================================
if api_key:
    df_historical, target_price_col = get_beef_data(api_key, target_code)

    if not df_historical.empty:
        fingerprint = f"{len(df_historical)}|{df_historical['report_date'].max()}"
        model, forecast = run_prophet(
            df_historical, target_price_col, forecast_weeks, target_code, fingerprint
        )

        current_price = df_historical[target_price_col].iloc[-1]
        predicted_price = forecast.iloc[-1]['yhat']

        total_storage = storage_cost_per_week * forecast_weeks
        total_carrying_cost = freight_cost + total_storage
        break_even_price = current_price + total_carrying_cost
        margin = predicted_price - break_even_price

        # --- Reliability gate: skill vs. naive carry-forward ---
        bt = backtest_model(model, df_historical, target_price_col,
                            forecast_weeks, target_code, fingerprint)

        SKILL_THRESHOLD = 0.05  # must beat "buy at today's price" by 5% on MAE

        if bt is None:
            data_is_reliable = False
            st.error("🚨 **INSUFFICIENT HISTORY** for a valid backtest at this horizon.")
            st.warning("Shorten the forecast window or pick a cut with a longer USDA series.")
        elif bt['skill'] < SKILL_THRESHOLD:
            data_is_reliable = False
            st.error(f"🚨 **NO FORECAST SKILL** (vs. naive: {bt['skill']:+.1%}, out-of-sample R²: {bt['r2']:.2f})")
            st.warning(
                f"Across {bt['n_folds']} backtests, the model's average error "
                f"(${bt['mae']:.2f}/lb) is not meaningfully better than simply assuming "
                f"today's price holds (${bt['mae_naive']:.2f}/lb). Do not use this for "
                f"purchasing decisions on **{selected_cut_name}**."
            )
        else:
            data_is_reliable = True
            st.success(
                f"✅ **Beats naive baseline by {bt['skill']:.1%}** "
                f"(out-of-sample R²: {bt['r2']:.2f}, avg error ±${bt['mae']:.2f}/lb, "
                f"MAPE {bt['mape']:.1f}%, {bt['n_folds']} folds)"
            )

        if data_is_reliable:
            if margin > (current_price * 0.05):
                signal_color, signal_text = "green", "BUY NOW"
            elif margin < -(current_price * 0.03):
                signal_color, signal_text = "red", "SELL / HOLD OFF"
            else:
                signal_color, signal_text = "orange", "FLAT - WATCH MARKET"
        else:
            signal_color, signal_text = "gray", "DATA UNRELIABLE"

        # --- Top Level Metrics ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Market Price", f"${current_price:.2f}/lb")
        with col2:
            st.metric(f"Predicted Price (in {forecast_weeks} weeks)",
                      f"${predicted_price:.2f}/lb",
                      delta=f"${(predicted_price - current_price):.2f}")
        with col3:
            st.markdown(
                f"<h3 style='text-align: center; color: {signal_color}; "
                f"border: 2px solid {signal_color}; padding: 10px;'>SIGNAL: {signal_text}</h3>",
                unsafe_allow_html=True)

        with st.expander("Show Margin & Cost Breakdown (Buy Today)"):
            st.write(f"**Action:** If you buy today and hold for {forecast_weeks} weeks:")
            st.write(f"+ Freight Cost: **${freight_cost:.3f}/lb**")
            st.write(f"+ Storage Cost (${storage_cost_per_week:.2f}/lb/wk): **${total_storage:.3f}/lb**")
            st.write(f"**Total Break-Even Price:** ${break_even_price:.2f}/lb")
            if margin > 0:
                st.success(f"**Estimated Profit Margin:** +${margin:.2f} per lb")
            else:
                st.error(f"**Estimated Loss:** ${margin:.2f} per lb")

        st.markdown("---")

        # --- TARGET EVENT REVERSE CALCULATOR ---
        st.subheader("🎯 Optimal Execution Calculator")
        st.markdown("Work backward from a target date to find the mathematically cheapest week to procure product, factoring in storage fees.")

        future_dates = forecast[forecast['ds'] > pd.to_datetime('today')]['ds'].dt.strftime('%Y-%m-%d').tolist()

        if future_dates and data_is_reliable:
            target_date_str = st.selectbox("When do you need the product in-house?",
                                           future_dates, index=len(future_dates) - 1)
            target_date = pd.to_datetime(target_date_str)

            actionable_window = forecast[
                (forecast['ds'] > pd.to_datetime('today')) & (forecast['ds'] <= target_date)
            ].copy()

            if not actionable_window.empty:
                actionable_window['weeks_to_hold'] = (target_date - actionable_window['ds']).dt.days / 7
                actionable_window['holding_cost'] = actionable_window['weeks_to_hold'] * storage_cost_per_week
                actionable_window['total_landed_cost'] = (
                    actionable_window['yhat'] + freight_cost + actionable_window['holding_cost']
                )

                optimal_buy = actionable_window.loc[actionable_window['total_landed_cost'].idxmin()]

                opt_date = optimal_buy['ds'].strftime('%B %d, %Y')
                opt_price = optimal_buy['yhat']
                opt_landed = optimal_buy['total_landed_cost']

                weeks_to_hold_from_today = (target_date - pd.to_datetime('today')).days / 7
                buy_today_cost = current_price + freight_cost + (weeks_to_hold_from_today * storage_cost_per_week)
                savings = buy_today_cost - opt_landed

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.info(f"**Optimal Buying Date:** Execute this trade on **{opt_date}**.")
                    st.write(f"Predicted Price: **${opt_price:.2f}/lb** + Storage/Freight = Landed Cost: **${opt_landed:.2f}/lb**")
                with col_b:
                    if savings > 0.02:
                        st.success(f"💡 **WAIT TO BUY.**\nYou will save roughly **${savings:.2f}/lb** by executing on {opt_date} rather than buying today.")
                    else:
                        st.error("🚨 **BUY TODAY.**\nThe market is rising faster than storage costs. Secure the product immediately.")
        elif not data_is_reliable:
            st.warning("Optimal execution calculator disabled — the model does not beat a naive price assumption at this horizon.")

        st.markdown("---")

        # ==========================================
        # INTERACTIVE TABS
        # ==========================================
        tab1, tab2 = st.tabs(["📈 Main Forecast", "⚖️ Market Spread Analyzer"])

        with tab1:
            st.subheader(f"{selected_cut_name} Projection")
            fig = plot_forecast(df_historical, target_price_col, forecast, selected_cut_name)

            if target_promo_price > 0:
                fig.add_hline(y=target_promo_price, line_dash="dash", line_color="green",
                              annotation_text=f" Buyer Target: ${target_promo_price:.2f}",
                              annotation_position="bottom right")

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Price Spread Comparison")
            baseline_name = st.selectbox("Select Secondary Cut to Compare:", list(cut_options.keys()), index=0)
            baseline_code = cut_options[baseline_name]

            if baseline_code == target_code:
                st.warning("Please select two different cuts to compare.")
            else:
                df_baseline, base_price_col = get_beef_data(api_key, baseline_code)

                if not df_baseline.empty:
                    df_target_subset = df_historical[['report_date', target_price_col]].rename(
                        columns={target_price_col: 'Target_Price'})
                    df_base_subset = df_baseline[['report_date', base_price_col]].rename(
                        columns={base_price_col: 'Base_Price'})

                    df_spread = pd.merge(df_target_subset, df_base_subset, on='report_date', how='inner')
                    df_spread['Spread'] = df_spread['Target_Price'] - df_spread['Base_Price']

                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Scatter(x=df_spread['report_date'], y=df_spread['Target_Price'],
                                                    name=selected_cut_name, line=dict(color='blue')))
                    fig_spread.add_trace(go.Scatter(x=df_spread['report_date'], y=df_spread['Base_Price'],
                                                    name=baseline_name, line=dict(color='red')))
                    fig_spread.update_layout(title="Historical Price Comparison",
                                             yaxis_title="Price ($/lb)", hovermode="x unified")
                    st.plotly_chart(fig_spread, use_container_width=True)

                    st.markdown("##### The Dollar Spread Over Time")
                    fig_diff = go.Figure(data=[go.Bar(x=df_spread['report_date'],
                                                      y=df_spread['Spread'], marker_color='green')])
                    fig_diff.update_layout(yaxis_title="Spread ($/lb)", hovermode="x unified")
                    st.plotly_chart(fig_diff, use_container_width=True)

    else:
        st.error(f"Failed to load data for {selected_cut_name}. The USDA may not have published this code recently.")
else:
    st.info("👈 Please enter your USDA API key to load the forecaster.")
