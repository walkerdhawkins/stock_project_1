# app.py
# -------------------------------------------------------
# Multi-Stock Analysis Dashboard
# Built with: Streamlit, YFinance, Scipy, Plotly
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, probplot, jarque_bera
from datetime import date, timedelta
import math

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Multi-Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker_input = st.sidebar.text_input("Enter 2-5 Tickers (separated by spaces)", value="AAPL MSFT").upper().strip()
tickers = ticker_input.split()

if ticker_input:
    if len(tickers) < 2:
        st.sidebar.error("Error: Please enter at least 2 tickers.")
        st.stop()
    elif len(tickers) > 5:
        st.sidebar.error("Error: Please enter no more than 5 tickers.")
        st.stop()

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1976,1,1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1976,1,1))

if start_date and end_date:
    date_range = end_date - start_date
    if date_range.days < 365:
        st.sidebar.error("Error: The date range must be at least 1 year (365 days).")
        st.stop()

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

vol_window = st.sidebar.slider("Rolling Volatility Window (Days)", min_value=10, max_value=120, value=30, step=5)

# -- Data download with caching ---------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker_list, start, end):
    benchmark = ["^GSPC"]
    all_tickers = ticker_list + benchmark
    data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
    
    if data.empty:
        return None, ticker_list
    
    downloaded_tickers = data.columns.get_level_values(1).unique()
    failed_tickers = [t for t in ticker_list if t not in downloaded_tickers]
    
    return data, failed_tickers

# -- Expensive calculations with caching ------------------
@st.cache_data
def calculate_metrics(returns_df, ticker_list, benchmark_ticker):
    returns_df['Equal-Weight Portfolio'] = returns_df[ticker_list].mean(axis=1)
    all_cols = ticker_list + [benchmark_ticker, 'Equal-Weight Portfolio']
    stats = []
    for t in all_cols:
        stats.append({
            "Ticker": t,
            "Ann. Mean Return": f"{(returns_df[t].mean() * 252):.2%}",
            "Ann. Volatility": f"{(returns_df[t].std() * math.sqrt(252)):.2%}",
            "Skewness": f"{returns_df[t].skew():.4f}",
            "Kurtosis": f"{returns_df[t].kurtosis():.4f}",
            "Min Daily Return": f"{returns_df[t].min():.2%}",
            "Max Daily Return": f"{returns_df[t].max():.2%}"
        })
    return pd.DataFrame(stats).set_index("Ticker"), returns_df

# -- Main logic -------------------------------------------
if tickers:
    df_raw, failed = load_data(tickers, start_date, end_date)
    
    if failed:
        for f_ticker in failed:
            st.error(f"Error: Failed to download data for **{f_ticker}**.")
        st.stop()
    
    if df_raw is not None:
        close_data = df_raw["Close"]
        benchmark_sym = "^GSPC"
        raw_returns = close_data.pct_change().dropna()
        
        stats_df, daily_returns = calculate_metrics(raw_returns, tickers, benchmark_sym)

        # 1. STATISTICAL PERFORMANCE SUMMARY
        st.subheader("Statistical Performance Summary")
        st.table(stats_df)

        st.divider()

        # 2. HYPOTHETICAL $10,000 INVESTMENT GROWTH
        st.subheader("Growth of Hypothetical $10,000 Investment")
        investment_start = 10000
        plot_cols = tickers + [benchmark_sym, 'Equal-Weight Portfolio']
        cum_growth = (1 + daily_returns[plot_cols]).cumprod() * investment_start
        start_row = pd.DataFrame(investment_start, index=[close_data.index[0]], columns=plot_cols)
        cum_growth = pd.concat([start_row, cum_growth])

        fig_growth = go.Figure()
        for col in plot_cols:
            l_width = 4 if col == 'Equal-Weight Portfolio' else 1.5
            l_dash = 'dash' if col == benchmark_sym else 'solid'
            l_color = 'black' if col == 'Equal-Weight Portfolio' else None
            fig_growth.add_trace(go.Scatter(x=cum_growth.index, y=cum_growth[col], name=col,
                                            line=dict(width=l_width, dash=l_dash, color=l_color)))
        fig_growth.update_layout(template="plotly_white", height=450, yaxis=dict(tickprefix="$", tickformat=","), hovermode="x unified")
        st.plotly_chart(fig_growth, use_container_width=True)

        st.divider()

        # 3. VISUAL ANALYSIS (Normalized Prices & Volatility)
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Normalized Prices (Base 100)")
            norm_df = (close_data / close_data.iloc[0]) * 100
            fig_price = go.Figure()
            for t in tickers:
                fig_price.add_trace(go.Scatter(x=norm_df.index, y=norm_df[t], name=t))
            fig_price.add_trace(go.Scatter(x=norm_df.index, y=norm_df[benchmark_sym], name="S&P 500", line=dict(dash='dash', color='gray')))
            fig_price.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_price, use_container_width=True)
        with col_right:
            st.subheader(f"Rolling {vol_window}-Day Volatility")
            rolling_vol = daily_returns.rolling(window=vol_window).std() * math.sqrt(252)
            fig_vol = go.Figure()
            for t in tickers:
                fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t))
            fig_vol.update_layout(template="plotly_white", height=400, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_vol, use_container_width=True)
        
        st.divider()

        # =======================================================
        # NEW: SIDE-BY-SIDE BOX PLOT
        # =======================================================
        st.subheader("Daily Return Distribution Comparison (Box Plot)")
        fig_box = go.Figure()
        for t in (tickers + [benchmark_sym, 'Equal-Weight Portfolio']):
            fig_box.add_trace(go.Box(
                y=daily_returns[t],
                name=t,
                boxpoints='outliers', # Show only outliers
                notched=True # Notches represent confidence interval of the median
            ))
        fig_box.update_layout(
            template="plotly_white",
            height=500,
            yaxis_title="Daily Return (%)",
            yaxis=dict(tickformat=".2%")
        )
        st.plotly_chart(fig_box, use_container_width=True)
        # =======================================================

        st.divider()

        # 4. DISTRIBUTION ANALYSIS (Deep Dive)
        st.subheader("Detailed Normality Analysis")
        selected_stock = st.selectbox("Select Ticker for Distribution Analysis", options=tickers + [benchmark_sym, 'Equal-Weight Portfolio'])
        s_rets = daily_returns[selected_stock]

        jb_stat, jb_p = jarque_bera(s_rets)
        jb_col1, jb_col2, jb_col3 = st.columns(3)
        jb_col1.metric("Jarque-Bera Stat", f"{jb_stat:.2f}")
        jb_col2.metric("p-value", f"{jb_p:.4f}")
        
        if jb_p < 0.05:
            jb_col3.error("Rejects normality (p < 0.05)")
        else:
            jb_col3.success("Fails to reject normality (p >= 0.05)")

        tab1, tab2 = st.tabs(["Histogram & Fit", "Q-Q Plot"])
        with tab1:
            mu, std = norm.fit(s_rets)
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=s_rets, histnorm='probability density', name='Actual', nbinsx=60))
            xr = np.linspace(s_rets.min(), s_rets.max(), 100)
            fig_h.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu, std), name='Normal Fit', line=dict(color='red', width=3)))
            fig_h.update_layout(template="plotly_white", title=f"Distribution for {selected_stock}")
            st.plotly_chart(fig_h, use_container_width=True)
        with tab2:
            (osm, osr), (slope, intercept, r) = probplot(s_rets, dist="norm")
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Quantiles', marker=dict(color='blue', size=4)))
            lx = np.array([osm.min(), osm.max()])
            fig_q.add_trace(go.Scatter(x=lx, y=intercept + slope*lx, mode='lines', name='Ref Line', line=dict(color='red')))
            fig_q.update_layout(template="plotly_white", title=f"Q-Q Plot for {selected_stock}")
            st.plotly_chart(fig_q, use_container_width=True)
            st.write(f"**R-squared value:** {r**2:.4f}")

else:
    st.info("Please enter stock tickers in the sidebar to begin.")