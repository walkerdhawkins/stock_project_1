# app.py
# -------------------------------------------------------
# Multi-Stock Analysis Dashboard
# Built with: Streamlit, YFinance, Scipy, Plotly
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import norm, probplot, jarque_bera
from datetime import date, timedelta
import math

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Multi-Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

# --- NEW: About / Methodology Section ---
with st.sidebar.expander("About & Methodology"):
    st.write("""
    **Overview**
    This dashboard provides interactive performance analytics for a custom selection of stocks. 
    It compares individual assets against an Equal-Weight Portfolio and the S&P 500 (^GSPC).
    
    **Key Assumptions**
    * **Annualization:** All annualized metrics (Mean Return and Volatility) assume **252 trading days** in a year.
    * **Returns:** Calculations are based on **simple arithmetic daily returns** (percentage change in adjusted closing price).
    * **Hypothetical Growth:** The growth chart assumes a starting investment of $10,000 with daily rebalancing for the Equal-Weight Portfolio.
    
    **Data Source**
    Market data is fetched in real-time via the [Yahoo Finance API](https://finance.yahoo.com/) (using the `yfinance` library).
    """)

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
corr_window = st.sidebar.slider("Rolling Correlation Window (Days)", min_value=20, max_value=150, value=60, step=5)

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
            "Ann. Mean Return": returns_df[t].mean() * 252,
            "Ann. Volatility": returns_df[t].std() * math.sqrt(252),
            "Skewness": returns_df[t].skew(),
            "Kurtosis": returns_df[t].kurtosis(),
            "Min Daily Return": returns_df[t].min(),
            "Max Daily Return": returns_df[t].max()
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
        
        stats_raw, daily_returns = calculate_metrics(raw_returns, tickers, benchmark_sym)

        # 1. STATISTICAL PERFORMANCE SUMMARY
        st.subheader("Statistical Performance Summary")
        disp_stats = stats_raw.copy()
        for col in ["Ann. Mean Return", "Ann. Volatility", "Min Daily Return", "Max Daily Return"]:
            disp_stats[col] = disp_stats[col].apply(lambda x: f"{x:.2%}")
        st.table(disp_stats)

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
        
        fig_growth.update_layout(
            template="plotly_white", 
            height=450, 
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            yaxis=dict(tickprefix="$", tickformat=","), 
            hovermode="x unified"
        )
        st.plotly_chart(fig_growth, width= "content")

        st.divider()

        # 3. VISUAL ANALYSIS & COMPARISONS
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Normalized Prices (Base 100)")
            norm_df = (close_data / close_data.iloc[0]) * 100
            fig_price = go.Figure()
            for t in tickers:
                fig_price.add_trace(go.Scatter(x=norm_df.index, y=norm_df[t], name=t))
            fig_price.add_trace(go.Scatter(x=norm_df.index, y=norm_df[benchmark_sym], name="S&P 500", line=dict(dash='dash', color='gray')))
            
            fig_price.update_layout(
                template="plotly_white", 
                height=400,
                xaxis_title="Date",
                yaxis_title="Price (Indexed to 100)"
            )
            st.plotly_chart(fig_price, width="content")
            
        with col_right:
            st.subheader(f"Rolling {vol_window}-Day Volatility")
            rolling_vol = daily_returns.rolling(window=vol_window).std() * math.sqrt(252)
            fig_vol = go.Figure()
            for t in tickers:
                fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t))
            
            fig_vol.update_layout(
                template="plotly_white", 
                height=400, 
                xaxis_title="Date",
                yaxis_title="Ann. Volatility (%)",
                yaxis=dict(tickformat=".0%")
            )
            st.plotly_chart(fig_vol, width="content")

        # =======================================================
        # TWO-ASSET PORTFOLIO EXPLORER
        # =======================================================
        st.divider()
        st.subheader("Two-Asset Portfolio Explorer")
        
        pexp_col1, pexp_col2 = st.columns([1, 2])
        
        with pexp_col1:
            st.write("Construct a custom 2-asset portfolio:")
            stock_a = st.selectbox("Select Asset A", options=tickers, index=0)
            stock_b = st.selectbox("Select Asset B", options=tickers, index=min(1, len(tickers)-1))
            
            weight_a = st.slider(f"Weight on {stock_a} (%)", 0, 100, 50) / 100
            weight_b = 1.0 - weight_a
            st.write(f"Weight on {stock_b}: {weight_b:.0%}")
            
            mu_a = stats_raw.loc[stock_a, "Ann. Mean Return"]
            mu_b = stats_raw.loc[stock_b, "Ann. Mean Return"]
            sigma_a = stats_raw.loc[stock_a, "Ann. Volatility"]
            sigma_b = stats_raw.loc[stock_b, "Ann. Volatility"]
            rho = daily_returns[stock_a].corr(daily_returns[stock_b])
            
            p_ret = (weight_a * mu_a) + (weight_b * mu_b)
            p_vol = math.sqrt(
                (weight_a**2 * sigma_a**2) + 
                (weight_b**2 * sigma_b**2) + 
                (2 * weight_a * weight_b * sigma_a * sigma_b * rho)
            )
            
            st.metric("Portfolio Ann. Return", f"{p_ret:.2%}")
            st.metric("Portfolio Ann. Volatility", f"{p_vol:.2%}")

        with pexp_col2:
            w_range = np.linspace(0, 1, 101)
            curve_vols = []
            for w in w_range:
                v = math.sqrt(
                    (w**2 * sigma_a**2) + 
                    ((1-w)**2 * sigma_b**2) + 
                    (2 * w * (1-w) * sigma_a * sigma_b * rho)
                )
                curve_vols.append(v)
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=w_range, y=curve_vols, mode='lines', name='Volatility Curve', line=dict(color='blue')))
            fig_curve.add_trace(go.Scatter(x=[weight_a], y=[p_vol], mode='markers', name='Current Allocation', marker=dict(size=12, color='red')))
            
            fig_curve.update_layout(
                title=f"Diversification Effect: {stock_a} & {stock_b}",
                xaxis_title=f"Weight on {stock_a} (0.0 to 1.0)",
                yaxis_title="Annualized Volatility (%)",
                xaxis=dict(tickformat=".0%"),
                yaxis=dict(tickformat=".2%"),
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_curve, width="content")
            st.caption(f"**Insight:** When correlation ({rho:.2f}) < 1.0, diversification can reduce total risk.")

        st.divider()

        # BOX PLOT & CORRELATION HEATMAP
        col_box, col_corr = st.columns(2)
        with col_box:
            st.subheader("Return Distributions (Box Plot)")
            fig_box = go.Figure()
            for t in (tickers + [benchmark_sym, 'Equal-Weight Portfolio']):
                fig_box.add_trace(go.Box(y=daily_returns[t], name=t, boxpoints='outliers'))
            
            fig_box.update_layout(
                template="plotly_white", 
                height=450, 
                xaxis_title="Ticker / Asset",
                yaxis_title="Daily Return (%)",
                yaxis=dict(tickformat=".2%")
            )
            st.plotly_chart(fig_box, width="content")

        with col_corr:
            st.subheader("Correlation Matrix")
            corr_matrix = daily_returns[tickers + [benchmark_sym]].corr()
            fig_heat = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', color_continuous_midpoint=0)
            
            fig_heat.update_layout(
                height=450,
                xaxis_title="Ticker",
                yaxis_title="Ticker"
            )
            st.plotly_chart(fig_heat, width="content")

        # PAIRWISE RELATIONSHIP ANALYSIS
        st.subheader("Pairwise Relationship Analysis")
        all_options = tickers + [benchmark_sym, 'Equal-Weight Portfolio']
        pair_col1, pair_col2 = st.columns([1, 3])
        with pair_col1:
            stock_x = st.selectbox("Stock X", options=all_options, key="sx")
            stock_y = st.selectbox("Stock Y", options=all_options, key="sy", index=1)
            current_corr = daily_returns[stock_x].corr(daily_returns[stock_y])
            st.metric(f"Overall Correlation", f"{current_corr:.4f}")

        with pair_col2:
            tab_scatter, tab_roll_corr = st.tabs(["Scatter Plot", "Rolling Correlation"])
            with tab_scatter:
                fig_scatter = px.scatter(daily_returns, x=stock_x, y=stock_y, trendline="ols", trendline_color_override="red", template="plotly_white")
                fig_scatter.update_layout(
                    xaxis_title=f"{stock_x} Daily Returns",
                    yaxis_title=f"{stock_y} Daily Returns"
                )
                st.plotly_chart(fig_scatter, width="content")
            with tab_roll_corr:
                rolling_corr_series = daily_returns[stock_x].rolling(window=corr_window).corr(daily_returns[stock_y]).dropna()
                fig_rolling_corr = go.Figure()
                fig_rolling_corr.add_trace(go.Scatter(x=rolling_corr_series.index, y=rolling_corr_series, mode='lines'))
                fig_rolling_corr.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Correlation Coefficient",
                    yaxis=dict(range=[-1.1, 1.1]), 
                    template="plotly_white"
                )
                st.plotly_chart(fig_rolling_corr, width="content")

        st.divider()

        # 4. DETAILED NORMALITY ANALYSIS
        st.subheader("Detailed Normality Analysis")
        selected_stock = st.selectbox("Select Ticker for Analysis", options=tickers + [benchmark_sym, 'Equal-Weight Portfolio'], key="norm_sel")
        s_rets = daily_returns[selected_stock]
        jb_stat, jb_p = jarque_bera(s_rets)
        jb_col1, jb_col2, jb_col3 = st.columns(3)
        jb_col1.metric("Jarque-Bera Stat", f"{jb_stat:.2f}")
        jb_col2.metric("p-value", f"{jb_p:.4f}")
        if jb_p < 0.05:
            jb_col3.error("Rejects normality (p < 0.05)")
        else:
            jb_col3.success("Fails to reject normality (p >= 0.05)")

        tab_hist, tab_qq = st.tabs(["Histogram & Fit", "Q-Q Plot"])
        with tab_hist:
            mu, std = norm.fit(s_rets)
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=s_rets, histnorm='probability density', nbinsx=60))
            xr = np.linspace(s_rets.min(), s_rets.max(), 100)
            fig_h.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu, std), line=dict(color='red', width=3)))
            fig_h.update_layout(
                template="plotly_white",
                xaxis_title="Daily Return",
                yaxis_title="Probability Density"
            )
            st.plotly_chart(fig_h, width="content")
        with tab_qq:
            (osm, osr), (slope, intercept, r) = probplot(s_rets, dist="norm")
            # Ensure osm is a numpy array to avoid the ValueError
            osm = np.array(osm) 
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(size=4)))
            lx = np.array([osm.min(), osm.max()])
            fig_q.add_trace(go.Scatter(x=lx, y=intercept + slope*lx, mode='lines', line=dict(color='red')))
            fig_q.update_layout(
                template="plotly_white",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Ordered Values (Actual Returns)"
            )
            st.plotly_chart(fig_q, width="content")
            st.write(f"**R-squared value:** {r**2:.4f}")

else:
    st.info("Please enter stock tickers in the sidebar to begin.")