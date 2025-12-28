import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from analytics import compute_metrics, compute_rolling_metrics, compute_top_kpis, compute_time_analysis, compute_risk_pain_metrics, compute_trading_insights, generate_trading_insights_summary
from charts import (equity_curve, drawdown_curve, pnl_distribution, win_loss_pie, 
                   pnl_growth_over_time, rolling_performance_charts, time_analysis_charts, 
                   monthly_heatmap, risk_pain_charts, create_time_tables, create_trading_insights_charts, create_pip_analysis_chart)
from diagnosis import diagnose, comprehensive_ai_diagnosis, generate_executive_summary
from style import inject_css
from tradelocker_api import TradeLockerAPI, test_tradelocker_connection, fetch_tradelocker_data, get_tradelocker_accounts

def get_date_range(period, custom_start=None, custom_end=None):
    """Get start and end dates for different time periods"""
    today = datetime.now().date()
    
    if period == "Custom Range":
        return custom_start, custom_end
    elif period == "Today":
        return today, today
    elif period == "Yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period == "This Week":
        start_week = today - timedelta(days=today.weekday())
        return start_week, today
    elif period == "Last Week":
        start_last_week = today - timedelta(days=today.weekday() + 7)
        end_last_week = today - timedelta(days=today.weekday() + 1)
        return start_last_week, end_last_week
    elif period == "This Month":
        start_month = today.replace(day=1)
        return start_month, today
    elif period == "Last Month":
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end
    elif period == "This Year":
        start_year = today.replace(month=1, day=1)
        return start_year, today
    elif period == "Last Year":
        last_year = today.year - 1
        start_last_year = datetime(last_year, 1, 1).date()
        end_last_year = datetime(last_year, 12, 31).date()
        return start_last_year, end_last_year
    elif period == "All Time":
        return None, None
    
    return None, None

def filter_dataframe_by_date(df, start_date, end_date):
    """Filter dataframe by date range"""
    if start_date is None or end_date is None:
        return df
    
    # Try to find datetime columns
    datetime_cols = []
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'datetime', 'open_time', 'close_time']):
            try:
                df[f'{col}_parsed'] = pd.to_datetime(df[col])
                datetime_cols.append(f'{col}_parsed')
            except:
                continue
    
    if not datetime_cols:
        st.warning("⚠️ No valid datetime columns found for filtering. Showing all data.")
        return df
    
    # Use the first datetime column found (usually close_time or open_time)
    date_col = datetime_cols[0]
    
    # Filter by date range
    mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    return filtered_df

st.set_page_config(layout="wide", page_title=" FundedBeco Trading Performance Intelligence")
st.markdown(inject_css(), unsafe_allow_html=True)
st.title("🎯 FundedBeco Trading Performance Intelligence")
st.markdown("**FundedBeco Strategy Diagnostics Console**")

# === DATA SOURCE SELECTION ===
st.markdown("### 📊 Data Source")
data_source = st.radio(
    "Choose your data source:",
    ["📁 Upload CSV File", "🔗 TradeLocker API"],
    horizontal=True
)

df = None
pnl = None
starting_balance = 10000

if data_source == "📁 Upload CSV File":
    file = st.file_uploader("Upload Broker Statement (CSV)", type="csv")
    
    if file:
        df_raw = pd.read_csv(file)
        df_raw.columns = [c.lower().replace(" ", "_") for c in df_raw.columns]

        # Separate balance entries from trades for proper analysis
        balance_df = pd.DataFrame()
        if 'type' in df_raw.columns:
            balance_df = df_raw[df_raw['type'].str.contains('balance', case=False, na=False)].copy()
            df = df_raw[~df_raw['type'].str.contains('balance', case=False, na=False)].copy()
        elif 'comment' in df_raw.columns:
            balance_df = df_raw[df_raw['comment'].str.contains('Initial|balance', case=False, na=False)].copy()
            df = df_raw[~df_raw['comment'].str.contains('Initial|balance', case=False, na=False)].copy()
        else:
            df = df_raw.copy()

        # Clean numeric columns and calculate REAL PnL (including commissions and swaps)
        numeric_cols = []
        for col in ['profit', 'pnl', 'commission', 'swaps']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)
                numeric_cols.append(col)

        # Calculate real PnL = Profit + Commission + Swaps
        if 'profit' in df.columns:
            pnl = 'real_pnl'
            df['real_pnl'] = df['profit'].copy()
            
            if 'commission' in df.columns:
                df['real_pnl'] += df['commission']
            if 'swaps' in df.columns:
                df['real_pnl'] += df['swaps']
                
            st.info(f"📊 **Real PnL Calculation**: Profit + Commission + Swaps")
            
            # Show breakdown
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Raw Profit", f"${df['profit'].sum():,.2f}")
            with col2:
                st.metric("Total Commission", f"${df['commission'].sum():,.2f}" if 'commission' in df.columns else "$0.00")
            with col3:
                st.metric("Total Swaps", f"${df['swaps'].sum():,.2f}" if 'swaps' in df.columns else "$0.00")
            with col4:
                st.metric("Real PnL", f"${df['real_pnl'].sum():,.2f}")
                
        else:
            # Fallback to existing PnL column
            pnl = next(c for c in df.columns if "pnl" in c or "profit" in c)
            df[pnl] = pd.to_numeric(df[pnl].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

        # Clean balance data for starting balance detection
        if not balance_df.empty:
            for col in ['profit', 'pnl']:
                if col in balance_df.columns:
                    balance_df[col] = pd.to_numeric(balance_df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

        # Calculate starting balance
        if not balance_df.empty:
            balance_amount = balance_df['profit'].iloc[0] if 'profit' in balance_df.columns else balance_df[list(balance_df.columns)[0]].iloc[0]
            if balance_amount > 0:
                starting_balance = balance_amount

elif data_source == "🔗 TradeLocker API":
    st.markdown("### 🔗 TradeLocker API Connection")
    
    with st.expander("📋 TradeLocker API Setup Instructions", expanded=False):
        st.markdown("""
        **How to connect your TradeLocker account:**
        
        1. **Email**: Your TradeLocker login email
        2. **Password**: Your TradeLocker password  
        3. **Server Name**: Your broker's server name (e.g., "GATESFX")
        4. **Account ID**: Your specific account ID (e.g., "1691721")
        5. **Account Number**: Select which account (1st, 2nd, 3rd, etc.) if you have multiple accounts
        6. **Account Type**: Select "Demo" or "Live" based on your account
        7. **History Period**: How many days of trading history to fetch
        
        **Finding Your Details:**
        - **Server Name**: Check your TradeLocker platform or broker documentation
        - **Account ID**: Usually shown in your trading platform (unique identifier like "1691721")
        - **Account Number**: If you have multiple accounts, select 1 for first account, 2 for second, etc.
        - **Account Type**: Demo accounts for testing, Live for real trading
        
        **API Endpoints Used:**
        - **Demo**: `https://demo.tradelocker.com/backend-api/`
        - **Live**: `https://live.tradelocker.com/backend-api/`
        
        **Security Notes:**
        - Your credentials are only used to fetch data and are not stored
        - Uses official TradeLocker backend API with secure authentication
        - All data processing happens locally in your browser
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        tl_email = st.text_input("📧 TradeLocker Email", type="default")
        tl_password = st.text_input("🔒 TradeLocker Password", type="password")
    
    with col2:
        tl_server = st.text_input("🖥️ Server Name", value="GATESFX", help="e.g., GATESFX, LIVE, DEMO")
        tl_account_id = st.text_input("🆔 Account ID", value="1691721", help="Your TradeLocker account ID")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        tl_is_live = st.selectbox("🔴 Account Type", [False, True], format_func=lambda x: "Demo" if not x else "Live")
    with col4:
        tl_acc_num = st.selectbox("🔢 Account Number", [1, 2, 3, 4, 5], index=1, help="Select which account (1st, 2nd, 3rd, etc.) if you have multiple accounts")
    with col5:
        tl_days = st.number_input("📅 Days of History", min_value=1, max_value=365, value=90)
    
    if st.button("🔗 Connect & Fetch Data", type="primary"):
        if tl_email and tl_password and tl_server and tl_account_id:
            with st.spinner("🔄 Connecting to TradeLocker..."):
                try:
                    # Test connection first
                    connection_test = test_tradelocker_connection(tl_email, tl_password, tl_server, tl_account_id, tl_acc_num, tl_is_live)
                    
                    if connection_test['success']:
                        st.success("✅ Connected successfully!")
                        
                        # Show account info
                        account_info = connection_test['account_info']
                        balance_info = connection_test['balance']
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Account ID", account_info['account_id'])
                        with col2:
                            st.metric("Account #", account_info['account_number'])
                        with col3:
                            st.metric("Server", account_info['server'])
                        with col4:
                            st.metric("Type", "Live" if account_info['is_live'] else "Demo")
                        with col5:
                            balance_value = balance_info.get('balance', 0.0)
                            currency = balance_info.get('currency', 'USD')
                            st.metric("Balance", f"${balance_value:,.2f} {currency}")
                        
                        # Fetch trading data
                        with st.spinner("📊 Fetching trading history..."):
                            try:
                                df = fetch_tradelocker_data(tl_email, tl_password, tl_server, tl_account_id, tl_acc_num, tl_is_live, tl_days)
                                
                                if df.empty:
                                    st.warning(f"⚠️ No trading history found for the last {tl_days} days.")
                                    st.info("💡 **Possible reasons:**")
                                    st.write("• No trades executed in the selected time period")
                                    st.write("• Different API endpoint needed for your broker")
                                    st.write("• Historical data may be stored differently")
                                    
                                    # Suggest trying different periods
                                    st.write("**Try:**")
                                    st.write("• Increase the 'Days of History' to 180 or 365")
                                    st.write("• Check if you have trades in your TradeLocker platform")
                                    st.write("• Contact your broker about API access to historical data")
                                else:
                                    st.success(f"✅ Fetched {len(df)} trades from the last {tl_days} days")
                                    
                                    # Set up variables for analysis
                                    pnl = 'real_pnl'
                                    starting_balance = balance_value - df[pnl].sum()  # Calculate starting balance
                                    
                                    # Show data preview
                                    with st.expander("📊 Data Preview", expanded=False):
                                        st.dataframe(df.head(), use_container_width=True)
                                        
                                        # Show column info for debugging
                                        st.write("**Available columns:**", list(df.columns))
                                        if len(df) > 0:
                                            st.write("**Date range in data:**", 
                                                   f"{df['open_time'].min()} to {df['close_time'].max()}" if 'open_time' in df.columns and 'close_time' in df.columns else "Date columns not found")
                            
                            except Exception as fetch_error:
                                st.error(f"❌ Error fetching trading data: {str(fetch_error)}")
                                st.info("💡 **Debug Information:**")
                                st.write("• Connection was successful but data fetching failed")
                                st.write("• This might be due to different API endpoints for historical data")
                                st.write("• Try uploading a CSV export from your TradeLocker platform instead")
                    else:
                        st.error(f"❌ Connection failed: {connection_test['error']}")
                        
                        # Show debug information
                        with st.expander("🔍 Debug Information", expanded=False):
                            st.write("**Error Details:**")
                            st.code(connection_test['error'])
                            
                            st.write("**Troubleshooting Steps:**")
                            st.write("1. **Check Credentials**: Verify email and password in TradeLocker platform")
                            st.write("2. **Verify Account ID**: Check your account ID in the trading platform")
                            st.write("3. **Try Different Account Number**: If you have multiple accounts, try 1, 2, or 3")
                            st.write("4. **Check Server Name**: Ensure server name matches your broker (e.g., GATESFX)")
                            st.write("5. **Account Type**: Make sure Demo/Live selection matches your account")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please fill in all required fields: Email, Password, Server Name, and Account ID")

# Continue with analysis only if we have data
if df is not None and not df.empty:

    # Clean numeric columns and calculate REAL PnL (including commissions and swaps)
    numeric_cols = []
    for col in ['profit', 'pnl', 'commission', 'swaps']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)
            numeric_cols.append(col)

    # Calculate real PnL = Profit + Commission + Swaps
    if 'profit' in df.columns:
        pnl = 'real_pnl'
        df['real_pnl'] = df['profit'].copy()
        
        if 'commission' in df.columns:
            df['real_pnl'] += df['commission']
        if 'swaps' in df.columns:
            df['real_pnl'] += df['swaps']
            
        st.info(f"📊 **Real PnL Calculation**: Profit + Commission + Swaps")
        
        # Show breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Raw Profit", f"${df['profit'].sum():,.2f}")
        with col2:
            st.metric("Total Commission", f"${df['commission'].sum():,.2f}" if 'commission' in df.columns else "$0.00")
        with col3:
            st.metric("Total Swaps", f"${df['swaps'].sum():,.2f}" if 'swaps' in df.columns else "$0.00")
        with col4:
            st.metric("Real PnL", f"${df['real_pnl'].sum():,.2f}")
            
    else:
        # Fallback to existing PnL column
        pnl = next(c for c in df.columns if "pnl" in c or "profit" in c)
        df[pnl] = pd.to_numeric(df[pnl].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

    # Clean balance data for starting balance detection
    if not balance_df.empty:
        for col in ['profit', 'pnl']:
            if col in balance_df.columns:
                balance_df[col] = pd.to_numeric(balance_df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

    # Calculate KPIs - first detect starting balance, then calculate with real PnL
    starting_balance = 10000  # Default
    if not balance_df.empty:
        balance_amount = balance_df['profit'].iloc[0] if 'profit' in balance_df.columns else balance_df[list(balance_df.columns)[0]].iloc[0]
        if balance_amount > 0:
            starting_balance = balance_amount
    
    # === DATE FILTERING SECTION ===
    st.markdown("---")
    st.markdown("### 📅 Date Range Filter")
    
    # Quick period buttons
    st.markdown("**Quick Select:**")
    quick_cols = st.columns(8)
    quick_periods = ["Today", "Yesterday", "This Week", "This Month", "Last Month", "This Year", "Last Year", "All Time"]
    
    selected_quick = None
    for i, period in enumerate(quick_periods):
        if quick_cols[i].button(period, key=f"quick_{period}"):
            selected_quick = period
    
    # Main period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        period_options = [
            "All Time", "Today", "Yesterday", "This Week", "Last Week", 
            "This Month", "Last Month", "This Year", "Last Year", "Custom Range"
        ]
        
        # Use quick selection if clicked, otherwise use selectbox
        if selected_quick:
            selected_period = selected_quick
            # Update session state to reflect the selection
            if 'period_selection' not in st.session_state:
                st.session_state.period_selection = selected_quick
        else:
            default_index = period_options.index(st.session_state.get('period_selection', 'All Time'))
            selected_period = st.selectbox("Select Time Period", period_options, index=default_index)
            st.session_state.period_selection = selected_period
    
    custom_start, custom_end = None, None
    if selected_period == "Custom Range":
        with col2:
            custom_start = st.date_input("Start Date")
        with col3:
            custom_end = st.date_input("End Date")
    
    # Get date range and filter data
    start_date, end_date = get_date_range(selected_period, custom_start, custom_end)
    
    # Show selected period info
    if start_date and end_date:
        if start_date == end_date:
            st.info(f"📊 **Analyzing**: {selected_period} ({start_date})")
        else:
            st.info(f"📊 **Analyzing**: {selected_period} ({start_date} to {end_date})")
        
        # Filter the dataframe
        original_df = df.copy()
        df = filter_dataframe_by_date(df, start_date, end_date)
        
        if len(df) == 0:
            st.error(f"❌ **No trades found** in the selected period ({start_date} to {end_date})")
            st.stop()
        elif len(df) < len(original_df):
            st.success(f"✅ **Filtered**: {len(df)} trades out of {len(original_df)} total trades")
    else:
        st.info(f"📊 **Analyzing**: All Time ({len(df)} trades)")
    
    # Recalculate equity curve for filtered data
    df["equity"] = df[pnl].cumsum()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] - df["peak"]
    df["drawdown_pct"] = df["drawdown"] / df["peak"].replace(0, 1) * 100
    
    # === PERIOD PERFORMANCE SUMMARY ===
    if selected_period == "All Time":
        st.markdown("---")
        st.markdown("### 📊 Performance by Period")
        
        # Calculate performance for different periods
        periods_to_analyze = ["Today", "Yesterday", "This Week", "Last Week", "This Month", "Last Month", "This Year"]
        period_summary = []
        
        original_df_for_summary = df.copy()
        
        for period in periods_to_analyze:
            start_p, end_p = get_date_range(period)
            if start_p and end_p:
                period_df = filter_dataframe_by_date(original_df_for_summary, start_p, end_p)
                if len(period_df) > 0:
                    period_pnl = period_df[pnl].sum()
                    period_trades = len(period_df)
                    win_rate = (period_df[pnl] > 0).sum() / len(period_df) * 100
                    
                    period_summary.append({
                        'Period': period,
                        'Trades': period_trades,
                        'PnL': f"${period_pnl:,.2f}",
                        'Win Rate': f"{win_rate:.1f}%",
                        'Avg/Trade': f"${period_pnl/period_trades:.2f}" if period_trades > 0 else "$0.00"
                    })
        
        if period_summary:
            summary_df = pd.DataFrame(period_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Calculate KPIs for the selected period
    if start_date and end_date and selected_period != "All Time":
        # For filtered periods, calculate the correct starting balance
        from analytics import compute_period_starting_balance
        
        # We need the original unfiltered dataframe to calculate cumulative PnL
        period_starting_balance = compute_period_starting_balance(original_df, df, pnl, starting_balance)
        top_kpis = compute_top_kpis(df, pnl, starting_balance_override=period_starting_balance)
        
        # Show additional context for filtered periods
        if len(df) > 0:
            first_trade_date = df.iloc[0]['close_time_parsed'].date() if 'close_time_parsed' in df.columns else "Unknown"
            last_trade_date = df.iloc[-1]['close_time_parsed'].date() if 'close_time_parsed' in df.columns else "Unknown"
            
            # Calculate how much PnL was accumulated before this period
            trades_before_count = original_df.index.get_loc(df.index[0]) if len(df) > 0 else 0
            pnl_before_period = period_starting_balance - starting_balance
            
            st.info(f"""
            📊 **Period Analysis**: {first_trade_date} to {last_trade_date}  
            **Starting Balance Calculation**: Original ${starting_balance:,.2f} + PnL from {trades_before_count} previous trades (${pnl_before_period:,.2f}) = **${period_starting_balance:,.2f}**
            """)
    else:
        top_kpis = compute_top_kpis(df, pnl, starting_balance_override=starting_balance)

    # Compute all metrics on trade data only
    metrics = compute_metrics(df, pnl)
    df = compute_rolling_metrics(df, pnl)
    time_data = compute_time_analysis(df, pnl)
    risk_data = compute_risk_pain_metrics(df, pnl)
    trading_insights = compute_trading_insights(df, pnl)
    insights_summary = generate_trading_insights_summary(trading_insights)

    # === TOP KPIs SECTION ===
    st.markdown("---")
    st.markdown("### 📊 Account Overview")
    
    # Show period comparison if not "All Time"
    if selected_period != "All Time" and start_date and end_date:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            kpi_cols = st.columns(4)
            for col, (k, v) in zip(kpi_cols, top_kpis.items()):
                col.metric(k, v)
        
        with col2:
            st.markdown("**📈 Period Comparison**")
            
            # Calculate comparison metrics
            period_pnl = df[pnl].sum()
            period_trades = len(df)
            period_days = (end_date - start_date).days + 1 if start_date != end_date else 1
            
            # Show period starting balance calculation for transparency
            if 'original_df' in locals() and len(df) < len(original_df):
                trades_before_period = len(original_df) - len(df) - (len(original_df) - original_df.index[-1] - 1) + (df.index[0] if len(df) > 0 else 0)
                st.write(f"• **Period Starting Balance**: ${period_starting_balance:,.2f}")
                st.write(f"• **Trades Before Period**: {original_df.index.get_loc(df.index[0]) if len(df) > 0 else 0}")
            
            st.write(f"• **Period PnL**: ${period_pnl:,.2f}")
            st.write(f"• **Trades**: {period_trades}")
            st.write(f"• **Days**: {period_days}")
            if period_days > 0:
                st.write(f"• **Avg PnL/Day**: ${period_pnl/period_days:.2f}")
            if period_trades > 0:
                st.write(f"• **Avg PnL/Trade**: ${period_pnl/period_trades:.2f}")
    else:
        kpi_cols = st.columns(4)
        for col, (k, v) in zip(kpi_cols, top_kpis.items()):
            col.metric(k, v)
    
    st.markdown("---")

    # === CORE PERFORMANCE METRICS ===
    st.markdown("### 📈 Core Performance Metrics")
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)

    # === EQUITY & PNL GROWTH ===
    st.markdown("### 💰 Equity & PnL Growth")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(equity_curve(df, selected_period), use_container_width=True)
    with col2:
        st.plotly_chart(pnl_growth_over_time(df, pnl, selected_period), use_container_width=True)

    # === ROLLING PERFORMANCE (EDGE DECAY DETECTION) ===
    st.markdown("### 🔍 Rolling Performance Analysis")
    st.markdown("**Detects edge decay and performance degradation over time**")
    st.plotly_chart(rolling_performance_charts(df), use_container_width=True)
    
    # Edge decay warning
    recent_expectancy = df["rolling_expectancy"].tail(10).mean()
    if recent_expectancy <= 0:
        st.error("⚠️ **EDGE DECAY DETECTED**: Recent rolling expectancy is negative. Consider stopping this strategy.")
    elif recent_expectancy < df["rolling_expectancy"].head(20).mean() * 0.5:
        st.warning("⚠️ **PERFORMANCE DECLINE**: Rolling expectancy has dropped significantly from early performance.")

    st.markdown("---")

    # === TIME ANALYSIS (KILLER SESSIONS) ===
    st.markdown("### ⏰ Time Analysis - Session Killers")
    st.markdown("**Identify when you should NOT trade**")
    
    hourly_chart, daily_chart = time_analysis_charts(time_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hourly_chart, use_container_width=True)
    with col2:
        st.plotly_chart(daily_chart, use_container_width=True)
    
    # Time analysis insights
    time_tables = create_time_tables(time_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚨 Worst Performing Hours (AVOID):**")
        for hour, data in time_tables['worst_hours'].iterrows():
            if data['Total_PnL'] < 0:
                st.write(f"• **{hour}:00** - Loss: ${data['Total_PnL']:.0f} ({data['Trade_Count']:.0f} trades)")
    
    with col2:
        st.markdown("**✅ Best Performing Hours:**")
        for hour, data in time_tables['best_hours'].iterrows():
            if data['Total_PnL'] > 0:
                st.write(f"• **{hour}:00** - Profit: ${data['Total_PnL']:.0f} ({data['Trade_Count']:.0f} trades)")

    st.markdown("---")

    # === MONTHLY SEASONALITY ===
    st.markdown("### 📅 Monthly Seasonality")
    st.plotly_chart(monthly_heatmap(time_data), use_container_width=True)
    
    # Monthly insights
    monthly_stats = time_data['monthly']
    best_month = monthly_stats.loc[monthly_stats['Total_PnL'].idxmax()]
    worst_month = monthly_stats.loc[monthly_stats['Total_PnL'].idxmin()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🏆 Best Month:** {best_month.name[1]} (${best_month['Total_PnL']:.0f})")
        st.markdown(f"**📉 Worst Month:** {worst_month.name[1]} (${worst_month['Total_PnL']:.0f})")
    
    with col2:
        profitable_months = (monthly_stats['Total_PnL'] > 0).sum()
        total_months = len(monthly_stats)
        st.markdown(f"**📊 Monthly Win Rate:** {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

    st.markdown("---")

    # === RISK & PAIN METRICS ===
    st.markdown("### 🩹 Risk & Pain Analysis")
    st.markdown("**Psychological and risk tolerance metrics**")
    
    # Risk metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Consecutive Losses", f"{risk_data['max_consecutive_losses']}")
    with col2:
        st.metric("Avg Drawdown Duration", f"{risk_data['avg_drawdown_duration']:.1f} trades")
    with col3:
        st.metric("Recovery Factor", f"{risk_data['recovery_factor']:.2f}")
    with col4:
        st.metric("Ulcer Index (Pain)", f"{risk_data['ulcer_index']:.2f}%")
    
    # Risk charts
    dd_chart, recovery_chart, consec_chart = risk_pain_charts(risk_data, df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(dd_chart, use_container_width=True)
    with col2:
        st.plotly_chart(recovery_chart, use_container_width=True)
    with col3:
        st.plotly_chart(consec_chart, use_container_width=True)

    # Pain tolerance warnings
    if risk_data['max_consecutive_losses'] > 5:
        st.error(f"🚨 **HIGH PAIN RISK**: Max consecutive losses ({risk_data['max_consecutive_losses']}) may exceed psychological tolerance")
    
    if risk_data['avg_drawdown_duration'] > 10:
        st.warning(f"⚠️ **LONG RECOVERY PERIODS**: Average drawdown lasts {risk_data['avg_drawdown_duration']:.1f} trades")

    st.markdown("---")

    # === TRADING INSIGHTS ANALYSIS ===
    st.markdown("### 🎯 Advanced Trading Insights")
    st.markdown("**Deep dive into lot sizing, risk-reward, pip analysis, and directional bias**")
    
    # Display insights summary
    if insights_summary:
        st.markdown("#### 📊 Key Insights Summary")
        for insight in insights_summary:
            if "✅" in insight:
                st.success(insight)
            elif "🚨" in insight:
                st.error(insight)
            elif "⚠️" in insight:
                st.warning(insight)
            else:
                st.info(insight)
    
    # Create and display trading insights charts
    insights_charts = create_trading_insights_charts(trading_insights)
    
    # Display charts in organized sections
    if 'lot_analysis' in insights_charts:
        st.markdown("#### 📏 Lot Size Analysis")
        st.plotly_chart(insights_charts['lot_analysis'], use_container_width=True)
        
        # Display lot size statistics
        if trading_insights['lot_analysis']['has_lot_data']:
            lot_stats = trading_insights['lot_analysis']['stats']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Lot Size", f"{lot_stats['avg_lot_size']:.2f}")
            with col2:
                st.metric("Lot Consistency", f"{lot_stats['lot_consistency']:.1%}")
            with col3:
                st.metric("Min/Max Lots", f"{lot_stats['min_lot_size']:.2f} / {lot_stats['max_lot_size']:.2f}")
            with col4:
                st.metric("Lot Std Dev", f"{lot_stats['lot_std']:.3f}")
    
    if 'direction_analysis' in insights_charts:
        st.markdown("#### 📈📉 Buy vs Sell Performance")
        st.plotly_chart(insights_charts['direction_analysis'], use_container_width=True)
        
        # Display direction comparison table
        if trading_insights['direction_analysis']['has_direction_data']:
            direction_perf = trading_insights['direction_analysis']['performance']
            st.markdown("**Performance Comparison:**")
            
            # Format the dataframe for display
            display_direction = direction_perf.copy()
            display_direction['Total_PnL'] = display_direction['Total_PnL'].apply(lambda x: f"${x:,.2f}")
            display_direction['Avg_PnL'] = display_direction['Avg_PnL'].apply(lambda x: f"${x:.2f}")
            display_direction['Win_Rate'] = display_direction['Win_Rate'].apply(lambda x: f"{x:.1f}%")
            display_direction['Profit_Factor'] = display_direction['Profit_Factor'].apply(lambda x: f"{x:.2f}")
            display_direction['Consistency'] = display_direction['Consistency'].apply(lambda x: f"{x:.1%}")
            
            # Rename columns for display
            display_direction.columns = ['Total PnL', 'Avg PnL', 'Trades', 'PnL Std', 'Win Rate', 'Profit Factor', 'Consistency']
            display_direction = display_direction[['Total PnL', 'Avg PnL', 'Trades', 'Win Rate', 'Profit Factor', 'Consistency']]
            
            st.dataframe(display_direction, use_container_width=True)
    
    if 'rr_analysis' in insights_charts:
        st.markdown("#### ⚖️ Risk-Reward Analysis")
        st.plotly_chart(insights_charts['rr_analysis'], use_container_width=True)
        
        if trading_insights['rr_analysis']['has_rr_data']:
            rr_stats = trading_insights['rr_analysis']['stats']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg R:R Ratio", f"{rr_stats['avg_rr_ratio']:.2f}:1")
            with col2:
                st.metric("Best R:R", f"{rr_stats['best_rr']:.2f}")
            with col3:
                st.metric("RR Consistency", f"{rr_stats['rr_consistency']:.1%}")
    
    # Pip Analysis
    pip_chart = create_pip_analysis_chart(trading_insights)
    if pip_chart:
        st.markdown("#### 📊 Pip Analysis")
        
        # Show calculation method info
        if trading_insights['pip_analysis'].get('estimated', False):
            calculation_method = trading_insights['pip_analysis'].get('calculation_method', 'estimated')
            if calculation_method == 'symbol_aware':
                st.info("💡 **Pip calculation**: Symbol-aware calculation (NAS100=points, XAUUSD=0.1 moves, JPY=0.01 moves, Forex=0.0001 moves)")
            else:
                st.info("💡 **Pip calculation**: Estimated from price data (assuming standard forex pairs)")
        
        st.plotly_chart(pip_chart, use_container_width=True)
        
        pip_stats = trading_insights['pip_analysis']['stats']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Pips/Trade", f"{pip_stats['avg_pips_per_trade']:.1f}")
        with col2:
            st.metric("Total Pips", f"{pip_stats['total_pips']:.0f}")
        with col3:
            st.metric("Best Trade", f"{pip_stats['best_pip_trade']:.1f} pips")
        with col4:
            st.metric("Worst Trade", f"{pip_stats['worst_pip_trade']:.1f} pips")
    
    if 'symbol_analysis' in insights_charts:
        st.markdown("#### 🎯 Symbol Performance Analysis")
        st.plotly_chart(insights_charts['symbol_analysis'], use_container_width=True)
        
        # Display top/worst performing symbols
        if trading_insights['symbol_analysis']['has_symbol_data']:
            symbol_perf = trading_insights['symbol_analysis']['performance']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏆 Top 3 Performing Symbols:**")
                top_symbols = symbol_perf.head(3)
                for symbol, data in top_symbols.iterrows():
                    st.write(f"• **{symbol}**: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
            
            with col2:
                st.markdown("**📉 Bottom 3 Performing Symbols:**")
                bottom_symbols = symbol_perf.tail(3)
                for symbol, data in bottom_symbols.iterrows():
                    st.write(f"• **{symbol}**: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
    
    if 'position_sizing' in insights_charts:
        st.markdown("#### 📏 Position Sizing Analysis")
        st.plotly_chart(insights_charts['position_sizing'], use_container_width=True)

    st.markdown("---")

    # === DRAWDOWN & DISTRIBUTION ===
    st.markdown("### 📉 Risk Analysis")
    st.plotly_chart(drawdown_curve(df, selected_period), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(pnl_distribution(df, pnl, selected_period), use_container_width=True)
    c2.plotly_chart(win_loss_pie(df, pnl, selected_period), use_container_width=True)

    # === COMPREHENSIVE AI DIAGNOSIS ===
    st.markdown("---")
    ai_diagnosis = comprehensive_ai_diagnosis(df, metrics, risk_data, time_data, pnl, selected_period)
    
    # Executive Summary
    executive_summary = generate_executive_summary(ai_diagnosis, metrics, selected_period)
    st.markdown(executive_summary)
    
    # Detailed Analysis Sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Critical Issues
        if ai_diagnosis['critical_issues']:
            st.markdown("### 🚨 Critical Issues")
            for issue in ai_diagnosis['critical_issues']:
                st.error(issue)
        
        # Warnings
        if ai_diagnosis['warnings']:
            st.markdown("### ⚠️ Warnings")
            for warning in ai_diagnosis['warnings']:
                st.warning(warning)
        
        # Strengths
        if ai_diagnosis['strengths']:
            st.markdown("### ✅ Strengths")
            for strength in ai_diagnosis['strengths']:
                st.success(strength)
    
    with col2:
        # Performance Insights
        if ai_diagnosis['performance_insights']:
            st.markdown("### 📊 Performance Insights")
            for insight in ai_diagnosis['performance_insights']:
                st.info(insight)
        
        # Psychological Analysis
        if ai_diagnosis['psychological_analysis']:
            st.markdown("### 🧠 Psychological Analysis")
            for analysis in ai_diagnosis['psychological_analysis']:
                st.info(analysis)
        
        # Risk Assessment
        if ai_diagnosis['risk_assessment']:
            st.markdown("### 🛡️ Risk Assessment")
            for risk in ai_diagnosis['risk_assessment']:
                if "HIGH RISK" in risk:
                    st.error(risk)
                elif "MODERATE RISK" in risk:
                    st.warning(risk)
                else:
                    st.success(risk)
    
    # === SYMBOL/ASSET ANALYSIS SECTION ===
    if 'symbol_analysis' in ai_diagnosis and 'symbol_performance' in ai_diagnosis['symbol_analysis']:
        symbol_perf = ai_diagnosis['symbol_analysis']['symbol_performance']
        
        if not symbol_perf.empty:
            st.markdown("---")
            st.markdown("### 📈 Symbol/Asset Performance Analysis")
            
            # Symbol performance table
            st.markdown("**Performance by Symbol:**")
            
            # Format the dataframe for display
            display_df = symbol_perf.copy()
            display_df['Total_PnL'] = display_df['Total_PnL'].apply(lambda x: f"${x:,.2f}")
            display_df['Avg_PnL'] = display_df['Avg_PnL'].apply(lambda x: f"${x:.2f}")
            display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1f}%")
            display_df['Max_DD_Pct'] = display_df['Max_DD_Pct'].apply(lambda x: f"{x:.1f}%")
            
            # Rename columns for display
            display_df.columns = ['Trades', 'Total PnL', 'Avg PnL/Trade', 'PnL Volatility', 'Win Rate', 'Profit Factor', 'Max Drawdown', 'Expectancy']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Symbol insights in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Symbol Insights:**")
                symbol_insights = ai_diagnosis['symbol_analysis']['symbol_insights']
                for insight in symbol_insights[:len(symbol_insights)//2 + 1]:
                    st.write(f"• {insight}")
            
            with col2:
                st.markdown("**💡 Symbol Recommendations:**")
                symbol_recs = ai_diagnosis['symbol_analysis']['symbol_recommendations']
                for rec in symbol_recs:
                    if "FOCUS ON" in rec or "AVOID" in rec:
                        st.write(f"• {rec}")
                    else:
                        st.write(f"• {rec}")
            
            # Asset class breakdown if available
            if ai_diagnosis['symbol_analysis']['asset_class_insights']:
                st.markdown("**📊 Asset Class Breakdown:**")
                for insight in ai_diagnosis['symbol_analysis']['asset_class_insights']:
                    st.info(insight)
    
    # Recommendations Section
    if ai_diagnosis['recommendations']:
        st.markdown("### 💡 AI Recommendations")
        rec_cols = st.columns(2)
        for i, rec in enumerate(ai_diagnosis['recommendations']):
            rec_cols[i % 2].info(rec)
    
    # Action Items
    if ai_diagnosis['action_items']:
        st.markdown("### 🎯 Action Items")
        for i, action in enumerate(ai_diagnosis['action_items'], 1):
            st.markdown(f"**{i}.** {action}")

    # === ORIGINAL DIAGNOSIS (Legacy) ===
    st.markdown("### 🩺 Quick Diagnosis")
    diagnosis_result = diagnose(metrics)
    if "stable" in diagnosis_result.lower():
        st.success(f"✅ {diagnosis_result}")
    else:
        st.error(f"🚨 {diagnosis_result}")
        
        
    # === ADDITIONAL INSIGHTS ===
    st.markdown("### 🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rolling Performance Summary:**")
        avg_rolling_exp = df["rolling_expectancy"].mean()
        avg_rolling_wr = df["rolling_win_rate"].mean()
        
        st.write(f"• Average Rolling Expectancy: **${avg_rolling_exp:.2f}**")
        st.write(f"• Average Rolling Win Rate: **{avg_rolling_wr:.1f}%**")
        
        # Consistency check
        expectancy_std = df["rolling_expectancy"].std()
        if expectancy_std > abs(avg_rolling_exp):
            st.write("• ⚠️ **High expectancy volatility** - inconsistent edge")
        else:
            st.write("• ✅ **Consistent expectancy** - stable edge")
    
    with col2:
        st.markdown("**Performance Trends:**")
        
        # Trend analysis
        early_performance = df["rolling_expectancy"].head(20).mean()
        recent_performance = df["rolling_expectancy"].tail(20).mean()
        
        if recent_performance > early_performance * 1.1:
            st.write("• 📈 **Improving performance** over time")
        elif recent_performance < early_performance * 0.9:
            st.write("• 📉 **Declining performance** - review strategy")
        else:
            st.write("• ➡️ **Stable performance** maintained")
            
        # Win rate stability
        wr_early = df["rolling_win_rate"].head(20).mean()
        wr_recent = df["rolling_win_rate"].tail(20).mean()
        
        if abs(wr_recent - wr_early) > 10:
            st.write("• ⚠️ **Win rate changed significantly**")
        else:
            st.write("• ✅ **Win rate remains stable**")