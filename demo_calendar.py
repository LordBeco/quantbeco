#!/usr/bin/env python3
"""
Demo script to showcase the daily calendar chart
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from charts import create_daily_calendar_chart

def create_demo_data():
    """Create realistic demo trading data"""
    
    # Generate 60 days of trading data
    start_date = datetime.now() - timedelta(days=60)
    trades = []
    
    # Create more realistic trading patterns
    for day in range(60):
        current_date = start_date + timedelta(days=day)
        
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() >= 5:
            continue
            
        # Random number of trades per day (0-8 trades)
        num_trades = np.random.poisson(2)  # Average 2 trades per day
        
        for trade in range(num_trades):
            # Add some time variation within the day
            trade_time = current_date + timedelta(
                hours=np.random.randint(8, 18),  # Trading hours 8-18
                minutes=np.random.randint(0, 60)
            )
            
            # More realistic P&L distribution
            # 60% win rate with varying sizes
            if np.random.random() < 0.6:  # Win
                profit = np.random.lognormal(3, 0.8)  # Positive skew
            else:  # Loss
                profit = -np.random.lognormal(2.8, 0.6)  # Smaller losses on average
            
            trades.append({
                'close_time': trade_time,
                'profit': profit,
                'lots': np.random.uniform(0.1, 2.0),
                'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'], p=[0.4, 0.3, 0.2, 0.1])
            })
    
    df = pd.DataFrame(trades)
    df = df.sort_values('close_time').reset_index(drop=True)
    
    return df

def main():
    st.set_page_config(page_title="Calendar Chart Demo", layout="wide")
    
    st.title("📅 Daily Trading Calendar Demo")
    st.markdown("**Interactive calendar showing daily trading performance**")
    
    # Create demo data
    df = create_demo_data()
    
    st.markdown(f"### Demo Data: {len(df)} trades over 60 days")
    
    # Show data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", len(df))
    with col2:
        total_pnl = df['profit'].sum()
        st.metric("Total P&L", f"${total_pnl:,.2f}")
    with col3:
        win_rate = (df['profit'] > 0).mean() * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        avg_daily = df.groupby(df['close_time'].dt.date)['profit'].sum().mean()
        st.metric("Avg Daily P&L", f"${avg_daily:.2f}")
    
    # Create and display calendar chart
    st.markdown("### 📅 Daily Trading Calendar")
    st.markdown("**See at one glance which how many days you are making or losing money. Click a day to look at the trades.**")
    
    # Add month/year navigation for demo
    col1, col2 = st.columns(2)
    with col1:
        demo_year = st.selectbox("Year", [2024, 2025], index=1)
    with col2:
        demo_month = st.selectbox("Month", list(range(1, 13)), 
                                 format_func=lambda x: calendar.month_name[x], 
                                 index=11)  # December
    
    calendar_chart = create_daily_calendar_chart(df, 'profit', f'{calendar.month_name[demo_month]} {demo_year}', demo_year, demo_month)
    st.plotly_chart(calendar_chart, use_container_width=True)
    
    # Additional insights
    st.markdown("### 📊 Calendar Insights")
    
    # Daily statistics
    df_copy = df.copy()
    df_copy['trade_date'] = df_copy['close_time'].dt.date
    daily_stats = df_copy.groupby('trade_date').agg({
        'profit': ['count', 'sum']
    })
    daily_stats.columns = ['trades', 'pnl']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        profitable_days = (daily_stats['pnl'] > 0).sum()
        total_trading_days = len(daily_stats)
        st.metric("Profitable Days", f"{profitable_days}/{total_trading_days}")
        st.metric("Daily Win Rate", f"{profitable_days/total_trading_days*100:.1f}%")
    
    with col2:
        best_day = daily_stats['pnl'].max()
        worst_day = daily_stats['pnl'].min()
        st.metric("Best Day", f"${best_day:.2f}")
        st.metric("Worst Day", f"${worst_day:.2f}")
    
    with col3:
        avg_trades_per_day = daily_stats['trades'].mean()
        max_trades_day = daily_stats['trades'].max()
        st.metric("Avg Trades/Day", f"{avg_trades_per_day:.1f}")
        st.metric("Max Trades/Day", f"{max_trades_day}")
    
    # Show raw data sample
    with st.expander("📊 Sample Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("• **Color Coding**: Green for profitable days, red for loss days, gray for no trading")
    st.markdown("• **Trade Count**: Shows number of trades executed each day")
    st.markdown("• **Daily P&L**: Displays end-of-day profit/loss amount")
    st.markdown("• **Interactive**: Hover over any day to see detailed information")
    st.markdown("• **Time Filtering**: Respects the date range filters in the main application")

if __name__ == "__main__":
    main()