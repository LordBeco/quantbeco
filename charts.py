import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

def equity_curve(df, period_name="All Time"):
    fig = px.line(df, y="equity", title=f"Equity Curve - {period_name}", template="plotly_dark")
    return fig

def drawdown_curve(df, period_name="All Time"):
    fig = px.area(df, y="drawdown", title=f"Drawdown - {period_name}", template="plotly_dark")
    return fig

def pnl_distribution(df, pnl, period_name="All Time"):
    fig = px.histogram(df, x=pnl, title=f"PnL Distribution - {period_name}", template="plotly_dark")
    return fig

def win_loss_pie(df, pnl, period_name="All Time"):
    fig = px.pie(
        names=["Wins", "Losses"],
        values=[len(df[df[pnl]>0]), len(df[df[pnl]<0])],
        title=f"Win vs Loss - {period_name}",
        template="plotly_dark"
    )
    return fig

def pnl_growth_over_time(df, pnl, period_name="All Time"):
    """Show PnL accumulation over time (trade by trade)"""
    fig = go.Figure()
    
    # Add cumulative PnL line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["equity"],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#00ff88', width=2)
    ))
    
    # Add individual trade markers (wins/losses)
    wins = df[df[pnl] > 0]
    losses = df[df[pnl] < 0]
    
    if len(wins) > 0:
        fig.add_trace(go.Scatter(
            x=wins.index + 1,
            y=wins["equity"],
            mode='markers',
            name='Wins',
            marker=dict(color='green', size=6, symbol='triangle-up')
        ))
    
    if len(losses) > 0:
        fig.add_trace(go.Scatter(
            x=losses.index + 1,
            y=losses["equity"],
            mode='markers',
            name='Losses',
            marker=dict(color='red', size=6, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f"PnL Growth Over Time - {period_name}",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative PnL ($)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig

def rolling_performance_charts(df):
    """
    Create professional-grade rolling performance charts for edge decay detection
    Based on risk management principles, not Twitter trader fluff
    """
    
    # Create comprehensive subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Rolling Expectancy (20-trade) - THE ONLY METRIC THAT MATTERS',
            'Rolling R:R Ratio - Win Rate Means Nothing Without This',
            'Rolling Win Rate (%) - Worship This & You\'ll Blow Up', 
            'Rolling Profit Factor - Gross Profit / Gross Loss',
            'Rolling Profit (20-trade) - Path Dependent, Not Edge',
            'Rolling Drawdown % - Pain Within Window',
            'Edge Decay Signals - Expectancy Trend & WR/Exp Divergence',
            'Profit Concentration Risk - Are 20% of Trades Carrying You?'
        ),
        vertical_spacing=0.06,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    trade_numbers = list(range(1, len(df) + 1))
    
    # === ROW 1: EXPECTANCY & R:R RATIO ===
    
    # Rolling Expectancy (MOST IMPORTANT)
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_expectancy"],
        mode='lines',
        name='Rolling Expectancy',
        line=dict(color='#ff6b6b', width=3),
        hovertemplate='Trade %{x}<br>Expectancy: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Add zero line for expectancy (CRITICAL)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=1,
                  annotation_text="EDGE THRESHOLD", annotation_position="top right")
    
    # Rolling R:R Ratio
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_rr_ratio"],
        mode='lines',
        name='Rolling R:R',
        line=dict(color='#4ecdc4', width=2),
        hovertemplate='Trade %{x}<br>R:R Ratio: %{y:.2f}:1<extra></extra>'
    ), row=1, col=2)
    
    # Add 1:1 reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", opacity=0.7, row=1, col=2,
                  annotation_text="1:1 RATIO", annotation_position="top right")
    
    # === ROW 2: WIN RATE & PROFIT FACTOR ===
    
    # Rolling Win Rate (with warning context)
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_win_rate"],
        mode='lines',
        name='Rolling Win Rate',
        line=dict(color='#45b7d1', width=2),
        hovertemplate='Trade %{x}<br>Win Rate: %{y:.1f}%<extra></extra>'
    ), row=2, col=1)
    
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1,
                  annotation_text="50%", annotation_position="top right")
    
    # Rolling Profit Factor
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_profit_factor"],
        mode='lines',
        name='Rolling PF',
        line=dict(color='#96ceb4', width=2),
        hovertemplate='Trade %{x}<br>Profit Factor: %{y:.2f}<extra></extra>'
    ), row=2, col=2)
    
    # Add 1.0 reference line (break-even)
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=2,
                  annotation_text="BREAK-EVEN", annotation_position="top right")
    
    # === ROW 3: PROFIT & DRAWDOWN ===
    
    # Rolling Profit
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_profit"],
        mode='lines',
        name='Rolling Profit',
        line=dict(color='#feca57', width=2),
        hovertemplate='Trade %{x}<br>20-Trade Profit: $%{y:.2f}<extra></extra>'
    ), row=3, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=3, col=1)
    
    # Rolling Drawdown %
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_drawdown_pct"],
        mode='lines',
        name='Rolling DD %',
        line=dict(color='#ff7675', width=2),
        fill='tonexty',
        fillcolor='rgba(255, 118, 117, 0.1)',
        hovertemplate='Trade %{x}<br>Rolling DD: %{y:.1f}%<extra></extra>'
    ), row=3, col=2)
    
    # === ROW 4: EDGE DECAY SIGNALS ===
    
    # Expectancy Trend & WR/Exp Divergence
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["expectancy_trend"],
        mode='lines',
        name='Expectancy Trend',
        line=dict(color='#fd79a8', width=2),
        hovertemplate='Trade %{x}<br>Expectancy Trend: $%{y:.3f}<extra></extra>'
    ), row=4, col=1)
    
    # Add zero line for trend
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=4, col=1)
    
    # Profit Concentration Risk
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df["rolling_profit_concentration"],
        mode='lines',
        name='Profit Concentration',
        line=dict(color='#a29bfe', width=2),
        hovertemplate='Trade %{x}<br>Top 20% Contribution: %{y:.1f}%<extra></extra>'
    ), row=4, col=2)
    
    # Add danger zone (>80% concentration is risky)
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=4, col=2,
                  annotation_text="DANGER ZONE", annotation_position="top right")
    
    # === PROFESSIONAL LAYOUT ===
    fig.update_layout(
        height=1000,
        template="plotly_dark",
        showlegend=False,
        title=dict(
            text="<b>Professional Edge Decay Analysis</b><br><span style='font-size:14px;'>Risk Manager's View - No Twitter Trader Fluff</span>",
            x=0.5,
            font=dict(size=20)
        ),
        margin=dict(l=60, r=60, t=100, b=140),  # Even larger bottom margin for two rows of annotations
        annotations=[
            # Professional warnings positioned in two rows to avoid overlap
            dict(
                text="⚠️ CRITICAL: If expectancy trends down for 2-3 windows → CUT SIZE",
                xref="paper", yref="paper",
                x=0.5, y=-0.08,  # First row, centered
                showarrow=False,
                font=dict(color="red", size=11),
                bgcolor="rgba(255,0,0,0.1)",
                bordercolor="red",
                borderwidth=1,
                xanchor="center"
            ),
            dict(
                text="📊 PRO TIP: Rising win rate + falling expectancy = STRATEGY DEATH",
                xref="paper", yref="paper", 
                x=0.5, y=-0.15,  # Second row, centered, further down
                showarrow=False,
                font=dict(color="orange", size=11),
                bgcolor="rgba(255,165,0,0.1)",
                bordercolor="orange",
                borderwidth=1,
                xanchor="center"
            )
        ]
    )
    
    # Update y-axis labels for each subplot
    fig.update_yaxes(title_text="Expectancy ($)", row=1, col=1)
    fig.update_yaxes(title_text="R:R Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Profit Factor", row=2, col=2)
    fig.update_yaxes(title_text="Rolling Profit ($)", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=2)
    fig.update_yaxes(title_text="Trend ($)", row=4, col=1)
    fig.update_yaxes(title_text="Concentration (%)", row=4, col=2)
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Trade Number", row=4, col=1)
    fig.update_xaxes(title_text="Trade Number", row=4, col=2)
    
    return fig

def time_analysis_charts(time_data):
    """Create time-based performance analysis charts"""
    
    # Hourly Performance Chart
    hourly_fig = go.Figure()
    
    hourly_fig.add_trace(go.Bar(
        x=time_data['hourly'].index,
        y=time_data['hourly']['Total_PnL'],
        name='Hourly PnL',
        marker_color=['red' if x < 0 else 'green' for x in time_data['hourly']['Total_PnL']]
    ))
    
    hourly_fig.update_layout(
        title="Performance by Hour of Day (Killer Hours Detection)",
        xaxis_title="Hour",
        yaxis_title="Total PnL ($)",
        template="plotly_dark"
    )
    
    # Daily Performance Chart
    daily_fig = go.Figure()
    
    # Reorder days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_ordered = time_data['daily'].reindex([day for day in day_order if day in time_data['daily'].index])
    
    daily_fig.add_trace(go.Bar(
        x=daily_ordered.index,
        y=daily_ordered['Total_PnL'],
        name='Daily PnL',
        marker_color=['red' if x < 0 else 'green' for x in daily_ordered['Total_PnL']]
    ))
    
    daily_fig.update_layout(
        title="Performance by Day of Week",
        xaxis_title="Day",
        yaxis_title="Total PnL ($)",
        template="plotly_dark"
    )
    
    return hourly_fig, daily_fig

def monthly_heatmap(time_data):
    """Create monthly performance heatmap"""
    
    monthly_data = time_data['monthly'].reset_index()
    
    # Create heatmap data
    fig = go.Figure(data=go.Heatmap(
        x=monthly_data['month_name'],
        y=['PnL'],
        z=[monthly_data['Total_PnL'].tolist()],
        colorscale='RdYlGn',
        text=[[f"${x:.0f}" for x in monthly_data['Total_PnL']]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Monthly Performance Heatmap",
        xaxis_title="Month",
        template="plotly_dark",
        height=200
    )
    
    return fig

def risk_pain_charts(risk_data, df):
    """Create risk and pain analysis charts"""
    
    # Drawdown Duration Histogram
    dd_fig = go.Figure()
    
    if risk_data['drawdown_periods']:
        dd_fig.add_trace(go.Histogram(
            x=risk_data['drawdown_periods'],
            nbinsx=10,
            name='Drawdown Duration',
            marker_color='red',
            opacity=0.7
        ))
    
    dd_fig.update_layout(
        title="Drawdown Duration Distribution (Pain Analysis)",
        xaxis_title="Duration (Trades)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    
    # Recovery Time Analysis
    recovery_fig = go.Figure()
    
    if risk_data['recovery_times']:
        recovery_fig.add_trace(go.Histogram(
            x=risk_data['recovery_times'],
            nbinsx=10,
            name='Recovery Time',
            marker_color='orange',
            opacity=0.7
        ))
    
    recovery_fig.update_layout(
        title="Recovery Time Distribution",
        xaxis_title="Recovery Time (Trades)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    
    # Consecutive Losses Chart
    consec_fig = go.Figure()
    
    if risk_data['consecutive_losses']:
        consec_fig.add_trace(go.Histogram(
            x=risk_data['consecutive_losses'],
            nbinsx=max(10, max(risk_data['consecutive_losses'])),
            name='Consecutive Losses',
            marker_color='darkred',
            opacity=0.7
        ))
    
    consec_fig.update_layout(
        title="Consecutive Losses Distribution (Psychological Pain)",
        xaxis_title="Consecutive Losses",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    
    return dd_fig, recovery_fig, consec_fig

def create_time_tables(time_data):
    """Create summary tables for time analysis"""
    
    # Find worst performing hours/days
    worst_hours = time_data['hourly'].nsmallest(3, 'Total_PnL')
    best_hours = time_data['hourly'].nlargest(3, 'Total_PnL')
    
    worst_days = time_data['daily'].nsmallest(3, 'Total_PnL')
    best_days = time_data['daily'].nlargest(3, 'Total_PnL')
    
    return {
        'worst_hours': worst_hours,
        'best_hours': best_hours,
        'worst_days': worst_days,
        'best_days': best_days
    }

def create_trading_insights_charts(insights):
    """Create comprehensive trading insights charts"""
    
    charts = {}
    
    # === LOT SIZE ANALYSIS CHARTS ===
    if insights['lot_analysis']['has_lot_data']:
        lot_perf = insights['lot_analysis']['performance_by_lot']
        
        # Lot Size vs Performance
        lot_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PnL by Lot Size', 'Win Rate by Lot Size', 'Trade Count by Lot Size', 'PnL per Lot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # PnL by Lot Size
        lot_fig.add_trace(go.Bar(
            x=lot_perf.index,
            y=lot_perf['Total_PnL'],
            name='Total PnL',
            marker_color=['red' if x < 0 else 'green' for x in lot_perf['Total_PnL']]
        ), row=1, col=1)
        
        # Win Rate by Lot Size
        lot_fig.add_trace(go.Scatter(
            x=lot_perf.index,
            y=lot_perf['Win_Rate'],
            mode='lines+markers',
            name='Win Rate %',
            line=dict(color='orange', width=3)
        ), row=1, col=2)
        
        # Trade Count by Lot Size
        lot_fig.add_trace(go.Bar(
            x=lot_perf.index,
            y=lot_perf['Trade_Count'],
            name='Trade Count',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # PnL per Lot
        lot_fig.add_trace(go.Bar(
            x=lot_perf.index,
            y=lot_perf['PnL_Per_Lot'],
            name='PnL per Lot',
            marker_color=['red' if x < 0 else 'green' for x in lot_perf['PnL_Per_Lot']]
        ), row=2, col=2)
        
        lot_fig.update_layout(
            height=600,
            template="plotly_dark",
            title_text="Lot Size Analysis",
            showlegend=False
        )
        
        charts['lot_analysis'] = lot_fig
    
    # === BUY VS SELL COMPARISON ===
    if insights['direction_analysis']['has_direction_data']:
        direction_perf = insights['direction_analysis']['performance']
        
        # Create comparison chart
        direction_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total PnL Comparison', 'Win Rate Comparison', 'Average PnL per Trade', 'Profit Factor'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        directions = direction_perf.index.tolist()
        colors = ['#1f77b4', '#ff7f0e']  # Blue for first, Orange for second
        
        # Total PnL
        direction_fig.add_trace(go.Bar(
            x=directions,
            y=direction_perf['Total_PnL'],
            name='Total PnL',
            marker_color=[colors[i] for i in range(len(directions))]
        ), row=1, col=1)
        
        # Win Rate
        direction_fig.add_trace(go.Bar(
            x=directions,
            y=direction_perf['Win_Rate'],
            name='Win Rate %',
            marker_color=[colors[i] for i in range(len(directions))]
        ), row=1, col=2)
        
        # Average PnL
        direction_fig.add_trace(go.Bar(
            x=directions,
            y=direction_perf['Avg_PnL'],
            name='Avg PnL',
            marker_color=[colors[i] for i in range(len(directions))]
        ), row=2, col=1)
        
        # Profit Factor
        direction_fig.add_trace(go.Bar(
            x=directions,
            y=direction_perf['Profit_Factor'],
            name='Profit Factor',
            marker_color=[colors[i] for i in range(len(directions))]
        ), row=2, col=2)
        
        direction_fig.update_layout(
            height=600,
            template="plotly_dark",
            title_text="Buy vs Sell Performance Analysis",
            showlegend=False
        )
        
        charts['direction_analysis'] = direction_fig
    
    # === SYMBOL PERFORMANCE CHART ===
    if insights['symbol_analysis']['has_symbol_data']:
        symbol_perf = insights['symbol_analysis']['performance'].head(10)  # Top 10 symbols
        
        symbol_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total PnL by Symbol', 'Win Rate by Symbol', 'Trade Count by Symbol', 'Consistency by Symbol'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total PnL by Symbol
        symbol_fig.add_trace(go.Bar(
            x=symbol_perf.index,
            y=symbol_perf['Total_PnL'],
            name='Total PnL',
            marker_color=['red' if x < 0 else 'green' for x in symbol_perf['Total_PnL']]
        ), row=1, col=1)
        
        # Win Rate by Symbol
        symbol_fig.add_trace(go.Scatter(
            x=symbol_perf.index,
            y=symbol_perf['Win_Rate'],
            mode='lines+markers',
            name='Win Rate %',
            line=dict(color='orange', width=2)
        ), row=1, col=2)
        
        # Trade Count by Symbol
        symbol_fig.add_trace(go.Bar(
            x=symbol_perf.index,
            y=symbol_perf['Trade_Count'],
            name='Trade Count',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # Consistency by Symbol
        symbol_fig.add_trace(go.Bar(
            x=symbol_perf.index,
            y=symbol_perf['Consistency'],
            name='Consistency',
            marker_color='purple'
        ), row=2, col=2)
        
        symbol_fig.update_layout(
            height=600,
            template="plotly_dark",
            title_text="Symbol Performance Analysis",
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        symbol_fig.update_xaxes(tickangle=45)
        
        charts['symbol_analysis'] = symbol_fig
    
    # === POSITION SIZING ANALYSIS ===
    if insights['position_sizing']['has_sizing_data']:
        sizing_perf = insights['position_sizing']['performance_by_size']
        
        sizing_fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('PnL by Position Size', 'Win Rate by Size', 'Average Lot Size'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # PnL by Position Size
        sizing_fig.add_trace(go.Bar(
            x=sizing_perf.index,
            y=sizing_perf['Total_PnL'],
            name='Total PnL',
            marker_color=['red' if x < 0 else 'green' for x in sizing_perf['Total_PnL']]
        ), row=1, col=1)
        
        # Win Rate by Size
        sizing_fig.add_trace(go.Bar(
            x=sizing_perf.index,
            y=sizing_perf['Win_Rate'],
            name='Win Rate %',
            marker_color='orange'
        ), row=1, col=2)
        
        # Average Lot Size
        sizing_fig.add_trace(go.Bar(
            x=sizing_perf.index,
            y=sizing_perf['Avg_Lot_Size'],
            name='Avg Lot Size',
            marker_color='lightblue'
        ), row=1, col=3)
        
        sizing_fig.update_layout(
            height=400,
            template="plotly_dark",
            title_text="Position Sizing Analysis",
            showlegend=False
        )
        
        charts['position_sizing'] = sizing_fig
    
    # === RISK-REWARD DISTRIBUTION ===
    if insights['rr_analysis']['has_rr_data']:
        rr_stats = insights['rr_analysis']['stats']
        
        # Create R:R summary chart
        rr_fig = go.Figure()
        
        # Add R:R ratio bar
        rr_fig.add_trace(go.Bar(
            x=['Risk:Reward Ratio'],
            y=[rr_stats['avg_rr_ratio']],
            name='R:R Ratio',
            marker_color='green' if rr_stats['avg_rr_ratio'] >= 1.5 else 'orange' if rr_stats['avg_rr_ratio'] >= 1.0 else 'red',
            text=[f"{rr_stats['avg_rr_ratio']:.2f}:1"],
            textposition='auto'
        ))
        
        # Add reference lines
        rr_fig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Break-even")
        rr_fig.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Good R:R")
        rr_fig.add_hline(y=2.0, line_dash="dash", line_color="darkgreen", opacity=0.5, annotation_text="Excellent R:R")
        
        rr_fig.update_layout(
            title="Risk:Reward Ratio Analysis",
            yaxis_title="Ratio",
            template="plotly_dark",
            height=400
        )
        
        charts['rr_analysis'] = rr_fig
    
    return charts

def create_pip_analysis_chart(insights):
    """Create pip analysis chart if pip data is available"""
    
    if not insights['pip_analysis']['has_pip_data']:
        return None
    
    pip_stats = insights['pip_analysis']['stats']
    
    # Create pip summary chart
    pip_fig = go.Figure()
    
    # Pip metrics
    metrics = ['Avg Pips/Trade', 'Best Trade (Pips)', 'Worst Trade (Pips)']
    values = [
        pip_stats['avg_pips_per_trade'],
        pip_stats['best_pip_trade'],
        pip_stats['worst_pip_trade']
    ]
    colors = ['blue', 'green', 'red']
    
    pip_fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition='auto'
    ))
    
    pip_fig.update_layout(
        title="Pip Analysis Summary",
        yaxis_title="Pips",
        template="plotly_dark",
        height=400
    )
    
    return pip_fig

def create_daily_calendar_chart(df, pnl_col, period_name="All Time", selected_year=None, selected_month=None):
    """
    Create a professional calendar-style chart showing daily trading performance
    Similar to the reference image with monthly view, year navigation, and clean layout
    """
    
    # Find datetime column
    datetime_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'datetime', 'close_time']):
            try:
                df[f'{col}_parsed'] = pd.to_datetime(df[col])
                datetime_col = f'{col}_parsed'
                break
            except:
                continue
    
    if datetime_col is None:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid datetime column found for calendar view",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template="plotly_white",
            title="Daily Trading Calendar",
            height=600,
            paper_bgcolor='white'
        )
        return fig
    
    # Group by date
    df_copy = df.copy()
    df_copy['trade_date'] = df_copy[datetime_col].dt.date
    
    # Aggregate daily data
    daily_data = df_copy.groupby('trade_date').agg({
        pnl_col: ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    daily_data.columns = ['date', 'trade_count', 'daily_pnl']
    
    if len(daily_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No trading data available for calendar view",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template="plotly_white",
            title="Daily Trading Calendar",
            height=600,
            paper_bgcolor='white'
        )
        return fig
    
    # Determine target year and month
    if selected_year and selected_month:
        target_year = selected_year
        target_month = selected_month
    else:
        # Use the most recent month with data
        latest_date = daily_data['date'].max()
        target_year = latest_date.year
        target_month = latest_date.month
    
    # Create month view
    first_day_of_month = datetime(target_year, target_month, 1).date()
    
    # Get last day of month
    if target_month == 12:
        last_day_of_month = datetime(target_year + 1, 1, 1).date() - timedelta(days=1)
    else:
        last_day_of_month = datetime(target_year, target_month + 1, 1).date() - timedelta(days=1)
    
    # Generate all dates in the month
    month_dates = pd.date_range(start=first_day_of_month, end=last_day_of_month, freq='D')
    
    # Create complete month dataset
    month_data = pd.DataFrame({'date': month_dates.date})
    month_data = month_data.merge(daily_data, on='date', how='left')
    month_data['trade_count'] = month_data['trade_count'].fillna(0)
    month_data['daily_pnl'] = month_data['daily_pnl'].fillna(0)
    
    # Add calendar information
    month_data['day'] = pd.to_datetime(month_data['date']).dt.day
    month_data['weekday'] = pd.to_datetime(month_data['date']).dt.weekday  # Monday = 0
    month_data['weekday_name'] = pd.to_datetime(month_data['date']).dt.day_name()
    
    # Create the calendar grid (7 columns for days of week, multiple rows for weeks)
    fig = go.Figure()
    
    # Calendar dimensions - SIGNIFICANTLY INCREASED for no text overlap
    cell_width = 2.0   # Increased from 1.4 to 2.0
    cell_height = 1.8  # Increased from 1.2 to 1.8
    margin = 0.1       # Increased margin for better separation
    
    # Get first day of month weekday (0=Monday, 6=Sunday)
    first_weekday = datetime(target_year, target_month, 1).weekday()
    # Convert to Sunday=0 format for calendar display
    first_weekday = (first_weekday + 1) % 7
    
    # Calculate max P&L for color scaling
    max_pnl = max(abs(month_data['daily_pnl'].min()), abs(month_data['daily_pnl'].max()))
    if max_pnl == 0:
        max_pnl = 100  # Default scale
    
    # Create calendar grid
    for idx, row in month_data.iterrows():
        day = row['day']
        trades = int(row['trade_count'])
        pnl = row['daily_pnl']
        
        # Calculate position in grid
        days_from_start = day - 1
        total_days_from_start = first_weekday + days_from_start
        
        week_row = total_days_from_start // 7
        day_col = total_days_from_start % 7
        
        # Calculate actual positions (flip y-axis so week 0 is at top)
        x_pos = day_col * (cell_width + margin)
        y_pos = -(week_row * (cell_height + margin))  # Negative for top-to-bottom
        
        # Determine cell color based on P&L (matching reference image style)
        if trades == 0:
            cell_color = '#f8f9fa'  # Light gray for no trades
            border_color = '#e9ecef'
            text_color = '#6c757d'
        elif pnl > 0:
            # Green shades for profit (more vibrant like reference)
            intensity = min(abs(pnl) / max_pnl, 1.0) if max_pnl > 0 else 0.3
            # Use a more vibrant green similar to reference
            green_base = 76  # Base green value
            green_intensity = int(green_base + intensity * 100)  # 76-176 range
            cell_color = f'rgb(40, {green_intensity}, 40)'
            border_color = '#28a745'
            text_color = 'white'
        elif pnl < 0:
            # Red shades for loss (more vibrant like reference)
            intensity = min(abs(pnl) / max_pnl, 1.0) if max_pnl > 0 else 0.3
            # Use a more vibrant red similar to reference
            red_base = 76
            red_intensity = int(red_base + intensity * 100)  # 76-176 range
            cell_color = f'rgb({red_intensity}, 40, 40)'
            border_color = '#dc3545'
            text_color = 'white'
        else:
            # Neutral gray for breakeven with trades
            cell_color = '#e9ecef'
            border_color = '#adb5bd'
            text_color = '#495057'
        
        # Add cell rectangle
        fig.add_shape(
            type="rect",
            x0=x_pos, y0=y_pos,
            x1=x_pos + cell_width, y1=y_pos + cell_height,
            fillcolor=cell_color,
            line=dict(color=border_color, width=1),
        )
        
        # Add day number (tiny superscript style in top-left corner)
        fig.add_annotation(
            x=x_pos + 0.08, y=y_pos + 1.75,  # Very top-left corner
            text=str(day),
            showarrow=False,
            font=dict(color=text_color, size=10, family="Arial"),  # Much smaller, like true superscript
            xanchor="left", yanchor="top"
        )
        
        # Add P&L amount (center, more prominent now)
        if pnl != 0:
            pnl_text = f"${pnl:.2f}" if abs(pnl) < 100 else f"${pnl:.0f}"
            fig.add_annotation(
                x=x_pos + 1.0, y=y_pos + 1.1,  # Slightly higher center
                text=pnl_text,
                showarrow=False,
                font=dict(color=text_color, size=18, family="Arial Bold"),  # Larger, more prominent
                xanchor="center", yanchor="middle"
            )
        else:
            if trades > 0:
                fig.add_annotation(
                    x=x_pos + 1.0, y=y_pos + 1.1,  # Slightly higher center
                    text="$0.00",
                    showarrow=False,
                    font=dict(color=text_color, size=16),
                    xanchor="center", yanchor="middle"
                )
        
        # Add trade count (bottom center, more space now)
        if trades > 0:
            trade_text = f"{trades} trade{'s' if trades != 1 else ''}"
            fig.add_annotation(
                x=x_pos + 1.0, y=y_pos + 0.4,  # Bottom with more space
                text=trade_text,
                showarrow=False,
                font=dict(color=text_color, size=13),  # Slightly larger
                xanchor="center", yanchor="bottom"
            )
        else:
            fig.add_annotation(
                x=x_pos + 1.0, y=y_pos + 0.4,  # Bottom center
                text="0 trades",
                showarrow=False,
                font=dict(color=text_color, size=12),
                xanchor="center", yanchor="bottom"
            )
        
        # Add invisible hover area
        hover_text = (
            f"<b>{row['weekday_name']}, {calendar.month_name[target_month]} {day}, {target_year}</b><br>"
            f"Trades: {trades}<br>"
            f"P&L: ${pnl:.2f}<br>"
            f"Status: {'Profitable' if pnl > 0 else 'Loss' if pnl < 0 else 'Breakeven' if trades > 0 else 'No Trading'}"
        )
        
        fig.add_trace(go.Scatter(
            x=[x_pos + 1.0], y=[y_pos + 0.9],  # Adjusted for larger cells
            mode='markers',
            marker=dict(size=80, color='rgba(0,0,0,0)'),  # Larger invisible area
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False,
            name=""
        ))
    
    # Add weekday headers (matching reference style)
    weekday_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for i, day_name in enumerate(weekday_labels):
        fig.add_annotation(
            x=i * (cell_width + margin) + 1.0, y=1.6,  # Adjusted for much larger cells
            text=day_name,
            showarrow=False,
            font=dict(color="#495057", size=18, family="Arial Bold"),  # Larger font
            xanchor="center", yanchor="bottom"
        )
    
    # Calculate layout dimensions
    max_weeks = 6  # Maximum weeks in a month view
    total_width = 7 * (cell_width + margin) - margin
    total_height = max_weeks * (cell_height + margin) + 2
    
    # Update layout (clean white background like reference)
    fig.update_layout(
        title=dict(
            text=f"<b>{calendar.month_name[target_month]} {target_year}</b><br><span style='font-size:14px; color:#6c757d;'>See at one glance which how many days you are making or losing money. Click a day to look at the trades.</span>",
            x=0.5,
            font=dict(size=24, color="#212529")
        ),
        template="plotly_white",
        height=900,   # Increased height for much larger cells
        width=1400,   # Increased width for much larger cells
        xaxis=dict(
            range=[-0.3, total_width + 0.3],  # More padding
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[-total_height, 2.0],  # Adjusted for much larger cells
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=120, b=80),
        hovermode='closest'
    )
    
    # Add summary statistics at the bottom
    total_trades = int(month_data['trade_count'].sum())
    total_pnl = month_data['daily_pnl'].sum()
    profitable_days = (month_data['daily_pnl'] > 0).sum()
    trading_days = (month_data['trade_count'] > 0).sum()
    loss_days = (month_data['daily_pnl'] < 0).sum()
    
    # Create summary similar to reference image
    summary_text = (
        f"<b>Month Summary:</b> {total_trades} total trades • "
        f"${total_pnl:.2f} total P&L • "
        f"{profitable_days} profitable days • "
        f"{loss_days} loss days • "
        f"{trading_days - profitable_days - loss_days} breakeven days"
    )
    
    fig.add_annotation(
        x=0.5, y=-0.08,
        text=summary_text,
        showarrow=False,
        font=dict(color="#6c757d", size=12),
        xref="paper", yref="paper",
        xanchor="center", yanchor="top"
    )
    
    return fig

def create_kelly_criterion_charts(kelly_metrics):
    """
    Create comprehensive Kelly Criterion analysis charts.
    
    Args:
        kelly_metrics: Dictionary from compute_kelly_metrics function
    
    Returns:
        Dictionary containing Kelly-related charts
    """
    
    if not kelly_metrics.get('has_kelly_data', False):
        return {}
    
    charts = {}
    
    # 1. Kelly Fraction Overview Chart
    kelly_overview_fig = create_kelly_overview_chart(kelly_metrics)
    if kelly_overview_fig:
        charts['kelly_overview'] = kelly_overview_fig
    
    # 2. Position Size Recommendation Chart
    position_size_fig = create_position_size_chart(kelly_metrics)
    if position_size_fig:
        charts['position_sizing'] = position_size_fig
    
    # 3. Risk-Reward Analysis Chart
    risk_reward_fig = create_risk_reward_chart(kelly_metrics)
    if risk_reward_fig:
        charts['risk_reward'] = risk_reward_fig
    
    return charts

def create_kelly_overview_chart(kelly_metrics):
    """Create Kelly Criterion overview chart showing key metrics"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Kelly Fraction vs Conservative Kelly",
            "Win Rate vs Odds Ratio",
            "Edge Analysis",
            "Risk Level Assessment"
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "indicator"}, {"type": "bar"}]]
    )
    
    # 1. Kelly Fraction Comparison (Top Left)
    kelly_fraction = kelly_metrics['kelly_fraction']
    conservative_kelly = kelly_metrics['conservative_kelly']
    
    fig.add_trace(
        go.Bar(
            x=['Full Kelly', 'Conservative Kelly (25%)', 'Your Current Risk'],
            y=[kelly_fraction, conservative_kelly, 0.02],  # Assuming 2% current risk
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
            name="Kelly Fractions"
        ),
        row=1, col=1
    )
    
    # 2. Win Rate vs Odds Ratio (Top Right)
    win_rate_pct = kelly_metrics['win_rate'] * 100
    odds_ratio = kelly_metrics['odds_ratio']
    
    fig.add_trace(
        go.Scatter(
            x=[odds_ratio],
            y=[win_rate_pct],
            mode='markers',
            marker=dict(
                size=20,
                color=kelly_fraction,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Kelly Fraction")
            ),
            name="Performance Point",
            text=[f"Kelly: {kelly_fraction:.3f}"],
            textposition="top center"
        ),
        row=1, col=2
    )
    
    # Add reference lines for good performance
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(x=1.5, line_dash="dash", line_color="gray", row=1, col=2)
    
    # 3. Edge Indicator (Bottom Left)
    edge_value = kelly_metrics['edge']
    edge_color = "green" if edge_value > 0 else "red"
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=edge_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Expected Value per Trade"},
            gauge={
                'axis': {'range': [None, max(abs(edge_value) * 2, 10)]},
                'bar': {'color': edge_color},
                'steps': [
                    {'range': [0, max(abs(edge_value) * 2, 10) * 0.5], 'color': "lightgray"},
                    {'range': [max(abs(edge_value) * 2, 10) * 0.5, max(abs(edge_value) * 2, 10)], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ),
        row=2, col=1
    )
    
    # 4. Risk Level Assessment (Bottom Right)
    risk_levels = ['LOW', 'MODERATE', 'HIGH', 'EXTREME', 'NO EDGE']
    risk_colors = ['#4ecdc4', '#45b7d1', '#ffa726', '#ff6b6b', '#8e24aa']
    current_risk = kelly_metrics['risk_level']
    
    risk_values = [1 if level == current_risk else 0.3 for level in risk_levels]
    
    fig.add_trace(
        go.Bar(
            x=risk_levels,
            y=risk_values,
            marker_color=[risk_colors[i] if risk_levels[i] == current_risk else 'lightgray' 
                         for i in range(len(risk_levels))],
            name="Risk Level"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Kelly Criterion Analysis Overview",
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Fraction of Capital", row=1, col=1)
    fig.update_yaxes(title_text="Kelly Fraction", row=1, col=1)
    
    fig.update_xaxes(title_text="Odds Ratio (Avg Win / Avg Loss)", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Risk Level", row=2, col=2)
    fig.update_yaxes(title_text="Current Level", row=2, col=2)
    
    return fig

def create_position_size_chart(kelly_metrics):
    """Create position sizing recommendation chart"""
    
    lot_rec = kelly_metrics['lot_recommendation']
    current_equity = kelly_metrics['current_equity']
    
    # Create comparison of different position sizing methods
    fig = go.Figure()
    
    # Traditional method (0.02 per 1k)
    traditional_lots = (current_equity / 1000) * 0.02
    
    # Kelly-based method
    kelly_lots = lot_rec['recommended_lot_size']
    
    # Conservative Kelly
    conservative_lots = lot_rec['base_lot_size'] * kelly_metrics['conservative_kelly']
    
    methods = ['Traditional\n(0.02/1k)', 'Kelly Optimal', 'Conservative Kelly', 'Recommended']
    lot_sizes = [traditional_lots, 
                lot_rec['base_lot_size'] * kelly_metrics['kelly_fraction'],
                conservative_lots, 
                kelly_lots]
    
    colors = ['#45b7d1', '#ff6b6b', '#4ecdc4', '#ffa726']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=lot_sizes,
        marker_color=colors,
        text=[f"{size:.3f}" for size in lot_sizes],
        textposition='auto',
        name="Lot Sizes"
    ))
    
    # Add risk level annotations
    for i, (method, size) in enumerate(zip(methods, lot_sizes)):
        risk_pct = (size / (current_equity / 1000)) * 2  # Rough risk estimate
        fig.add_annotation(
            x=i, y=size + max(lot_sizes) * 0.05,
            text=f"~{risk_pct:.1f}% risk",
            showarrow=False,
            font=dict(size=10, color="white")
        )
    
    fig.update_layout(
        title=f"Position Sizing Recommendations (Current Equity: ${current_equity:,.2f})",
        xaxis_title="Sizing Method",
        yaxis_title="Recommended Lot Size",
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_risk_reward_chart(kelly_metrics):
    """Create risk-reward analysis chart"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Win Rate vs Kelly Fraction", "Odds Ratio Impact"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Win Rate vs Kelly Fraction sensitivity analysis
    win_rates = np.arange(0.3, 0.8, 0.05)
    kelly_fractions = []
    
    avg_win = kelly_metrics['avg_win']
    avg_loss = abs(kelly_metrics['avg_loss'])
    odds_ratio = kelly_metrics['odds_ratio']
    
    for wr in win_rates:
        # Kelly formula: (bp - q) / b where b = odds_ratio, p = win_rate, q = 1-p
        kelly_f = (odds_ratio * wr - (1 - wr)) / odds_ratio
        kelly_fractions.append(max(0, kelly_f))  # Don't go negative
    
    fig.add_trace(
        go.Scatter(
            x=win_rates * 100,
            y=kelly_fractions,
            mode='lines+markers',
            name='Kelly Fraction',
            line=dict(color='#4ecdc4', width=3)
        ),
        row=1, col=1
    )
    
    # Highlight current position
    current_wr = kelly_metrics['win_rate'] * 100
    current_kelly = kelly_metrics['kelly_fraction']
    
    fig.add_trace(
        go.Scatter(
            x=[current_wr],
            y=[current_kelly],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Your Current Performance'
        ),
        row=1, col=1
    )
    
    # 2. Odds Ratio Impact
    odds_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    kelly_for_odds = []
    
    current_wr_decimal = kelly_metrics['win_rate']
    for odds in odds_ratios:
        kelly_f = (odds * current_wr_decimal - (1 - current_wr_decimal)) / odds
        kelly_for_odds.append(max(0, kelly_f))
    
    colors = ['red' if odds < 1.5 else 'orange' if odds < 2.0 else 'green' for odds in odds_ratios]
    
    fig.add_trace(
        go.Bar(
            x=[f"{odds}:1" for odds in odds_ratios],
            y=kelly_for_odds,
            marker_color=colors,
            name='Kelly Fraction'
        ),
        row=1, col=2
    )
    
    # Highlight current odds ratio
    current_odds_idx = min(range(len(odds_ratios)), 
                          key=lambda i: abs(odds_ratios[i] - kelly_metrics['odds_ratio']))
    
    fig.update_layout(
        title="Risk-Reward Sensitivity Analysis",
        template="plotly_dark",
        height=500,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Win Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Kelly Fraction", row=1, col=1)
    
    fig.update_xaxes(title_text="Risk:Reward Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Kelly Fraction", row=1, col=2)
    
    return fig

def create_kelly_insights_summary_chart(kelly_metrics):
    """Create a summary chart showing key Kelly insights"""
    
    insights = kelly_metrics.get('insights', [])
    
    if not insights:
        return None
    
    # Create a text-based summary chart
    fig = go.Figure()
    
    # Add invisible scatter plot for layout
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    # Add insights as annotations
    y_positions = np.linspace(0.9, 0.1, len(insights))
    
    for i, insight in enumerate(insights):
        # Determine color based on insight type
        if "🚨" in insight or "❌" in insight:
            color = "#ff6b6b"
        elif "⚠️" in insight:
            color = "#ffa726"
        elif "✅" in insight or "🎯" in insight:
            color = "#4ecdc4"
        else:
            color = "#45b7d1"
        
        fig.add_annotation(
            x=0.05, y=y_positions[i],
            text=insight,
            showarrow=False,
            font=dict(size=12, color=color),
            xref="paper", yref="paper",
            xanchor="left", yanchor="middle",
            bgcolor="rgba(0,0,0,0.1)",
            bordercolor=color,
            borderwidth=1
        )
    
    fig.update_layout(
        title="Kelly Criterion Insights & Recommendations",
        template="plotly_dark",
        height=max(400, len(insights) * 50),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig