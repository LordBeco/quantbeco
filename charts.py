import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Create rolling performance charts to detect edge decay"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Rolling Expectancy (20-trade)', 'Rolling Win Rate (%)', 'Rolling Profit (20-trade)'),
        vertical_spacing=0.08
    )
    
    # Rolling Expectancy
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["rolling_expectancy"],
        mode='lines',
        name='Rolling Expectancy',
        line=dict(color='#ff6b6b', width=2)
    ), row=1, col=1)
    
    # Add zero line for expectancy
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)
    
    # Rolling Win Rate
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["rolling_win_rate"],
        mode='lines',
        name='Rolling Win Rate',
        line=dict(color='#4ecdc4', width=2)
    ), row=2, col=1)
    
    # Rolling Profit
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["rolling_profit"],
        mode='lines',
        name='Rolling Profit',
        line=dict(color='#45b7d1', width=2)
    ), row=3, col=1)
    
    # Add zero line for profit
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=False,
        title_text="Rolling Performance Analysis (Edge Decay Detection)"
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Trade Number", row=3, col=1)
    
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