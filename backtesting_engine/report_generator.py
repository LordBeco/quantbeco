"""
Report Generator

Generates broker statements and performance reports compatible with trade_analyzer_pro.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
from datetime import datetime
import numpy as np
from .backtest_engine import BacktestResults, Trade


class ReportGenerator:
    """Generates broker statements and performance reports"""
    
    def generate_broker_statement(self, results: BacktestResults, 
                                 starting_balance: float = 10000.0,
                                 account_currency: str = "USD") -> pd.DataFrame:
        """Generate CSV compatible with trade_analyzer_pro format
        
        Args:
            results: BacktestResults containing trades and performance data
            starting_balance: Starting account balance for calculations
            account_currency: Account currency for P&L calculations
            
        Returns:
            DataFrame with columns matching trade_analyzer_pro CSV format:
            Ticket, Symbol, Type, Lots, Open Time, Open Price, Close Time, 
            Close Price, Swaps, Commission, Profit, Comment
        """
        if not results.trades:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'Ticket', 'Symbol', 'Type', 'Lots', 'Open Time', 'Open Price',
                'Close Time', 'Close Price', 'Swaps', 'Commission', 'Profit', 'Comment'
            ])
        
        # Generate broker statement records
        statement_records = []
        
        for i, trade in enumerate(results.trades):
            # Generate unique ticket number (timestamp-based)
            ticket = int(trade.entry_time.timestamp() * 1000000) + i
            
            # Format trade type
            trade_type = trade.side.lower()  # 'buy' or 'sell'
            
            # Format timestamps
            open_time = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')
            close_time = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate commission (simplified - could be enhanced based on instrument)
            commission = -0.7  # Default commission similar to sample data
            
            # Swaps (overnight financing - simplified)
            swaps = 0.0  # Default to 0 for simplicity
            
            # Generate comment (could include exit reason)
            comment = f"backtest_{trade.exit_reason}_{i}"
            
            # Create record
            record = {
                'Ticket': ticket,
                'Symbol': trade.instrument,
                'Type': trade_type,
                'Lots': trade.quantity,  # Keep as numeric
                'Open Time': open_time,
                'Open Price': trade.entry_price,
                'Close Time': close_time,
                'Close Price': trade.exit_price,
                'Swaps': swaps,
                'Commission': commission,
                'Profit': round(trade.pnl, 2),  # Round to 2 decimal places
                'Comment': comment
            }
            
            statement_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(statement_records)
        
        # Ensure proper data types
        df['Ticket'] = df['Ticket'].astype(str)
        df['Lots'] = pd.to_numeric(df['Lots'], errors='coerce')
        df['Open Price'] = pd.to_numeric(df['Open Price'], errors='coerce')
        df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
        df['Swaps'] = pd.to_numeric(df['Swaps'], errors='coerce')
        df['Commission'] = pd.to_numeric(df['Commission'], errors='coerce')
        df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
        
        return df
    
    def calculate_accurate_pnl(self, trade: Trade, instrument: str, 
                              account_currency: str = "USD") -> float:
        """Calculate accurate profit/loss values in account currency
        
        Args:
            trade: Trade object with entry/exit prices and quantity
            instrument: Trading instrument (e.g., 'EURUSD', 'NAS100.R')
            account_currency: Account currency for P&L calculation
            
        Returns:
            Accurate P&L value in account currency
        """
        # This is already calculated in the Trade object during backtesting
        # The BacktestEngine handles pip value calculations and currency conversion
        return trade.pnl
    
    def validate_broker_statement_format(self, df: pd.DataFrame) -> bool:
        """Validate that the generated DataFrame matches trade_analyzer_pro format
        
        Args:
            df: Generated broker statement DataFrame
            
        Returns:
            True if format is valid, False otherwise
        """
        required_columns = [
            'Ticket', 'Symbol', 'Type', 'Lots', 'Open Time', 'Open Price',
            'Close Time', 'Close Price', 'Swaps', 'Commission', 'Profit', 'Comment'
        ]
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check data types and formats
        try:
            # Validate timestamps
            pd.to_datetime(df['Open Time'])
            pd.to_datetime(df['Close Time'])
            
            # Validate numeric columns
            pd.to_numeric(df['Open Price'])
            pd.to_numeric(df['Close Price'])
            pd.to_numeric(df['Profit'])
            
            # Validate trade types
            valid_types = {'buy', 'sell'}
            if not all(trade_type.lower() in valid_types for trade_type in df['Type']):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def create_trading_charts(self, data: pd.DataFrame, 
                             trades: List[Trade]) -> go.Figure:
        """Generate interactive trading charts with entry/exit points
        
        Args:
            data: OHLCV price data with datetime index
            trades: List of Trade objects with entry/exit information
            
        Returns:
            Plotly Figure with candlestick chart and trade markers
        """
        if data.empty:
            # Return empty figure for empty data
            fig = go.Figure()
            fig.update_layout(
                title="No Data Available",
                xaxis_title="Time",
                yaxis_title="Price"
            )
            return fig
        
        # Create subplots with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Chart with Trades', 'Volume'),
            row_heights=[0.8, 0.2]
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add volume bars if volume data exists
        if 'volume' in data.columns:
            volume_colors = ['green' if close >= open else 'red' 
                           for close, open in zip(data['close'], data['open'])]
            
            volume_bars = go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.6
            )
            fig.add_trace(volume_bars, row=2, col=1)
        
        # Add trade markers
        if trades:
            # Separate buy and sell trades for different colors
            buy_trades = [t for t in trades if t.side == 'buy']
            sell_trades = [t for t in trades if t.side == 'sell']
            
            # Add buy entry markers (green arrows up)
            if buy_trades:
                buy_entry_times = [t.entry_time for t in buy_trades]
                buy_entry_prices = [t.entry_price for t in buy_trades]
                buy_entry_hover = [
                    f"Buy Entry<br>"
                    f"Time: {t.entry_time}<br>"
                    f"Price: {t.entry_price:.5f}<br>"
                    f"Quantity: {t.quantity}<br>"
                    f"Instrument: {t.instrument}"
                    for t in buy_trades
                ]
                
                fig.add_trace(go.Scatter(
                    x=buy_entry_times,
                    y=buy_entry_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Entry',
                    hovertext=buy_entry_hover,
                    hoverinfo='text'
                ), row=1, col=1)
                
                # Add buy exit markers (green arrows down)
                buy_exit_times = [t.exit_time for t in buy_trades]
                buy_exit_prices = [t.exit_price for t in buy_trades]
                buy_exit_hover = [
                    f"Buy Exit<br>"
                    f"Time: {t.exit_time}<br>"
                    f"Price: {t.exit_price:.5f}<br>"
                    f"P&L: {t.pnl:.2f}<br>"
                    f"Duration: {t.duration}<br>"
                    f"Exit Reason: {t.exit_reason}"
                    for t in buy_trades
                ]
                
                fig.add_trace(go.Scatter(
                    x=buy_exit_times,
                    y=buy_exit_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='lightgreen',
                        line=dict(width=2, color='green')
                    ),
                    name='Buy Exit',
                    hovertext=buy_exit_hover,
                    hoverinfo='text'
                ), row=1, col=1)
                
                # Add connecting lines for buy trades
                for trade in buy_trades:
                    line_color = 'green' if trade.pnl > 0 else 'red'
                    fig.add_trace(go.Scatter(
                        x=[trade.entry_time, trade.exit_time],
                        y=[trade.entry_price, trade.exit_price],
                        mode='lines',
                        line=dict(color=line_color, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=1)
            
            # Add sell entry markers (red arrows down)
            if sell_trades:
                sell_entry_times = [t.entry_time for t in sell_trades]
                sell_entry_prices = [t.entry_price for t in sell_trades]
                sell_entry_hover = [
                    f"Sell Entry<br>"
                    f"Time: {t.entry_time}<br>"
                    f"Price: {t.entry_price:.5f}<br>"
                    f"Quantity: {t.quantity}<br>"
                    f"Instrument: {t.instrument}"
                    for t in sell_trades
                ]
                
                fig.add_trace(go.Scatter(
                    x=sell_entry_times,
                    y=sell_entry_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Entry',
                    hovertext=sell_entry_hover,
                    hoverinfo='text'
                ), row=1, col=1)
                
                # Add sell exit markers (red arrows up)
                sell_exit_times = [t.exit_time for t in sell_trades]
                sell_exit_prices = [t.exit_price for t in sell_trades]
                sell_exit_hover = [
                    f"Sell Exit<br>"
                    f"Time: {t.exit_time}<br>"
                    f"Price: {t.exit_price:.5f}<br>"
                    f"P&L: {t.pnl:.2f}<br>"
                    f"Duration: {t.duration}<br>"
                    f"Exit Reason: {t.exit_reason}"
                    for t in sell_trades
                ]
                
                fig.add_trace(go.Scatter(
                    x=sell_exit_times,
                    y=sell_exit_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='lightcoral',
                        line=dict(width=2, color='red')
                    ),
                    name='Sell Exit',
                    hovertext=sell_exit_hover,
                    hoverinfo='text'
                ), row=1, col=1)
                
                # Add connecting lines for sell trades
                for trade in sell_trades:
                    line_color = 'green' if trade.pnl > 0 else 'red'
                    fig.add_trace(go.Scatter(
                        x=[trade.entry_time, trade.exit_time],
                        y=[trade.entry_price, trade.exit_price],
                        mode='lines',
                        line=dict(color=line_color, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Trading Chart - {trades[0].instrument if trades else 'Backtest Results'}",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Update x-axis to remove gaps (weekends, holidays)
        fig.update_xaxes(
            rangeslider_visible=False,
            type='date'
        )
        
        # Update y-axis for price chart
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        # Update y-axis for volume chart
        if 'volume' in data.columns:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def add_stop_loss_take_profit_levels(self, fig: go.Figure, trades: List[Trade], 
                                       data: pd.DataFrame) -> go.Figure:
        """Add stop loss and take profit levels to the chart
        
        Args:
            fig: Existing Plotly figure
            trades: List of trades with SL/TP information
            data: Price data for reference
            
        Returns:
            Updated figure with SL/TP levels
        """
        for trade in trades:
            # Find the data range for this trade
            trade_data = data[
                (data.index >= trade.entry_time) & 
                (data.index <= trade.exit_time)
            ]
            
            if trade_data.empty:
                continue
            
            # Add stop loss level if available
            if hasattr(trade, 'stop_loss') and not pd.isna(getattr(trade, 'stop_loss', np.nan)):
                stop_loss = getattr(trade, 'stop_loss')
                fig.add_hline(
                    y=stop_loss,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text=f"SL: {stop_loss:.5f}",
                    annotation_position="bottom right"
                )
            
            # Add take profit level if available
            if hasattr(trade, 'take_profit') and not pd.isna(getattr(trade, 'take_profit', np.nan)):
                take_profit = getattr(trade, 'take_profit')
                fig.add_hline(
                    y=take_profit,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.7,
                    annotation_text=f"TP: {take_profit:.5f}",
                    annotation_position="top right"
                )
        
        return fig