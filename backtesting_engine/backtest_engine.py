"""
Backtest Engine

Core backtesting execution using VectorBT for high-performance strategy testing.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from .instrument_manager import InstrumentManager
from error_handling import (
    ErrorHandler, BacktestingEngineError, ErrorDetails, 
    ErrorCategory, ErrorSeverity, safe_execute
)

# Suppress VectorBT warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')


@dataclass
class BacktestConfig:
    """Comprehensive backtesting configuration"""
    start_date: datetime
    end_date: datetime
    timeframe: str = "1h"
    timezone: str = "UTC"
    starting_balance: float = 10000.0
    leverage: float = 1.0
    commission: float = 0.0
    slippage: float = 0.0
    lot_size: float = 1.0
    compounding: bool = False
    risk_per_trade: float = 0.02
    instrument: str = "EURUSD"


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    instrument: str
    side: str  # 'buy', 'sell'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal'


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtest results"""
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    performance_metrics: PerformanceMetrics
    drawdown_series: pd.Series
    monthly_returns: pd.DataFrame


class BacktestEngine:
    """Core backtesting execution using VectorBT"""
    
    def __init__(self):
        """Initialize the backtest engine with error handling"""
        self.instrument_manager = InstrumentManager()
        self.error_handler = ErrorHandler("backtest_engine")
        self._current_equity = 0.0
        self._trades_log = []
        
    def execute_backtest(self, strategy: Callable, data: pd.DataFrame, 
                        config: BacktestConfig) -> BacktestResults:
        """Execute strategy against tick data with comprehensive error handling
        
        Args:
            strategy: Strategy function that returns buy/sell signals
            data: OHLCV tick data with datetime index
            config: Backtesting configuration
            
        Returns:
            BacktestResults with trades, equity curve, and performance metrics
        """
        try:
            # Validate inputs
            self._validate_backtest_inputs(strategy, data, config)
            
            # Initialize tracking variables
            self._current_equity = config.starting_balance
            self._trades_log = []
            
            # Prepare data for backtesting
            prepared_data = self._prepare_data(data, config)
            
            # Generate strategy signals with error handling
            signals = self._generate_signals(strategy, prepared_data, config)
            
            # Execute trades based on signals
            trades = self._execute_trades(signals, prepared_data, config)
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(trades, config)
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(trades, equity_curve, config)
            
            # Calculate drawdown series
            drawdown_series = self._calculate_drawdown_series(equity_curve)
            
            # Generate monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            return BacktestResults(
                trades=trades,
                equity_curve=equity_curve,
                performance_metrics=performance_metrics,
                drawdown_series=drawdown_series,
                monthly_returns=monthly_returns
            )
            
        except BacktestingEngineError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            error_details = ErrorDetails(
                category=ErrorCategory.EXECUTION_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Unexpected error during backtest execution: {str(e)}",
                technical_details=str(e),
                user_message="An unexpected error occurred during backtesting.",
                suggestions=[
                    "Check your strategy code for errors",
                    "Verify your data format",
                    "Try with a smaller dataset",
                    "Contact support if the issue persists"
                ]
            )
            self.error_handler.handle_error(e, {
                'data_shape': data.shape,
                'config': config.__dict__
            })
            raise BacktestingEngineError(error_details)
    
    def _validate_backtest_inputs(self, strategy: Callable, data: pd.DataFrame, config: BacktestConfig):
        """Validate backtest inputs with comprehensive checks"""
        errors = []
        
        # Validate strategy
        if not callable(strategy):
            errors.append("Strategy must be a callable function")
        
        # Validate data
        if data.empty:
            errors.append("Data cannot be empty")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Data missing required columns: {missing_columns}")
        
        # Validate config
        if config.starting_balance <= 0:
            errors.append("Starting balance must be positive")
        
        if config.leverage <= 0:
            errors.append("Leverage must be positive")
        
        if config.lot_size <= 0:
            errors.append("Lot size must be positive")
        
        if config.risk_per_trade < 0 or config.risk_per_trade > 1:
            errors.append("Risk per trade must be between 0 and 1")
        
        # Check date range
        if config.start_date and config.end_date:
            if config.start_date >= config.end_date:
                errors.append("Start date must be before end date")
        
        if errors:
            error_details = ErrorDetails(
                category=ErrorCategory.VALIDATION_ERROR,
                severity=ErrorSeverity.ERROR,
                message="Backtest input validation failed",
                user_message="Please fix the following issues with your backtest configuration:",
                suggestions=[f"Fix: {error}" for error in errors]
            )
            error_details.technical_details = "; ".join(errors)
            raise BacktestingEngineError(error_details)
    
    def _prepare_data(self, data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        """Prepare data for backtesting with comprehensive error handling"""
        try:
            prepared_data = data.copy()
            
            # Ensure datetime index
            if not isinstance(prepared_data.index, pd.DatetimeIndex):
                if 'timestamp' in prepared_data.columns:
                    prepared_data = prepared_data.set_index('timestamp')
                elif 'datetime' in prepared_data.columns:
                    prepared_data = prepared_data.set_index('datetime')
                else:
                    error_details = ErrorDetails(
                        category=ErrorCategory.DATA_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message="Data must have datetime index or timestamp/datetime column",
                        user_message="Your data needs a timestamp column for backtesting.",
                        suggestions=[
                            "Add a 'timestamp' or 'datetime' column to your data",
                            "Ensure timestamps are in a recognizable format",
                            "Check your CSV file structure"
                        ]
                    )
                    raise BacktestingEngineError(error_details)
            
            # Ensure timezone consistency between data and config
            if hasattr(config, 'start_date') and config.start_date is not None:
                # Check if config dates are timezone-aware
                config_has_tz = getattr(config.start_date, 'tzinfo', None) is not None
                data_has_tz = prepared_data.index.tz is not None
                
                if config_has_tz and not data_has_tz:
                    # Config has timezone but data doesn't - localize data to UTC first
                    prepared_data.index = prepared_data.index.tz_localize('UTC')
                elif not config_has_tz and data_has_tz:
                    # Data has timezone but config doesn't - convert data to naive
                    prepared_data.index = prepared_data.index.tz_localize(None)
                elif config_has_tz and data_has_tz:
                    # Both have timezones - ensure they match
                    config_tz = config.start_date.tzinfo
                    if prepared_data.index.tz != config_tz:
                        prepared_data.index = prepared_data.index.tz_convert(config_tz)
            
            # Filter by date range
            original_length = len(prepared_data)
            if config.start_date:
                prepared_data = prepared_data[prepared_data.index >= config.start_date]
            if config.end_date:
                prepared_data = prepared_data[prepared_data.index <= config.end_date]
            
            # Check if filtering removed all data
            if prepared_data.empty:
                error_details = ErrorDetails(
                    category=ErrorCategory.DATA_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Date range filtering resulted in empty dataset",
                    user_message="No data found in the specified date range.",
                    suggestions=[
                        "Check your start and end dates",
                        "Verify your data covers the requested period",
                        "Expand your date range"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Log data filtering results
            filtered_length = len(prepared_data)
            if filtered_length < original_length:
                self.error_handler.logger.info(
                    f"Filtered data from {original_length} to {filtered_length} rows based on date range"
                )
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in prepared_data.columns]
            if missing_columns:
                error_details = ErrorDetails(
                    category=ErrorCategory.DATA_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Missing required columns: {missing_columns}",
                    user_message="Your data is missing required price columns.",
                    suggestions=[
                        f"Ensure your data has columns: {', '.join(required_columns)}",
                        "Check column names for typos",
                        "Verify your CSV file format"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Add volume if missing (set to 1000 as default)
            if 'volume' not in prepared_data.columns:
                prepared_data['volume'] = 1000
                self.error_handler.logger.info("Added default volume column (1000)")
            
            # Validate data quality
            self._validate_prepared_data(prepared_data)
            
            return prepared_data
            
        except BacktestingEngineError:
            raise
        except Exception as e:
            error_details = ErrorDetails(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Error preparing data for backtesting: {str(e)}",
                technical_details=str(e),
                user_message="Failed to prepare your data for backtesting.",
                suggestions=[
                    "Check your data format and structure",
                    "Verify timestamp and price columns",
                    "Try with a different dataset"
                ]
            )
            raise BacktestingEngineError(error_details)
    
    def _validate_prepared_data(self, data: pd.DataFrame):
        """Validate prepared data quality"""
        # Check for sufficient data points
        if len(data) < 10:
            error_details = ErrorDetails(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Insufficient data points ({len(data)}) for backtesting",
                user_message="Not enough data for reliable backtesting results.",
                suggestions=[
                    "Use a longer time period",
                    "Check your date range settings",
                    "Verify your data source"
                ]
            )
            raise BacktestingEngineError(error_details)
        
        # Check for missing values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        for col in critical_cols:
            if data[col].isnull().any():
                null_count = data[col].isnull().sum()
                error_details = ErrorDetails(
                    category=ErrorCategory.DATA_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Column '{col}' has {null_count} missing values",
                    user_message=f"Your {col} price data has missing values.",
                    suggestions=[
                        "Fill missing values in your data",
                        "Remove rows with missing prices",
                        "Check your data source for completeness"
                    ]
                )
                raise BacktestingEngineError(error_details)
    
    def _generate_signals(self, strategy: Callable, data: pd.DataFrame, 
                         config: BacktestConfig) -> pd.DataFrame:
        """Generate buy/sell signals from strategy with comprehensive error handling"""
        try:
            # Call strategy function with data
            signals = strategy(data)
            
            # Validate strategy output
            if signals is None:
                error_details = ErrorDetails(
                    category=ErrorCategory.STRATEGY_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Strategy function returned None",
                    user_message="Your strategy didn't generate any signals.",
                    suggestions=[
                        "Check your strategy function returns data",
                        "Ensure your strategy logic is correct",
                        "Add debug prints to your strategy"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Ensure signals is a DataFrame with proper structure
            if isinstance(signals, pd.Series):
                signals = pd.DataFrame({'signal': signals}, index=data.index)
            elif isinstance(signals, dict):
                signals = pd.DataFrame(signals, index=data.index)
            elif not isinstance(signals, pd.DataFrame):
                error_details = ErrorDetails(
                    category=ErrorCategory.STRATEGY_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Strategy must return DataFrame, Series, or dict",
                    user_message="Your strategy returned an invalid data type.",
                    suggestions=[
                        "Return a pandas DataFrame with signal columns",
                        "Return a pandas Series with signal values",
                        "Return a dictionary with signal data"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Validate signal structure
            if signals.empty:
                error_details = ErrorDetails(
                    category=ErrorCategory.STRATEGY_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Strategy generated empty signals",
                    user_message="Your strategy didn't generate any trading signals.",
                    suggestions=[
                        "Check your strategy logic",
                        "Verify your indicator calculations",
                        "Ensure your conditions can be met"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Ensure signal column exists
            if 'signal' not in signals.columns:
                if 'buy' in signals.columns and 'sell' in signals.columns:
                    # Convert buy/sell columns to signal column
                    signals['signal'] = 0
                    signals.loc[signals['buy'] == 1, 'signal'] = 1
                    signals.loc[signals['sell'] == 1, 'signal'] = -1
                else:
                    error_details = ErrorDetails(
                        category=ErrorCategory.STRATEGY_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message="Strategy signals must contain 'signal' column or 'buy'/'sell' columns",
                        user_message="Your strategy output is missing required signal columns.",
                        suggestions=[
                            "Add a 'signal' column with values 1 (buy), -1 (sell), 0 (hold)",
                            "Or add separate 'buy' and 'sell' columns with 1/0 values",
                            "Check your strategy function output"
                        ]
                    )
                    raise BacktestingEngineError(error_details)
            
            # Validate signal values
            unique_signals = signals['signal'].dropna().unique()
            valid_signals = {-1, 0, 1}
            invalid_signals = set(unique_signals) - valid_signals
            
            if invalid_signals:
                error_details = ErrorDetails(
                    category=ErrorCategory.STRATEGY_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Invalid signal values found: {invalid_signals}",
                    user_message="Your strategy generated invalid signal values.",
                    suggestions=[
                        "Use only -1 (sell), 0 (hold), 1 (buy) for signals",
                        "Check your signal generation logic",
                        "Ensure signals are properly calculated"
                    ]
                )
                raise BacktestingEngineError(error_details)
            
            # Add stop loss and take profit if provided by strategy
            if 'stop_loss' not in signals.columns:
                signals['stop_loss'] = np.nan
            if 'take_profit' not in signals.columns:
                signals['take_profit'] = np.nan
            
            # Log signal statistics
            signal_counts = signals['signal'].value_counts()
            self.error_handler.logger.info(f"Generated signals - Buy: {signal_counts.get(1, 0)}, Sell: {signal_counts.get(-1, 0)}, Hold: {signal_counts.get(0, 0)}")
            
            return signals
            
        except BacktestingEngineError:
            raise
        except Exception as e:
            error_details = ErrorDetails(
                category=ErrorCategory.STRATEGY_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Error generating strategy signals: {str(e)}",
                technical_details=str(e),
                user_message="Your strategy code encountered an error.",
                suggestions=[
                    "Check your strategy code for syntax errors",
                    "Verify your indicator calculations",
                    "Test your strategy with sample data",
                    "Add error handling to your strategy"
                ]
            )
            self.error_handler.handle_error(e, {'strategy_name': getattr(strategy, '__name__', 'unknown')})
            raise BacktestingEngineError(error_details)
    
    def _execute_trades(self, signals: pd.DataFrame, data: pd.DataFrame, 
                       config: BacktestConfig) -> List[Trade]:
        """Execute trades based on signals with realistic slippage and spread"""
        trades = []
        current_position = None
        current_position_size = 0.0
        
        # Get instrument specifications
        instrument_spec = self.instrument_manager.get_instrument_spec(config.instrument)
        spread = instrument_spec.typical_spread if instrument_spec else 2.0
        
        for timestamp, signal_row in signals.iterrows():
            if timestamp not in data.index:
                continue
                
            price_row = data.loc[timestamp]
            signal = signal_row['signal']
            
            # Check for exit conditions first (stop loss, take profit)
            if current_position is not None:
                exit_reason = self._check_exit_conditions(
                    current_position, price_row, signal_row, config
                )
                
                if exit_reason:
                    # Close current position
                    exit_price = self._calculate_exit_price(
                        current_position, price_row, spread, config.slippage, exit_reason
                    )
                    
                    trade = self._create_trade_record(
                        current_position, timestamp, exit_price, exit_reason, config
                    )
                    trades.append(trade)
                    
                    # Update equity
                    self._current_equity += trade.pnl
                    
                    # Reset position
                    current_position = None
                    current_position_size = 0.0
            
            # Check for new entry signals
            if signal != 0 and current_position is None:
                # Calculate position size
                position_size = self._calculate_position_size(
                    signals.loc[timestamp], config
                )
                
                if position_size > 0:
                    # Calculate entry price with slippage and spread
                    entry_price = self._calculate_entry_price(
                        price_row, signal, spread, config.slippage
                    )
                    
                    # Create position record
                    current_position = {
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'side': 'buy' if signal > 0 else 'sell',
                        'quantity': position_size,
                        'stop_loss': signal_row.get('stop_loss', np.nan),
                        'take_profit': signal_row.get('take_profit', np.nan)
                    }
                    current_position_size = position_size
        
        # Close any remaining open position at the end
        if current_position is not None:
            final_timestamp = data.index[-1]
            final_price = data.iloc[-1]
            
            exit_price = self._calculate_exit_price(
                current_position, final_price, spread, config.slippage, 'end_of_data'
            )
            
            trade = self._create_trade_record(
                current_position, final_timestamp, exit_price, 'end_of_data', config
            )
            trades.append(trade)
            self._current_equity += trade.pnl
        
        return trades
    
    def _check_exit_conditions(self, position: Dict, price_row: pd.Series, 
                              signal_row: pd.Series, config: BacktestConfig) -> Optional[str]:
        """Check if position should be closed due to stop loss, take profit, or signal"""
        current_price = price_row['close']
        high_price = price_row['high']
        low_price = price_row['low']
        
        # Check stop loss with intrabar execution
        if not pd.isna(position['stop_loss']):
            if position['side'] == 'buy':
                # For long positions, check if low price hit stop loss
                if low_price <= position['stop_loss']:
                    return 'stop_loss'
            elif position['side'] == 'sell':
                # For short positions, check if high price hit stop loss
                if high_price >= position['stop_loss']:
                    return 'stop_loss'
        
        # Check take profit with intrabar execution
        if not pd.isna(position['take_profit']):
            if position['side'] == 'buy':
                # For long positions, check if high price hit take profit
                if high_price >= position['take_profit']:
                    return 'take_profit'
            elif position['side'] == 'sell':
                # For short positions, check if low price hit take profit
                if low_price <= position['take_profit']:
                    return 'take_profit'
        
        # Check for opposite signal
        signal = signal_row['signal']
        if signal != 0:
            if (position['side'] == 'buy' and signal < 0) or \
               (position['side'] == 'sell' and signal > 0):
                return 'signal'
        
        return None
    
    def _calculate_entry_price(self, price_row: pd.Series, signal: float, 
                              spread: float, slippage: float) -> float:
        """Calculate entry price with spread and slippage"""
        base_price = price_row['close']
        
        # Apply spread (buy at ask, sell at bid)
        if signal > 0:  # Buy
            price_with_spread = base_price + (spread * 0.00001)  # Assuming 5-digit pricing
        else:  # Sell
            price_with_spread = base_price - (spread * 0.00001)
        
        # Apply slippage
        slippage_amount = price_with_spread * slippage
        if signal > 0:  # Buy - slippage increases price
            final_price = price_with_spread + slippage_amount
        else:  # Sell - slippage decreases price
            final_price = price_with_spread - slippage_amount
        
        return final_price
    
    def _calculate_exit_price(self, position: Dict, price_row: pd.Series, 
                             spread: float, slippage: float, exit_reason: str = None) -> float:
        """Calculate exit price with spread and slippage, considering exit reason"""
        
        # For stop loss and take profit, use the exact level if available
        if exit_reason == 'stop_loss' and not pd.isna(position['stop_loss']):
            base_price = position['stop_loss']
        elif exit_reason == 'take_profit' and not pd.isna(position['take_profit']):
            base_price = position['take_profit']
        else:
            # Use close price for signal exits or end of data
            base_price = price_row['close']
        
        # Apply spread (opposite of entry)
        if position['side'] == 'buy':  # Sell at bid
            price_with_spread = base_price - (spread * 0.00001)
        else:  # Buy at ask
            price_with_spread = base_price + (spread * 0.00001)
        
        # Apply slippage
        slippage_amount = price_with_spread * slippage
        if position['side'] == 'buy':  # Selling - slippage decreases price
            final_price = price_with_spread - slippage_amount
        else:  # Buying to cover - slippage increases price
            final_price = price_with_spread + slippage_amount
        
        return final_price
    
    def _calculate_position_size(self, signal_row: pd.Series, config: BacktestConfig) -> float:
        """Calculate position size based on configuration"""
        if config.compounding:
            # Use current equity for compounding
            return self.instrument_manager.calculate_position_size(
                self._current_equity, config.risk_per_trade, 
                50.0,  # Default stop distance in pips
                config.instrument, 'compounding'
            )
        else:
            # Use fixed lot size
            return config.lot_size
    
    def _create_trade_record(self, position: Dict, exit_time: datetime, 
                            exit_price: float, exit_reason: str, 
                            config: BacktestConfig) -> Trade:
        """Create a trade record"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        # Calculate P&L
        if side == 'buy':
            price_diff = exit_price - entry_price
        else:  # sell
            price_diff = entry_price - exit_price
        
        # Get pip value for P&L calculation
        pip_value = self.instrument_manager.get_pip_value(config.instrument)
        
        # Calculate P&L in account currency
        # For forex: P&L = (price_diff / 0.00001) * pip_value * quantity
        pips_gained = price_diff / 0.00001  # Assuming 5-digit pricing
        pnl = pips_gained * pip_value * quantity
        
        # Calculate percentage P&L
        pnl_pct = (pnl / self._current_equity) * 100 if self._current_equity > 0 else 0.0
        
        # Calculate duration
        duration = exit_time - position['entry_time']
        
        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            instrument=config.instrument,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=duration,
            exit_reason=exit_reason
        )
    
    def _calculate_equity_curve(self, trades: List[Trade], config: BacktestConfig) -> pd.Series:
        """Calculate equity curve from trades"""
        if not trades:
            # Return flat equity curve at starting balance
            return pd.Series([config.starting_balance], 
                           index=[pd.Timestamp.now()], name='equity')
        
        # Create equity curve
        equity_data = []
        equity_times = []
        current_equity = config.starting_balance
        
        # Add starting point
        equity_data.append(current_equity)
        equity_times.append(trades[0].entry_time)
        
        # Add equity after each trade
        for trade in trades:
            current_equity += trade.pnl
            equity_data.append(current_equity)
            equity_times.append(trade.exit_time)
        
        return pd.Series(equity_data, index=equity_times, name='equity')
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        if len(equity_curve) == 0:
            return pd.Series([], name='drawdown')
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown as percentage
        drawdown = ((equity_curve - running_max) / running_max) * 100
        
        return drawdown.fillna(0.0)
    
    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns"""
        if len(equity_curve) == 0:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly_equity = equity_curve.resample('M').last()
        
        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change() * 100
        
        # Create DataFrame with year and month columns
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        return monthly_df.dropna()
        
    def calculate_performance_metrics(self, trades: List[Trade], 
                                    equity_curve: pd.Series, 
                                    config: BacktestConfig) -> PerformanceMetrics:
        """Calculate comprehensive performance statistics"""
        if not trades:
            return PerformanceMetrics(
                total_return=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                total_trades=0
            )
        
        # Calculate total return
        final_equity = equity_curve.iloc[-1]
        total_return = ((final_equity - config.starting_balance) / config.starting_balance) * 100
        
        # Calculate win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
        
        # Calculate max drawdown
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        max_drawdown = abs(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualized Sharpe ratio assuming daily returns
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float('inf')  # All winning trades
        else:
            profit_factor = 0.0  # No profitable trades
        
        return PerformanceMetrics(
            total_return=total_return,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            total_trades=len(trades)
        )
    
    def calculate_additional_metrics(self, trades: List[Trade], 
                                   equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        if not trades:
            return {}
        
        # Average trade metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Largest win/loss
        largest_win = max([t.pnl for t in trades]) if trades else 0.0
        largest_loss = min([t.pnl for t in trades]) if trades else 0.0
        
        # Average trade duration
        avg_duration_hours = np.mean([t.duration.total_seconds() / 3600 for t in trades]) if trades else 0.0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif trade.pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Recovery factor (total return / max drawdown)
        max_dd = abs(self._calculate_drawdown_series(equity_curve).min()) if len(equity_curve) > 1 else 0.0
        total_return = ((equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]) * 100
        recovery_factor = total_return / max_dd if max_dd > 0 else 0.0
        
        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration_hours': avg_duration_hours,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'recovery_factor': recovery_factor,
            'expectancy': (len(winning_trades) / len(trades) * avg_win) + ((len(trades) - len(winning_trades)) / len(trades) * avg_loss) if trades else 0.0
        }