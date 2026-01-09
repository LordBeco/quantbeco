"""
Code Generator

Generates structured strategy code with proper interfaces and templates.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class RiskManagement:
    """Risk management parameters"""
    stop_loss_type: str = 'percentage'  # 'fixed', 'atr', 'percentage'
    stop_loss_value: float = 2.0
    take_profit_type: str = 'percentage'  # 'fixed', 'atr', 'percentage'
    take_profit_value: float = 4.0
    position_sizing: str = 'fixed'  # 'fixed', 'risk_based', 'kelly'
    max_risk_per_trade: float = 1.0  # Percentage of account


@dataclass
class StrategyComponents:
    """Components that make up a trading strategy"""
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: RiskManagement
    timeframe: str = "1h"
    
    def __post_init__(self):
        if not self.indicators:
            self.indicators = []
        if not self.entry_conditions:
            self.entry_conditions = []
        if not self.exit_conditions:
            self.exit_conditions = []
        if not isinstance(self.risk_management, RiskManagement):
            self.risk_management = RiskManagement()


class CodeGenerator:
    """Generates structured strategy code with proper interfaces"""
    
    def __init__(self):
        """Initialize the code generator"""
        self.template_version = "1.0"
    
    def _generate_imports(self) -> str:
        """Generate required imports for strategy code"""
        return '''import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')
'''
    
    def _generate_indicator_functions(self, indicators: List[str]) -> str:
        """Generate indicator calculation functions"""
        functions = []
        
        # Always include basic indicators
        functions.append('''
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    return {
        'upper': sma + (std * std_dev),
        'middle': sma,
        'lower': sma - (std * std_dev)
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()
''')
        
        return '\n'.join(functions)
    
    def _generate_main_indicator_function(self, components: StrategyComponents) -> str:
        """Generate the main calculate_indicators function"""
        function_code = '''
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for the strategy
    
    Args:
        data: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with additional indicator columns
    """
    df = data.copy()
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate indicators based on strategy components
'''
        
        # Add specific indicators based on components
        if 'sma' in str(components.indicators).lower() or 'moving average' in str(components.indicators).lower():
            function_code += '''    df['sma_20'] = calculate_sma(df['close'], 20)
    df['sma_50'] = calculate_sma(df['close'], 50)
'''
        
        if 'ema' in str(components.indicators).lower():
            function_code += '''    df['ema_12'] = calculate_ema(df['close'], 12)
    df['ema_26'] = calculate_ema(df['close'], 26)
'''
        
        if 'rsi' in str(components.indicators).lower():
            function_code += '''    df['rsi'] = calculate_rsi(df['close'])
'''
        
        if 'macd' in str(components.indicators).lower():
            function_code += '''    macd_data = calculate_macd(df['close'])
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_histogram'] = macd_data['histogram']
'''
        
        if 'bollinger' in str(components.indicators).lower():
            function_code += '''    bb_data = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_data['upper']
    df['bb_middle'] = bb_data['middle']
    df['bb_lower'] = bb_data['lower']
'''
        
        function_code += '''    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    return df
'''
        
        return function_code
    
    def _generate_signal_function(self, components: StrategyComponents) -> str:
        """Generate the signal generation function"""
        function_code = '''
def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals based on strategy logic
    
    Args:
        data: DataFrame with OHLCV and indicator columns
        
    Returns:
        DataFrame with signal column (1=buy, -1=sell, 0=hold)
    """
    df = data.copy()
    df['signal'] = 0
    
    # Initialize signal conditions
    buy_conditions = []
    sell_conditions = []
    
'''
        
        # Generate conditions based on components
        if 'rsi' in str(components.indicators).lower():
            function_code += '''    # RSI-based conditions
    buy_conditions.append(df['rsi'] < 30)  # Oversold
    sell_conditions.append(df['rsi'] > 70)  # Overbought
    
'''
        
        if 'sma' in str(components.indicators).lower() or 'moving average' in str(components.indicators).lower():
            function_code += '''    # Moving Average conditions
    buy_conditions.append(df['close'] > df['sma_20'])  # Price above SMA
    sell_conditions.append(df['close'] < df['sma_20'])  # Price below SMA
    
'''
        
        if 'macd' in str(components.indicators).lower():
            function_code += '''    # MACD conditions
    buy_conditions.append(df['macd'] > df['macd_signal'])  # MACD above signal
    sell_conditions.append(df['macd'] < df['macd_signal'])  # MACD below signal
    
'''
        
        function_code += '''    # Combine conditions (all must be true)
    if buy_conditions:
        buy_signal = buy_conditions[0]
        for condition in buy_conditions[1:]:
            buy_signal = buy_signal & condition
        df.loc[buy_signal, 'signal'] = 1
    
    if sell_conditions:
        sell_signal = sell_conditions[0]
        for condition in sell_conditions[1:]:
            sell_signal = sell_signal & condition
        df.loc[sell_signal, 'signal'] = -1
    
    return df[['signal']]
'''
        
        return function_code
    
    def _generate_risk_management_function(self, risk_mgmt: RiskManagement) -> str:
        """Generate risk management function"""
        function_code = f'''
def apply_risk_management(data: pd.DataFrame, signals: pd.DataFrame, 
                         stop_loss_pct: float = {risk_mgmt.stop_loss_value / 100:.4f},
                         take_profit_pct: float = {risk_mgmt.take_profit_value / 100:.4f}) -> pd.DataFrame:
    """
    Apply risk management rules to trading signals
    
    Args:
        data: DataFrame with OHLCV data
        signals: DataFrame with signal column
        stop_loss_pct: Stop loss percentage (default: {risk_mgmt.stop_loss_value}%)
        take_profit_pct: Take profit percentage (default: {risk_mgmt.take_profit_value}%)
        
    Returns:
        DataFrame with risk-adjusted signals and levels
    """
    df = data.copy()
    df['signal'] = signals['signal']
    df['stop_loss'] = 0.0
    df['take_profit'] = 0.0
    df['position_size'] = 0.0
    
    # Calculate stop loss and take profit levels
    for i in range(len(df)):
        if df.iloc[i]['signal'] == 1:  # Buy signal
            entry_price = df.iloc[i]['close']
            df.iloc[i, df.columns.get_loc('stop_loss')] = entry_price * (1 - stop_loss_pct)
            df.iloc[i, df.columns.get_loc('take_profit')] = entry_price * (1 + take_profit_pct)
            
        elif df.iloc[i]['signal'] == -1:  # Sell signal
            entry_price = df.iloc[i]['close']
            df.iloc[i, df.columns.get_loc('stop_loss')] = entry_price * (1 + stop_loss_pct)
            df.iloc[i, df.columns.get_loc('take_profit')] = entry_price * (1 - take_profit_pct)
    
    # Calculate position sizing based on risk management type
    if '{risk_mgmt.position_sizing}' == 'fixed':
        df.loc[df['signal'] != 0, 'position_size'] = 1.0
    elif '{risk_mgmt.position_sizing}' == 'risk_based':
        # Risk-based position sizing
        df['risk_amount'] = {risk_mgmt.max_risk_per_trade / 100:.4f}  # {risk_mgmt.max_risk_per_trade}% of account
        df.loc[df['signal'] != 0, 'position_size'] = df['risk_amount'] / stop_loss_pct
    else:
        df.loc[df['signal'] != 0, 'position_size'] = 1.0
    
    return df[['signal', 'stop_loss', 'take_profit', 'position_size']]
'''
        
        return function_code
    
    def _generate_main_strategy_function(self, components: StrategyComponents) -> str:
        """Generate the main strategy execution function"""
        function_code = f'''
def run_strategy(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute the complete trading strategy
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary containing strategy results
    """
    # Step 1: Calculate indicators
    data_with_indicators = calculate_indicators(data)
    
    # Step 2: Generate signals
    signals = generate_signals(data_with_indicators)
    
    # Step 3: Apply risk management
    final_signals = apply_risk_management(data_with_indicators, signals)
    
    # Combine all data
    result_data = data_with_indicators.copy()
    for col in final_signals.columns:
        result_data[col] = final_signals[col]
    
    # Calculate basic statistics
    total_signals = len(result_data[result_data['signal'] != 0])
    buy_signals = len(result_data[result_data['signal'] == 1])
    sell_signals = len(result_data[result_data['signal'] == -1])
    
    return {{
        'data': result_data,
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'timeframe': '{components.timeframe}',
        'risk_management': {{
            'stop_loss_type': '{components.risk_management.stop_loss_type}',
            'stop_loss_value': {components.risk_management.stop_loss_value},
            'take_profit_type': '{components.risk_management.take_profit_type}',
            'take_profit_value': {components.risk_management.take_profit_value},
            'position_sizing': '{components.risk_management.position_sizing}'
        }}
    }}
'''
        
        return function_code
    
    def _generate_example_usage(self) -> str:
        """Generate example usage code"""
        return '''
# Example usage:
if __name__ == "__main__":
    # Create sample data (replace with your actual data)
    import datetime
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Run the strategy
    results = run_strategy(sample_data)
    
    print(f"Strategy Results:")
    print(f"Total Signals: {results['total_signals']}")
    print(f"Buy Signals: {results['buy_signals']}")
    print(f"Sell Signals: {results['sell_signals']}")
    print(f"Timeframe: {results['timeframe']}")
'''
    
    def generate_strategy_template(self, components: StrategyComponents) -> str:
        """Generate complete strategy code with all required methods"""
        try:
            if not components:
                raise ValueError("Strategy components cannot be None")
            
            if not components.indicators:
                # Provide default indicators if none specified
                components.indicators = ['sma', 'rsi']
            
            # Generate all code sections
            imports = self._generate_imports()
            indicator_functions = self._generate_indicator_functions(components.indicators)
            main_indicator_function = self._generate_main_indicator_function(components)
            signal_function = self._generate_signal_function(components)
            risk_management_function = self._generate_risk_management_function(components.risk_management)
            main_strategy_function = self._generate_main_strategy_function(components)
            example_usage = self._generate_example_usage()
            
            # Combine all sections
            complete_code = f'''"""
Generated Trading Strategy
Generated by AI Strategy Builder v{self.template_version}

Strategy Components:
- Indicators: {', '.join(components.indicators)}
- Timeframe: {components.timeframe}
- Risk Management: {components.risk_management.stop_loss_type} stop loss, {components.risk_management.take_profit_type} take profit
- Position Sizing: {components.risk_management.position_sizing}
"""

{imports}

{indicator_functions}

{main_indicator_function}

{signal_function}

{risk_management_function}

{main_strategy_function}

{example_usage}
'''
            
            return complete_code
            
        except Exception as e:
            raise Exception(f"Failed to generate strategy template: {str(e)}")
    
    def validate_components(self, components: StrategyComponents) -> List[str]:
        """Validate strategy components and return any issues"""
        issues = []
        
        if not components:
            issues.append("Strategy components cannot be None")
            return issues
        
        if not components.indicators:
            issues.append("No indicators specified")
        
        if not components.entry_conditions and not components.exit_conditions:
            issues.append("No entry or exit conditions specified")
        
        if not isinstance(components.risk_management, RiskManagement):
            issues.append("Invalid risk management configuration")
        
        # Validate risk management values
        if components.risk_management.stop_loss_value <= 0:
            issues.append("Stop loss value must be positive")
        
        if components.risk_management.take_profit_value <= 0:
            issues.append("Take profit value must be positive")
        
        if components.risk_management.max_risk_per_trade <= 0 or components.risk_management.max_risk_per_trade > 100:
            issues.append("Max risk per trade must be between 0 and 100 percent")
        
        return issues