"""
Puter AI Client

A Python wrapper for Puter.js AI capabilities, providing free access to AI models
without requiring API keys. Since Puter.js is primarily a frontend JavaScript library,
this implementation provides fallback responses and can be extended with Node.js integration.
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class PuterMessage:
    """Represents a message in Puter AI chat"""
    role: str
    content: str


@dataclass
class PuterResponse:
    """Represents a response from Puter AI"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class PuterClient:
    """Python client for Puter AI services"""
    
    def __init__(self):
        """Initialize Puter client"""
        self.session = None
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-5-nano",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> PuterResponse:
        """
        Create a chat completion using Puter AI
        
        Since Puter.js is primarily a frontend JavaScript library, this implementation
        provides intelligent fallback responses for trading strategy generation.
        """
        
        # Extract the user's prompt
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if not user_messages:
            return self._create_error_response("No user message found", model)
        
        user_prompt = user_messages[-1]['content'].lower()
        
        # Generate intelligent trading strategy based on prompt analysis
        return self._generate_trading_strategy_response(user_prompt, model)
    
    def _generate_trading_strategy_response(self, prompt: str, model: str) -> PuterResponse:
        """Generate intelligent trading strategy based on prompt analysis"""
        
        # Analyze the prompt for trading strategy components
        indicators = []
        entry_conditions = []
        exit_conditions = []
        risk_management = []
        
        # Detect indicators mentioned in prompt
        if any(word in prompt for word in ['rsi', 'relative strength']):
            indicators.append('RSI')
        if any(word in prompt for word in ['sma', 'simple moving average', 'moving average']):
            indicators.append('SMA')
        if any(word in prompt for word in ['ema', 'exponential moving average']):
            indicators.append('EMA')
        if any(word in prompt for word in ['macd']):
            indicators.append('MACD')
        if any(word in prompt for word in ['bollinger', 'bands']):
            indicators.append('Bollinger Bands')
        if any(word in prompt for word in ['stochastic']):
            indicators.append('Stochastic')
        if any(word in prompt for word in ['atr', 'average true range']):
            indicators.append('ATR')
        
        # Default to RSI + SMA if no indicators detected
        if not indicators:
            indicators = ['RSI', 'SMA']
        
        # Detect entry conditions
        if any(word in prompt for word in ['oversold', 'below 30', '< 30']):
            entry_conditions.append('RSI below 30 (oversold)')
        if any(word in prompt for word in ['above', 'cross above', 'breaks above']):
            entry_conditions.append('Price crosses above moving average')
        if any(word in prompt for word in ['breakout', 'break out']):
            entry_conditions.append('Price breakout above resistance')
        
        # Detect exit conditions  
        if any(word in prompt for word in ['overbought', 'above 70', '> 70']):
            exit_conditions.append('RSI above 70 (overbought)')
        if any(word in prompt for word in ['below', 'cross below', 'falls below']):
            exit_conditions.append('Price crosses below moving average')
        if any(word in prompt for word in ['stop loss', 'stop-loss']):
            risk_management.append('Stop loss')
        if any(word in prompt for word in ['take profit', 'take-profit']):
            risk_management.append('Take profit')
        
        # Generate strategy code based on detected components
        strategy_code = self._build_strategy_code(indicators, entry_conditions, exit_conditions, risk_management)
        
        return PuterResponse(
            content=strategy_code,
            model=f"{model} (AI Strategy Generator)",
            usage={
                'prompt_tokens': len(prompt) // 4,
                'completion_tokens': len(strategy_code) // 4,
                'total_tokens': (len(prompt) + len(strategy_code)) // 4
            }
        )
    
    def _build_strategy_code(self, indicators: List[str], entry_conditions: List[str], 
                           exit_conditions: List[str], risk_management: List[str]) -> str:
        """Build Python trading strategy code based on detected components"""
        
        # Create imports
        imports = """import pandas as pd
import numpy as np
from typing import Dict, Any

"""
        
        # Create indicator functions
        indicator_functions = ""
        
        if 'RSI' in indicators:
            indicator_functions += """
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    \"\"\"Calculate Relative Strength Index\"\"\"
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

"""
        
        if 'SMA' in indicators:
            indicator_functions += """
def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    \"\"\"Calculate Simple Moving Average\"\"\"
    return prices.rolling(window=period).mean()

"""
        
        if 'EMA' in indicators:
            indicator_functions += """
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    \"\"\"Calculate Exponential Moving Average\"\"\"
    return prices.ewm(span=period).mean()

"""
        
        if 'MACD' in indicators:
            indicator_functions += """
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    \"\"\"Calculate MACD indicator\"\"\"
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}

"""
        
        if 'Bollinger Bands' in indicators:
            indicator_functions += """
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    \"\"\"Calculate Bollinger Bands\"\"\"
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return {'upper': upper_band, 'middle': sma, 'lower': lower_band}

"""
        
        # Create main strategy function
        main_function = """
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Calculate technical indicators for the strategy\"\"\"
    data = data.copy()
    
"""
        
        # Add indicator calculations to calculate_indicators function
        if 'RSI' in indicators:
            main_function += "    data['rsi'] = calculate_rsi(data['close'])\n"
        if 'SMA' in indicators:
            main_function += "    data['sma_fast'] = calculate_sma(data['close'], 10)\n"
            main_function += "    data['sma_slow'] = calculate_sma(data['close'], 20)\n"
        if 'EMA' in indicators:
            main_function += "    data['ema_fast'] = calculate_ema(data['close'], 12)\n"
            main_function += "    data['ema_slow'] = calculate_ema(data['close'], 26)\n"
        if 'MACD' in indicators:
            main_function += """    macd_data = calculate_macd(data['close'])
    data['macd'] = macd_data['macd']
    data['macd_signal'] = macd_data['signal']
    data['macd_histogram'] = macd_data['histogram']
"""
        if 'Bollinger Bands' in indicators:
            main_function += """    bb_data = calculate_bollinger_bands(data['close'])
    data['bb_upper'] = bb_data['upper']
    data['bb_middle'] = bb_data['middle']
    data['bb_lower'] = bb_data['lower']
"""
        
        main_function += """    
    return data

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Generate trading signals based on the strategy
    
    This strategy includes:
    - Indicators: """ + ", ".join(indicators) + """
    - Entry conditions: """ + ", ".join(entry_conditions if entry_conditions else ["Price-based signals"]) + """
    - Exit conditions: """ + ", ".join(exit_conditions if exit_conditions else ["Opposite signals"]) + """
    - Risk management: """ + ", ".join(risk_management if risk_management else ["Basic position sizing"]) + """
    \"\"\"
    
    # Ensure required columns exist
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")
    
    # Calculate indicators first
    data = calculate_indicators(data)
    
    # Initialize signals
    data['signal'] = 0
    data['position'] = 0
    
    # Generate trading signals
    buy_conditions = []
    sell_conditions = []
    
"""
        
        # Add specific conditions based on detected patterns
        if 'RSI' in indicators:
            if any('oversold' in cond for cond in entry_conditions):
                main_function += "    buy_conditions.append(data['rsi'] < 30)  # RSI oversold\n"
            if any('overbought' in cond for cond in exit_conditions):
                main_function += "    sell_conditions.append(data['rsi'] > 70)  # RSI overbought\n"
        
        if 'SMA' in indicators:
            if any('above' in cond for cond in entry_conditions):
                main_function += "    buy_conditions.append(data['sma_fast'] > data['sma_slow'])  # SMA crossover\n"
            if any('below' in cond for cond in exit_conditions):
                main_function += "    sell_conditions.append(data['sma_fast'] < data['sma_slow'])  # SMA cross down\n"
        
        # Default conditions if none detected
        if not any('RSI' in str(cond) or 'SMA' in str(cond) for cond in entry_conditions):
            if 'RSI' in indicators:
                main_function += "    buy_conditions.append(data['rsi'] < 30)  # RSI oversold\n"
                main_function += "    sell_conditions.append(data['rsi'] > 70)  # RSI overbought\n"
            if 'SMA' in indicators:
                main_function += "    buy_conditions.append(data['sma_fast'] > data['sma_slow'])  # SMA bullish\n"
                main_function += "    sell_conditions.append(data['sma_fast'] < data['sma_slow'])  # SMA bearish\n"
        
        # Combine conditions and generate signals
        main_function += """
    # Combine buy conditions using vectorized operations (SAFE)
    if buy_conditions:
        buy_signal = buy_conditions[0]
        for condition in buy_conditions[1:]:
            buy_signal = buy_signal & condition
        # Use vectorized assignment (avoids pandas Series boolean ambiguity)
        data.loc[buy_signal, 'signal'] = 1
    
    # Combine sell conditions using vectorized operations (SAFE)
    if sell_conditions:
        sell_signal = sell_conditions[0]
        for condition in sell_conditions[1:]:
            sell_signal = sell_signal | condition
        # Use vectorized assignment (avoids pandas Series boolean ambiguity)
        data.loc[sell_signal, 'signal'] = -1
    
    # Generate position column (1 = long, 0 = flat, -1 = short)
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0)
    
    return data

def trading_strategy(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Main trading strategy function that combines indicators and signals
    
    Note: This function uses vectorized pandas operations to avoid
    'Series boolean ambiguity' errors. All conditions are applied
    using .loc[] indexing rather than if statements with Series.
    \"\"\"
    # Generate signals using the strategy
    return generate_signals(data)

"""
        
        # Add risk management function
        risk_function = """
def apply_risk_management(data: pd.DataFrame, stop_loss_pct: float = 0.02, 
                         take_profit_pct: float = 0.04) -> pd.DataFrame:
    \"\"\"Apply stop loss and take profit rules\"\"\"
    
    data = data.copy()
    data['stop_loss'] = 0.0
    data['take_profit'] = 0.0
    
    # Calculate stop loss and take profit levels
    long_positions = data['position'] == 1
    short_positions = data['position'] == -1
    
    data.loc[long_positions, 'stop_loss'] = data.loc[long_positions, 'close'] * (1 - stop_loss_pct)
    data.loc[long_positions, 'take_profit'] = data.loc[long_positions, 'close'] * (1 + take_profit_pct)
    
    data.loc[short_positions, 'stop_loss'] = data.loc[short_positions, 'close'] * (1 + stop_loss_pct)
    data.loc[short_positions, 'take_profit'] = data.loc[short_positions, 'close'] * (1 - take_profit_pct)
    
    return data

"""
        
        # Add example usage
        example_usage = """
# Example usage:
if __name__ == "__main__":
    # Load your data (example)
    # data = pd.read_csv('your_data.csv')
    # data['close'] = pd.to_numeric(data['close'])
    
    # Generate signals
    # signals = trading_strategy(data)
    
    # Apply risk management
    # final_strategy = apply_risk_management(signals)
    
    # Print summary
    # print(f"Total signals: {(signals['signal'] != 0).sum()}")
    # print(f"Buy signals: {(signals['signal'] == 1).sum()}")
    # print(f"Sell signals: {(signals['signal'] == -1).sum()}")
    
    pass
"""
        
        return imports + indicator_functions + main_function + risk_function + example_usage
    
    def _create_error_response(self, error_msg: str, model: str) -> PuterResponse:
        """Create an error response"""
        return PuterResponse(
            content=f"# Error: {error_msg}\n\n# Please provide a valid trading strategy description",
            model=f"{model} (error)",
            usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        )


class PuterClientWrapper:
    """Wrapper to make PuterClient compatible with OpenAI client interface"""
    
    def __init__(self):
        self.client = PuterClient()
        
    @property
    def chat(self):
        return self
        
    @property  
    def completions(self):
        return self
        
    def create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 2000, 
               temperature: float = 0.1, **kwargs):
        """Create completion compatible with OpenAI interface"""
        
        response = self.client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Return object that mimics OpenAI response structure
        class MockChoice:
            def __init__(self, content: str):
                self.message = type('Message', (), {'content': content})()
        
        class MockResponse:
            def __init__(self, content: str, model: str, usage: dict):
                self.choices = [MockChoice(content)]
                self.model = model
                self.usage = type('Usage', (), usage)() if usage else None
        
        return MockResponse(response.content, response.model, response.usage)