"""
Pine Script Converter

Converts Python strategy code to Pine Script format for TradingView compatibility.
Generates proper Pine Script v5 syntax.
"""

import re
from typing import Optional, Dict, List, Tuple
from .strategy_prompt_processor import ValidationResult


class PineScriptConverter:
    """Converts Python strategy code to Pine Script v5"""
    
    def __init__(self):
        """Initialize the Pine Script converter"""
        self.pine_version = "5"
    
    def _analyze_python_strategy(self, python_code: str) -> Dict[str, any]:
        """Analyze Python code to extract strategy components"""
        analysis = {
            'indicators': [],
            'conditions': [],
            'signals': {'long': [], 'short': []},
            'risk_management': {'stop_loss': 2.0, 'take_profit': 4.0}
        }
        
        # Detect indicators
        if 'rsi' in python_code.lower():
            analysis['indicators'].append('RSI')
        if 'sma' in python_code.lower() or 'moving' in python_code.lower():
            analysis['indicators'].append('SMA')
        if 'ema' in python_code.lower():
            analysis['indicators'].append('EMA')
        if 'macd' in python_code.lower():
            analysis['indicators'].append('MACD')
        if 'bollinger' in python_code.lower():
            analysis['indicators'].append('BB')
        if 'stochastic' in python_code.lower():
            analysis['indicators'].append('STOCH')
        
        # Detect conditions
        if 'rsi' in python_code.lower() and ('< 30' in python_code or 'oversold' in python_code.lower()):
            analysis['conditions'].append('rsi_oversold')
        if 'rsi' in python_code.lower() and ('> 70' in python_code or 'overbought' in python_code.lower()):
            analysis['conditions'].append('rsi_overbought')
        if 'cross' in python_code.lower():
            analysis['conditions'].append('ma_crossover')
        if 'breakout' in python_code.lower():
            analysis['conditions'].append('breakout')
        
        # Extract risk management values
        stop_loss_match = re.search(r'stop_loss.*?(\d+\.?\d*)', python_code, re.IGNORECASE)
        if stop_loss_match:
            analysis['risk_management']['stop_loss'] = float(stop_loss_match.group(1))
        
        take_profit_match = re.search(r'take_profit.*?(\d+\.?\d*)', python_code, re.IGNORECASE)
        if take_profit_match:
            analysis['risk_management']['take_profit'] = float(take_profit_match.group(1))
        
        return analysis
    
    def _generate_pine_header(self, strategy_name: str = "AI Generated Strategy") -> str:
        """Generate Pine Script v5 header"""
        return f'''//@version={self.pine_version}
strategy("{strategy_name}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Input parameters
stop_loss_pct = input.float(2.0, title="Stop Loss %", minval=0.1, maxval=10.0) / 100
take_profit_pct = input.float(4.0, title="Take Profit %", minval=0.1, maxval=20.0) / 100
rsi_period = input.int(14, title="RSI Period", minval=1, maxval=50)
sma_fast = input.int(10, title="Fast SMA Period", minval=1, maxval=100)
sma_slow = input.int(20, title="Slow SMA Period", minval=1, maxval=200)
ema_fast = input.int(12, title="Fast EMA Period", minval=1, maxval=100)
ema_slow = input.int(26, title="Slow EMA Period", minval=1, maxval=200)

'''
    
    def _generate_indicators(self, indicators: List[str]) -> str:
        """Generate Pine Script indicator calculations"""
        code = "// Technical Indicators\n"
        
        if 'RSI' in indicators:
            code += "rsi = ta.rsi(close, rsi_period)\n"
        
        if 'SMA' in indicators:
            code += "sma_fast_line = ta.sma(close, sma_fast)\n"
            code += "sma_slow_line = ta.sma(close, sma_slow)\n"
        
        if 'EMA' in indicators:
            code += "ema_fast_line = ta.ema(close, ema_fast)\n"
            code += "ema_slow_line = ta.ema(close, ema_slow)\n"
        
        if 'MACD' in indicators:
            code += "ema12 = ta.ema(close, 12)\n"
            code += "ema26 = ta.ema(close, 26)\n"
            code += "macdLine = ema12 - ema26\n"
            code += "signalLine = ta.ema(macdLine, 9)\n"
            code += "histogram = macdLine - signalLine\n"
        
        if 'BB' in indicators:
            code += "bb_basis = ta.sma(close, 20)\n"
            code += "bb_dev = ta.stdev(close, 20)\n"
            code += "bb_upper = bb_basis + bb_dev * 2\n"
            code += "bb_lower = bb_basis - bb_dev * 2\n"
        
        if 'STOCH' in indicators:
            code += "k = ta.stoch(close, high, low, 14)\n"
            code += "d = ta.sma(k, 3)\n"
        
        return code + "\n"
    
    def _generate_conditions(self, analysis: Dict[str, any]) -> Tuple[str, str]:
        """Generate Pine Script conditions for long and short entries"""
        conditions = analysis['conditions']
        indicators = analysis['indicators']
        
        long_conditions = []
        short_conditions = []
        
        # RSI conditions
        if 'rsi_oversold' in conditions and 'RSI' in indicators:
            long_conditions.append("rsi < 30")
        if 'rsi_overbought' in conditions and 'RSI' in indicators:
            short_conditions.append("rsi > 70")
        
        # Moving average crossover
        if 'ma_crossover' in conditions:
            if 'SMA' in indicators:
                long_conditions.append("ta.crossover(sma_fast_line, sma_slow_line)")
                short_conditions.append("ta.crossunder(sma_fast_line, sma_slow_line)")
            elif 'EMA' in indicators:
                long_conditions.append("ta.crossover(ema_fast_line, ema_slow_line)")
                short_conditions.append("ta.crossunder(ema_fast_line, ema_slow_line)")
        
        # MACD conditions
        if 'MACD' in indicators:
            long_conditions.append("ta.crossover(macdLine, signalLine)")
            short_conditions.append("ta.crossunder(macdLine, signalLine)")
        
        # Bollinger Bands conditions
        if 'BB' in indicators:
            long_conditions.append("close < bb_lower")
            short_conditions.append("close > bb_upper")
        
        # Default conditions if none detected
        if not long_conditions and not short_conditions:
            if 'RSI' in indicators:
                long_conditions.append("rsi < 30")
                short_conditions.append("rsi > 70")
            elif 'SMA' in indicators:
                long_conditions.append("close > sma_fast_line and sma_fast_line > sma_slow_line")
                short_conditions.append("close < sma_fast_line and sma_fast_line < sma_slow_line")
            else:
                # Very basic price action
                long_conditions.append("close > open")
                short_conditions.append("close < open")
        
        long_condition = " and ".join(long_conditions) if long_conditions else "false"
        short_condition = " and ".join(short_conditions) if short_conditions else "false"
        
        return long_condition, short_condition
    
    def _generate_strategy_logic(self, analysis: Dict[str, any]) -> str:
        """Generate Pine Script strategy execution logic"""
        long_condition, short_condition = self._generate_conditions(analysis)
        
        code = f'''// Signal Generation
long_condition = {long_condition}
short_condition = {short_condition}

// Strategy Execution
if long_condition
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close * (1 - stop_loss_pct), limit=close * (1 + take_profit_pct))

if short_condition
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close * (1 + stop_loss_pct), limit=close * (1 - take_profit_pct))

// Plot signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.labelup, text="BUY", size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.labeldown, text="SELL", size=size.small)
'''
        
        return code
    
    def _generate_plots(self, indicators: List[str]) -> str:
        """Generate Pine Script plot statements"""
        code = "\n// Optional: Plot indicators for reference\n"
        
        if 'SMA' in indicators:
            code += "plot(sma_fast_line, color=color.blue, title=\"Fast SMA\", linewidth=1)\n"
            code += "plot(sma_slow_line, color=color.red, title=\"Slow SMA\", linewidth=1)\n"
        
        if 'EMA' in indicators:
            code += "plot(ema_fast_line, color=color.blue, title=\"Fast EMA\", linewidth=1)\n"
            code += "plot(ema_slow_line, color=color.red, title=\"Slow EMA\", linewidth=1)\n"
        
        if 'BB' in indicators:
            code += "plot(bb_upper, color=color.gray, title=\"BB Upper\")\n"
            code += "plot(bb_basis, color=color.orange, title=\"BB Basis\")\n"
            code += "plot(bb_lower, color=color.gray, title=\"BB Lower\")\n"
        
        return code
    
    def convert_to_pine(self, python_code: str) -> str:
        """Convert Python strategy to Pine Script v5"""
        try:
            if not python_code or not python_code.strip():
                raise ValueError("Python code cannot be empty")
            
            # Analyze the Python strategy
            analysis = self._analyze_python_strategy(python_code)
            
            # Generate Pine Script components
            header = self._generate_pine_header()
            indicators_code = self._generate_indicators(analysis['indicators'])
            strategy_logic = self._generate_strategy_logic(analysis)
            plots_code = self._generate_plots(analysis['indicators'])
            
            # Combine all parts
            pine_script = f"{header}{indicators_code}{strategy_logic}{plots_code}"
            
            return pine_script
            
        except Exception as e:
            # Return a basic template if conversion fails
            return self._generate_basic_template(str(e))
    
    def convert_to_pine_advanced(self, python_code: str, options: Dict[str, any] = None) -> str:
        """Advanced Pine Script conversion with options"""
        try:
            if options is None:
                options = {}
            
            # Use AI refinement if requested
            if options.get('use_ai_refinement', False):
                ai_client = options.get('ai_client')
                provider = options.get('provider', 'puter')
                return self.convert_to_pine_with_ai_refinement(python_code, ai_client, provider)
            else:
                return self.convert_to_pine(python_code)
                
        except Exception as e:
            return self._generate_basic_template(f"Advanced conversion failed: {str(e)}")
    
    def convert_to_pine_with_ai_refinement(self, python_code: str, ai_client=None, provider: str = "puter", options: Dict[str, any] = None) -> str:
        """Convert Python strategy to Pine Script v5 with AI refinement"""
        try:
            # First, do the basic conversion with options
            if options:
                initial_pine = self.convert_to_pine_advanced(python_code, options)
            else:
                initial_pine = self.convert_to_pine(python_code)
            
            # If no AI client provided, return basic conversion
            if not ai_client:
                return initial_pine
            
            # Use AI to refine the Pine Script
            refined_pine = self._refine_pine_with_ai(initial_pine, python_code, ai_client, provider)
            
            return refined_pine
            
        except Exception as e:
            # Fallback to basic conversion if AI refinement fails
            if options:
                return self.convert_to_pine_advanced(python_code, options)
            else:
                return self.convert_to_pine(python_code)
    
    def _refine_pine_with_ai(self, initial_pine: str, original_python: str, ai_client, provider: str) -> str:
        """Use AI to refine and improve the Pine Script code"""
        try:
            refinement_prompt = f"""You are a Pine Script v5 expert. Please review and improve this Pine Script code.

ORIGINAL PYTHON STRATEGY:
```python
{original_python[:1000]}...
```

GENERATED PINE SCRIPT:
```pinescript
{initial_pine}
```

Please improve the Pine Script by:
1. Fixing any syntax errors
2. Optimizing the logic for better performance
3. Adding proper error handling
4. Ensuring all conditions are correctly translated
5. Adding helpful comments
6. Making sure it follows Pine Script v5 best practices

Return ONLY the improved Pine Script code, no explanations:"""

            # Get model parameters based on provider
            if provider == 'openai':
                from config import Config
                model = Config.OPENAI_MODEL
                max_tokens = min(Config.OPENAI_MAX_TOKENS, 3000)  # Pine Script shouldn't be too long
                temperature = 0.1  # Low temperature for code generation
            elif provider == 'openrouter':
                from config import Config
                model = Config.OPENROUTER_MODEL
                max_tokens = min(Config.OPENROUTER_MAX_TOKENS, 3000)
                temperature = 0.1
            else:  # puter or fallback
                model = "gpt-5-nano"
                max_tokens = 2000
                temperature = 0.1
            
            # Make AI request
            response = ai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Pine Script v5 expert. Generate clean, optimized Pine Script code."},
                    {"role": "user", "content": refinement_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.choices and response.choices[0].message.content:
                refined_code = response.choices[0].message.content.strip()
                
                # Clean up the response (remove markdown formatting if present)
                refined_code = re.sub(r'```pinescript\n?', '', refined_code)
                refined_code = re.sub(r'```\n?', '', refined_code)
                refined_code = refined_code.strip()
                
                # Validate the refined code
                validation = self.validate_pine_script(refined_code)
                
                if validation.is_valid or len(validation.errors) <= 2:  # Allow minor errors
                    return refined_code
                else:
                    # If AI refinement introduced errors, return original
                    return initial_pine
            else:
                return initial_pine
                
        except Exception as e:
            # If AI refinement fails, return the initial conversion
            return initial_pine
    
    def _generate_basic_template(self, error_msg: str = "") -> str:
        """Generate a basic Pine Script template when conversion fails"""
        return f'''//@version=5
strategy("Basic AI Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Input parameters
stop_loss_pct = input.float(2.0, title="Stop Loss %", minval=0.1, maxval=10.0) / 100
take_profit_pct = input.float(4.0, title="Take Profit %", minval=0.1, maxval=20.0) / 100
rsi_period = input.int(14, title="RSI Period", minval=1, maxval=50)

// Technical Indicators
rsi = ta.rsi(close, rsi_period)
sma_fast = ta.sma(close, 10)
sma_slow = ta.sma(close, 20)

// Signal Generation
long_condition = ta.crossover(sma_fast, sma_slow) and rsi < 70
short_condition = ta.crossunder(sma_fast, sma_slow) and rsi > 30

// Strategy Execution
if long_condition
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close * (1 - stop_loss_pct), limit=close * (1 + take_profit_pct))

if short_condition
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close * (1 + stop_loss_pct), limit=close * (1 - take_profit_pct))

// Plot signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.labelup, text="BUY", size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.labeldown, text="SELL", size=size.small)

// Plot moving averages
plot(sma_fast, color=color.blue, title="Fast SMA", linewidth=1)
plot(sma_slow, color=color.red, title="Slow SMA", linewidth=1)

// Conversion note: {error_msg}
'''
    
    def validate_pine_script(self, pine_code: str) -> ValidationResult:
        """Validate Pine Script syntax and structure"""
        errors = []
        warnings = []
        
        try:
            if not pine_code or not pine_code.strip():
                errors.append("Pine Script code is empty")
                return ValidationResult(False, errors, warnings)
            
            # Check for Pine Script v5 requirements
            if not re.search(r'//@version=5', pine_code):
                errors.append("Missing Pine Script version 5 declaration")
            
            if 'strategy(' not in pine_code and 'indicator(' not in pine_code:
                errors.append("Missing strategy() or indicator() declaration")
            
            # Check for Python syntax that shouldn't be there
            python_issues = [
                (r'import\s+', "Python import statements found"),
                (r'def\s+\w+\(', "Python function definitions found"),
                (r'\.rolling\(', "Pandas rolling operations found"),
                (r'pd\.', "Pandas references found"),
                (r'np\.', "NumPy references found"),
                (r'data\[\'', "Python DataFrame syntax found"),
                (r'\.loc\[', "Pandas .loc syntax found"),
                (r'\.iloc\[', "Pandas .iloc syntax found")
            ]
            
            for pattern, message in python_issues:
                if re.search(pattern, pine_code):
                    errors.append(message)
            
            # Check for proper Pine Script syntax
            if 'ta.' not in pine_code and ('rsi' in pine_code.lower() or 'sma' in pine_code.lower() or 'ema' in pine_code.lower()):
                warnings.append("Consider using Pine Script built-in technical analysis functions (ta.*)")
            
            # Check for strategy execution
            if 'strategy.entry' not in pine_code:
                warnings.append("No strategy entry calls found")
            
            if 'strategy.exit' not in pine_code:
                warnings.append("No strategy exit calls found")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
            
            for pattern, message in python_issues:
                if re.search(pattern, pine_code):
                    errors.append(message)
            
            # Check for proper Pine Script syntax
            if 'ta.' not in pine_code and ('rsi' in pine_code.lower() or 'sma' in pine_code.lower() or 'ema' in pine_code.lower()):
                warnings.append("Consider using Pine Script built-in technical analysis functions (ta.*)")
            
            # Check for strategy execution
            if 'strategy.entry' not in pine_code:
                warnings.append("No strategy entry calls found")
            
            if 'strategy.exit' not in pine_code:
                warnings.append("No strategy exit calls found")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def ai_refine_pine_script(self, pine_code: str, ai_client=None, provider: str = "puter") -> str:
        """Use AI to refine existing Pine Script code"""
        try:
            if not ai_client:
                return pine_code  # Return original if no AI client
            
            refinement_prompt = f"""You are a Pine Script v5 expert. Please review and improve this Pine Script code.

PINE SCRIPT CODE:
```pinescript
{pine_code}
```

Please improve the Pine Script by:
1. Fixing any syntax errors
2. Optimizing the logic for better performance
3. Adding proper error handling
4. Improving code readability and comments
5. Ensuring it follows Pine Script v5 best practices
6. Adding useful features if appropriate

Return ONLY the improved Pine Script code, no explanations:"""

            # Get model parameters based on provider
            if provider == 'openai':
                from config import Config
                model = Config.OPENAI_MODEL
                max_tokens = min(Config.OPENAI_MAX_TOKENS, 3000)
                temperature = 0.1
            elif provider == 'openrouter':
                from config import Config
                model = Config.OPENROUTER_MODEL
                max_tokens = min(Config.OPENROUTER_MAX_TOKENS, 3000)
                temperature = 0.1
            else:  # puter or fallback
                model = "gpt-5-nano"
                max_tokens = 2000
                temperature = 0.1
            
            # Make AI request
            response = ai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Pine Script v5 expert. Generate clean, optimized Pine Script code."},
                    {"role": "user", "content": refinement_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.choices and response.choices[0].message.content:
                refined_code = response.choices[0].message.content.strip()
                
                # Clean up the response (remove markdown formatting if present)
                refined_code = re.sub(r'```pinescript\n?', '', refined_code)
                refined_code = re.sub(r'```\n?', '', refined_code)
                refined_code = refined_code.strip()
                
                # Validate the refined code
                validation = self.validate_pine_script(refined_code)
                
                if validation.is_valid or len(validation.errors) <= 2:  # Allow minor errors
                    return refined_code
                else:
                    # If AI refinement introduced errors, return original
                    return pine_code
            else:
                return pine_code
                
        except Exception as e:
            # If AI refinement fails, return the original code
            return pine_code