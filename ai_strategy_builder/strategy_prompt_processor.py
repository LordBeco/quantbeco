"""
Strategy Prompt Processor

Handles conversion of natural language prompts into executable trading strategy code
using OpenAI GPT API or Puter AI (free alternative).
"""

from typing import Optional, Dict, Any, Tuple
import ast
import re
import time
import random
from openai import OpenAI
from dataclasses import dataclass
from config import Config
from .puter_client import PuterClientWrapper
from .openrouter_client import OpenRouterClientWrapper
from error_handling import (
    ErrorHandler, RetryHandler, ValidationErrorCollector,
    AIStrategyBuilderError, APIError, ErrorDetails, ErrorCategory, ErrorSeverity
)


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class StrategyCode:
    """Generated strategy code with metadata"""
    python_code: str
    pine_script_code: Optional[str] = None
    indicators: list[str] = None
    entry_conditions: list[str] = None
    exit_conditions: list[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []
        if self.entry_conditions is None:
            self.entry_conditions = []
        if self.exit_conditions is None:
            self.exit_conditions = []
        if self.metadata is None:
            self.metadata = {}


class StrategyPromptProcessor:
    """Processes natural language prompts and generates strategy code"""
    
    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None):
        """Initialize with AI provider (OpenAI, Puter, or OpenRouter)"""
        self.provider = provider or Config.AI_PROVIDER
        self.api_key = api_key or (Config.OPENAI_API_KEY if self.provider == 'openai' else Config.OPENROUTER_API_KEY)
        self.model = model  # Allow override of default model
        self.client = None
        self.error_handler = ErrorHandler("strategy_prompt_processor")
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        # Initialize AI client based on provider
        try:
            if self.provider == 'openai':
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
                    self.error_handler.logger.info("OpenAI client initialized successfully")
                else:
                    self.error_handler.logger.warning("No OpenAI API key provided")
                    raise ValueError("OpenAI API key required when using OpenAI provider")
            elif self.provider == 'puter':
                self.client = PuterClientWrapper()
                self.error_handler.logger.info("Puter AI client initialized successfully (free mode)")
            elif self.provider == 'openrouter':
                self.client = OpenRouterClientWrapper(api_key=self.api_key)
                self.error_handler.logger.info("OpenRouter client initialized successfully")
            else:
                raise ValueError(f"Unsupported AI provider: {self.provider}")
                
        except Exception as e:
            error_details = ErrorDetails(
                category=ErrorCategory.CONFIGURATION_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Failed to initialize {self.provider} client",
                technical_details=str(e),
                user_message=f"AI configuration error. Please check your {self.provider} settings.",
                suggestions=[
                    f"Verify your {self.provider} configuration",
                    "Check your internet connection" if self.provider == 'puter' else "Verify your API key is correct",
                    "Try switching AI providers in settings"
                ]
            )
            raise AIStrategyBuilderError(error_details)
    
    def _validate_api_key(self):
        """Validate OpenAI API key by making a test request"""
        if not self.client:
            return
        
        try:
            # Make a minimal test request
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
        except Exception as e:
            error_message = str(e).lower()
            
            if "invalid api key" in error_message or "unauthorized" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.CONFIGURATION_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Invalid OpenAI API key",
                    user_message="Your OpenAI API key is invalid or expired.",
                    suggestions=[
                        "Check your API key in the OpenAI dashboard",
                        "Generate a new API key if needed",
                        "Ensure the key has proper permissions"
                    ]
                )
            elif "quota" in error_message or "billing" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.CONFIGURATION_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="OpenAI API quota exceeded",
                    user_message="Your OpenAI API quota has been exceeded.",
                    suggestions=[
                        "Check your OpenAI billing dashboard",
                        "Add credits to your account",
                        "Wait for quota reset if on free tier"
                    ]
                )
            else:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"OpenAI API validation failed: {str(e)}",
                    user_message="Unable to connect to OpenAI API. Please try again later.",
                    suggestions=[
                        "Check your internet connection",
                        "Try again in a few minutes",
                        "Contact support if the issue persists"
                    ]
                )
            
            raise APIError(error_details)
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize and validate the input prompt"""
        validator = ValidationErrorCollector()
        
        # Check for empty prompt
        if not prompt or not prompt.strip():
            validator.add_error(
                "Prompt cannot be empty",
                suggestions=["Please provide a description of your trading strategy"]
            )
        
        sanitized = prompt.strip()
        
        # Length validation
        if len(sanitized) > 5000:
            validator.add_error(
                "Prompt too long (max 5000 characters)",
                suggestions=[
                    "Shorten your strategy description",
                    "Focus on key trading rules and indicators",
                    "Remove unnecessary details"
                ]
            )
        elif len(sanitized) < 10:
            validator.add_warning(
                "Prompt is very short - consider adding more details",
                suggestions=[
                    "Describe your entry and exit conditions",
                    "Mention specific indicators you want to use",
                    "Include risk management preferences"
                ]
            )
        
        # Check for basic trading strategy keywords
        trading_keywords = [
            'buy', 'sell', 'strategy', 'indicator', 'signal', 'trade', 'price', 
            'moving average', 'rsi', 'macd', 'bollinger', 'entry', 'exit',
            'stop loss', 'take profit', 'trend', 'support', 'resistance'
        ]
        
        if not any(keyword.lower() in sanitized.lower() for keyword in trading_keywords):
            validator.add_error(
                "Prompt should describe a trading strategy",
                suggestions=[
                    "Include trading terms like 'buy', 'sell', 'indicator'",
                    "Describe entry and exit conditions",
                    "Mention specific technical indicators"
                ]
            )
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'import\s+os', r'import\s+sys', r'exec\s*\(', r'eval\s*\(',
            r'__import__', r'subprocess', r'system\s*\(', r'open\s*\('
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                validator.add_error(
                    "Prompt contains potentially harmful code patterns",
                    suggestions=[
                        "Focus on trading logic only",
                        "Avoid system commands or file operations",
                        "Describe strategy in natural language"
                    ]
                )
                break
        
        # Raise errors if any found
        if validator.has_errors():
            error_details = ErrorDetails(
                category=ErrorCategory.VALIDATION_ERROR,
                severity=ErrorSeverity.ERROR,
                message=validator.get_error_summary(),
                user_message="Please fix the following issues with your prompt:",
                suggestions=[suggestion for error in validator.errors for suggestion in error.suggestions]
            )
            raise AIStrategyBuilderError(error_details)
        
        # Log warnings if any
        if validator.has_warnings():
            self.error_handler.logger.warning(validator.get_warning_summary())
        
        return sanitized
    
    def _parse_strategy_components(self, code: str) -> Dict[str, list]:
        """Parse strategy components from generated code with error handling"""
        components = {
            'indicators': [],
            'entry_conditions': [],
            'exit_conditions': []
        }
        
        try:
            # Extract indicators (basic pattern matching)
            indicator_patterns = [
                (r'sma\s*\(', 'Simple Moving Average'),
                (r'ema\s*\(', 'Exponential Moving Average'),
                (r'rsi\s*\(', 'Relative Strength Index'),
                (r'macd\s*\(', 'MACD'),
                (r'bollinger\s*\(', 'Bollinger Bands'),
                (r'moving_average\s*\(', 'Moving Average'),
                (r'\.rolling\s*\(', 'Rolling Window'),
                (r'\.ewm\s*\(', 'Exponentially Weighted Moving'),
                (r'stochastic\s*\(', 'Stochastic Oscillator'),
                (r'atr\s*\(', 'Average True Range')
            ]
            
            for pattern, name in indicator_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    components['indicators'].append(name)
            
            # Extract conditions (basic pattern matching)
            condition_patterns = [
                (r'>\s*', 'Price Above Threshold'),
                (r'<\s*', 'Price Below Threshold'),
                (r'cross', 'Indicator Crossover'),
                (r'breakout', 'Breakout Pattern'),
                (r'support', 'Support Level'),
                (r'resistance', 'Resistance Level')
            ]
            
            for pattern, name in condition_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    if 'entry' in code.lower() or 'buy' in code.lower():
                        components['entry_conditions'].append(name)
                    if 'exit' in code.lower() or 'sell' in code.lower():
                        components['exit_conditions'].append(name)
            
        except Exception as e:
            self.error_handler.logger.warning(f"Error parsing strategy components: {str(e)}")
        
        return components
    
    def _generate_code_with_api(self, prompt: str) -> Tuple[str, str]:
        """Generate Python code using AI API (OpenAI, Puter, or OpenRouter) with comprehensive error handling
        
        Returns:
            tuple: (generated_code, model_used)
        """
        if not self.client:
            error_details = ErrorDetails(
                category=ErrorCategory.CONFIGURATION_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"{self.provider} API client not initialized",
                user_message=f"{self.provider} API is not configured. Please check your settings.",
                suggestions=[
                    f"Configure your {self.provider} settings",
                    "Check the .env file configuration",
                    "Try switching to a different AI provider"
                ]
            )
            raise AIStrategyBuilderError(error_details)
        
        system_prompt = """You are an expert trading strategy developer. Convert the natural language trading strategy description into clean, executable Python code.

Requirements:
1. Use pandas and numpy for calculations
2. Include proper function definitions for signal generation
3. Implement clear entry and exit conditions
4. Include risk management (stop loss, take profit)
5. Use standard technical indicators (TA-Lib style)
6. Return only the Python code, no explanations

The code should follow this structure:
```python
import pandas as pd
import numpy as np

def calculate_indicators(data):
    # Calculate technical indicators
    pass

def generate_signals(data):
    # Generate buy/sell signals
    # Return DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
    pass

def apply_risk_management(data, signals, stop_loss_pct=0.02, take_profit_pct=0.04):
    # Apply stop loss and take profit rules
    pass
```"""
        
        try:
            # Get model and parameters based on provider
            if self.provider == 'openai':
                model = self.model or Config.OPENAI_MODEL
                max_tokens = Config.OPENAI_MAX_TOKENS
                temperature = Config.OPENAI_TEMPERATURE
            elif self.provider == 'puter':
                model = self.model or Config.PUTER_MODEL
                max_tokens = Config.PUTER_MAX_TOKENS
                temperature = Config.PUTER_TEMPERATURE
            else:  # openrouter
                model = self.model or Config.OPENROUTER_MODEL
                max_tokens = Config.OPENROUTER_MAX_TOKENS
                temperature = Config.OPENROUTER_TEMPERATURE
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not response.choices or not response.choices[0].message.content:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"{self.provider} API returned empty response",
                    user_message="The AI service returned an empty response. Please try again.",
                    suggestions=[
                        "Try rephrasing your strategy description",
                        "Make your prompt more specific",
                        "Try again in a few moments"
                    ]
                )
                raise APIError(error_details)
            
            return response.choices[0].message.content.strip(), model
            
        except Exception as e:
            # Handle API errors (works for both providers)
            error_message = str(e).lower()
            
            if "rate limit" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"{self.provider} API rate limit exceeded",
                    user_message="Too many requests. Please wait a moment before trying again.",
                    suggestions=[
                        "Wait 1-2 minutes before retrying",
                        f"Consider upgrading your {self.provider} plan for higher limits" if self.provider == 'openai' else "Try again later"
                    ]
                )
            elif "quota" in error_message or "billing" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"{self.provider} API quota exceeded",
                    user_message=f"Your {self.provider} API quota has been exceeded.",
                    suggestions=[
                        f"Check your {self.provider} billing dashboard" if self.provider == 'openai' else "Try again later",
                        "Add credits to your account" if self.provider == 'openai' else "Switch to OpenAI if you have credits",
                        "Wait for quota reset if on free tier"
                    ]
                )
            elif "timeout" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"{self.provider} API request timed out",
                    user_message="The request timed out. Please try again.",
                    suggestions=[
                        "Check your internet connection",
                        "Try with a shorter prompt",
                        "Retry in a few moments"
                    ]
                )
            elif "connection" in error_message:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Connection error to {self.provider} API",
                    user_message="Unable to connect to the AI service. Please check your connection.",
                    suggestions=[
                        "Check your internet connection",
                        "Try again in a few minutes",
                        "Contact support if the issue persists"
                    ]
                )
            else:
                error_details = ErrorDetails(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"OpenAI API error: {str(e)}",
                    user_message="An error occurred while generating the strategy code.",
                    suggestions=[
                        "Try again with a different prompt",
                        "Check your API configuration",
                        "Contact support if the issue persists"
                    ]
                )
            
            raise APIError(error_details)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from API response with error handling"""
        if not response or not response.strip():
            error_details = ErrorDetails(
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.ERROR,
                message="Empty response from OpenAI API",
                user_message="The AI service returned an empty response. Please try again.",
                suggestions=[
                    "Try rephrasing your strategy description",
                    "Make your prompt more detailed",
                    "Try again in a few moments"
                ]
            )
            raise AIStrategyBuilderError(error_details)
        
        # Remove markdown code blocks
        code_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            extracted_code = matches[0].strip()
            if extracted_code:
                return extracted_code
        
        # Try alternative code block patterns
        alt_patterns = [
            r'```\s*(.*?)\s*```',  # Generic code blocks
            r'`([^`]+)`',  # Inline code
        ]
        
        for pattern in alt_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Take the longest match (likely the main code)
                longest_match = max(matches, key=len).strip()
                if len(longest_match) > 50:  # Reasonable minimum code length
                    return longest_match
        
        # If no code blocks found, try to extract code-like content
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # Start of code detection
            if stripped_line.startswith(('import ', 'def ', 'class ', 'if ', 'for ', 'while ')):
                in_code = True
            
            # Skip explanatory text
            if stripped_line.startswith(('Here', 'This', 'The', 'Note:', 'Remember:')):
                in_code = False
                continue
            
            if in_code and stripped_line:
                code_lines.append(line)
        
        extracted_code = '\n'.join(code_lines) if code_lines else response
        
        # Validate that we have some code-like content
        if not any(keyword in extracted_code for keyword in ['def ', 'import ', '=']):
            error_details = ErrorDetails(
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.ERROR,
                message="No valid Python code found in API response",
                user_message="The AI service didn't generate valid Python code. Please try again.",
                suggestions=[
                    "Try being more specific about your trading strategy",
                    "Mention specific indicators or conditions",
                    "Ask for Python code explicitly in your prompt"
                ]
            )
            raise AIStrategyBuilderError(error_details)
        
        return extracted_code
    
    def _parse_strategy_components(self, code: str) -> Dict[str, list]:
        """Parse strategy components from generated code"""
        components = {
            'indicators': [],
            'entry_conditions': [],
            'exit_conditions': []
        }
        
        # Extract indicators (basic pattern matching)
        indicator_patterns = [
            r'sma\s*\(',
            r'ema\s*\(',
            r'rsi\s*\(',
            r'macd\s*\(',
            r'bollinger\s*\(',
            r'moving_average\s*\(',
            r'\.rolling\s*\(',
            r'\.ewm\s*\('
        ]
        
        for pattern in indicator_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                indicator_name = pattern.replace(r'\s*\(', '').replace('r\'', '').replace('\\', '')
                components['indicators'].append(indicator_name)
        
        # Extract conditions (basic pattern matching)
        if re.search(r'>\s*', code):
            components['entry_conditions'].append('price_above_threshold')
        if re.search(r'<\s*', code):
            components['entry_conditions'].append('price_below_threshold')
        if re.search(r'cross', code, re.IGNORECASE):
            components['entry_conditions'].append('indicator_crossover')
        
        return components
    
    def process_prompt(self, prompt: str) -> StrategyCode:
        """Convert natural language to Python strategy code with comprehensive error handling"""
        try:
            # Sanitize input with detailed validation
            sanitized_prompt = self._sanitize_prompt(prompt)
            
            # Generate code with retry logic and error handling
            def generate_code():
                return self._generate_code_with_api(sanitized_prompt)
            
            raw_response, model_used = self.retry_handler.retry_with_backoff(
                generate_code,
                retryable_exceptions=(APIError, ConnectionError, TimeoutError)
            )
            
            # Extract clean code with validation
            python_code = self._extract_code_from_response(raw_response)
            
            # Validate the generated code
            validation_result = self.validate_strategy(python_code)
            if not validation_result.is_valid:
                # Log detailed validation errors for debugging
                self.error_handler.logger.error(f"Validation failed. Errors: {validation_result.errors}")
                self.error_handler.logger.warning(f"Validation warnings: {validation_result.warnings}")
                
                error_details = ErrorDetails(
                    category=ErrorCategory.STRATEGY_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Generated code failed validation",
                    technical_details=f"Validation errors: {'; '.join(validation_result.errors)}",
                    user_message="The generated strategy code has issues that need to be fixed.",
                    suggestions=[
                        "Try rephrasing your strategy description",
                        "Be more specific about entry and exit conditions",
                        "Check the generated code and fix syntax errors manually"
                    ]
                )
                raise AIStrategyBuilderError(error_details)
            
            # Parse components with error handling
            try:
                components = self._parse_strategy_components(python_code)
            except Exception as e:
                self.error_handler.logger.warning(f"Failed to parse strategy components: {str(e)}")
                components = {'indicators': [], 'entry_conditions': [], 'exit_conditions': []}
            
            # Create strategy code object
            strategy_code = StrategyCode(
                python_code=python_code,
                indicators=components['indicators'],
                entry_conditions=components['entry_conditions'],
                exit_conditions=components['exit_conditions'],
                metadata={
                    'original_prompt': sanitized_prompt,
                    'generated_at': time.time(),
                    'model_used': model_used,  # Use the actual model that was used
                    'provider': self.provider,
                    'validation_warnings': validation_result.warnings
                }
            )
            
            # Log warnings if any
            if validation_result.warnings:
                self.error_handler.logger.warning(f"Strategy validation warnings: {validation_result.warnings}")
            
            return strategy_code
            
        except (AIStrategyBuilderError, APIError) as e:
            # Re-raise our custom errors
            raise e
        except Exception as e:
            # Handle unexpected errors
            error_details = ErrorDetails(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Unexpected error processing prompt: {str(e)}",
                technical_details=str(e),
                user_message="An unexpected error occurred while processing your strategy. Please try again.",
                suggestions=[
                    "Try again with a simpler strategy description",
                    "Check your internet connection",
                    "Contact support if the issue persists"
                ]
            )
            self.error_handler.handle_error(e, {'prompt_length': len(prompt)})
            raise AIStrategyBuilderError(error_details)
        
    def validate_strategy(self, code: str) -> ValidationResult:
        """Validate generated strategy code for syntax and logic with comprehensive checks"""
        validator = ValidationErrorCollector()
        
        # Check if code is not empty
        if not code or not code.strip():
            validator.add_error("Code is empty")
            return ValidationResult(False, [error.message for error in validator.errors], [])
        
        # Syntax validation - be more lenient
        syntax_errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Only treat severe syntax errors as failures
            error_msg = str(e).lower()
            if any(severe in error_msg for severe in ['invalid syntax', 'unexpected eof', 'invalid character']):
                validator.add_error(
                    f"Syntax error: {str(e)}",
                    suggestions=[
                        "Check for missing parentheses or brackets",
                        "Verify proper indentation",
                        "Look for unclosed strings or comments"
                    ]
                )
            else:
                # Treat minor syntax issues as warnings
                validator.add_warning(
                    f"Minor syntax issue: {str(e)}",
                    suggestions=["Code may need minor adjustments for execution"]
                )
        except Exception as e:
            # Only add as warning for parsing errors
            validator.add_warning(
                f"Code parsing warning: {str(e)}",
                suggestions=["Code structure may need review"]
            )
        
        # Check for required functions (make this a warning, not an error)
        required_functions = ['calculate_indicators', 'generate_signals']
        missing_functions = []
        for func_name in required_functions:
            if f'def {func_name}' not in code:
                missing_functions.append(func_name)
        
        # Only warn if both functions are missing (allow alternative structures)
        if len(missing_functions) == len(required_functions):
            # Check if there's at least a main trading function
            if 'def trading_strategy' not in code and 'def ' not in code:
                validator.add_error(
                    "No trading functions found in the code",
                    suggestions=["Add at least one function that processes trading data"]
                )
            else:
                validator.add_warning(
                    f"Missing recommended functions: {', '.join(missing_functions)}",
                    suggestions=[f"Consider adding '{func}' function for better structure" for func in missing_functions]
                )
        
        # Check for imports
        required_imports = ['pandas', 'numpy']
        missing_imports = []
        for imp in required_imports:
            if f'import {imp}' not in code and f'from {imp}' not in code:
                missing_imports.append(imp)
        
        if missing_imports:
            validator.add_warning(
                f"Missing recommended imports: {', '.join(missing_imports)}",
                suggestions=[f"Add 'import {imp}' to your code" for imp in missing_imports]
            )
        
        # Check for basic trading logic
        trading_keywords = ['buy', 'sell', 'signal', 'entry', 'exit', 'long', 'short']
        if not any(keyword in code.lower() for keyword in trading_keywords):
            validator.add_warning(
                "Code may not contain trading logic",
                suggestions=[
                    "Add buy/sell signal generation",
                    "Include entry and exit conditions",
                    "Define trading signals clearly"
                ]
            )
        
        # Check for risk management
        risk_keywords = ['stop_loss', 'take_profit', 'risk', 'position_size']
        if not any(keyword in code.lower() for keyword in risk_keywords):
            validator.add_warning(
                "No risk management detected",
                suggestions=[
                    "Add stop loss logic",
                    "Include take profit conditions",
                    "Consider position sizing rules"
                ]
            )
        
        # Check for potential issues
        potential_issues = [
            (r'while\s+True:', "Infinite loop detected - may cause performance issues"),
            (r'time\.sleep\s*\(', "Sleep statements may not work in backtesting"),
            (r'print\s*\(', "Print statements should be replaced with proper logging"),
            (r'input\s*\(', "Input statements will not work in automated trading"),
        ]
        
        for pattern, warning_msg in potential_issues:
            if re.search(pattern, code):
                validator.add_warning(
                    warning_msg,
                    suggestions=["Review and modify the flagged code section"]
                )
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'exec\s*\(', "Dangerous exec() function detected"),
            (r'eval\s*\(', "Dangerous eval() function detected"),
            (r'__import__', "Dynamic imports detected"),
            (r'open\s*\(.*[\'\"]\w+[\'\"]\s*,\s*[\'\"]\w*w', "File write operations detected"),
        ]
        
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, code):
                validator.add_error(
                    error_msg,
                    suggestions=[
                        "Remove dangerous code patterns",
                        "Focus on trading logic only",
                        "Avoid system operations"
                    ]
                )
        
        # Return validation result
        is_valid = not validator.has_errors()
        return ValidationResult(
            is_valid=is_valid,
            errors=[error.message for error in validator.errors],
            warnings=[warning.message for warning in validator.warnings]
        )
    
    def generate_custom_indicator(self, prompt: str) -> str:
        """Generate custom indicator code from natural language prompt"""
        try:
            # Sanitize the prompt
            sanitized_prompt = self._sanitize_prompt(prompt)
            
            # Enhanced prompt for indicator generation
            indicator_prompt = f"""You are an expert technical indicator developer. Create a custom Python indicator based on this description:

{sanitized_prompt}

Requirements:
1. Use pandas and numpy for calculations
2. Return a function that takes OHLCV data and returns indicator values
3. Include proper parameter validation
4. Add clear comments explaining the logic
5. Use standard technical analysis principles
6. Return only the Python code, no explanations

The code should follow this structure:
```python
import pandas as pd
import numpy as np

def custom_indicator(data, period=14, **kwargs):
    '''
    Custom indicator description
    
    Parameters:
    - data: DataFrame with OHLCV columns
    - period: Calculation period
    - **kwargs: Additional parameters
    
    Returns:
    - Series with indicator values
    '''
    # Indicator calculation logic here
    pass
```"""

            # Generate code using the same API method
            raw_response, model_used = self._generate_code_with_api(indicator_prompt)
            
            # Extract clean code
            indicator_code = self._extract_code_from_response(raw_response)
            
            # Basic validation for indicator code
            if 'def ' not in indicator_code:
                raise ValueError("Generated code doesn't contain a function definition")
            
            return indicator_code
            
        except Exception as e:
            # Return a basic template if generation fails
            return self._generate_basic_indicator_template(str(e))
    
    def _generate_basic_indicator_template(self, error_msg: str = "") -> str:
        """Generate a basic indicator template when generation fails"""
        return f'''import pandas as pd
import numpy as np

def custom_indicator(data, period=14):
    """
    Basic custom indicator template
    
    Parameters:
    - data: DataFrame with OHLCV columns (must have 'close' column)
    - period: Calculation period (default: 14)
    
    Returns:
    - Series with indicator values
    """
    try:
        # Simple moving average as default indicator
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        indicator_values = data['close'].rolling(window=period).mean()
        
        return indicator_values
        
    except Exception as e:
        # Return NaN series if calculation fails
        return pd.Series([np.nan] * len(data), index=data.index)

# Generation note: {error_msg}
'''