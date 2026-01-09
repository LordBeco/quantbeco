"""
Configuration Module

Handles environment variables and API configurations for the trade analyzer pro application.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration management"""
    
    # AI Provider Configuration
    AI_PROVIDER: str = os.getenv('AI_PROVIDER', 'puter').lower()
    
    # OpenAI API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_MAX_TOKENS: int = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
    OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # Puter Configuration (free alternative)
    PUTER_MODEL: str = os.getenv('PUTER_MODEL', 'gpt-5-nano')
    PUTER_MAX_TOKENS: int = int(os.getenv('PUTER_MAX_TOKENS', '2000'))
    PUTER_TEMPERATURE: float = float(os.getenv('PUTER_TEMPERATURE', '0.1'))
    
    # OpenRouter Configuration (free models available)
    OPENROUTER_API_KEY: Optional[str] = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_MODEL: str = os.getenv('OPENROUTER_MODEL', 'nex-agi/deepseek-v3.1-nex-n1:free')
    OPENROUTER_MAX_TOKENS: int = int(os.getenv('OPENROUTER_MAX_TOKENS', '2000'))
    OPENROUTER_TEMPERATURE: float = float(os.getenv('OPENROUTER_TEMPERATURE', '0.1'))
    OPENROUTER_SITE_URL: str = os.getenv('OPENROUTER_SITE_URL', 'http://localhost:8507')
    OPENROUTER_SITE_NAME: str = os.getenv('OPENROUTER_SITE_NAME', 'Trade Analyzer Pro')
    
    # Backtesting Configuration
    DEFAULT_STARTING_BALANCE: float = float(os.getenv('DEFAULT_STARTING_BALANCE', '10000'))
    DEFAULT_LEVERAGE: float = float(os.getenv('DEFAULT_LEVERAGE', '1.0'))
    DEFAULT_COMMISSION: float = float(os.getenv('DEFAULT_COMMISSION', '0.0'))
    DEFAULT_SLIPPAGE: float = float(os.getenv('DEFAULT_SLIPPAGE', '0.0'))
    
    # Data Processing Configuration
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv('MAX_UPLOAD_SIZE_MB', '2048'))
    DEFAULT_TIMEZONE: str = os.getenv('DEFAULT_TIMEZONE', 'UTC')
    
    @classmethod
    def validate_ai_config(cls) -> bool:
        """Validate AI configuration based on selected provider"""
        if cls.AI_PROVIDER == 'openai':
            return cls.OPENAI_API_KEY is not None and len(cls.OPENAI_API_KEY.strip()) > 0
        elif cls.AI_PROVIDER == 'puter':
            return True  # Puter doesn't require API key
        elif cls.AI_PROVIDER == 'openrouter':
            return True  # OpenRouter has free models that don't require API key
        return False
    
    @classmethod
    def validate_openai_config(cls) -> bool:
        """Validate OpenAI API configuration (legacy method for compatibility)"""
        return cls.validate_ai_config()
    
    @classmethod
    def get_ai_config(cls) -> dict:
        """Get AI configuration as dictionary"""
        if cls.AI_PROVIDER == 'openai':
            return {
                'provider': 'openai',
                'api_key': cls.OPENAI_API_KEY,
                'model': cls.OPENAI_MODEL,
                'max_tokens': cls.OPENAI_MAX_TOKENS,
                'temperature': cls.OPENAI_TEMPERATURE
            }
        elif cls.AI_PROVIDER == 'puter':
            return {
                'provider': 'puter',
                'model': cls.PUTER_MODEL,
                'max_tokens': cls.PUTER_MAX_TOKENS,
                'temperature': cls.PUTER_TEMPERATURE
            }
        elif cls.AI_PROVIDER == 'openrouter':
            return {
                'provider': 'openrouter',
                'api_key': cls.OPENROUTER_API_KEY,
                'model': cls.OPENROUTER_MODEL,
                'max_tokens': cls.OPENROUTER_MAX_TOKENS,
                'temperature': cls.OPENROUTER_TEMPERATURE,
                'site_url': cls.OPENROUTER_SITE_URL,
                'site_name': cls.OPENROUTER_SITE_NAME
            }
        return {}
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration as dictionary (legacy method for compatibility)"""
        return cls.get_ai_config()