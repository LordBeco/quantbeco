"""
Configuration Module

Handles environment variables and API configurations for the trade analyzer pro application.
Supports both local development (.env) and production deployment (Streamlit secrets).
"""

import os
from typing import Optional

# Try to import streamlit for secrets management
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Load environment variables from .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Config:
    """Application configuration management with support for local and production environments"""
    
    @staticmethod
    def _get_config_value(key: str, default: str = None) -> Optional[str]:
        """
        Get configuration value from environment or Streamlit secrets
        
        Priority:
        1. Environment variable (for local development)
        2. Streamlit secrets (for production deployment)
        3. Default value
        """
        # First try environment variables (local development)
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        
        # Then try Streamlit secrets (production deployment)
        if STREAMLIT_AVAILABLE:
            try:
                import streamlit as st
                return st.secrets.get(key, default)
            except (FileNotFoundError, KeyError, AttributeError):
                # secrets.toml not found or key not in secrets
                pass
        
        return default
    
    @staticmethod
    def is_production() -> bool:
        """Check if running in production environment"""
        # Check for common production environment indicators
        return (
            os.getenv('ENVIRONMENT') == 'production' or
            os.getenv('STREAMLIT_SHARING') == 'true' or
            os.getenv('HEROKU') is not None or
            Config._check_streamlit_production()
        )
    
    @staticmethod
    def _check_streamlit_production() -> bool:
        """Check if Streamlit secrets indicate production"""
        if STREAMLIT_AVAILABLE:
            try:
                import streamlit as st
                return hasattr(st, 'secrets') and 'ENVIRONMENT' in st.secrets and st.secrets['ENVIRONMENT'] == 'production'
            except:
                return False
        return False
    
    @staticmethod
    def get_environment() -> str:
        """Get current environment (local/production)"""
        return 'production' if Config.is_production() else 'local'
    
    @classmethod
    def validate_ai_config(cls) -> bool:
        """Validate AI configuration based on selected provider"""
        if cls.AI_PROVIDER == 'openai':
            return cls.OPENAI_API_KEY is not None and len(cls.OPENAI_API_KEY.strip()) > 0
        elif cls.AI_PROVIDER == 'puter':
            return True  # Puter doesn't require API key
        elif cls.AI_PROVIDER == 'openrouter':
            return cls.OPENROUTER_API_KEY is not None and len(cls.OPENROUTER_API_KEY.strip()) > 0
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


# Initialize configuration values
def _init_config():
    """Initialize configuration values"""
    # AI Provider Configuration
    Config.AI_PROVIDER = Config._get_config_value('AI_PROVIDER', 'puter').lower()
    
    # OpenAI API Configuration
    Config.OPENAI_API_KEY = Config._get_config_value('OPENAI_API_KEY')
    Config.OPENAI_MODEL = Config._get_config_value('OPENAI_MODEL', 'gpt-4')
    Config.OPENAI_MAX_TOKENS = int(Config._get_config_value('OPENAI_MAX_TOKENS', '2000'))
    Config.OPENAI_TEMPERATURE = float(Config._get_config_value('OPENAI_TEMPERATURE', '0.1'))
    
    # Puter Configuration (free alternative)
    Config.PUTER_MODEL = Config._get_config_value('PUTER_MODEL', 'gpt-5-nano')
    Config.PUTER_MAX_TOKENS = int(Config._get_config_value('PUTER_MAX_TOKENS', '2000'))
    Config.PUTER_TEMPERATURE = float(Config._get_config_value('PUTER_TEMPERATURE', '0.1'))
    
    # OpenRouter Configuration (free models available)
    Config.OPENROUTER_API_KEY = Config._get_config_value('OPENROUTER_API_KEY')
    Config.OPENROUTER_MODEL = Config._get_config_value('OPENROUTER_MODEL', 'nex-agi/deepseek-v3.1-nex-n1:free')
    Config.OPENROUTER_MAX_TOKENS = int(Config._get_config_value('OPENROUTER_MAX_TOKENS', '2000'))
    Config.OPENROUTER_TEMPERATURE = float(Config._get_config_value('OPENROUTER_TEMPERATURE', '0.1'))
    Config.OPENROUTER_SITE_URL = Config._get_config_value('OPENROUTER_SITE_URL', 'http://localhost:8507')
    Config.OPENROUTER_SITE_NAME = Config._get_config_value('OPENROUTER_SITE_NAME', 'Trade Analyzer Pro')
    
    # Backtesting Configuration
    Config.DEFAULT_STARTING_BALANCE = float(Config._get_config_value('DEFAULT_STARTING_BALANCE', '10000'))
    Config.DEFAULT_LEVERAGE = float(Config._get_config_value('DEFAULT_LEVERAGE', '1.0'))
    Config.DEFAULT_COMMISSION = float(Config._get_config_value('DEFAULT_COMMISSION', '0.0'))
    Config.DEFAULT_SLIPPAGE = float(Config._get_config_value('DEFAULT_SLIPPAGE', '0.0'))
    
    # Data Processing Configuration
    Config.MAX_UPLOAD_SIZE_MB = int(Config._get_config_value('MAX_UPLOAD_SIZE_MB', '2048'))
    Config.DEFAULT_TIMEZONE = Config._get_config_value('DEFAULT_TIMEZONE', 'UTC')

# Initialize configuration when module is imported
_init_config()