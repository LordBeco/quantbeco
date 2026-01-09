"""
AI Strategy Builder Module

This module provides functionality for converting natural language trading strategy
descriptions into executable Python code and optionally to Pine Script format.
"""

__version__ = "1.0.0"
__author__ = "Trade Analyzer Pro"

# Import main classes for easy access
from .strategy_prompt_processor import StrategyPromptProcessor
from .pine_script_converter import PineScriptConverter
from .code_generator import CodeGenerator

__all__ = [
    'StrategyPromptProcessor',
    'PineScriptConverter', 
    'CodeGenerator'
]