"""
Backtesting Engine Module

This module provides comprehensive backtesting functionality for trading strategies,
including data processing, strategy execution, and results generation.
"""

__version__ = "1.0.0"
__author__ = "Trade Analyzer Pro"

# Import main classes for easy access
from .data_processor import DataProcessor
from .backtest_engine import BacktestEngine
from .instrument_manager import InstrumentManager
from .report_generator import ReportGenerator

__all__ = [
    'DataProcessor',
    'BacktestEngine',
    'InstrumentManager',
    'ReportGenerator'
]