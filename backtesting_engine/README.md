# Backtesting Engine Module

This module provides comprehensive backtesting functionality for trading strategies, including data processing, strategy execution using VectorBT, and results generation.

## Components

- **DataProcessor**: Handles tick data upload, validation, and preprocessing
- **BacktestEngine**: Core backtesting execution using VectorBT for high-performance testing
- **InstrumentManager**: Manages instrument specifications and pip calculations
- **ReportGenerator**: Generates broker statements and performance reports compatible with trade_analyzer_pro

## Features

- High-performance backtesting using VectorBT
- Support for various data formats (CSV with OHLCV columns)
- Realistic slippage and spread calculations
- Comprehensive performance metrics
- Interactive chart generation with Plotly
- Broker statement generation compatible with existing analytics

## Configuration

Default configuration can be customized via environment variables:

```bash
export DEFAULT_STARTING_BALANCE=10000
export DEFAULT_LEVERAGE=1.0
export DEFAULT_COMMISSION=0.0
export DEFAULT_SLIPPAGE=0.0
```

## Usage

```python
from backtesting_engine import BacktestEngine, DataProcessor

# Process data
processor = DataProcessor()
validated_data = processor.validate_tick_data(your_data)

# Run backtest
engine = BacktestEngine()
results = engine.execute_backtest(strategy, data, config)
```

## Implementation Status

- ✅ Module structure created
- ⏳ Core functionality (to be implemented in subsequent tasks)