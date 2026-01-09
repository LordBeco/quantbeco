import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import calendar
from datetime import datetime, timedelta
from analytics import compute_metrics, compute_rolling_metrics, compute_top_kpis, compute_time_analysis, compute_risk_pain_metrics, compute_trading_insights, generate_trading_insights_summary, analyze_edge_decay, compute_kelly_metrics
from charts import (equity_curve, drawdown_curve, pnl_distribution, win_loss_pie, 
                   pnl_growth_over_time, rolling_performance_charts, time_analysis_charts, 
                   monthly_heatmap, risk_pain_charts, create_time_tables, create_trading_insights_charts, create_pip_analysis_chart, create_daily_calendar_chart, create_kelly_criterion_charts, create_kelly_insights_summary_chart)
try:
    from charts import create_deposit_withdrawal_analysis_charts, create_clean_vs_raw_equity_chart
    DEPOSIT_CHARTS_AVAILABLE = True
except ImportError:
    DEPOSIT_CHARTS_AVAILABLE = False
    
from transaction_handler import process_broker_statement
from diagnosis import diagnose, comprehensive_ai_diagnosis, generate_executive_summary
from style import inject_css
from tradelocker_api import TradeLockerAPI, test_tradelocker_connection, fetch_tradelocker_data, get_tradelocker_accounts

# Import AI Strategy Builder components
from ai_strategy_builder.strategy_prompt_processor import StrategyPromptProcessor, StrategyCode
from ai_strategy_builder.pine_script_converter import PineScriptConverter
from ai_strategy_builder.code_generator import CodeGenerator, StrategyComponents, RiskManagement

# Import Backtesting Engine components
from backtesting_engine.data_processor import DataProcessor, DataConfig, ValidationResult
from backtesting_engine.backtest_engine import BacktestEngine, BacktestConfig, BacktestResults
from backtesting_engine.instrument_manager import InstrumentManager
from backtesting_engine.report_generator import ReportGenerator

from config import Config

def get_date_range(period, custom_start=None, custom_end=None):
    """Get start and end dates for different time periods"""
    today = datetime.now().date()
    
    if period == "Custom Range":
        return custom_start, custom_end
    elif period == "Today":
        return today, today
    elif period == "Yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period == "This Week":
        start_week = today - timedelta(days=today.weekday())
        return start_week, today
    elif period == "Last Week":
        start_last_week = today - timedelta(days=today.weekday() + 7)
        end_last_week = today - timedelta(days=today.weekday() + 1)
        return start_last_week, end_last_week
    elif period == "This Month":
        start_month = today.replace(day=1)
        return start_month, today
    elif period == "Last Month":
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end
    elif period == "This Year":
        start_year = today.replace(month=1, day=1)
        return start_year, today
    elif period == "Last Year":
        last_year = today.year - 1
        start_last_year = datetime(last_year, 1, 1).date()
        end_last_year = datetime(last_year, 12, 31).date()
        return start_last_year, end_last_year
    elif period == "All Time":
        return None, None
    
    return None, None

def filter_dataframe_by_date(df, start_date, end_date):
    """Filter dataframe by date range"""
    if start_date is None or end_date is None:
        return df
    
    # Try to find datetime columns
    datetime_cols = []
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'datetime', 'open_time', 'close_time']):
            try:
                df[f'{col}_parsed'] = pd.to_datetime(df[col])
                datetime_cols.append(f'{col}_parsed')
            except:
                continue
    
    if not datetime_cols:
        st.warning("⚠️ No valid datetime columns found for filtering. Showing all data.")
        return df
    
    # Use the first datetime column found (usually close_time or open_time)
    date_col = datetime_cols[0]
    
    # Filter by date range
    mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    return filtered_df

def ai_strategy_builder_interface():
    """AI Strategy Builder interface implementation"""
    st.markdown("### 🤖 AI Strategy Builder")
    st.markdown("**Convert your trading ideas into executable code using natural language**")
    
    # Check OpenAI API configuration
    if not Config.validate_openai_config():
        st.error("🚨 **OpenAI API Key Required**")
        st.markdown("""
        To use the AI Strategy Builder, you need to configure your OpenAI API key:
        
        1. **Get an API key** from [OpenAI Platform](https://platform.openai.com/api-keys)
        2. **Set environment variable**: `OPENAI_API_KEY=your_key_here`
        3. **Restart the application**
        
        Or create a `.env` file in the project directory with:
        ```
        OPENAI_API_KEY=your_key_here
        ```
        """)
        return
    
    # Initialize components
    if 'strategy_processor' not in st.session_state:
        st.session_state.strategy_processor = StrategyPromptProcessor()
    if 'pine_converter' not in st.session_state:
        st.session_state.pine_converter = PineScriptConverter()
    if 'code_generator' not in st.session_state:
        st.session_state.code_generator = CodeGenerator()
    
    # Initialize session state for generated code
    if 'generated_strategy' not in st.session_state:
        st.session_state.generated_strategy = None
    if 'pine_script_code' not in st.session_state:
        st.session_state.pine_script_code = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    
    # === NATURAL LANGUAGE PROMPT INPUT ===
    st.markdown("#### 💬 Describe Your Trading Strategy")
    
    # Example prompts for user guidance
    with st.expander("💡 Example Prompts", expanded=False):
        st.markdown("""
        **Simple Moving Average Strategy:**
        "Buy when price crosses above 20-period SMA and RSI is below 70. Sell when price crosses below 20-period SMA or RSI is above 30. Use 2% stop loss and 4% take profit."
        
        **MACD + RSI Strategy:**
        "Enter long when MACD crosses above signal line and RSI is oversold (below 30). Exit when MACD crosses below signal line or RSI reaches overbought (above 70). Risk 1% per trade."
        
        **Bollinger Bands Mean Reversion:**
        "Buy when price touches lower Bollinger Band and RSI is below 30. Sell when price reaches middle Bollinger Band or RSI is above 70. Use ATR-based stop loss."
        
        **Breakout Strategy:**
        "Enter long when price breaks above 20-period high with volume confirmation. Exit on 3% stop loss or when price falls below 10-period EMA. Use fixed position sizing."
        """)
    
    # Prompt input area
    prompt = st.text_area(
        "Describe your trading strategy in natural language:",
        height=150,
        placeholder="Example: Buy when RSI is below 30 and price is above 50-period moving average. Sell when RSI is above 70. Use 2% stop loss and 3% take profit...",
        help="Be specific about entry/exit conditions, indicators, and risk management rules"
    )
    
    # Strategy generation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Strategy Code", type="primary", disabled=not prompt.strip())
    
    with col2:
        if st.session_state.generated_strategy:
            clear_button = st.button("🗑️ Clear", help="Clear generated code")
            if clear_button:
                st.session_state.generated_strategy = None
                st.session_state.pine_script_code = None
                st.session_state.validation_results = None
                st.rerun()
    
    with col3:
        # Model selection
        model_options = ["gpt-4", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Model", model_options, index=0, help="Choose AI model")
    
    # Generate strategy code
    if generate_button and prompt.strip():
        with st.spinner("🤖 Generating strategy code..."):
            try:
                # Update model in config if different
                if selected_model != Config.OPENAI_MODEL:
                    Config.OPENAI_MODEL = selected_model
                
                # Process the prompt
                strategy_code = st.session_state.strategy_processor.process_prompt(prompt)
                st.session_state.generated_strategy = strategy_code
                
                # Validate the generated code
                validation = st.session_state.strategy_processor.validate_strategy(strategy_code.python_code)
                st.session_state.validation_results = validation
                
                st.success("✅ Strategy code generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating strategy: {str(e)}")
                st.info("💡 **Troubleshooting Tips:**")
                st.write("• Make your prompt more specific about trading rules")
                st.write("• Include clear entry and exit conditions")
                st.write("• Mention specific indicators (RSI, SMA, MACD, etc.)")
                st.write("• Check your OpenAI API key and quota")
    
    # === DISPLAY GENERATED CODE ===
    if st.session_state.generated_strategy:
        strategy = st.session_state.generated_strategy
        
        st.markdown("---")
        st.markdown("#### 📝 Generated Python Strategy Code")
        
        # Display validation results
        if st.session_state.validation_results:
            validation = st.session_state.validation_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if validation.is_valid:
                    st.success("✅ Code is valid")
                else:
                    st.error("❌ Code has errors")
            
            with col2:
                st.metric("Errors", len(validation.errors))
            
            with col3:
                st.metric("Warnings", len(validation.warnings))
            
            # Show errors and warnings
            if validation.errors:
                st.error("**Errors:**")
                for error in validation.errors:
                    st.write(f"• {error}")
            
            if validation.warnings:
                st.warning("**Warnings:**")
                for warning in validation.warnings:
                    st.write(f"• {warning}")
        
        # Display strategy metadata
        if strategy.metadata:
            with st.expander("📊 Strategy Information", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model Used:** {strategy.metadata.get('model_used', 'Unknown')}")
                    st.write(f"**Generated:** {datetime.fromtimestamp(strategy.metadata.get('generated_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    if strategy.indicators:
                        st.write(f"**Indicators:** {', '.join(strategy.indicators)}")
                    if strategy.entry_conditions:
                        st.write(f"**Entry Conditions:** {len(strategy.entry_conditions)}")
                    if strategy.exit_conditions:
                        st.write(f"**Exit Conditions:** {len(strategy.exit_conditions)}")
        
        # Code display with syntax highlighting
        st.code(strategy.python_code, language="python")
        
        # Code actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download Python code
            st.download_button(
                label="📥 Download Python Code",
                data=strategy.python_code,
                file_name="ai_generated_strategy.py",
                mime="text/plain"
            )
        
        with col2:
            # Convert to Pine Script
            convert_to_pine = st.button("🌲 Convert to Pine Script")
        
        with col3:
            # Validate code again
            revalidate = st.button("🔍 Re-validate Code")
        
        # Handle Pine Script conversion
        if convert_to_pine:
            with st.spinner("🌲 Converting to Pine Script..."):
                try:
                    pine_code = st.session_state.pine_converter.convert_to_pine(strategy.python_code)
                    st.session_state.pine_script_code = pine_code
                    
                    # Validate Pine Script
                    pine_validation = st.session_state.pine_converter.validate_pine_script(pine_code)
                    
                    if pine_validation.is_valid:
                        st.success("✅ Pine Script conversion successful!")
                    else:
                        st.warning("⚠️ Pine Script converted with warnings")
                        
                except Exception as e:
                    st.error(f"❌ Pine Script conversion failed: {str(e)}")
                    st.info("💡 **Note:** Pine Script conversion is experimental and may not work for all strategies")
        
        # Handle code re-validation
        if revalidate:
            with st.spinner("🔍 Re-validating code..."):
                try:
                    validation = st.session_state.strategy_processor.validate_strategy(strategy.python_code)
                    st.session_state.validation_results = validation
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Validation error: {str(e)}")
    
    # === PINE SCRIPT DISPLAY ===
    if st.session_state.pine_script_code:
        st.markdown("---")
        st.markdown("#### 🌲 Pine Script Code (TradingView)")
        
        # Pine Script validation info
        try:
            pine_validation = st.session_state.pine_converter.validate_pine_script(st.session_state.pine_script_code)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if pine_validation.is_valid:
                    st.success("✅ Pine Script is valid")
                else:
                    st.error("❌ Pine Script has issues")
            
            with col2:
                st.metric("Errors", len(pine_validation.errors))
            
            with col3:
                st.metric("Warnings", len(pine_validation.warnings))
            
            # Show Pine Script issues
            if pine_validation.errors:
                st.error("**Pine Script Errors:**")
                for error in pine_validation.errors:
                    st.write(f"• {error}")
            
            if pine_validation.warnings:
                st.warning("**Pine Script Warnings:**")
                for warning in pine_validation.warnings:
                    st.write(f"• {warning}")
                    
        except Exception as e:
            st.warning(f"⚠️ Could not validate Pine Script: {str(e)}")
        
        # Display Pine Script code
        st.code(st.session_state.pine_script_code, language="javascript")  # Pine Script syntax highlighting
        
        # Pine Script actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Pine Script",
                data=st.session_state.pine_script_code,
                file_name="ai_generated_strategy.pine",
                mime="text/plain"
            )
        
        with col2:
            st.markdown("**📋 How to use in TradingView:**")
            st.write("1. Copy the Pine Script code")
            st.write("2. Open TradingView Pine Editor")
            st.write("3. Paste and save the code")
            st.write("4. Add to chart and configure parameters")
    
    # === STRATEGY TESTING SECTION ===
    if st.session_state.generated_strategy:
        st.markdown("---")
        st.markdown("#### 🧪 Strategy Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Next Steps:**")
            st.write("• Test your strategy with the Backtesting Engine")
            st.write("• Upload historical data to validate performance")
            st.write("• Adjust parameters based on backtest results")
            st.write("• Deploy to TradingView using Pine Script")
        
        with col2:
            st.warning("**Important Notes:**")
            st.write("• Generated code is for educational purposes")
            st.write("• Always validate strategies with historical data")
            st.write("• Consider transaction costs and slippage")
            st.write("• Never risk more than you can afford to lose")
        
        # Quick strategy summary
        if st.session_state.generated_strategy.indicators or st.session_state.generated_strategy.entry_conditions:
            with st.expander("📋 Strategy Summary", expanded=False):
                strategy = st.session_state.generated_strategy
                
                if strategy.indicators:
                    st.write(f"**Indicators Used:** {', '.join(strategy.indicators)}")
                
                if strategy.entry_conditions:
                    st.write(f"**Entry Conditions:** {', '.join(strategy.entry_conditions)}")
                
                if strategy.exit_conditions:
                    st.write(f"**Exit Conditions:** {', '.join(strategy.exit_conditions)}")
                
                if strategy.metadata and 'original_prompt' in strategy.metadata:
                    st.write(f"**Original Prompt:** {strategy.metadata['original_prompt'][:200]}...")

def backtesting_engine_interface():
    """Backtesting Engine interface implementation"""
    st.markdown("### ⚡ Backtesting Engine")
    st.markdown("**Test your trading strategies against historical tick data**")
    
    # Initialize components
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'backtest_engine' not in st.session_state:
        st.session_state.backtest_engine = BacktestEngine()
    if 'instrument_manager' not in st.session_state:
        st.session_state.instrument_manager = InstrumentManager()
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    
    # Initialize session state for backtesting
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'backtest_config' not in st.session_state:
        st.session_state.backtest_config = None
    
    # === DATA UPLOAD SECTION ===
    st.markdown("#### 📊 Upload Tick Data")
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file with tick data",
        type=['csv'],
        help="Upload CSV file with columns: timestamp, open, high, low, close, volume (optional)"
    )
    
    # Data upload and validation
    if uploaded_file is not None:
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file)
            
            # Validate data
            validation_result = st.session_state.data_processor.validate_tick_data(data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if validation_result.is_valid:
                    st.success("✅ Data is valid")
                else:
                    st.error("❌ Data validation failed")
            
            with col2:
                st.metric("Rows", len(data))
            
            with col3:
                st.metric("Columns", len(data.columns))
            
            # Show validation details
            if validation_result.errors:
                st.error("**Validation Errors:**")
                for error in validation_result.errors:
                    st.write(f"• {error}")
            
            if validation_result.warnings:
                st.warning("**Validation Warnings:**")
                for warning in validation_result.warnings:
                    st.write(f"• {warning}")
            
            # Display data preview if valid
            if validation_result.is_valid:
                st.session_state.uploaded_data = data
                
                with st.expander("📋 Data Preview", expanded=False):
                    st.dataframe(data.head(10))
                
                # Basic statistics
                with st.expander("📈 Data Statistics", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'timestamp' in data.columns:
                            st.write(f"**Date Range:**")
                            try:
                                timestamps = pd.to_datetime(data['timestamp'])
                                st.write(f"From: {timestamps.min()}")
                                st.write(f"To: {timestamps.max()}")
                                st.write(f"Duration: {timestamps.max() - timestamps.min()}")
                            except:
                                st.write("Could not parse timestamps")
                    
                    with col2:
                        if 'close' in data.columns:
                            st.write(f"**Price Statistics:**")
                            st.write(f"Min: {data['close'].min():.5f}")
                            st.write(f"Max: {data['close'].max():.5f}")
                            st.write(f"Mean: {data['close'].mean():.5f}")
                            st.write(f"Std: {data['close'].std():.5f}")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 **File Format Requirements:**")
            st.write("• CSV format with headers")
            st.write("• Required columns: timestamp, open, high, low, close")
            st.write("• Optional columns: volume")
            st.write("• Timestamp format: YYYY-MM-DD HH:MM:SS or similar")
    
    # === CONFIGURATION SECTION ===
    if st.session_state.uploaded_data is not None:
        st.markdown("---")
        st.markdown("#### ⚙️ Backtesting Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Date Range & Timeframe**")
            
            # Date range selection
            try:
                data = st.session_state.uploaded_data
                timestamps = pd.to_datetime(data['timestamp'])
                min_date = timestamps.min().date()
                max_date = timestamps.max().date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.date_input(
                    "End Date", 
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
            except:
                st.error("Could not parse timestamps for date selection")
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
            
            # Timezone selection
            timezone_options = [
                "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
                "Europe/London", "Europe/Berlin", "Europe/Zurich", "Asia/Tokyo",
                "Asia/Hong_Kong", "Asia/Singapore", "Australia/Sydney"
            ]
            timezone = st.selectbox("Timezone", timezone_options, index=0)
            
            # Timeframe selection
            timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            timeframe = st.selectbox("Timeframe", timeframe_options, index=4)
        
        with col2:
            st.markdown("**💰 Account Configuration**")
            
            # Starting balance
            starting_balance = st.number_input(
                "Starting Balance ($)",
                min_value=100.0,
                max_value=1000000.0,
                value=Config.DEFAULT_STARTING_BALANCE,
                step=1000.0
            )
            
            # Leverage
            leverage = st.number_input(
                "Leverage",
                min_value=1.0,
                max_value=500.0,
                value=Config.DEFAULT_LEVERAGE,
                step=1.0
            )
            
            # Commission and slippage
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=Config.DEFAULT_COMMISSION,
                step=0.01,
                format="%.3f"
            )
            
            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=Config.DEFAULT_SLIPPAGE,
                step=0.01,
                format="%.3f"
            )
        
        # === INSTRUMENT CONFIGURATION ===
        st.markdown("**🎯 Instrument & Position Sizing**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Instrument selection
            available_instruments = list(st.session_state.instrument_manager.get_available_instruments().keys())
            instrument = st.selectbox(
                "Instrument",
                available_instruments,
                index=0 if available_instruments else None
            )
            
            # Display instrument info
            if instrument:
                spec = st.session_state.instrument_manager.get_instrument_spec(instrument)
                if spec:
                    st.info(f"**{instrument}**\n"
                           f"Pip Value: ${spec.pip_value}\n"
                           f"Spread: {spec.typical_spread} pips\n"
                           f"Max Leverage: {spec.max_leverage}:1")
        
        with col2:
            # Position sizing method
            position_sizing_methods = ["fixed", "risk_based", "compounding"]
            position_sizing = st.selectbox(
                "Position Sizing",
                position_sizing_methods,
                index=1
            )
            
            if position_sizing == "fixed":
                lot_size = st.number_input(
                    "Lot Size",
                    min_value=0.01,
                    max_value=100.0,
                    value=0.1,
                    step=0.01
                )
            else:
                lot_size = 0.1  # Will be calculated dynamically
        
        with col3:
            # Risk management
            if position_sizing in ["risk_based", "compounding"]:
                risk_per_trade = st.number_input(
                    "Risk per Trade (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1
                ) / 100
            else:
                risk_per_trade = 0.02
            
            # Compounding option
            compounding = st.checkbox(
                "Enable Compounding",
                value=(position_sizing == "compounding"),
                help="Reinvest profits to increase position sizes"
            )
        
        # === STRATEGY CODE INPUT ===
        st.markdown("---")
        st.markdown("#### 💻 Strategy Code")
        
        # Check if there's generated strategy code from AI Builder
        strategy_source = st.radio(
            "Strategy Source",
            ["Manual Input", "From AI Strategy Builder"],
            horizontal=True
        )
        
        if strategy_source == "From AI Strategy Builder":
            if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
                st.success("✅ Using strategy from AI Strategy Builder")
                strategy_code = st.session_state.generated_strategy.python_code
                
                # Display the strategy code (read-only)
                st.code(strategy_code, language="python")
                
                # Option to edit
                if st.checkbox("Edit Strategy Code"):
                    strategy_code = st.text_area(
                        "Strategy Code (Python)",
                        value=strategy_code,
                        height=300,
                        help="Modify the generated strategy code if needed"
                    )
            else:
                st.warning("⚠️ No strategy found from AI Strategy Builder")
                st.info("💡 Go to the AI Strategy Builder tab first to generate a strategy")
                strategy_code = ""
        else:
            # Manual strategy input
            default_strategy = '''import pandas as pd
import numpy as np

def run_strategy(data):
    """
    Simple moving average crossover strategy
    """
    # Calculate indicators
    data['sma_fast'] = data['close'].rolling(10).mean()
    data['sma_slow'] = data['close'].rolling(20).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
    data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
    
    return data[['signal']]
'''
            
            strategy_code = st.text_area(
                "Strategy Code (Python)",
                value=default_strategy,
                height=300,
                help="Write your strategy function that returns buy/sell signals"
            )
        
        # === BACKTEST EXECUTION ===
        st.markdown("---")
        st.markdown("#### 🚀 Run Backtest")
        
        # Create backtest configuration
        if st.button("🚀 Start Backtest", type="primary", disabled=not strategy_code.strip()):
            
            # Create configuration
            config = BacktestConfig(
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                timeframe=timeframe,
                timezone=timezone,
                starting_balance=starting_balance,
                leverage=leverage,
                commission=commission / 100,  # Convert to decimal
                slippage=slippage / 100,      # Convert to decimal
                lot_size=lot_size,
                compounding=compounding,
                risk_per_trade=risk_per_trade,
                instrument=instrument
            )
            
            st.session_state.backtest_config = config
            
            # Preprocess data
            with st.spinner("📊 Preprocessing data..."):
                try:
                    data_config = DataConfig(
                        timezone=timezone,
                        fill_gaps=True,
                        remove_outliers=True
                    )
                    
                    processed_data = st.session_state.data_processor.preprocess_data(
                        st.session_state.uploaded_data, data_config
                    )
                    
                    st.success("✅ Data preprocessing completed")
                    
                except Exception as e:
                    st.error(f"❌ Data preprocessing failed: {str(e)}")
                    st.stop()
            
            # Execute strategy
            with st.spinner("⚡ Running backtest..."):
                try:
                    # Create strategy function from code
                    exec_globals = {'pd': pd, 'np': np}
                    exec(strategy_code, exec_globals)
                    
                    if 'run_strategy' not in exec_globals:
                        st.error("❌ Strategy code must contain a 'run_strategy' function")
                        st.stop()
                    
                    strategy_func = exec_globals['run_strategy']
                    
                    # Run backtest
                    results = st.session_state.backtest_engine.execute_backtest(
                        strategy_func, processed_data, config
                    )
                    
                    st.session_state.backtest_results = results
                    st.success("✅ Backtest completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Backtest execution failed: {str(e)}")
                    st.info("💡 **Common Issues:**")
                    st.write("• Check strategy function syntax")
                    st.write("• Ensure function returns DataFrame with 'signal' column")
                    st.write("• Verify data has required OHLCV columns")
                    st.write("• Check date range covers available data")

    # === RESULTS DISPLAY SECTION ===
    if st.session_state.backtest_results is not None:
        st.markdown("---")
        st.markdown("#### 📊 Backtest Results")
        
        results = st.session_state.backtest_results
        config = st.session_state.backtest_config
        
        # === PERFORMANCE METRICS ===
        st.markdown("**📈 Performance Metrics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{results.performance_metrics.total_return:.2f}%",
                delta=f"{results.performance_metrics.total_return:.2f}%"
            )
        
        with col2:
            st.metric(
                "Win Rate", 
                f"{results.performance_metrics.win_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{results.performance_metrics.max_drawdown:.2f}%",
                delta=f"-{results.performance_metrics.max_drawdown:.2f}%"
            )
        
        with col4:
            st.metric(
                "Total Trades",
                f"{results.performance_metrics.total_trades}"
            )
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{results.performance_metrics.sharpe_ratio:.2f}"
            )
        
        with col2:
            st.metric(
                "Profit Factor",
                f"{results.performance_metrics.profit_factor:.2f}"
            )
        
        with col3:
            final_balance = config.starting_balance + sum(t.pnl for t in results.trades)
            st.metric(
                "Final Balance",
                f"${final_balance:,.2f}",
                delta=f"${sum(t.pnl for t in results.trades):,.2f}"
            )
        
        with col4:
            if results.trades:
                avg_trade = sum(t.pnl for t in results.trades) / len(results.trades)
                st.metric(
                    "Avg Trade P&L",
                    f"${avg_trade:.2f}"
                )
        
        # === DETAILED STATISTICS ===
        with st.expander("📊 Detailed Statistics", expanded=False):
            if results.trades:
                winning_trades = [t for t in results.trades if t.pnl > 0]
                losing_trades = [t for t in results.trades if t.pnl < 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Winning Trades:**")
                    st.write(f"Count: {len(winning_trades)}")
                    if winning_trades:
                        st.write(f"Average: ${np.mean([t.pnl for t in winning_trades]):.2f}")
                        st.write(f"Largest: ${max([t.pnl for t in winning_trades]):.2f}")
                        avg_duration = np.mean([t.duration.total_seconds() / 3600 for t in winning_trades])
                        st.write(f"Avg Duration: {avg_duration:.1f} hours")
                
                with col2:
                    st.write("**Losing Trades:**")
                    st.write(f"Count: {len(losing_trades)}")
                    if losing_trades:
                        st.write(f"Average: ${np.mean([t.pnl for t in losing_trades]):.2f}")
                        st.write(f"Largest: ${min([t.pnl for t in losing_trades]):.2f}")
                        avg_duration = np.mean([t.duration.total_seconds() / 3600 for t in losing_trades])
                        st.write(f"Avg Duration: {avg_duration:.1f} hours")
        
        # === INTERACTIVE TRADING CHART ===
        st.markdown("**📈 Interactive Trading Chart**")
        
        try:
            # Get the processed data for charting
            if st.session_state.uploaded_data is not None:
                chart_data = st.session_state.uploaded_data.copy()
                
                # Ensure datetime index
                if 'timestamp' in chart_data.columns:
                    chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
                    chart_data = chart_data.set_index('timestamp')
                
                # Filter data to backtest period
                if config.start_date and config.end_date:
                    chart_data = chart_data[
                        (chart_data.index >= config.start_date) & 
                        (chart_data.index <= config.end_date)
                    ]
                
                # Generate trading chart
                fig = st.session_state.report_generator.create_trading_charts(
                    chart_data, results.trades
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error generating chart: {str(e)}")
            st.info("💡 Chart generation failed, but backtest results are still valid")
        
        # === EQUITY CURVE ===
        st.markdown("**💰 Equity Curve**")
        
        if len(results.equity_curve) > 1:
            # Create equity curve chart
            equity_fig = go.Figure()
            
            equity_fig.add_trace(go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
            
            # Add starting balance line
            equity_fig.add_hline(
                y=config.starting_balance,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Starting Balance: ${config.starting_balance:,.2f}"
            )
            
            equity_fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Time",
                yaxis_title="Account Balance ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(equity_fig, use_container_width=True)
        
        # === DRAWDOWN CHART ===
        if len(results.drawdown_series) > 1:
            st.markdown("**📉 Drawdown Analysis**")
            
            dd_fig = go.Figure()
            
            dd_fig.add_trace(go.Scatter(
                x=results.drawdown_series.index,
                y=results.drawdown_series.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tonexty'
            ))
            
            dd_fig.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Time", 
                yaxis_title="Drawdown (%)",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(dd_fig, use_container_width=True)
        
        # === TRADE LIST ===
        with st.expander("📋 Trade List", expanded=False):
            if results.trades:
                # Create trade list DataFrame
                trade_data = []
                for i, trade in enumerate(results.trades, 1):
                    trade_data.append({
                        '#': i,
                        'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Side': trade.side.upper(),
                        'Instrument': trade.instrument,
                        'Quantity': f"{trade.quantity:.2f}",
                        'Entry Price': f"{trade.entry_price:.5f}",
                        'Exit Price': f"{trade.exit_price:.5f}",
                        'P&L': f"${trade.pnl:.2f}",
                        'P&L %': f"{trade.pnl_pct:.2f}%",
                        'Duration': str(trade.duration).split('.')[0],  # Remove microseconds
                        'Exit Reason': trade.exit_reason.replace('_', ' ').title()
                    })
                
                trade_df = pd.DataFrame(trade_data)
                st.dataframe(trade_df, use_container_width=True)
            else:
                st.info("No trades were executed during the backtest period")
        
        # === CSV DOWNLOAD AND ANALYTICS INTEGRATION ===
        st.markdown("---")
        st.markdown("**💾 Export & Integration**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate broker statement CSV
            try:
                broker_statement = st.session_state.report_generator.generate_broker_statement(
                    results, config.starting_balance
                )
                
                csv_data = broker_statement.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Broker Statement CSV",
                    data=csv_data,
                    file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download CSV file compatible with trade_analyzer_pro analytics"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating CSV: {str(e)}")
        
        with col2:
            # Export configuration
            config_data = {
                'backtest_config': {
                    'start_date': config.start_date.isoformat(),
                    'end_date': config.end_date.isoformat(),
                    'timeframe': config.timeframe,
                    'timezone': config.timezone,
                    'starting_balance': config.starting_balance,
                    'leverage': config.leverage,
                    'commission': config.commission,
                    'slippage': config.slippage,
                    'instrument': config.instrument,
                    'compounding': config.compounding,
                    'risk_per_trade': config.risk_per_trade
                },
                'performance_metrics': {
                    'total_return': results.performance_metrics.total_return,
                    'win_rate': results.performance_metrics.win_rate,
                    'max_drawdown': results.performance_metrics.max_drawdown,
                    'sharpe_ratio': results.performance_metrics.sharpe_ratio,
                    'profit_factor': results.performance_metrics.profit_factor,
                    'total_trades': results.performance_metrics.total_trades
                }
            }
            
            import json
            config_json = json.dumps(config_data, indent=2)
            
            st.download_button(
                label="📋 Download Configuration",
                data=config_json,
                file_name=f"backtest_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download backtest configuration for reproducibility"
            )
        
        with col3:
            # Integration with existing analytics
            st.info("**🔗 Analytics Integration**")
            st.write("• Download CSV and upload to Analytics tab")
            st.write("• Use existing charts and analysis tools")
            st.write("• Compare with live trading results")
            
            if st.button("🔄 Clear Results", help="Clear current backtest results"):
                st.session_state.backtest_results = None
                st.session_state.backtest_config = None
                st.rerun()
        
        # === BACKTEST SUMMARY ===
        with st.expander("📄 Backtest Summary", expanded=False):
            st.markdown(f"""
            **Backtest Configuration Summary:**
            - **Period:** {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}
            - **Instrument:** {config.instrument}
            - **Timeframe:** {config.timeframe}
            - **Starting Balance:** ${config.starting_balance:,.2f}
            - **Leverage:** {config.leverage}:1
            - **Commission:** {config.commission*100:.3f}%
            - **Slippage:** {config.slippage*100:.3f}%
            - **Position Sizing:** {'Compounding' if config.compounding else 'Fixed'}
            - **Risk per Trade:** {config.risk_per_trade*100:.1f}%
            
            **Performance Summary:**
            - **Total Return:** {results.performance_metrics.total_return:.2f}%
            - **Total Trades:** {results.performance_metrics.total_trades}
            - **Win Rate:** {results.performance_metrics.win_rate:.1f}%
            - **Profit Factor:** {results.performance_metrics.profit_factor:.2f}
            - **Max Drawdown:** {results.performance_metrics.max_drawdown:.2f}%
            - **Sharpe Ratio:** {results.performance_metrics.sharpe_ratio:.2f}
            
            **Risk Assessment:**
            - {'🟢 Low Risk' if results.performance_metrics.max_drawdown < 10 else '🟡 Medium Risk' if results.performance_metrics.max_drawdown < 20 else '🔴 High Risk'}
            - {'🟢 Good Performance' if results.performance_metrics.total_return > 10 else '🟡 Moderate Performance' if results.performance_metrics.total_return > 0 else '🔴 Poor Performance'}
            - {'🟢 Consistent' if results.performance_metrics.win_rate > 60 else '🟡 Average' if results.performance_metrics.win_rate > 40 else '🔴 Inconsistent'}
            """)

def analytics_dashboard():
    """Original analytics dashboard functionality"""
    
    # === DATA SOURCE SELECTION ===
    st.markdown("### 📊 Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["📁 Upload CSV File", "🔗 TradeLocker API"],
        horizontal=True
    )

    df = None
    pnl = None
    starting_balance = 10000  # Default fallback, will be updated

    if data_source == "📁 Upload CSV File":
        file = st.file_uploader("Upload Broker Statement (CSV)", type="csv")
        
        if file:
            df_raw = pd.read_csv(file)
            df_raw.columns = [c.lower().replace(" ", "_") for c in df_raw.columns]

            # === STARTING BALANCE DETECTION ===
            st.markdown("### 💰 Account Starting Balance")
            
            # First, try to detect starting balance from the data
            detected_starting_balance = None
            balance_detection_info = ""
            
            # Quick check for balance entries
            if 'type' in df_raw.columns:
                balance_entries = df_raw[df_raw['type'].str.contains('balance', case=False, na=False)]
                if len(balance_entries) > 0:
                    # Clean the profit column for detection
                    balance_entries = balance_entries.copy()
                    balance_entries['profit_clean'] = pd.to_numeric(
                        balance_entries['profit'].astype(str).str.replace(',', ''), 
                        errors="coerce"
                    ).fillna(0)
                    
                    # Use the first positive balance entry
                    positive_balances = balance_entries[balance_entries['profit_clean'] > 0]
                    if len(positive_balances) > 0:
                        detected_starting_balance = positive_balances['profit_clean'].iloc[0]
                        balance_comment = positive_balances['comment'].iloc[0] if 'comment' in positive_balances.columns else "Balance entry"
                        balance_detection_info = f"✅ **Auto-detected from CSV**: ${detected_starting_balance:,.2f} ({balance_comment})"
            
            # Show detection result
            if balance_detection_info:
                st.success(balance_detection_info)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                default_balance = detected_starting_balance if detected_starting_balance else 5000.0
                
                user_starting_balance = st.number_input(
                    "Confirm or adjust your starting balance:", 
                    min_value=100.0, 
                    max_value=1000000.0, 
                    value=default_balance,
                    step=100.0,
                    format="%.2f",
                    help="Auto-detected from your CSV or enter manually"
                )
            
            with col2:
                st.markdown("**Why this matters:**")
                st.write("• Accurate equity calculations")
                st.write("• Proper Kelly Criterion analysis")
                st.write("• Correct performance metrics")
                if detected_starting_balance:
                    st.write("• ✅ Found in your CSV data")
            
            # Update starting balance
            starting_balance = user_starting_balance
            
            # Show quick calculation preview
            pnl_col_preview = None
            for col in ['profit', 'pnl']:
                if col in df_raw.columns:
                    pnl_col_preview = col
                    break
            
            if pnl_col_preview:
                # Clean the P&L column for preview
                df_raw_preview = df_raw.copy()
                df_raw_preview[pnl_col_preview] = pd.to_numeric(
                    df_raw_preview[pnl_col_preview].astype(str).str.replace(',', ''), 
                    errors="coerce"
                ).fillna(0)
                
                # Exclude balance entries from P&L calculation
                trading_only_preview = df_raw_preview.copy()
                if 'type' in df_raw_preview.columns:
                    # Remove balance entries
                    trading_only_preview = trading_only_preview[
                        ~trading_only_preview['type'].str.contains('balance', case=False, na=False)
                    ]
                
                # Calculate trading-only P&L
                trading_pnl = trading_only_preview[pnl_col_preview].sum()
                estimated_current = starting_balance + trading_pnl
                
                st.info(f"""
                📊 **Quick Preview**: Starting ${starting_balance:,.2f} + Trading P&L ${trading_pnl:,.2f} = **${estimated_current:,.2f}** current equity
                """)
            else:
                st.info(f"📊 **Starting Balance Set**: ${starting_balance:,.2f}")

            # === TRANSACTION PROCESSING (DEPOSITS/WITHDRAWALS) ===
            st.markdown("### 🔍 Transaction Analysis")
            
            # Process broker statement to separate trading from non-trading transactions
            pnl_col_for_processing = None
            for col in ['profit', 'pnl']:
                if col in df_raw.columns:
                    pnl_col_for_processing = col
                    break
            
            if pnl_col_for_processing:
                # Clean numeric data first
                df_raw[pnl_col_for_processing] = pd.to_numeric(
                    df_raw[pnl_col_for_processing].astype(str).str.replace(',', ''), 
                    errors="coerce"
                ).fillna(0)
                
                # Process transactions
                processed_data = process_broker_statement(df_raw, pnl_col_for_processing, starting_balance)
                
                # Display transaction analysis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Transactions", processed_data['summary']['total_transactions'])
                    st.metric("Trading Transactions", processed_data['summary']['trading_transactions'])
                
                with col2:
                    st.metric("Deposits", processed_data['summary']['deposits'])
                    st.metric("Withdrawals", processed_data['summary']['withdrawals'])
                
                with col3:
                    net_deposits = processed_data.get('net_deposits', 0)
                    st.metric("Net Deposits", f"${net_deposits:,.2f}")
                    st.metric("Clean Trading P&L", f"${processed_data['clean_trading_pnl']:,.2f}")
                
                # Show insights
                if processed_data['insights']:
                    st.markdown("#### 📊 Transaction Insights")
                    for insight in processed_data['insights']:
                        if "✅" in insight:
                            st.success(insight)
                        elif "⚠️" in insight or "📉" in insight:
                            st.warning(insight)
                        else:
                            st.info(insight)
                
                # Use cleaned trading data for analysis
                df = processed_data['trading_df']
                
                # Show comparison charts if there are deposits/withdrawals
                if processed_data['has_deposits_withdrawals'] and DEPOSIT_CHARTS_AVAILABLE:
                    st.markdown("#### 📈 Equity Curve Impact Analysis")
                    
                    # Raw vs Clean comparison
                    try:
                        comparison_chart = create_clean_vs_raw_equity_chart(df_raw, df, pnl_col_for_processing)
                        st.plotly_chart(comparison_chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create comparison chart: {str(e)}")
                    
                    # Detailed deposit/withdrawal analysis
                    try:
                        dw_charts = create_deposit_withdrawal_analysis_charts(processed_data)
                        
                        if 'equity_comparison' in dw_charts:
                            st.plotly_chart(dw_charts['equity_comparison'], use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        if 'deposit_timeline' in dw_charts:
                            with col1:
                                st.plotly_chart(dw_charts['deposit_timeline'], use_container_width=True)
                        
                        if 'performance_impact' in dw_charts:
                            with col2:
                                st.plotly_chart(dw_charts['performance_impact'], use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create deposit/withdrawal charts: {str(e)}")
                    
                    # Important note about analysis
                    st.info("""
                    📊 **Important**: All performance metrics below are calculated using **trading-only data** 
                    (deposits and withdrawals excluded). This provides accurate trading performance analysis.
                    """)
                elif processed_data['has_deposits_withdrawals']:
                    st.warning("⚠️ Deposits/withdrawals detected but visualization charts not available. Analysis still uses clean trading data.")
                    st.info("""
                    📊 **Important**: All performance metrics below are calculated using **trading-only data** 
                    (deposits and withdrawals excluded). This provides accurate trading performance analysis.
                    """)
                
            else:
                # Fallback to original logic if no P&L column found
                df = df_raw.copy()

            # Continue with original balance separation logic for compatibility
            # (This is now mainly for starting balance detection)
            balance_df = pd.DataFrame()
            if 'type' in df.columns:
                balance_df = df[df['type'].str.contains('balance', case=False, na=False)].copy()
                # Don't filter out balance entries here since transaction_handler already did it
            elif 'comment' in df.columns:
                balance_df = df[df['comment'].str.contains('Initial|balance', case=False, na=False)].copy()
                # Don't filter out balance entries here since transaction_handler already did it

            # Clean numeric columns and calculate REAL PnL (including commissions and swaps)
            numeric_cols = []
            for col in ['profit', 'pnl', 'commission', 'swaps']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)
                    numeric_cols.append(col)

            # Calculate real PnL = Profit + Commission + Swaps
            if 'profit' in df.columns:
                pnl = 'real_pnl'
                df['real_pnl'] = df['profit'].copy()
                
                if 'commission' in df.columns:
                    df['real_pnl'] += df['commission']
                if 'swaps' in df.columns:
                    df['real_pnl'] += df['swaps']
                    
                st.info(f"📊 **Real PnL Calculation**: Profit + Commission + Swaps")
                
                # Show breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Raw Profit", f"${df['profit'].sum():,.2f}")
                with col2:
                    st.metric("Total Commission", f"${df['commission'].sum():,.2f}" if 'commission' in df.columns else "$0.00")
                with col3:
                    st.metric("Total Swaps", f"${df['swaps'].sum():,.2f}" if 'swaps' in df.columns else "$0.00")
                with col4:
                    st.metric("Real PnL", f"${df['real_pnl'].sum():,.2f}")
                    
            else:
                # Fallback to existing PnL column
                pnl = next(c for c in df.columns if "pnl" in c or "profit" in c)
                df[pnl] = pd.to_numeric(df[pnl].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

            # Clean balance data for starting balance detection
            if not balance_df.empty:
                for col in ['profit', 'pnl']:
                    if col in balance_df.columns:
                        balance_df[col] = pd.to_numeric(balance_df[col].astype(str).str.replace(',', ''), errors="coerce").fillna(0)

            # Calculate starting balance
            if not balance_df.empty:
                balance_amount = balance_df['profit'].iloc[0] if 'profit' in balance_df.columns else balance_df[list(balance_df.columns)[0]].iloc[0]
                if balance_amount > 0:
                    starting_balance = balance_amount

    elif data_source == "🔗 TradeLocker API":
        st.markdown("### 🔗 TradeLocker API Connection")
        
        with st.expander("📋 TradeLocker API Setup Instructions", expanded=False):
            st.markdown("""
            **How to connect your TradeLocker account:**
            
            1. **Email**: Your TradeLocker login email
            2. **Password**: Your TradeLocker password  
            3. **Server Name**: Your broker's server name (e.g., "GATESFX")
            4. **Account ID**: Your specific account ID (e.g., "1691721")
            5. **Account Number**: Select which account (1st, 2nd, 3rd, etc.) if you have multiple accounts
            6. **Account Type**: Select "Demo" or "Live" based on your account
            7. **History Period**: How many days of trading history to fetch
            
            **Finding Your Details:**
            - **Server Name**: Check your TradeLocker platform or broker documentation
            - **Account ID**: Usually shown in your trading platform (unique identifier like "1691721")
            - **Account Number**: If you have multiple accounts, select 1 for first account, 2 for second, etc.
            - **Account Type**: Demo accounts for testing, Live for real trading
            
            **API Endpoints Used:**
            - **Demo**: `https://demo.tradelocker.com/backend-api/`
            - **Live**: `https://live.tradelocker.com/backend-api/`
            
            **Security Notes:**
            - Your credentials are only used to fetch data and are not stored
            - Uses official TradeLocker backend API with secure authentication
            - All data processing happens locally in your browser
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            tl_email = st.text_input("📧 TradeLocker Email", type="default")
            tl_password = st.text_input("🔒 TradeLocker Password", type="password")
        
        with col2:
            tl_server = st.text_input("🖥️ Server Name", value="GATESFX", help="e.g., GATESFX, LIVE, DEMO")
            tl_account_id = st.text_input("🆔 Account ID", value="1691721", help="Your TradeLocker account ID")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            tl_is_live = st.selectbox("🔴 Account Type", [False, True], format_func=lambda x: "Demo" if not x else "Live")
        with col4:
            tl_acc_num = st.selectbox("🔢 Account Number", [1, 2, 3, 4, 5], index=1, help="Select which account (1st, 2nd, 3rd, etc.) if you have multiple accounts")
        with col5:
            tl_days = st.number_input("📅 Days of History", min_value=1, max_value=365, value=90)
        
        if st.button("🔗 Connect & Fetch Data", type="primary"):
            if tl_email and tl_password and tl_server and tl_account_id:
                with st.spinner("🔄 Connecting to TradeLocker..."):
                    try:
                        # Test connection first
                        connection_test = test_tradelocker_connection(tl_email, tl_password, tl_server, tl_account_id, tl_acc_num, tl_is_live)
                        
                        if connection_test['success']:
                            st.success("✅ Connected successfully!")
                            
                            # Show account info
                            account_info = connection_test['account_info']
                            balance_info = connection_test['balance']
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Account ID", account_info['account_id'])
                            with col2:
                                st.metric("Account #", account_info['account_number'])
                            with col3:
                                st.metric("Server", account_info['server'])
                            with col4:
                                st.metric("Type", "Live" if account_info['is_live'] else "Demo")
                            with col5:
                                balance_value = balance_info.get('balance', 0.0)
                                currency = balance_info.get('currency', 'USD')
                                st.metric("Balance", f"${balance_value:,.2f} {currency}")
                            
                            # Fetch trading data
                            with st.spinner("📊 Fetching trading history..."):
                                try:
                                    df = fetch_tradelocker_data(tl_email, tl_password, tl_server, tl_account_id, tl_acc_num, tl_is_live, tl_days)
                                    
                                    if df.empty:
                                        st.warning(f"⚠️ No trading history found for the last {tl_days} days.")
                                        st.info("💡 **Possible reasons:**")
                                        st.write("• No trades executed in the selected time period")
                                        st.write("• Different API endpoint needed for your broker")
                                        st.write("• Historical data may be stored differently")
                                        
                                        # Suggest trying different periods
                                        st.write("**Try:**")
                                        st.write("• Increase the 'Days of History' to 180 or 365")
                                        st.write("• Check if you have trades in your TradeLocker platform")
                                        st.write("• Contact your broker about API access to historical data")
                                    else:
                                        st.success(f"✅ Fetched {len(df)} trades from the last {tl_days} days")
                                        
                                        # Set up variables for analysis
                                        pnl = 'real_pnl'
                                        starting_balance = balance_value - df[pnl].sum()  # Calculate starting balance
                                        
                                        # Show data preview
                                        with st.expander("📊 Data Preview", expanded=False):
                                            st.dataframe(df.head(), use_container_width=True)
                                            
                                            # Show column info for debugging
                                            st.write("**Available columns:**", list(df.columns))
                                            if len(df) > 0:
                                                st.write("**Date range in data:**", 
                                                       f"{df['open_time'].min()} to {df['close_time'].max()}" if 'open_time' in df.columns and 'close_time' in df.columns else "Date columns not found")
                                
                                except Exception as fetch_error:
                                    st.error(f"❌ Error fetching trading data: {str(fetch_error)}")
                                    st.info("💡 **Debug Information:**")
                                    st.write("• Connection was successful but data fetching failed")
                                    st.write("• This might be due to different API endpoints for historical data")
                                    st.write("• Try uploading a CSV export from your TradeLocker platform instead")
                        else:
                            st.error(f"❌ Connection failed: {connection_test['error']}")
                            
                            # Show debug information
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.write("**Error Details:**")
                                st.code(connection_test['error'])
                                
                                st.write("**Troubleshooting Steps:**")
                                st.write("1. **Check Credentials**: Verify email and password in TradeLocker platform")
                                st.write("2. **Verify Account ID**: Check your account ID in the trading platform")
                                st.write("3. **Try Different Account Number**: If you have multiple accounts, try 1, 2, or 3")
                                st.write("4. **Check Server Name**: Ensure server name matches your broker (e.g., GATESFX)")
                                st.write("5. **Account Type**: Make sure Demo/Live selection matches your account")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please fill in all required fields: Email, Password, Server Name, and Account ID")

    # Continue with analysis only if we have data
    if df is not None and not df.empty:
        
        # === DATE FILTERING SECTION ===
        st.markdown("---")
        st.markdown("### 📅 Date Range Filter")
        
        # Quick period buttons
        st.markdown("**Quick Select:**")
        quick_cols = st.columns(8)
        quick_periods = ["Today", "Yesterday", "This Week", "This Month", "Last Month", "This Year", "Last Year", "All Time"]
        
        selected_quick = None
        for i, period in enumerate(quick_periods):
            if quick_cols[i].button(period, key=f"quick_{period}"):
                selected_quick = period
        
        # Main period selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            period_options = [
                "All Time", "Today", "Yesterday", "This Week", "Last Week", 
                "This Month", "Last Month", "This Year", "Last Year", "Custom Range"
            ]
            
            # Use quick selection if clicked, otherwise use selectbox
            if selected_quick:
                selected_period = selected_quick
                # Update session state to reflect the selection
                if 'period_selection' not in st.session_state:
                    st.session_state.period_selection = selected_quick
            else:
                default_index = period_options.index(st.session_state.get('period_selection', 'All Time'))
                selected_period = st.selectbox("Select Time Period", period_options, index=default_index)
                st.session_state.period_selection = selected_period
        
        custom_start, custom_end = None, None
        if selected_period == "Custom Range":
            with col2:
                custom_start = st.date_input("Start Date")
            with col3:
                custom_end = st.date_input("End Date")
        
        # Get date range and filter data
        start_date, end_date = get_date_range(selected_period, custom_start, custom_end)
        
        # Show selected period info
        # Store original dataframe before any filtering for calendar use
        original_df = df.copy()
        
        if start_date and end_date:
            if start_date == end_date:
                st.info(f"📊 **Analyzing**: {selected_period} ({start_date})")
            else:
                st.info(f"📊 **Analyzing**: {selected_period} ({start_date} to {end_date})")
            
            # Filter the dataframe
            df = filter_dataframe_by_date(df, start_date, end_date)
            
            if len(df) == 0:
                st.error(f"❌ **No trades found** in the selected period ({start_date} to {end_date})")
                st.stop()
            elif len(df) < len(original_df):
                st.success(f"✅ **Filtered**: {len(df)} trades out of {len(original_df)} total trades")
        else:
            st.info(f"📊 **Analyzing**: All Time ({len(df)} trades)")
        
        # Recalculate equity curve for filtered data
        df["equity"] = df[pnl].cumsum()
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = df["equity"] - df["peak"]
        df["drawdown_pct"] = df["drawdown"] / df["peak"].replace(0, 1) * 100
        
        # Calculate KPIs and metrics
        if start_date and end_date and selected_period != "All Time":
            # For filtered periods, calculate the correct starting balance
            from analytics import compute_period_starting_balance
            
            # We need the original unfiltered dataframe to calculate cumulative PnL
            period_starting_balance = compute_period_starting_balance(original_df, df, pnl, starting_balance)
            top_kpis = compute_top_kpis(df, pnl, starting_balance_override=period_starting_balance)
        else:
            top_kpis = compute_top_kpis(df, pnl, starting_balance_override=starting_balance)

        # Compute all metrics on trade data only
        metrics = compute_metrics(df, pnl)
        df = compute_rolling_metrics(df, pnl)
        time_data = compute_time_analysis(df, pnl)
        risk_data = compute_risk_pain_metrics(df, pnl)
        trading_insights = compute_trading_insights(df, pnl)
        insights_summary = generate_trading_insights_summary(trading_insights)
        
        # Compute Kelly Criterion metrics
        current_equity_for_kelly = top_kpis.get('Current Equity', starting_balance)
        # Extract numeric value from the formatted string
        if isinstance(current_equity_for_kelly, str):
            current_equity_for_kelly = float(current_equity_for_kelly.replace('$', '').replace(',', ''))
        
        # Add user input for current lot size
        st.markdown("#### 💰 Your Current Position Sizing")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_current_lots = st.number_input(
                "What lot size are you currently using?", 
                min_value=0.001, 
                max_value=10.0, 
                value=0.1,  # Default based on user's mention
                step=0.001,
                format="%.3f",
                help="Enter your typical lot size to get personalized Kelly recommendations"
            )
        
        with col2:
            st.markdown("**Quick Reference:**")
            st.write(f"• Your equity: ${current_equity_for_kelly:,.2f}")
            st.write(f"• Traditional rate: {user_current_lots / (current_equity_for_kelly / 1000):.3f} per 1k")
            st.write(f"• Standard rate: 0.020 per 1k")
        
        try:
            kelly_metrics = compute_kelly_metrics(df, pnl, current_equity_for_kelly, user_current_lots)
        except Exception as e:
            st.error(f"❌ Error computing Kelly metrics: {str(e)}")
            kelly_metrics = None

        # === TOP KPIs SECTION ===
        st.markdown("---")
        st.markdown("### 📊 Account Overview")
        
        # Display KPIs in a grid
        kpi_cols = st.columns(4)
        kpi_items = list(top_kpis.items())
        
        for i, (key, value) in enumerate(kpi_items):
            with kpi_cols[i % 4]:
                st.metric(key, value)
        
        # === CHARTS SECTION ===
        st.markdown("---")
        st.markdown("### 📈 Performance Charts")
        
        # Equity curve
        fig_equity = equity_curve(df, pnl)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Drawdown curve
        fig_drawdown = drawdown_curve(df)
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        # Additional charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pnl_dist = pnl_distribution(df, pnl)
            st.plotly_chart(fig_pnl_dist, use_container_width=True)
        
        with col2:
            fig_win_loss = win_loss_pie(df, pnl)
            st.plotly_chart(fig_win_loss, use_container_width=True)
        
        # === DETAILED METRICS ===
        with st.expander("📊 Detailed Metrics", expanded=False):
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # === KELLY CRITERION ANALYSIS ===
        if kelly_metrics and kelly_metrics.get('has_kelly_data', False):
            st.markdown("---")
            st.markdown("### 🎯 Kelly Criterion Analysis")
            
            # Kelly metrics display
            kelly_cols = st.columns(4)
            
            try:
                with kelly_cols[0]:
                    st.metric("Kelly %", f"{kelly_metrics['kelly_fraction'] * 100:.2f}%")
                with kelly_cols[1]:
                    st.metric("Optimal Lot Size", f"{kelly_metrics['lot_recommendation']['recommended_lot_size']:.3f}")
                with kelly_cols[2]:
                    current_vs_optimal = kelly_metrics['analysis_summary'].get('user_vs_kelly_ratio', 1.0) or 1.0
                    st.metric("Current vs Optimal", f"{current_vs_optimal:.1f}x")
                with kelly_cols[3]:
                    st.metric("Risk Level", kelly_metrics['lot_recommendation']['risk_assessment'])
                
                # Kelly charts
                try:
                    kelly_charts = create_kelly_criterion_charts(kelly_metrics)
                    for chart_name, chart in kelly_charts.items():
                        if chart is not None:
                            st.plotly_chart(chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not create Kelly Criterion charts: {str(e)}")
                    st.info("📊 Kelly analysis data is still available in the metrics above.")
                    
            except Exception as e:
                st.error(f"❌ Error displaying Kelly metrics: {str(e)}")
                
        elif kelly_metrics and kelly_metrics.get('error'):
            st.warning(f"⚠️ Kelly Criterion Analysis: {kelly_metrics['error']}")
        else:
            st.info("📊 Kelly Criterion analysis requires sufficient trading data with both wins and losses.")
    
    else:
        st.info("📊 Upload a CSV file or connect to TradeLocker API to start analyzing your trading performance.")

# === MAIN APP ===
st.set_page_config(layout="wide", page_title=" FundedBeco Trading Performance Intelligence")
st.markdown(inject_css(), unsafe_allow_html=True)
st.title("🎯 FundedBeco Trading Performance Intelligence")
st.markdown("**FundedBeco Strategy Diagnostics Console**")

# === TAB NAVIGATION ===
tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🤖 AI Strategy Builder", "⚡ Backtesting Engine"])

with tab1:
    # Original analytics dashboard content goes here
    analytics_dashboard()

with tab2:
    # AI Strategy Builder interface
    ai_strategy_builder_interface()

with tab3:
    # Backtesting Engine interface
    backtesting_engine_interface()