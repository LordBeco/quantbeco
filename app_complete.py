import streamlit as st
import pandas as pd
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
    # Backtesting Engine interface (placeholder for now)
    st.markdown("### ⚡ Backtesting Engine")
    st.info("🚧 Backtesting Engine interface will be implemented in the next subtask")
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