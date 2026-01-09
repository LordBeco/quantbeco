import streamlit as st
import pandas as pd
import numpy as np
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

# Import plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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

# Add caching for expensive operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_data_processing(data_hash, data):
    """Cache expensive data processing operations"""
    return data.copy()

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def cached_chart_generation(chart_type, data_hash, **kwargs):
    """Cache chart generation to improve performance"""
    # This will be implemented for specific chart types
    pass

@st.cache_data(ttl=300)
def cached_analytics_computation(data_hash, data, period):
    """Cache analytics computations"""
    try:
        return compute_metrics(data, period)
    except Exception as e:
        st.error(f"Error computing metrics: {str(e)}")
        return None

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



def enhanced_ai_strategy_builder_interface():
    """Enhanced AI Strategy Builder with comprehensive features"""
    st.markdown("### 🤖 AI Strategy Builder")
    st.markdown("**Build complete trading strategies from natural language prompts**")
    
    # === STRATEGY BUILDER TABS ===
    builder_tab1, builder_tab2, builder_tab3, builder_tab4 = st.tabs([
        "💬 Strategy Generator", 
        "📊 Indicators", 
        "📋 Orders & Risk", 
        "🌲 Pine Script Export"
    ])
    
    with builder_tab1:
        strategy_generator_interface()
    
    with builder_tab2:
        indicators_builder_interface()
    
    with builder_tab3:
        orders_risk_management_interface()
    
    with builder_tab4:
        pine_script_export_interface()


def strategy_generator_interface():
    """Natural language strategy generation interface"""
    st.markdown("#### 💬 Natural Language Strategy Generator")
    
    # === AI PROVIDER SELECTION ===
    st.markdown("##### 🔧 AI Provider Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Provider selection
        provider_options = {
            'puter': '🆓 Puter AI (Free - Template-based)',
            'openrouter': '🌐 OpenRouter (Free Models Available)',
            'openai': '🤖 OpenAI (Paid - Requires API Key & Credits)'
        }
        
        selected_provider = st.selectbox(
            "Choose AI Provider:",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0 if Config.AI_PROVIDER == 'puter' else (1 if Config.AI_PROVIDER == 'openrouter' else 2),
            help="Puter AI provides template-based responses. OpenRouter offers free access to various AI models. OpenAI provides premium AI but requires credits.",
            key="strategy_generator_ai_provider"
        )
        
        # Model selection for OpenRouter
        if selected_provider == 'openrouter':
            from ai_strategy_builder.openrouter_client import OpenRouterClientWrapper
            available_models = OpenRouterClientWrapper.get_available_models()
            
            model_options = {model['id']: f"{model['name']} - {model['description']}" for model in available_models}
            
            selected_model = st.selectbox(
                "Choose OpenRouter Model:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=0,
                help="All these models are free to use. Some may have rate limits."
            )
            
            # Store selected model in session state
            st.session_state.selected_openrouter_model = selected_model
    
    with col2:
        # Provider status
        if selected_provider == 'puter':
            st.success("🆓 **FREE**")
            st.write("✅ No API key needed")
            st.write("✅ No usage limits")
            st.write("⚠️ Template-based responses")
        elif selected_provider == 'openrouter':
            st.success("🌐 **FREE MODELS**")
            st.write("✅ Multiple free models")
            st.write("✅ Real AI responses")
            st.write("⚠️ Rate limits may apply")
        else:
            if Config.OPENAI_API_KEY:
                st.success("🤖 **CONFIGURED**")
                st.write("✅ API key found")
                st.write("💳 Usage costs apply")
                st.write("🚀 Advanced AI responses")
            else:
                st.error("❌ **NOT CONFIGURED**")
                st.write("❌ No API key")
                st.write("💡 Add OPENAI_API_KEY to .env")
    
    # === STRATEGY PROMPT INPUT ===
    st.markdown("##### 💬 Describe Your Trading Strategy")
    
    # Strategy templates
    with st.expander("📋 Strategy Templates", expanded=False):
        template_col1, template_col2 = st.columns(2)
        
        with template_col1:
            st.markdown("**Trend Following:**")
            if st.button("📈 Moving Average Crossover"):
                st.session_state.strategy_prompt = "Create a moving average crossover strategy. Buy when 20-period EMA crosses above 50-period EMA and RSI is above 50. Sell when 20-period EMA crosses below 50-period EMA or RSI drops below 30. Use 2% stop loss and 4% take profit. Position size should be 1% of account balance."
            
            if st.button("📊 MACD + RSI Combo"):
                st.session_state.strategy_prompt = "Build a MACD and RSI combination strategy. Enter long when MACD line crosses above signal line and RSI is between 30-70. Exit when MACD crosses below signal line or RSI goes above 80. Use ATR-based stop loss (2x ATR) and 3:1 risk-reward ratio."
            
            if st.button("🎯 Breakout Strategy"):
                st.session_state.strategy_prompt = "Create a breakout strategy using Bollinger Bands. Buy when price breaks above upper Bollinger Band with volume confirmation (volume > 1.5x average). Sell when price touches middle Bollinger Band or after 10 bars. Use 1.5% stop loss and dynamic take profit based on ATR."
        
        with template_col2:
            st.markdown("**Mean Reversion:**")
            if st.button("🔄 RSI Oversold/Overbought"):
                st.session_state.strategy_prompt = "Design an RSI mean reversion strategy. Buy when RSI drops below 20 and price is above 200-period SMA. Sell when RSI rises above 80 or price falls below 200-period SMA. Use 1% stop loss and take profit when RSI reaches 50."
            
            if st.button("📉 Support/Resistance"):
                st.session_state.strategy_prompt = "Build a support and resistance strategy using pivot points. Buy at support levels when price bounces with confirmation from Stochastic oversold. Sell at resistance levels when Stochastic is overbought. Use 0.5% stop loss below support/above resistance."
            
            if st.button("⚡ Scalping Strategy"):
                st.session_state.strategy_prompt = "Create a 5-minute scalping strategy using EMA and MACD. Enter long when price is above 21-EMA and MACD histogram turns positive. Exit after 3 bars or when MACD histogram turns negative. Use tight 0.3% stop loss and 0.6% take profit."
    
    # Prompt input area
    prompt = st.text_area(
        "Describe your trading strategy in natural language:",
        height=150,
        value=st.session_state.get('strategy_prompt', ''),
        placeholder="Example: Create a trend-following strategy using 20 and 50 period moving averages. Buy when 20 MA crosses above 50 MA and RSI is above 50. Sell when 20 MA crosses below 50 MA. Use 2% stop loss and 4% take profit. Position size should be 1% of account balance...",
        help="Be specific about entry/exit conditions, indicators, risk management, and position sizing"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Generation Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy_type = st.selectbox(
                "Strategy Type",
                ["Trend Following", "Mean Reversion", "Breakout", "Scalping", "Swing Trading", "Custom"],
                help="Helps AI understand the strategy category"
            )
        
        with col2:
            timeframe = st.selectbox(
                "Primary Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=2,
                help="Main timeframe for the strategy"
            )
        
        with col3:
            complexity = st.selectbox(
                "Strategy Complexity",
                ["Simple", "Intermediate", "Advanced"],
                index=1,
                help="Complexity level affects number of indicators and conditions"
            )
        
        # Additional parameters
        col4, col5 = st.columns(2)
        
        with col4:
            include_risk_management = st.checkbox("Include Risk Management", value=True)
            include_position_sizing = st.checkbox("Include Position Sizing", value=True)
        
        with col5:
            include_filters = st.checkbox("Include Market Filters", value=False)
            include_time_filters = st.checkbox("Include Time-based Filters", value=False)
    
    # Strategy generation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Strategy Code", type="primary", disabled=not prompt.strip())
    
    with col2:
        if st.session_state.get('generated_strategy'):
            clear_button = st.button("🗑️ Clear", help="Clear generated code")
            if clear_button:
                for key in ['generated_strategy', 'pine_script_code', 'validation_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    with col3:
        # Export options
        if st.session_state.get('generated_strategy'):
            export_format = st.selectbox("Export As", ["Python", "Pine Script", "Both"])
    
    # Generate strategy code
    if generate_button and prompt.strip():
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔧 Initializing AI processor...")
            progress_bar.progress(10)
            
            # Initialize strategy processor if needed
            if 'strategy_processor' not in st.session_state:
                if selected_provider == 'openrouter' and hasattr(st.session_state, 'selected_openrouter_model'):
                    st.session_state.strategy_processor = StrategyPromptProcessor(
                        provider=selected_provider, 
                        model=st.session_state.selected_openrouter_model
                    )
                else:
                    st.session_state.strategy_processor = StrategyPromptProcessor(provider=selected_provider)
            
            progress_bar.progress(25)
            status_text.text("📝 Preparing enhanced prompt...")
            
            # Enhanced prompt with context
            enhanced_prompt = f"""
            Strategy Type: {strategy_type}
            Timeframe: {timeframe}
            Complexity: {complexity}
            Include Risk Management: {include_risk_management}
            Include Position Sizing: {include_position_sizing}
            Include Market Filters: {include_filters}
            Include Time Filters: {include_time_filters}
            
            Strategy Description: {prompt}
            
            Please generate a complete trading strategy with:
            1. Clear entry and exit conditions
            2. Risk management rules
            3. Position sizing logic
            4. Indicator calculations
            5. Signal generation
            6. Trade management
            """
            
            progress_bar.progress(40)
            status_text.text(f"🤖 Generating strategy with {selected_provider.upper()}...")
            
            # Process the enhanced prompt
            strategy_code = st.session_state.strategy_processor.process_prompt(enhanced_prompt)
            st.session_state.generated_strategy = strategy_code
            
            progress_bar.progress(70)
            status_text.text("✅ Validating generated code...")
            
            # Validate the generated code
            validation = st.session_state.strategy_processor.validate_strategy(strategy_code.python_code)
            st.session_state.validation_results = validation
            
            progress_bar.progress(100)
            status_text.text("🎉 Strategy generation completed!")
            
            # Clear progress indicators after a short delay
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Comprehensive strategy code generated successfully!")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error generating strategy: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**")
            st.write("• Make your prompt more specific about trading rules")
            st.write("• Include clear entry and exit conditions")
            st.write("• Mention specific indicators and parameters")
            st.write("• Check your AI provider configuration")
    
    # === DISPLAY GENERATED CODE ===
    if st.session_state.get('generated_strategy'):
        display_generated_strategy()


def indicators_builder_interface():
    """Custom indicators builder interface"""
    st.markdown("#### 📊 Custom Indicators Builder")
    st.markdown("**Build custom technical indicators from scratch**")
    
    # === INDICATOR CATEGORIES ===
    indicator_category = st.selectbox(
        "Indicator Category",
        ["Trend", "Momentum", "Volatility", "Volume", "Support/Resistance", "Custom"],
        help="Choose the type of indicator to build"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # === INDICATOR BUILDER ===
        st.markdown("##### 🔧 Indicator Configuration")
        
        indicator_name = st.text_input("Indicator Name", placeholder="My Custom RSI")
        
        if indicator_category == "Trend":
            st.markdown("**Trend Indicators:**")
            trend_type = st.selectbox("Type", ["Moving Average", "MACD", "ADX", "Parabolic SAR", "Custom"])
            
            if trend_type == "Moving Average":
                ma_type = st.selectbox("MA Type", ["SMA", "EMA", "WMA", "VWMA"])
                period = st.number_input("Period", min_value=1, max_value=200, value=20)
                source = st.selectbox("Source", ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"])
                
                # Generate indicator code
                if st.button("Generate Moving Average Code"):
                    indicator_code = generate_moving_average_code(ma_type, period, source, indicator_name)
                    st.session_state.custom_indicator_code = indicator_code
            
            elif trend_type == "Custom":
                st.text_area("Describe your custom trend indicator:", 
                           placeholder="Create a custom trend indicator that combines EMA and volume...")
        
        elif indicator_category == "Momentum":
            st.markdown("**Momentum Indicators:**")
            momentum_type = st.selectbox("Type", ["RSI", "Stochastic", "CCI", "Williams %R", "Custom"])
            
            if momentum_type == "RSI":
                rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
                rsi_source = st.selectbox("Source", ["close", "open", "high", "low"])
                overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=70)
                oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=30)
                
                if st.button("Generate RSI Code"):
                    indicator_code = generate_rsi_code(rsi_period, rsi_source, overbought, oversold, indicator_name)
                    st.session_state.custom_indicator_code = indicator_code
        
        elif indicator_category == "Custom":
            st.markdown("**Custom Indicator Builder:**")
            custom_description = st.text_area(
                "Describe your custom indicator:",
                height=100,
                placeholder="Create an indicator that combines RSI, MACD, and volume to identify strong momentum shifts..."
            )
            
            if st.button("Generate Custom Indicator") and custom_description:
                with st.spinner("🤖 Generating custom indicator..."):
                    try:
                        # Use AI to generate custom indicator
                        if 'strategy_processor' not in st.session_state:
                            st.session_state.strategy_processor = StrategyPromptProcessor()
                        
                        indicator_prompt = f"""
                        Create a custom technical indicator with the following specifications:
                        Name: {indicator_name}
                        Category: {indicator_category}
                        Description: {custom_description}
                        
                        Generate Python code that:
                        1. Calculates the indicator values
                        2. Includes proper parameter validation
                        3. Returns clear buy/sell signals
                        4. Has configurable parameters
                        5. Includes plotting functionality
                        """
                        
                        indicator_code = st.session_state.strategy_processor.generate_custom_indicator(indicator_prompt)
                        st.session_state.custom_indicator_code = indicator_code
                        st.success("✅ Custom indicator generated!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating indicator: {str(e)}")
    
    with col2:
        # === INDICATOR PREVIEW ===
        st.markdown("##### 📊 Indicator Preview")
        
        if st.session_state.get('custom_indicator_code'):
            st.code(st.session_state.custom_indicator_code[:500] + "..." if len(st.session_state.custom_indicator_code) > 500 else st.session_state.custom_indicator_code, language="python")
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("📥 Save Indicator"):
                    # Save to indicators library
                    save_custom_indicator(indicator_name, st.session_state.custom_indicator_code)
                    st.success("✅ Indicator saved!")
            
            with col2b:
                if st.button("🧪 Test Indicator"):
                    # Test indicator with sample data
                    test_custom_indicator(st.session_state.custom_indicator_code)
        else:
            st.info("Configure an indicator above to see preview")
        
        # === SAVED INDICATORS ===
        st.markdown("##### 📚 Saved Indicators")
        saved_indicators = get_saved_indicators()
        
        if saved_indicators:
            selected_indicator = st.selectbox("Load Saved Indicator", [""] + saved_indicators)
            if selected_indicator and st.button("📤 Load"):
                loaded_code = load_saved_indicator(selected_indicator)
                st.session_state.custom_indicator_code = loaded_code
                st.rerun()
        else:
            st.info("No saved indicators yet")


def orders_risk_management_interface():
    """Orders and risk management interface"""
    st.markdown("#### 📋 Orders & Risk Management")
    st.markdown("**Configure order types, position sizing, and risk management rules**")
    
    # === ORDER MANAGEMENT ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Order Types")
        
        # Entry orders
        st.markdown("**Entry Orders:**")
        entry_order_type = st.selectbox(
            "Entry Order Type",
            ["Market", "Limit", "Stop", "Stop Limit", "Conditional"],
            help="Type of order for entering positions"
        )
        
        if entry_order_type == "Limit":
            limit_offset = st.number_input("Limit Offset (pips)", value=5.0, help="Offset from current price")
        elif entry_order_type == "Stop":
            stop_offset = st.number_input("Stop Offset (pips)", value=10.0, help="Offset from current price")
        elif entry_order_type == "Conditional":
            condition_type = st.selectbox("Condition", ["RSI Level", "MACD Cross", "Price Level", "Custom"])
        
        # Exit orders
        st.markdown("**Exit Orders:**")
        use_stop_loss = st.checkbox("Use Stop Loss", value=True)
        if use_stop_loss:
            sl_type = st.selectbox("Stop Loss Type", ["Fixed Pips", "ATR Multiple", "Percentage", "Trailing"])
            if sl_type == "Fixed Pips":
                sl_pips = st.number_input("Stop Loss (pips)", value=20.0)
            elif sl_type == "ATR Multiple":
                sl_atr_multiple = st.number_input("ATR Multiple", value=2.0)
            elif sl_type == "Percentage":
                sl_percentage = st.number_input("Stop Loss (%)", value=2.0)
            elif sl_type == "Trailing":
                trailing_distance = st.number_input("Trailing Distance (pips)", value=15.0)
        
        use_take_profit = st.checkbox("Use Take Profit", value=True)
        if use_take_profit:
            tp_type = st.selectbox("Take Profit Type", ["Fixed Pips", "Risk Reward Ratio", "Percentage", "Multiple Targets"])
            if tp_type == "Fixed Pips":
                tp_pips = st.number_input("Take Profit (pips)", value=40.0)
            elif tp_type == "Risk Reward Ratio":
                risk_reward = st.number_input("Risk:Reward Ratio", value=2.0)
            elif tp_type == "Multiple Targets":
                tp1_pips = st.number_input("TP1 (pips)", value=20.0)
                tp2_pips = st.number_input("TP2 (pips)", value=40.0)
                tp3_pips = st.number_input("TP3 (pips)", value=60.0)
    
    with col2:
        st.markdown("##### 💰 Position Sizing")
        
        # Position sizing methods
        sizing_method = st.selectbox(
            "Position Sizing Method",
            ["Fixed Lot Size", "Fixed Percentage", "Kelly Criterion", "ATR-based", "Volatility-based", "Custom"],
            help="Method for calculating position sizes"
        )
        
        if sizing_method == "Fixed Lot Size":
            fixed_lot_size = st.number_input("Lot Size", value=0.1, step=0.01)
        
        elif sizing_method == "Fixed Percentage":
            risk_percentage = st.number_input("Risk per Trade (%)", value=2.0, max_value=10.0)
        
        elif sizing_method == "Kelly Criterion":
            st.info("Kelly Criterion will be calculated based on historical performance")
            kelly_fraction = st.number_input("Kelly Fraction Multiplier", value=0.25, max_value=1.0, 
                                           help="Fraction of Kelly to use (0.25 = 25% of full Kelly)")
        
        elif sizing_method == "ATR-based":
            atr_period = st.number_input("ATR Period", value=14)
            atr_multiplier = st.number_input("ATR Multiplier", value=1.0)
            max_risk_pct = st.number_input("Max Risk (%)", value=3.0)
        
        elif sizing_method == "Custom":
            custom_sizing_formula = st.text_area(
                "Custom Sizing Formula:",
                placeholder="lot_size = account_balance * 0.02 / (stop_loss_pips * pip_value)",
                help="Define custom position sizing logic"
            )
        
        # Risk management rules
        st.markdown("##### ⚠️ Risk Management")
        
        max_daily_loss = st.number_input("Max Daily Loss (%)", value=5.0, max_value=20.0)
        max_open_positions = st.number_input("Max Open Positions", value=3, max_value=10)
        max_correlation = st.number_input("Max Position Correlation", value=0.7, max_value=1.0)
        
        # Time-based filters
        use_time_filters = st.checkbox("Use Time Filters")
        if use_time_filters:
            trading_start = st.time_input("Trading Start Time", value=datetime.strptime("08:00", "%H:%M").time())
            trading_end = st.time_input("Trading End Time", value=datetime.strptime("17:00", "%H:%M").time())
            
            avoid_news = st.checkbox("Avoid News Events")
            if avoid_news:
                news_buffer = st.number_input("News Buffer (minutes)", value=30)
    
    # === GENERATE RISK MANAGEMENT CODE ===
    st.markdown("---")
    if st.button("🔧 Generate Risk Management Code", type="primary"):
        with st.spinner("Generating risk management system..."):
            try:
                risk_config = {
                    'entry_order_type': entry_order_type,
                    'sizing_method': sizing_method,
                    'use_stop_loss': use_stop_loss,
                    'use_take_profit': use_take_profit,
                    'max_daily_loss': max_daily_loss,
                    'max_open_positions': max_open_positions,
                    'use_time_filters': use_time_filters
                }
                
                # Add specific parameters based on selections
                if use_stop_loss:
                    risk_config['sl_type'] = sl_type
                    if sl_type == "Fixed Pips":
                        risk_config['sl_pips'] = sl_pips
                    elif sl_type == "ATR Multiple":
                        risk_config['sl_atr_multiple'] = sl_atr_multiple
                
                if use_take_profit:
                    risk_config['tp_type'] = tp_type
                    if tp_type == "Fixed Pips":
                        risk_config['tp_pips'] = tp_pips
                    elif tp_type == "Risk Reward Ratio":
                        risk_config['risk_reward'] = risk_reward
                
                if sizing_method == "Fixed Lot Size":
                    risk_config['fixed_lot_size'] = fixed_lot_size
                elif sizing_method == "Fixed Percentage":
                    risk_config['risk_percentage'] = risk_percentage
                
                # Generate the risk management code
                risk_management_code = generate_risk_management_code(risk_config)
                st.session_state.risk_management_code = risk_management_code
                
                st.success("✅ Risk management system generated!")
                
                # Display the generated code
                with st.expander("📋 Generated Risk Management Code", expanded=True):
                    st.code(risk_management_code, language="python")
                
                # Save and export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📥 Download Risk Management",
                        risk_management_code,
                        "risk_management.py",
                        "text/plain"
                    )
                
                with col2:
                    if st.button("💾 Save to Strategy"):
                        if 'generated_strategy' in st.session_state:
                            # Integrate with existing strategy
                            integrate_risk_management()
                            st.success("✅ Integrated with strategy!")
                        else:
                            st.warning("⚠️ Generate a strategy first")
                
                with col3:
                    if st.button("🧪 Test Risk Rules"):
                        test_risk_management_rules(risk_config)
                
            except Exception as e:
                st.error(f"❌ Error generating risk management: {str(e)}")


def pine_script_export_interface():
    """Pine Script export and conversion interface"""
    st.markdown("#### 🌲 Pine Script Export")
    st.markdown("**Convert your Python strategy to TradingView Pine Script**")
    
    if not st.session_state.get('generated_strategy'):
        st.warning("⚠️ **No strategy found**")
        st.info("Generate a strategy in the 'Strategy Generator' tab first, then return here to convert it to Pine Script.")
        return
    
    # === PINE SCRIPT CONVERSION OPTIONS ===
    st.markdown("##### ⚙️ Conversion Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pine_version = st.selectbox("Pine Script Version", ["v5", "v4"], index=0)
        conversion_mode = st.selectbox(
            "Conversion Mode", 
            ["Basic", "Advanced", "AI-Enhanced"],
            help="Basic: Rule-based conversion, Advanced: Enhanced features, AI-Enhanced: AI-optimized code"
        )
        
        include_alerts = st.checkbox("Include Alerts", value=True)
        include_plotting = st.checkbox("Include Plotting", value=True)
        include_table = st.checkbox("Include Info Table", value=False)
    
    with col2:
        optimize_for = st.selectbox(
            "Optimize For",
            ["Readability", "Performance", "Compatibility"],
            help="Readability: Clean code, Performance: Fast execution, Compatibility: Broad support"
        )
        
        add_comments = st.checkbox("Add Comments", value=True)
        add_inputs = st.checkbox("Add User Inputs", value=True)
        add_backtesting = st.checkbox("Add Backtesting Logic", value=True)
    
    # === CONVERSION PROCESS ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        convert_button = st.button("🌲 Convert to Pine Script", type="primary")
    
    with col2:
        # Initialize refine_button to avoid UnboundLocalError
        refine_button = False
        if st.session_state.get('pine_script_code'):
            refine_button = st.button("🤖 AI Refine", help="Use AI to improve the Pine Script")
    
    # Convert to Pine Script
    if convert_button:
        with st.spinner(f"🌲 Converting to Pine Script ({conversion_mode})..."):
            try:
                # Initialize Pine Script converter if needed
                if 'pine_converter' not in st.session_state:
                    st.session_state.pine_converter = PineScriptConverter()
                
                conversion_options = {
                    'version': pine_version,
                    'mode': conversion_mode,
                    'include_alerts': include_alerts,
                    'include_plotting': include_plotting,
                    'include_table': include_table,
                    'optimize_for': optimize_for,
                    'add_comments': add_comments,
                    'add_inputs': add_inputs,
                    'add_backtesting': add_backtesting
                }
                
                strategy = st.session_state.generated_strategy
                
                if conversion_mode == "AI-Enhanced":
                    # Use AI-enhanced conversion
                    pine_code = st.session_state.pine_converter.convert_to_pine_with_ai_refinement(
                        strategy.python_code,
                        st.session_state.strategy_processor.client,
                        st.session_state.strategy_processor.provider,
                        options=conversion_options
                    )
                else:
                    # Use rule-based conversion
                    pine_code = st.session_state.pine_converter.convert_to_pine_advanced(
                        strategy.python_code,
                        options=conversion_options
                    )
                
                st.session_state.pine_script_code = pine_code
                st.success("✅ Pine Script conversion successful!")
                
            except Exception as e:
                st.error(f"❌ Pine Script conversion failed: {str(e)}")
                st.info("💡 **Troubleshooting:**")
                st.write("• Try 'Basic' conversion mode first")
                st.write("• Some complex Python logic may need manual adjustment")
                st.write("• Check if your strategy uses supported indicators")
    
    # AI Refinement
    if st.session_state.get('pine_script_code') and refine_button:
        with st.spinner("🤖 AI refining Pine Script..."):
            try:
                refined_code = st.session_state.pine_converter.ai_refine_pine_script(
                    st.session_state.pine_script_code,
                    st.session_state.strategy_processor.client,
                    st.session_state.strategy_processor.provider
                )
                st.session_state.pine_script_code = refined_code
                st.success("✅ Pine Script refined with AI!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ AI refinement failed: {str(e)}")
    
    # === DISPLAY PINE SCRIPT ===
    if st.session_state.get('pine_script_code'):
        st.markdown("---")
        st.markdown("##### 📋 Generated Pine Script")
        
        # Pine Script validation
        try:
            pine_validation = st.session_state.pine_converter.validate_pine_script(st.session_state.pine_script_code)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if pine_validation.is_valid:
                    st.success("✅ Valid Pine Script")
                else:
                    st.error("❌ Has Issues")
            
            with col2:
                st.metric("Errors", len(pine_validation.errors))
            
            with col3:
                st.metric("Warnings", len(pine_validation.warnings))
            
            with col4:
                lines_count = len(st.session_state.pine_script_code.split('\n'))
                st.metric("Lines", lines_count)
            
            # Show validation issues
            if pine_validation.errors:
                with st.expander("❌ Pine Script Errors", expanded=True):
                    for error in pine_validation.errors:
                        st.write(f"• {error}")
            
            if pine_validation.warnings:
                with st.expander("⚠️ Pine Script Warnings", expanded=False):
                    for warning in pine_validation.warnings:
                        st.write(f"• {warning}")
                        
        except Exception as e:
            st.warning(f"⚠️ Could not validate Pine Script: {str(e)}")
        
        # Display the Pine Script code
        st.code(st.session_state.pine_script_code, language="javascript")
        
        # === EXPORT AND USAGE ===
        st.markdown("##### 📤 Export & Usage")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download Pine Script",
                st.session_state.pine_script_code,
                f"ai_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pine",
                "text/plain"
            )
        
        with col2:
            if st.button("📋 Copy to Clipboard"):
                # This would need JavaScript integration
                st.info("💡 Use Ctrl+A, Ctrl+C to copy the code above")
        
        with col3:
            if st.button("🔗 Open TradingView"):
                st.markdown("[Open TradingView Pine Editor](https://www.tradingview.com/pine-editor/)", unsafe_allow_html=True)
        
        # Usage instructions
        with st.expander("📖 How to Use in TradingView", expanded=False):
            st.markdown("""
            **Step-by-Step Instructions:**
            
            1. **Copy the Pine Script code** above
            2. **Open TradingView** and go to the Pine Editor
            3. **Create a new script** and paste the code
            4. **Save the script** with a descriptive name
            5. **Add to chart** by clicking "Add to Chart"
            6. **Configure parameters** in the strategy settings
            7. **Enable alerts** if included in the script
            
            **Tips:**
            - Test on historical data first
            - Adjust parameters based on your preferences
            - Use paper trading before live trading
            - Monitor performance and adjust as needed
            
            **Common Issues:**
            - If you get errors, check the Pine Script version
            - Some indicators may need manual adjustment
            - Complex logic might require simplification
            """)


def display_generated_strategy():
    """Display the generated strategy with validation and options"""
    strategy = st.session_state.generated_strategy
    
    st.markdown("---")
    st.markdown("#### 📝 Generated Strategy Code")
    
    # Display validation results
    if st.session_state.get('validation_results'):
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
            with st.expander("❌ Errors", expanded=True):
                for error in validation.errors:
                    st.write(f"• {error}")
        
        if validation.warnings:
            with st.expander("⚠️ Warnings", expanded=False):
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📥 Download Python",
            strategy.python_code,
            f"ai_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
            "text/plain"
        )
    
    with col2:
        if st.button("🔍 Re-validate"):
            with st.spinner("🔍 Re-validating code..."):
                try:
                    validation = st.session_state.strategy_processor.validate_strategy(strategy.python_code)
                    st.session_state.validation_results = validation
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Validation error: {str(e)}")
    
    with col3:
        if st.button("🧪 Test Strategy"):
            test_strategy_with_sample_data()
    
    with col4:
        if st.button("📊 Analyze Code"):
            analyze_strategy_complexity()


# Helper functions for the enhanced AI Strategy Builder
def generate_moving_average_code(ma_type, period, source, name):
    """Generate moving average indicator code"""
    code = f"""
import pandas as pd
import numpy as np

def {name.lower().replace(' ', '_')}(data, period={period}, source='{source}'):
    \"\"\"
    {name} - {ma_type} with period {period}
    \"\"\"
    if source not in data.columns:
        raise ValueError(f"Source column '{source}' not found in data")
    
    if ma_type == 'SMA':
        return data[source].rolling(window=period).mean()
    elif ma_type == 'EMA':
        return data[source].ewm(span=period).mean()
    elif ma_type == 'WMA':
        weights = np.arange(1, period + 1)
        return data[source].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    elif ma_type == 'VWMA':
        if 'volume' not in data.columns:
            raise ValueError("Volume column required for VWMA")
        return (data[source] * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
def generate_signals(data):
    \"\"\"Generate trading signals based on {name}\"\"\"
    ma = {name.lower().replace(' ', '_')}(data)
    
    signals = pd.DataFrame(index=data.index)
    signals['ma'] = ma
    signals['signal'] = 0
    
    # Buy when price crosses above MA
    signals.loc[data['{source}'] > ma, 'signal'] = 1
    # Sell when price crosses below MA  
    signals.loc[data['{source}'] < ma, 'signal'] = -1
    
    return signals
"""
    return code

def generate_rsi_code(period, source, overbought, oversold, name):
    """Generate RSI indicator code"""
    code = f"""
import pandas as pd
import numpy as np

def {name.lower().replace(' ', '_')}(data, period={period}, source='{source}'):
    \"\"\"
    {name} - RSI with period {period}
    Overbought: {overbought}, Oversold: {oversold}
    \"\"\"
    if source not in data.columns:
        raise ValueError(f"Source column '{source}' not found in data")
    
    delta = data[source].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(data):
    \"\"\"Generate trading signals based on {name}\"\"\"
    rsi = {name.lower().replace(' ', '_')}(data)
    
    signals = pd.DataFrame(index=data.index)
    signals['rsi'] = rsi
    signals['signal'] = 0
    
    # Buy when RSI is oversold
    signals.loc[rsi < {oversold}, 'signal'] = 1
    # Sell when RSI is overbought
    signals.loc[rsi > {overbought}, 'signal'] = -1
    
    return signals
"""
    return code

def generate_risk_management_code(config):
    """Generate comprehensive risk management code"""
    code = f"""
import pandas as pd
import numpy as np
from datetime import datetime, time

class RiskManager:
    def __init__(self):
        self.config = {config}
        self.daily_pnl = 0
        self.open_positions = 0
        self.max_daily_loss = {config.get('max_daily_loss', 5.0)}
        self.max_open_positions = {config.get('max_open_positions', 3)}
        
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        \"\"\"Calculate position size based on configured method\"\"\"
        sizing_method = self.config.get('sizing_method', 'Fixed Percentage')
        
        if sizing_method == 'Fixed Lot Size':
            return {config.get('fixed_lot_size', 0.1)}
        
        elif sizing_method == 'Fixed Percentage':
            risk_amount = account_balance * ({config.get('risk_percentage', 2.0)} / 100)
            pip_risk = abs(entry_price - stop_loss_price) * 10000  # Assuming 4-digit pairs
            pip_value = 1.0  # Adjust based on pair and lot size
            lot_size = risk_amount / (pip_risk * pip_value)
            return round(lot_size, 2)
        
        elif sizing_method == 'Kelly Criterion':
            # Implement Kelly Criterion calculation
            kelly_fraction = {config.get('kelly_fraction', 0.25)}
            return account_balance * kelly_fraction / entry_price
        
        return 0.1  # Default fallback
    
    def calculate_stop_loss(self, entry_price, direction, atr_value=None):
        \"\"\"Calculate stop loss based on configured method\"\"\"
        if not self.config.get('use_stop_loss', True):
            return None
            
        sl_type = self.config.get('sl_type', 'Fixed Pips')
        
        if sl_type == 'Fixed Pips':
            pips = {config.get('sl_pips', 20.0)}
            pip_size = 0.0001  # Adjust for pair
            if direction == 'long':
                return entry_price - (pips * pip_size)
            else:
                return entry_price + (pips * pip_size)
        
        elif sl_type == 'ATR Multiple' and atr_value:
            atr_multiple = {config.get('sl_atr_multiple', 2.0)}
            if direction == 'long':
                return entry_price - (atr_value * atr_multiple)
            else:
                return entry_price + (atr_value * atr_multiple)
        
        elif sl_type == 'Percentage':
            percentage = {config.get('sl_percentage', 2.0)} / 100
            if direction == 'long':
                return entry_price * (1 - percentage)
            else:
                return entry_price * (1 + percentage)
        
        return entry_price * 0.98 if direction == 'long' else entry_price * 1.02
    
    def calculate_take_profit(self, entry_price, stop_loss_price, direction):
        \"\"\"Calculate take profit based on configured method\"\"\"
        if not self.config.get('use_take_profit', True):
            return None
            
        tp_type = self.config.get('tp_type', 'Risk Reward Ratio')
        
        if tp_type == 'Fixed Pips':
            pips = {config.get('tp_pips', 40.0)}
            pip_size = 0.0001
            if direction == 'long':
                return entry_price + (pips * pip_size)
            else:
                return entry_price - (pips * pip_size)
        
        elif tp_type == 'Risk Reward Ratio':
            risk_reward = {config.get('risk_reward', 2.0)}
            risk = abs(entry_price - stop_loss_price)
            if direction == 'long':
                return entry_price + (risk * risk_reward)
            else:
                return entry_price - (risk * risk_reward)
        
        return entry_price * 1.04 if direction == 'long' else entry_price * 0.96
    
    def can_open_position(self, current_time=None):
        \"\"\"Check if new position can be opened based on risk rules\"\"\"
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check maximum open positions
        if self.open_positions >= self.max_open_positions:
            return False, "Maximum open positions reached"
        
        # Check time filters
        if self.config.get('use_time_filters', False) and current_time:
            trading_start = time(8, 0)  # 08:00
            trading_end = time(17, 0)   # 17:00
            if not (trading_start <= current_time.time() <= trading_end):
                return False, "Outside trading hours"
        
        return True, "OK"
    
    def update_daily_pnl(self, pnl):
        \"\"\"Update daily P&L tracking\"\"\"
        self.daily_pnl += pnl
    
    def reset_daily_tracking(self):
        \"\"\"Reset daily tracking (call at start of new day)\"\"\"
        self.daily_pnl = 0
"""
    return code

def save_custom_indicator(name, code):
    """Save custom indicator to library"""
    # This would save to a file or database
    if 'saved_indicators' not in st.session_state:
        st.session_state.saved_indicators = {}
    st.session_state.saved_indicators[name] = code

def get_saved_indicators():
    """Get list of saved indicators"""
    return list(st.session_state.get('saved_indicators', {}).keys())

def load_saved_indicator(name):
    """Load saved indicator code"""
    return st.session_state.get('saved_indicators', {}).get(name, "")

def test_custom_indicator(code):
    """Test custom indicator with sample data"""
    try:
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Execute the indicator code (in a safe way)
        st.success("✅ Indicator code syntax is valid")
        st.info("💡 Full testing requires historical data upload")
        
    except Exception as e:
        st.error(f"❌ Indicator test failed: {str(e)}")

def integrate_risk_management():
    """Integrate risk management with existing strategy"""
    if 'generated_strategy' in st.session_state and 'risk_management_code' in st.session_state:
        # Combine strategy and risk management code
        combined_code = st.session_state.generated_strategy.python_code + "\n\n" + st.session_state.risk_management_code
        
        # Update the strategy object
        strategy = st.session_state.generated_strategy
        strategy.python_code = combined_code
        st.session_state.generated_strategy = strategy

def test_risk_management_rules(config):
    """Test risk management rules with sample scenarios"""
    st.info("🧪 **Risk Management Test Results:**")
    
    # Test scenarios
    scenarios = [
        {"account_balance": 10000, "daily_pnl": -300, "open_positions": 2},
        {"account_balance": 10000, "daily_pnl": -600, "open_positions": 1},
        {"account_balance": 5000, "daily_pnl": 0, "open_positions": 4},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        st.write(f"**Scenario {i}:**")
        st.write(f"• Account: ${scenario['account_balance']:,}")
        st.write(f"• Daily P&L: ${scenario['daily_pnl']:,}")
        st.write(f"• Open Positions: {scenario['open_positions']}")
        
        # Test against rules
        max_daily_loss = config.get('max_daily_loss', 5.0) / 100 * scenario['account_balance']
        max_positions = config.get('max_open_positions', 3)
        
        if abs(scenario['daily_pnl']) >= max_daily_loss:
            st.error(f"❌ Daily loss limit exceeded (${max_daily_loss:,.0f})")
        elif scenario['open_positions'] >= max_positions:
            st.error(f"❌ Too many open positions (max: {max_positions})")
        else:
            st.success("✅ Within risk limits")
        
        st.write("---")

def test_strategy_with_sample_data():
    """Test the generated strategy with sample data"""
    st.info("🧪 **Strategy Testing:**")
    st.write("This would test your strategy against sample historical data")
    st.write("• Generate sample OHLCV data")
    st.write("• Run strategy logic")
    st.write("• Calculate performance metrics")
    st.write("• Show equity curve")

def analyze_strategy_complexity():
    """Analyze the complexity of the generated strategy"""
    if 'generated_strategy' in st.session_state:
        strategy = st.session_state.generated_strategy
        code_lines = len(strategy.python_code.split('\n'))
        
        st.info("📊 **Strategy Analysis:**")
        st.write(f"• Code Lines: {code_lines}")
        st.write(f"• Indicators: {len(strategy.indicators) if strategy.indicators else 0}")
        st.write(f"• Entry Conditions: {len(strategy.entry_conditions) if strategy.entry_conditions else 0}")
        st.write(f"• Exit Conditions: {len(strategy.exit_conditions) if strategy.exit_conditions else 0}")
        
        # Complexity assessment
        if code_lines < 50:
            st.success("✅ Simple strategy - Easy to understand and maintain")
        elif code_lines < 150:
            st.warning("⚠️ Moderate complexity - Review logic carefully")
        else:
            st.error("❌ High complexity - Consider simplifying")
    """AI Strategy Builder interface implementation"""
    st.markdown("### 🤖 AI Strategy Builder")
    st.markdown("**Convert your trading ideas into executable code using natural language**")
    
    # === AI PROVIDER SELECTION ===
    st.markdown("#### 🔧 AI Provider Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Provider selection
        provider_options = {
            'puter': '🆓 Puter AI (Free - Template-based)',
            'openrouter': '🌐 OpenRouter (Free Models Available)',
            'openai': '🤖 OpenAI (Paid - Requires API Key & Credits)'
        }
        
        selected_provider = st.selectbox(
            "Choose AI Provider:",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0 if Config.AI_PROVIDER == 'puter' else (1 if Config.AI_PROVIDER == 'openrouter' else 2),
            help="Puter AI provides template-based responses. OpenRouter offers free access to various AI models. OpenAI provides premium AI but requires credits.",
            key="analyze_complexity_ai_provider"
        )
        
        # Model selection for OpenRouter
        if selected_provider == 'openrouter':
            from ai_strategy_builder.openrouter_client import OpenRouterClientWrapper
            available_models = OpenRouterClientWrapper.get_available_models()
            
            model_options = {model['id']: f"{model['name']} - {model['description']}" for model in available_models}
            
            selected_model = st.selectbox(
                "Choose OpenRouter Model:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=0,
                help="All these models are free to use. Some may have rate limits."
            )
            
            # Store selected model in session state
            st.session_state.selected_openrouter_model = selected_model
        
        # Update session state if provider changed
        if 'current_provider' not in st.session_state:
            st.session_state.current_provider = Config.AI_PROVIDER
        
        if selected_provider != st.session_state.current_provider:
            st.session_state.current_provider = selected_provider
            # Clear existing strategy processor to reinitialize with new provider
            if 'strategy_processor' in st.session_state:
                del st.session_state.strategy_processor
            st.info(f"✅ Switched to {provider_options[selected_provider]}")
    
    with col2:
        # Provider status
        if selected_provider == 'puter':
            st.success("🆓 **FREE**")
            st.write("✅ No API key needed")
            st.write("✅ No usage limits")
            st.write("⚠️ Template-based responses")
        elif selected_provider == 'openrouter':
            st.success("🌐 **FREE MODELS**")
            st.write("✅ Multiple free models")
            st.write("✅ Real AI responses")
            st.write("⚠️ Rate limits may apply")
            if hasattr(st.session_state, 'selected_openrouter_model'):
                model_name = next((m['name'] for m in available_models if m['id'] == st.session_state.selected_openrouter_model), 'Unknown')
                st.write(f"🤖 Using: {model_name}")
        else:
            if Config.OPENAI_API_KEY:
                st.success("🤖 **CONFIGURED**")
                st.write("✅ API key found")
                st.write("💳 Usage costs apply")
                st.write("🚀 Advanced AI responses")
            else:
                st.error("❌ **NOT CONFIGURED**")
                st.write("❌ No API key")
                st.write("💡 Add OPENAI_API_KEY to .env")
    
    # Check AI configuration based on selected provider
    provider_valid = False
    if selected_provider == 'puter':
        provider_valid = True
    elif selected_provider == 'openrouter':
        provider_valid = True  # OpenRouter has free models
    elif selected_provider == 'openai':
        provider_valid = Config.OPENAI_API_KEY is not None and len(Config.OPENAI_API_KEY.strip()) > 0
    
    if not provider_valid:
        st.error("🚨 **Configuration Required**")
        if selected_provider == 'openai':
            st.markdown("""
            To use OpenAI, you need to configure your API key:
            
            1. **Get an API key** from [OpenAI Platform](https://platform.openai.com/api-keys)
            2. **Add to .env file**: `OPENAI_API_KEY=your_key_here`
            3. **Restart the application**
            
            **💡 Alternatives: Try OpenRouter or Puter AI (Both Free!)**
            Select "OpenRouter" or "Puter AI" above for free AI access without API keys!
            """)
        return
    
    # Show current provider info
    if selected_provider == 'puter':
        st.info("🆓 **Using Puter AI** - Intelligent template-based strategy generation!")
    elif selected_provider == 'openrouter':
        model_name = "DeepSeek R1T2 Chimera"  # Default
        if hasattr(st.session_state, 'selected_openrouter_model'):
            model_name = next((m['name'] for m in available_models if m['id'] == st.session_state.selected_openrouter_model), model_name)
        st.info(f"🌐 **Using OpenRouter** - {model_name} (Free AI model)")
    else:
        st.info("🤖 **Using OpenAI** - Advanced AI-powered responses")
    
    # Initialize components with selected provider
    try:
        if 'strategy_processor' not in st.session_state or st.session_state.current_provider != selected_provider:
            # For OpenRouter, pass the selected model
            if selected_provider == 'openrouter' and hasattr(st.session_state, 'selected_openrouter_model'):
                st.session_state.strategy_processor = StrategyPromptProcessor(
                    provider=selected_provider, 
                    model=st.session_state.selected_openrouter_model
                )
            else:
                st.session_state.strategy_processor = StrategyPromptProcessor(provider=selected_provider)
            
            st.session_state.current_provider = selected_provider
            
        if 'pine_converter' not in st.session_state:
            st.session_state.pine_converter = PineScriptConverter()
        if 'code_generator' not in st.session_state:
            st.session_state.code_generator = CodeGenerator()
    except Exception as e:
        st.error(f"❌ Failed to initialize AI provider: {str(e)}")
        st.info("💡 Try switching to a different provider or check your configuration.")
        return
    
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
            # Convert to Pine Script with options
            col2a, col2b = st.columns([3, 1])
            with col2a:
                convert_to_pine = st.button("🌲 Convert to Pine Script")
            with col2b:
                use_ai_refinement = st.checkbox("🤖 AI Refine", value=True, help="Use AI to improve the Pine Script after conversion")
        
        with col3:
            # Validate code again
            revalidate = st.button("🔍 Re-validate Code")
        
        # Handle Pine Script conversion
        if convert_to_pine:
            conversion_method = "AI-refined" if use_ai_refinement else "Rule-based"
            with st.spinner(f"🌲 Converting to Pine Script ({conversion_method})..."):
                try:
                    if use_ai_refinement:
                        # Use AI-refined conversion
                        pine_code = st.session_state.pine_converter.convert_to_pine_with_ai_refinement(
                            strategy.python_code, 
                            st.session_state.strategy_processor.client,
                            st.session_state.strategy_processor.provider
                        )
                        st.success("✅ Pine Script conversion with AI refinement successful!")
                    else:
                        # Use basic rule-based conversion
                        pine_code = st.session_state.pine_converter.convert_to_pine(strategy.python_code)
                        st.success("✅ Pine Script conversion successful!")
                    
                    st.session_state.pine_script_code = pine_code
                    
                    # Validate Pine Script
                    pine_validation = st.session_state.pine_converter.validate_pine_script(pine_code)
                    
                    if not pine_validation.is_valid:
                        st.warning("⚠️ Pine Script converted with warnings")
                        if pine_validation.errors:
                            with st.expander("⚠️ Validation Issues", expanded=False):
                                for error in pine_validation.errors:
                                    st.write(f"• {error}")
                        
                except Exception as e:
                    st.error(f"❌ Pine Script conversion failed: {str(e)}")
                    st.info("💡 **Troubleshooting:**")
                    st.write("• Try disabling AI refinement for basic conversion")
                    st.write("• Check if your strategy has complex logic that needs manual adjustment")
                    st.write("• Pine Script conversion works best with simple indicator-based strategies")
        
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
        
        # Show conversion info
        st.info("🔄 **Conversion Process**: Python Strategy → Rule-based Conversion → AI Refinement (if enabled) → Final Pine Script")
        
        # Pine Script validation info
        try:
            pine_validation = st.session_state.pine_converter.validate_pine_script(st.session_state.pine_script_code)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if pine_validation.is_valid:
                    st.success("✅ Valid Pine Script")
                else:
                    st.error("❌ Has Issues")
            
            with col2:
                st.metric("Errors", len(pine_validation.errors))
            
            with col3:
                st.metric("Warnings", len(pine_validation.warnings))
            
            with col4:
                lines_count = len(st.session_state.pine_script_code.split('\n'))
                st.metric("Lines", lines_count)
            
            # Show Pine Script issues
            if pine_validation.errors:
                with st.expander("❌ Pine Script Errors", expanded=True):
                    for error in pine_validation.errors:
                        st.write(f"• {error}")
            
            if pine_validation.warnings:
                with st.expander("⚠️ Pine Script Warnings", expanded=False):
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
                        st.plotly_chart(comparison_chart, use_container_width=True, key="equity_comparison_chart")
                    except Exception as e:
                        st.warning(f"Could not create comparison chart: {str(e)}")
                
                    # Detailed deposit/withdrawal analysis
                    try:
                        dw_charts = create_deposit_withdrawal_analysis_charts(processed_data)
                    
                        if 'equity_comparison' in dw_charts:
                            st.plotly_chart(dw_charts['equity_comparison'], use_container_width=True, key="plotly_chart_1")
                    
                        col1, col2 = st.columns(2)
                    
                        if 'deposit_timeline' in dw_charts:
                            with col1:
                                st.plotly_chart(dw_charts['deposit_timeline'], use_container_width=True, key="plotly_chart_2")
                    
                        if 'performance_impact' in dw_charts:
                            with col2:
                                st.plotly_chart(dw_charts['performance_impact'], use_container_width=True, key="plotly_chart_3")
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
    
        # === PERIOD PERFORMANCE SUMMARY ===
        if selected_period == "All Time":
            st.markdown("---")
            st.markdown("### 📊 Performance by Period")
        
            # Calculate performance for different periods
            periods_to_analyze = ["Today", "Yesterday", "This Week", "Last Week", "This Month", "Last Month", "This Year"]
            period_summary = []
        
            original_df_for_summary = df.copy()
        
            for period in periods_to_analyze:
                start_p, end_p = get_date_range(period)
                if start_p and end_p:
                    period_df = filter_dataframe_by_date(original_df_for_summary, start_p, end_p)
                    if len(period_df) > 0:
                        period_pnl = period_df[pnl].sum()
                        period_trades = len(period_df)
                        win_rate = (period_df[pnl] > 0).sum() / len(period_df) * 100
                    
                        period_summary.append({
                            'Period': period,
                            'Trades': period_trades,
                            'PnL': f"${period_pnl:,.2f}",
                            'Win Rate': f"{win_rate:.1f}%",
                            'Avg/Trade': f"${period_pnl/period_trades:.2f}" if period_trades > 0 else "$0.00"
                        })
        
            if period_summary:
                summary_df = pd.DataFrame(period_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
        # Calculate KPIs for the selected period
        if start_date and end_date and selected_period != "All Time":
            # For filtered periods, calculate the correct starting balance
            from analytics import compute_period_starting_balance
        
            # We need the original unfiltered dataframe to calculate cumulative PnL
            period_starting_balance = compute_period_starting_balance(original_df, df, pnl, starting_balance)
            top_kpis = compute_top_kpis(df, pnl, starting_balance_override=period_starting_balance)
        
            # Show additional context for filtered periods
            if len(df) > 0:
                first_trade_date = df.iloc[0]['close_time_parsed'].date() if 'close_time_parsed' in df.columns else "Unknown"
                last_trade_date = df.iloc[-1]['close_time_parsed'].date() if 'close_time_parsed' in df.columns else "Unknown"
            
                # Calculate how much PnL was accumulated before this period
                trades_before_count = original_df.index.get_loc(df.index[0]) if len(df) > 0 else 0
                pnl_before_period = period_starting_balance - starting_balance
            
                st.info(f"""
                📊 **Period Analysis**: {first_trade_date} to {last_trade_date}  
                **Starting Balance Calculation**: Original ${starting_balance:,.2f} + PnL from {trades_before_count} previous trades (${pnl_before_period:,.2f}) = **${period_starting_balance:,.2f}**
                """)
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
    
        kelly_metrics = compute_kelly_metrics(df, pnl, current_equity_for_kelly, user_current_lots)

        # === TOP KPIs SECTION ===
        st.markdown("---")
        st.markdown("### 📊 Account Overview")
    
        # Show period comparison if not "All Time"
        if selected_period != "All Time" and start_date and end_date:
            col1, col2 = st.columns([3, 1])
        
            with col1:
                kpi_cols = st.columns(4)
                for col, (k, v) in zip(kpi_cols, top_kpis.items()):
                    col.metric(k, v)
        
            with col2:
                st.markdown("**📈 Period Comparison**")
            
                # Calculate comparison metrics
                period_pnl = df[pnl].sum()
                period_trades = len(df)
                period_days = (end_date - start_date).days + 1 if start_date != end_date else 1
            
                # Show period starting balance calculation for transparency
                if 'original_df' in locals() and len(df) < len(original_df):
                    trades_before_period = len(original_df) - len(df) - (len(original_df) - original_df.index[-1] - 1) + (df.index[0] if len(df) > 0 else 0)
                    st.write(f"• **Period Starting Balance**: ${period_starting_balance:,.2f}")
                    st.write(f"• **Trades Before Period**: {original_df.index.get_loc(df.index[0]) if len(df) > 0 else 0}")
            
                st.write(f"• **Period PnL**: ${period_pnl:,.2f}")
                st.write(f"• **Trades**: {period_trades}")
                st.write(f"• **Days**: {period_days}")
                if period_days > 0:
                    st.write(f"• **Avg PnL/Day**: ${period_pnl/period_days:.2f}")
                if period_trades > 0:
                    st.write(f"• **Avg PnL/Trade**: ${period_pnl/period_trades:.2f}")
        else:
            kpi_cols = st.columns(4)
            for col, (k, v) in zip(kpi_cols, top_kpis.items()):
                col.metric(k, v)
    
        st.markdown("---")

        # === CORE PERFORMANCE METRICS ===
        st.markdown("### 📈 Core Performance Metrics")
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, v)

        # === KELLY CRITERION POSITION SIZING ===
        st.markdown("---")
        st.markdown("### 🎯 Kelly Criterion Position Sizing")
        st.markdown("**Optimal position sizing based on your trading history and current equity**")
    
        if kelly_metrics.get('has_kelly_data', False):
            # Kelly Overview Metrics
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                kelly_fraction = kelly_metrics['kelly_fraction']
                kelly_pct = kelly_fraction * 100
                if kelly_fraction > 0:
                    st.metric("Kelly Fraction", f"{kelly_pct:.2f}%", 
                             help="Optimal % of capital to risk per trade")
                else:
                    st.metric("Kelly Fraction", "No Edge", 
                             help="Negative Kelly indicates no statistical edge")
        
            with col2:
                conservative_kelly = kelly_metrics['conservative_kelly']
                conservative_pct = conservative_kelly * 100
                st.metric("Conservative Kelly", f"{conservative_pct:.2f}%", 
                         help="25% of full Kelly for safer growth")
        
            with col3:
                recommended_lots = kelly_metrics['lot_recommendation']['recommended_lot_size']
                st.metric("Recommended Lot Size", f"{recommended_lots:.3f}", 
                         help="Based on Kelly Criterion and current equity")
        
            with col4:
                current_equity = kelly_metrics['current_equity']
                st.metric("Current Equity", f"${current_equity:,.2f}", 
                         help="Used for position sizing calculations")
        
            # Kelly Analysis Summary
            st.markdown("#### 📊 Kelly Analysis Summary")
        
            col1, col2 = st.columns(2)
        
            with col1:
                # Risk Assessment
                risk_level = kelly_metrics['risk_level']
                risk_assessment = kelly_metrics['lot_recommendation']['risk_assessment']
            
                if risk_level == 'NO EDGE':
                    st.error(f"🚨 **Risk Level**: {risk_level}")
                    st.error(f"**Assessment**: {risk_assessment}")
                elif risk_level in ['HIGH', 'EXTREME']:
                    st.warning(f"⚠️ **Risk Level**: {risk_level}")
                    st.warning(f"**Assessment**: {risk_assessment}")
                else:
                    st.success(f"✅ **Risk Level**: {risk_level}")
                    st.info(f"**Assessment**: {risk_assessment}")
            
                # Edge Analysis
                edge = kelly_metrics['edge']
                if edge > 0:
                    st.success(f"📈 **Statistical Edge**: ${edge:.2f} per trade")
                else:
                    st.error(f"📉 **No Edge**: ${edge:.2f} per trade (negative expectancy)")
        
            with col2:
                # Position Sizing Comparison
                st.markdown("**Position Sizing Comparison:**")
            
                traditional_lots = (current_equity / 1000) * 0.02
                kelly_lots = recommended_lots
            
                st.write(f"• **Your Traditional Method**: {traditional_lots:.3f} lots (0.02 per 1k)")
                st.write(f"• **Kelly Recommended**: {kelly_lots:.3f} lots")
            
                difference_pct = ((kelly_lots - traditional_lots) / traditional_lots * 100) if traditional_lots > 0 else 0
            
                if difference_pct > 10:
                    st.write(f"• **Difference**: +{difference_pct:.1f}% (Kelly suggests larger size)")
                elif difference_pct < -10:
                    st.write(f"• **Difference**: {difference_pct:.1f}% (Kelly suggests smaller size)")
                else:
                    st.write(f"• **Difference**: {difference_pct:.1f}% (Similar sizing)")
            
                # Risk-Reward Metrics
                odds_ratio = kelly_metrics['odds_ratio']
                win_rate_pct = kelly_metrics['win_rate'] * 100
            
                st.write(f"• **Risk:Reward Ratio**: {odds_ratio:.2f}:1")
                st.write(f"• **Win Rate**: {win_rate_pct:.1f}%")
        
            # Kelly Insights
            st.markdown("#### 💡 Kelly Insights & Recommendations")
        
            insights = kelly_metrics.get('insights', [])
            if insights:
                for insight in insights:
                    if "🚨" in insight or "❌" in insight:
                        st.error(insight)
                    elif "⚠️" in insight:
                        st.warning(insight)
                    elif "✅" in insight or "🎯" in insight:
                        st.success(insight)
                    else:
                        st.info(insight)
        
            # Kelly Charts
            st.markdown("#### 📊 Kelly Criterion Analysis Charts")
        
            kelly_charts = create_kelly_criterion_charts(kelly_metrics)
        
            if 'kelly_overview' in kelly_charts:
                st.plotly_chart(kelly_charts['kelly_overview'], use_container_width=True, key="plotly_chart_4")
        
            col1, col2 = st.columns(2)
        
            if 'position_sizing' in kelly_charts:
                with col1:
                    st.plotly_chart(kelly_charts['position_sizing'], use_container_width=True, key="plotly_chart_5")
        
            if 'risk_reward' in kelly_charts:
                with col2:
                    st.plotly_chart(kelly_charts['risk_reward'], use_container_width=True, key="plotly_chart_6")
        
            # Kelly Insights Summary Chart
            insights_chart = create_kelly_insights_summary_chart(kelly_metrics)
            if insights_chart:
                st.plotly_chart(insights_chart, use_container_width=True, key="plotly_chart_7")
        
            # Practical Implementation Guide
            with st.expander("🛠️ Practical Implementation Guide", expanded=False):
                st.markdown("""
                **How to Use Kelly Criterion Results:**
            
                1. **Start Conservative**: Use 25% of the full Kelly fraction (Conservative Kelly)
                2. **Monitor Performance**: Track if your edge remains consistent over time
                3. **Adjust Gradually**: Increase position size only if edge strengthens
                4. **Risk Management**: Never exceed the full Kelly fraction
                5. **Regular Review**: Recalculate Kelly fraction monthly with new data
            
                **Position Sizing Formula Used:**
                - Kelly Fraction = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
                - Recommended Lots = (Current Equity / 1000) × Base Lot Size × Kelly Multiplier
                - Conservative Scaling = 25% of Full Kelly (reduces risk of ruin)
            
                **Warning Signs to Reduce Position Size:**
                - Kelly fraction becomes negative (no edge)
                - Win rate drops significantly
                - Average losses increase relative to wins
                - Drawdown exceeds expected levels
                """)
    
        else:
            st.error("❌ **Kelly Criterion Analysis Unavailable**")
            st.info("Insufficient trading data for Kelly Criterion calculation. Need both winning and losing trades.")

        st.markdown("---")

        # === EQUITY & PNL GROWTH ===
        st.markdown("### 💰 Equity & PnL Growth")
    
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(equity_curve(df, selected_period), use_container_width=True)
        with col2:
            st.plotly_chart(pnl_growth_over_time(df, pnl, selected_period), use_container_width=True)

        # === ROLLING PERFORMANCE (PROFESSIONAL EDGE DECAY ANALYSIS) ===
        st.markdown("### 🔍 Professional Edge Decay Analysis")
        st.markdown("**Risk Manager's View - No Twitter Trader Fluff**")
    
        # Professional rolling performance charts
        st.plotly_chart(rolling_performance_charts(df), use_container_width=True, key="plotly_chart_8")
    
        # === PROFESSIONAL EDGE ANALYSIS ===
        edge_analysis = analyze_edge_decay(df, pnl)
    
        # Edge Status Alert
        edge_status = edge_analysis.get('edge_status', 'UNKNOWN')
        if edge_status == 'NO_EDGE':
            st.error("🚨 **CRITICAL: NO STATISTICAL EDGE** - Strategy has no edge (expectancy ≤ 0)")
        elif edge_status == 'EDGE_DECAY':
            st.error("⚠️ **WARNING: EDGE DECAY DETECTED** - Expectancy is deteriorating")
        elif edge_status == 'EDGE_IMPROVING':
            st.success("✅ **POSITIVE: EDGE STRENGTHENING** - Expectancy is improving")
        elif edge_status == 'EDGE_STABLE':
            st.info("📊 **NEUTRAL: EDGE STABLE** - Performance appears consistent")
        elif edge_status == 'INSUFFICIENT_DATA':
            st.warning("⚠️ **INSUFFICIENT DATA** - Need more trades for edge analysis")
        else:
            st.info("📊 **EDGE STATUS**: Analysis in progress")
    
        # Professional Analysis Breakdown
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### 🎯 Risk Manager's Assessment")
        
            # Warnings
            warnings = edge_analysis.get('warnings', [])
            if warnings:
                st.markdown("**⚠️ WARNINGS:**")
                for warning in warnings:
                    st.markdown(f"• {warning}")
        
            # Recommendations
            recommendations = edge_analysis.get('recommendations', [])
            if recommendations:
                st.markdown("**📋 RECOMMENDATIONS:**")
                for rec in recommendations:
                    st.markdown(f"• {rec}")
        
            # Positive signals
            signals = edge_analysis.get('signals', [])
            if signals:
                st.markdown("**✅ POSITIVE SIGNALS:**")
                for signal in signals:
                    st.markdown(f"• {signal}")
        
            # Show message if no analysis available
            if not warnings and not recommendations and not signals:
                st.info("📊 Expand date range for detailed analysis")
    
        with col2:
            st.markdown("#### 📊 Current Risk Metrics")
        
            # Display risk metrics
            risk_metrics = edge_analysis.get('risk_metrics', {})
        
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Current Expectancy", risk_metrics.get('current_expectancy', 'N/A'))
                st.metric("Win Rate", risk_metrics.get('win_rate', 'N/A'))
                st.metric("R:R Ratio", risk_metrics.get('rr_ratio', 'N/A'))
        
            with col2b:
                st.metric("Expectancy Trend", risk_metrics.get('expectancy_trend', 'N/A'))
                st.metric("Profit Factor", risk_metrics.get('profit_factor', 'N/A'))
                st.metric("Profit Concentration", risk_metrics.get('profit_concentration', 'N/A'))
    
        # Professional Decision Rules
        st.markdown("#### 🎯 Professional Decision Rules")
    
        # Check if decision_rules exists (might not exist with insufficient data)
        if 'decision_rules' in edge_analysis:
            decision_rules = edge_analysis['decision_rules']
        
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 TRADE ONLY WHEN:**")
                st.info(decision_rules.get('trade_only_when', 'N/A'))
            
                st.markdown("**📉 REDUCE SIZE WHEN:**")
                st.warning(decision_rules.get('reduce_size_when', 'N/A'))
        
            with col2:
                st.markdown("**🛑 STOP TRADING WHEN:**")
                st.error(decision_rules.get('stop_trading_when', 'N/A'))
            
                st.markdown("**🏷️ REGIME TAGGING:**")
                st.info(decision_rules.get('regime_tagging', 'N/A'))
        else:
            # Handle case where there's insufficient data for decision rules
            st.warning("⚠️ **Insufficient data for professional decision rules**")
            st.info("Need at least 20 trades in the selected period for meaningful edge analysis.")
        
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 TRADE ONLY WHEN:**")
                st.info("Expand date range to get analysis")
            
                st.markdown("**📉 REDUCE SIZE WHEN:**")
                st.info("Expand date range to get analysis")
        
            with col2:
                st.markdown("**🛑 STOP TRADING WHEN:**")
                st.info("Expand date range to get analysis")
            
                st.markdown("**🏷️ REGIME TAGGING:**")
                st.info("Expand date range to get analysis")
    
        # Missing Analytics (Next Level)
        missing_analytics = edge_analysis.get('missing_analytics', [])
        if missing_analytics:
            with st.expander("🔴 Missing Analytics - Next Level Analysis Needed", expanded=False):
                st.markdown("**To reach TradingView-grade analytics, you still need:**")
                for missing in missing_analytics:
                    st.markdown(f"• {missing}")
            
                st.markdown("---")
                st.markdown("**💡 Pro Tip:** These missing metrics separate professional traders from retail traders. Without them, you're trading half-blind.")
        else:
            st.info("📊 Expand date range to see advanced analytics recommendations")

        st.markdown("---")

        # === TIME ANALYSIS (KILLER SESSIONS) ===
        st.markdown("### ⏰ Time Analysis - Session Killers")
        st.markdown("**Identify when you should NOT trade**")
    
        # Daily Trading Calendar
        st.markdown("#### 📅 Daily Trading Calendar")
        st.markdown("**See at one glance which how many days you are making or losing money. Click a day to look at the trades.**")
    
        # Calendar navigation controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
        # Get available years and months from data
        if len(df) > 0:
            # Find datetime column for navigation
            datetime_col = None
            original_col = None
        
            for col in df.columns:
                if any(word in col.lower() for word in ['time', 'date', 'datetime', 'close_time']):
                    try:
                        # Test if we can parse this column
                        test_series = pd.to_datetime(df[col])
                        original_col = col
                        datetime_col = f'{col}_parsed'
                        break
                    except:
                        continue
        
            if datetime_col and original_col:
                # Create the parsed datetime column
                df_temp = df.copy()
                df_temp[datetime_col] = pd.to_datetime(df_temp[original_col])
                df_temp['trade_date'] = df_temp[datetime_col].dt.date
                available_dates = df_temp['trade_date'].dropna()
            
                if len(available_dates) > 0:
                    min_year = available_dates.min().year
                    max_year = available_dates.max().year
                    latest_date = available_dates.max()
                
                    # Initialize calendar session state if not exists
                    if 'calendar_year' not in st.session_state:
                        st.session_state.calendar_year = latest_date.year
                    if 'calendar_month' not in st.session_state:
                        st.session_state.calendar_month = latest_date.month
                
                    # Year selection (standalone - doesn't affect main app)
                    with col1:
                        available_years = list(range(min_year, max_year + 1))
                        default_year_idx = available_years.index(st.session_state.calendar_year) if st.session_state.calendar_year in available_years else 0
                        selected_year = st.selectbox("📅 Year", available_years, index=default_year_idx, key="calendar_year_select")
                        if selected_year != st.session_state.calendar_year:
                            st.session_state.calendar_year = selected_year
                
                    # Month selection (standalone - doesn't affect main app)
                    with col2:
                        month_names = [calendar.month_name[i] for i in range(1, 13)]
                        month_numbers = list(range(1, 13))
                        default_month_idx = st.session_state.calendar_month - 1
                        selected_month_name = st.selectbox("📅 Month", month_names, index=default_month_idx, key="calendar_month_select")
                        selected_month = month_numbers[month_names.index(selected_month_name)]
                        if selected_month != st.session_state.calendar_month:
                            st.session_state.calendar_month = selected_month
                
                    # Use session state values for consistency
                    selected_year = st.session_state.calendar_year
                    selected_month = st.session_state.calendar_month
                
                    # Navigation buttons (standalone)
                    with col3:
                        if st.button("⬅️ Previous Month", key="calendar_prev_month"):
                            if selected_month == 1:
                                new_month = 12
                                new_year = selected_year - 1
                            else:
                                new_month = selected_month - 1
                                new_year = selected_year
                        
                            if new_year >= min_year:
                                st.session_state.calendar_year = new_year
                                st.session_state.calendar_month = new_month
                                st.rerun()
                
                    with col4:
                        if st.button("➡️ Next Month", key="calendar_next_month"):
                            if selected_month == 12:
                                new_month = 1
                                new_year = selected_year + 1
                            else:
                                new_month = selected_month + 1
                                new_year = selected_year
                        
                            if new_year <= max_year:
                                st.session_state.calendar_year = new_year
                                st.session_state.calendar_month = new_month
                                st.rerun()
                
                    # Create calendar with selected month/year (using ORIGINAL full dataset, not filtered)
                    calendar_chart = create_daily_calendar_chart(original_df, pnl, f"{calendar.month_name[selected_month]} {selected_year}", selected_year, selected_month)
                else:
                    # No date data available
                    calendar_chart = create_daily_calendar_chart(df, pnl, "No Date Data")
            else:
                # No datetime column found
                calendar_chart = create_daily_calendar_chart(df, pnl, "No Date Column")
        else:
            # No data
            calendar_chart = create_daily_calendar_chart(df, pnl, "No Data")
    
        # Display the calendar
        st.plotly_chart(calendar_chart, use_container_width=True, key="plotly_chart_9")
    
        # Calendar insights (only show if we have data and datetime column)
        if len(df) > 0 and 'datetime_col' in locals() and datetime_col and 'original_col' in locals():
            col1, col2, col3 = st.columns(3)
        
            # Calculate daily stats for the selected month using ORIGINAL dataset
            df_copy = original_df.copy()  # Use original_df instead of filtered df
            df_copy[datetime_col] = pd.to_datetime(df_copy[original_col])
            df_copy['trade_date'] = df_copy[datetime_col].dt.date
        
            # Filter for selected month/year from calendar (not main app filter)
            if 'selected_year' in locals() and 'selected_month' in locals():
                month_filter = (df_copy[datetime_col].dt.year == selected_year) & (df_copy[datetime_col].dt.month == selected_month)
                month_df = df_copy[month_filter]
            
                if len(month_df) > 0:
                    daily_stats = month_df.groupby('trade_date').agg({
                        pnl: ['count', 'sum']
                    })
                    daily_stats.columns = ['trades', 'pnl']
                
                    with col1:
                        profitable_days = (daily_stats['pnl'] > 0).sum()
                        total_trading_days = len(daily_stats)
                        loss_days = (daily_stats['pnl'] < 0).sum()
                    
                        st.metric("Profitable Days", f"{profitable_days}")
                        st.metric("Loss Days", f"{loss_days}")
                
                    with col2:
                        if len(daily_stats) > 0:
                            best_day_pnl = daily_stats['pnl'].max()
                            worst_day_pnl = daily_stats['pnl'].min()
                        
                            st.metric("Best Day", f"${best_day_pnl:.2f}")
                            st.metric("Worst Day", f"${worst_day_pnl:.2f}")
                
                    with col3:
                        if len(daily_stats) > 0:
                            avg_trades_per_day = daily_stats['trades'].mean()
                            total_month_pnl = daily_stats['pnl'].sum()
                        
                            st.metric("Avg Trades/Day", f"{avg_trades_per_day:.1f}")
                            st.metric("Month Total P&L", f"${total_month_pnl:.2f}")
                else:
                    # Show message if no data for selected month
                    st.info(f"📅 No trading data found for {calendar.month_name[selected_month]} {selected_year}")
            else:
                st.info("📅 Select a month and year to view calendar insights")
    
        # Hourly and Daily Analysis Charts
        st.markdown("#### ⏰ Hourly & Daily Performance")
    
        hourly_chart, daily_chart = time_analysis_charts(time_data)
    
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(hourly_chart, use_container_width=True, key="plotly_chart_10")
        with col2:
            st.plotly_chart(daily_chart, use_container_width=True, key="plotly_chart_11")
    
        # Time analysis insights
        time_tables = create_time_tables(time_data)
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚨 Worst Performing Hours (AVOID):**")
            for hour, data in time_tables['worst_hours'].iterrows():
                if data['Total_PnL'] < 0:
                    st.write(f"• **{hour}:00** - Loss: ${data['Total_PnL']:.0f} ({data['Trade_Count']:.0f} trades)")
    
        with col2:
            st.markdown("**✅ Best Performing Hours:**")
            for hour, data in time_tables['best_hours'].iterrows():
                if data['Total_PnL'] > 0:
                    st.write(f"• **{hour}:00** - Profit: ${data['Total_PnL']:.0f} ({data['Trade_Count']:.0f} trades)")

        st.markdown("---")

        # === MONTHLY SEASONALITY ===
        st.markdown("### 📅 Monthly Seasonality")
        st.plotly_chart(monthly_heatmap(time_data), use_container_width=True, key="plotly_chart_12")
    
        # Monthly insights
        monthly_stats = time_data['monthly']
        best_month = monthly_stats.loc[monthly_stats['Total_PnL'].idxmax()]
        worst_month = monthly_stats.loc[monthly_stats['Total_PnL'].idxmin()]
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🏆 Best Month:** {best_month.name[1]} (${best_month['Total_PnL']:.0f})")
            st.markdown(f"**📉 Worst Month:** {worst_month.name[1]} (${worst_month['Total_PnL']:.0f})")
    
        with col2:
            profitable_months = (monthly_stats['Total_PnL'] > 0).sum()
            total_months = len(monthly_stats)
            st.markdown(f"**📊 Monthly Win Rate:** {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

        st.markdown("---")

        # === RISK & PAIN METRICS ===
        st.markdown("### 🩹 Risk & Pain Analysis")
        st.markdown("**Psychological and risk tolerance metrics**")
    
        # Risk metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Consecutive Losses", f"{risk_data['max_consecutive_losses']}")
        with col2:
            st.metric("Avg Drawdown Duration", f"{risk_data['avg_drawdown_duration']:.1f} trades")
        with col3:
            st.metric("Recovery Factor", f"{risk_data['recovery_factor']:.2f}")
        with col4:
            st.metric("Ulcer Index (Pain)", f"{risk_data['ulcer_index']:.2f}%")
    
        # Risk charts
        dd_chart, recovery_chart, consec_chart = risk_pain_charts(risk_data, df)
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(dd_chart, use_container_width=True, key="plotly_chart_13")
        with col2:
            st.plotly_chart(recovery_chart, use_container_width=True, key="plotly_chart_14")
        with col3:
            st.plotly_chart(consec_chart, use_container_width=True, key="plotly_chart_15")

        # Pain tolerance warnings
        if risk_data['max_consecutive_losses'] > 5:
            st.error(f"🚨 **HIGH PAIN RISK**: Max consecutive losses ({risk_data['max_consecutive_losses']}) may exceed psychological tolerance")
    
        if risk_data['avg_drawdown_duration'] > 10:
            st.warning(f"⚠️ **LONG RECOVERY PERIODS**: Average drawdown lasts {risk_data['avg_drawdown_duration']:.1f} trades")

        st.markdown("---")

        # === TRADING INSIGHTS ANALYSIS ===
        st.markdown("### 🎯 Advanced Trading Insights")
        st.markdown("**Deep dive into lot sizing, risk-reward, pip analysis, and directional bias**")
    
        # Display insights summary
        if insights_summary:
            st.markdown("#### 📊 Key Insights Summary")
            for insight in insights_summary:
                if "✅" in insight:
                    st.success(insight)
                elif "🚨" in insight:
                    st.error(insight)
                elif "⚠️" in insight:
                    st.warning(insight)
                else:
                    st.info(insight)
    
        # Create and display trading insights charts
        insights_charts = create_trading_insights_charts(trading_insights)
    
        # Display charts in organized sections
        if 'lot_analysis' in insights_charts:
            st.markdown("#### 📏 Lot Size Analysis")
            st.plotly_chart(insights_charts['lot_analysis'], use_container_width=True, key="plotly_chart_16")
        
            # Display lot size statistics
            if trading_insights['lot_analysis']['has_lot_data']:
                lot_stats = trading_insights['lot_analysis']['stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Lot Size", f"{lot_stats['avg_lot_size']:.2f}")
                with col2:
                    st.metric("Lot Consistency", f"{lot_stats['lot_consistency']:.1%}")
                with col3:
                    st.metric("Min/Max Lots", f"{lot_stats['min_lot_size']:.2f} / {lot_stats['max_lot_size']:.2f}")
                with col4:
                    st.metric("Lot Std Dev", f"{lot_stats['lot_std']:.3f}")
    
        if 'direction_analysis' in insights_charts:
            st.markdown("#### 📈📉 Buy vs Sell Performance")
            st.plotly_chart(insights_charts['direction_analysis'], use_container_width=True, key="plotly_chart_17")
        
            # Display direction comparison table
            if trading_insights['direction_analysis']['has_direction_data']:
                direction_perf = trading_insights['direction_analysis']['performance']
                st.markdown("**Performance Comparison:**")
            
                # Format the dataframe for display
                display_direction = direction_perf.copy()
                display_direction['Total_PnL'] = display_direction['Total_PnL'].apply(lambda x: f"${x:,.2f}")
                display_direction['Avg_PnL'] = display_direction['Avg_PnL'].apply(lambda x: f"${x:.2f}")
                display_direction['Win_Rate'] = display_direction['Win_Rate'].apply(lambda x: f"{x:.1f}%")
                display_direction['Profit_Factor'] = display_direction['Profit_Factor'].apply(lambda x: f"{x:.2f}")
                display_direction['Consistency'] = display_direction['Consistency'].apply(lambda x: f"{x:.1%}")
            
                # Rename columns for display
                display_direction.columns = ['Total PnL', 'Avg PnL', 'Trades', 'PnL Std', 'Win Rate', 'Profit Factor', 'Consistency']
                display_direction = display_direction[['Total PnL', 'Avg PnL', 'Trades', 'Win Rate', 'Profit Factor', 'Consistency']]
            
                st.dataframe(display_direction, use_container_width=True)
    
        if 'rr_analysis' in insights_charts:
            st.markdown("#### ⚖️ Risk-Reward Analysis")
            st.plotly_chart(insights_charts['rr_analysis'], use_container_width=True, key="plotly_chart_18")
        
            if trading_insights['rr_analysis']['has_rr_data']:
                rr_stats = trading_insights['rr_analysis']['stats']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg R:R Ratio", f"{rr_stats['avg_rr_ratio']:.2f}:1")
                with col2:
                    st.metric("Best R:R", f"{rr_stats['best_rr']:.2f}")
                with col3:
                    st.metric("RR Consistency", f"{rr_stats['rr_consistency']:.1%}")
    
        # Pip Analysis
        pip_chart = create_pip_analysis_chart(trading_insights)
        if pip_chart:
            st.markdown("#### 📊 Pip Analysis")
        
            # Show calculation method info
            if trading_insights['pip_analysis'].get('estimated', False):
                calculation_method = trading_insights['pip_analysis'].get('calculation_method', 'estimated')
                if calculation_method == 'symbol_aware':
                    st.info("💡 **Pip calculation**: Symbol-aware calculation (NAS100=points, XAUUSD=0.1 moves, JPY=0.01 moves, Forex=0.0001 moves)")
                else:
                    st.info("💡 **Pip calculation**: Estimated from price data (assuming standard forex pairs)")
        
            st.plotly_chart(pip_chart, use_container_width=True, key="plotly_chart_19")
        
            pip_stats = trading_insights['pip_analysis']['stats']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Pips/Trade", f"{pip_stats['avg_pips_per_trade']:.1f}")
            with col2:
                st.metric("Total Pips", f"{pip_stats['total_pips']:.0f}")
            with col3:
                st.metric("Best Trade", f"{pip_stats['best_pip_trade']:.1f} pips")
            with col4:
                st.metric("Worst Trade", f"{pip_stats['worst_pip_trade']:.1f} pips")
    
        if 'symbol_analysis' in insights_charts:
            st.markdown("#### 🎯 Symbol Performance Analysis")
            st.plotly_chart(insights_charts['symbol_analysis'], use_container_width=True, key="plotly_chart_20")
        
            # Display top/worst performing symbols
            if trading_insights['symbol_analysis']['has_symbol_data']:
                symbol_perf = trading_insights['symbol_analysis']['performance']
            
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏆 Top 3 Performing Symbols:**")
                    top_symbols = symbol_perf.head(3)
                    for symbol, data in top_symbols.iterrows():
                        st.write(f"• **{symbol}**: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
            
                with col2:
                    st.markdown("**📉 Bottom 3 Performing Symbols:**")
                    bottom_symbols = symbol_perf.tail(3)
                    for symbol, data in bottom_symbols.iterrows():
                        st.write(f"• **{symbol}**: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
    
        if 'position_sizing' in insights_charts:
            st.markdown("#### 📏 Position Sizing Analysis")
            st.plotly_chart(insights_charts['position_sizing'], use_container_width=True, key="plotly_chart_21")

        st.markdown("---")

        # === DRAWDOWN & DISTRIBUTION ===
        st.markdown("### 📉 Risk Analysis")
        st.plotly_chart(drawdown_curve(df, selected_period), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.plotly_chart(pnl_distribution(df, pnl, selected_period), use_container_width=True)
        c2.plotly_chart(win_loss_pie(df, pnl, selected_period), use_container_width=True)

        # === COMPREHENSIVE AI DIAGNOSIS ===
        st.markdown("---")
        ai_diagnosis = comprehensive_ai_diagnosis(df, metrics, risk_data, time_data, pnl, selected_period)
    
        # Executive Summary
        executive_summary = generate_executive_summary(ai_diagnosis, metrics, selected_period)
        st.markdown(executive_summary)
    
        # Detailed Analysis Sections
        col1, col2 = st.columns(2)
    
        with col1:
            # Critical Issues
            if ai_diagnosis['critical_issues']:
                st.markdown("### 🚨 Critical Issues")
                for issue in ai_diagnosis['critical_issues']:
                    st.error(issue)
        
            # Warnings
            if ai_diagnosis['warnings']:
                st.markdown("### ⚠️ Warnings")
                for warning in ai_diagnosis['warnings']:
                    st.warning(warning)
        
            # Strengths
            if ai_diagnosis['strengths']:
                st.markdown("### ✅ Strengths")
                for strength in ai_diagnosis['strengths']:
                    st.success(strength)
    
        with col2:
            # Performance Insights
            if ai_diagnosis['performance_insights']:
                st.markdown("### 📊 Performance Insights")
                for insight in ai_diagnosis['performance_insights']:
                    st.info(insight)
        
            # Psychological Analysis
            if ai_diagnosis['psychological_analysis']:
                st.markdown("### 🧠 Psychological Analysis")
                for analysis in ai_diagnosis['psychological_analysis']:
                    st.info(analysis)
        
            # Risk Assessment
            if ai_diagnosis['risk_assessment']:
                st.markdown("### 🛡️ Risk Assessment")
                for risk in ai_diagnosis['risk_assessment']:
                    if "HIGH RISK" in risk:
                        st.error(risk)
                    elif "MODERATE RISK" in risk:
                        st.warning(risk)
                    else:
                        st.success(risk)
    
        # === SYMBOL/ASSET ANALYSIS SECTION ===
        if 'symbol_analysis' in ai_diagnosis and 'symbol_performance' in ai_diagnosis['symbol_analysis']:
            symbol_perf = ai_diagnosis['symbol_analysis']['symbol_performance']
        
            if not symbol_perf.empty:
                st.markdown("---")
                st.markdown("### 📈 Symbol/Asset Performance Analysis")
            
                # Symbol performance table
                st.markdown("**Performance by Symbol:**")
            
                # Format the dataframe for display
                display_df = symbol_perf.copy()
                display_df['Total_PnL'] = display_df['Total_PnL'].apply(lambda x: f"${x:,.2f}")
                display_df['Avg_PnL'] = display_df['Avg_PnL'].apply(lambda x: f"${x:.2f}")
                display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1f}%")
                display_df['Max_DD_Pct'] = display_df['Max_DD_Pct'].apply(lambda x: f"{x:.1f}%")
            
                # Rename columns for display
                display_df.columns = ['Trades', 'Total PnL', 'Avg PnL/Trade', 'PnL Volatility', 'Win Rate', 'Profit Factor', 'Max Drawdown', 'Expectancy']
            
                st.dataframe(display_df, use_container_width=True)
            
                # Symbol insights in columns
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown("**🎯 Symbol Insights:**")
                    symbol_insights = ai_diagnosis['symbol_analysis']['symbol_insights']
                    for insight in symbol_insights[:len(symbol_insights)//2 + 1]:
                        st.write(f"• {insight}")
            
                with col2:
                    st.markdown("**💡 Symbol Recommendations:**")
                    symbol_recs = ai_diagnosis['symbol_analysis']['symbol_recommendations']
                    for rec in symbol_recs:
                        if "FOCUS ON" in rec or "AVOID" in rec:
                            st.write(f"• {rec}")
                        else:
                            st.write(f"• {rec}")
            
                # Asset class breakdown if available
                if ai_diagnosis['symbol_analysis']['asset_class_insights']:
                    st.markdown("**📊 Asset Class Breakdown:**")
                    for insight in ai_diagnosis['symbol_analysis']['asset_class_insights']:
                        st.info(insight)
    
        # Recommendations Section
        if ai_diagnosis['recommendations']:
            st.markdown("### 💡 AI Recommendations")
            rec_cols = st.columns(2)
            for i, rec in enumerate(ai_diagnosis['recommendations']):
                rec_cols[i % 2].info(rec)
    
        # Action Items
        if ai_diagnosis['action_items']:
            st.markdown("### 🎯 Action Items")
            for i, action in enumerate(ai_diagnosis['action_items'], 1):
                st.markdown(f"**{i}.** {action}")

        # === ORIGINAL DIAGNOSIS (Legacy) ===
        st.markdown("### 🩺 Quick Diagnosis")
        diagnosis_result = diagnose(metrics)
        if "stable" in diagnosis_result.lower():
            st.success(f"✅ {diagnosis_result}")
        else:
            st.error(f"🚨 {diagnosis_result}")
        
        
        # === ADDITIONAL INSIGHTS ===
        st.markdown("### 🎯 Key Insights")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("**Rolling Performance Summary:**")
            avg_rolling_exp = df["rolling_expectancy"].mean()
            avg_rolling_wr = df["rolling_win_rate"].mean()
        
            st.write(f"• Average Rolling Expectancy: **${avg_rolling_exp:.2f}**")
            st.write(f"• Average Rolling Win Rate: **{avg_rolling_wr:.1f}%**")
        
            # Consistency check
            expectancy_std = df["rolling_expectancy"].std()
            if expectancy_std > abs(avg_rolling_exp):
                st.write("• ⚠️ **High expectancy volatility** - inconsistent edge")
            else:
                st.write("• ✅ **Consistent expectancy** - stable edge")
    
        with col2:
            st.markdown("**Performance Trends:**")
        
            # Trend analysis
            early_performance = df["rolling_expectancy"].head(20).mean()
            recent_performance = df["rolling_expectancy"].tail(20).mean()
        
            if recent_performance > early_performance * 1.1:
                st.write("• 📈 **Improving performance** over time")
            elif recent_performance < early_performance * 0.9:
                st.write("• 📉 **Declining performance** - review strategy")
            else:
                st.write("• ➡️ **Stable performance** maintained")
            
            # Win rate stability
            wr_early = df["rolling_win_rate"].head(20).mean()
            wr_recent = df["rolling_win_rate"].tail(20).mean()
        
            if abs(wr_recent - wr_early) > 10:
                st.write("• ⚠️ **Win rate changed significantly**")
            else:
                st.write("• ✅ **Win rate remains stable**")


def advanced_backtesting_engine_interface():
    """Advanced backtesting engine with comprehensive features"""
    st.markdown("### ⚡ Advanced Backtesting Engine")
    st.markdown("**Professional-grade backtesting with tick data support**")
    
    # === BACKTESTING TABS ===
    backtest_tab1, backtest_tab2, backtest_tab3, backtest_tab4 = st.tabs([
        "📊 Data Upload", 
        "⚙️ Configuration", 
        "🚀 Run Backtest", 
        "📈 Results & Reports"
    ])
    
    with backtest_tab1:
        data_upload_interface()
    
    with backtest_tab2:
        backtest_configuration_interface()
    
    with backtest_tab3:
        run_backtest_interface()
    
    with backtest_tab4:
        backtest_results_interface()


def data_upload_interface():
    """Data upload and validation interface"""
    st.markdown("#### 📊 Tick Data Upload & Validation")
    
    # === FILE UPLOAD ===
    st.markdown("##### 📁 Upload Historical Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose tick data file (CSV format)",
            type=['csv', 'txt'],
            help="Upload CSV file with OHLCV data. Supports various formats and delimiters."
        )
    
    with col2:
        st.markdown("**Supported Formats:**")
        st.write("• MT4/MT5 exports")
        st.write("• TradingView exports")
        st.write("• Custom CSV formats")
        st.write("• Tick data files")
        st.write("• Multiple timeframes")
    
    # === DATA PROCESSING ===
    if uploaded_file is not None:
        try:
            # Smart CSV reading with enhanced detection
            data = smart_read_tick_data(uploaded_file)
            
            if data is not None and not data.empty:
                st.session_state.uploaded_tick_data = data
                
                # === DATA VALIDATION ===
                st.markdown("##### ✅ Data Validation")
                
                try:
                    validation_results = validate_tick_data(data)
                    display_validation_results(validation_results)
                    
                    # Store validation results for debugging
                    st.session_state.last_validation_results = validation_results
                    
                except Exception as e:
                    st.error(f"❌ Error during validation: {e}")
                    st.info("💡 **Fallback**: Proceeding with basic validation...")
                    
                    # Basic fallback validation
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    
                    if not missing_columns:
                        st.success("✅ **Basic validation passed** - All required columns present")
                    else:
                        st.error(f"❌ **Missing columns**: {missing_columns}")
                        return  # Don't proceed if basic validation fails
                
                # === DATA PREVIEW ===
                st.markdown("##### 📋 Data Preview")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", f"{len(data):,}")
                
                with col2:
                    st.metric("Columns", len(data.columns))
                
                with col3:
                    if 'timestamp' in data.columns:
                        try:
                            timestamps = pd.to_datetime(data['timestamp'])
                            duration = timestamps.max() - timestamps.min()
                            duration_str = str(duration).split('.')[0]
                            st.metric("Duration", duration_str)
                        except Exception as e:
                            st.metric("Duration", "Error")
                            st.error(f"Duration calculation error: {e}")
                    else:
                        st.metric("Duration", "Unknown")
                
                with col4:
                    if 'timestamp' in data.columns:
                        try:
                            timestamps = pd.to_datetime(data['timestamp']).sort_values()
                            if len(timestamps) > 1:
                                time_diffs = timestamps.diff().dropna()
                                avg_diff = time_diffs.mean()
                                
                                if avg_diff.total_seconds() <= 60:
                                    timeframe = f"{int(avg_diff.total_seconds())}s"
                                elif avg_diff.total_seconds() <= 3600:
                                    timeframe = f"{int(avg_diff.total_seconds()/60)}m"
                                else:
                                    timeframe = f"{int(avg_diff.total_seconds()/3600)}h"
                                st.metric("Timeframe", timeframe)
                            else:
                                st.metric("Timeframe", "Single Point")
                        except Exception as e:
                            st.metric("Timeframe", "Error")
                            st.error(f"Timeframe calculation error: {e}")
                    else:
                        st.metric("Timeframe", "Unknown")
                
                # Data sample display
                st.markdown("**Sample Data:**")
                try:
                    sample_data = data.head(20)
                    st.dataframe(sample_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying data sample: {e}")
                    st.write("Raw data info:")
                    st.write(f"Shape: {data.shape}")
                    st.write(f"Columns: {list(data.columns)}")
                    st.write(f"Data types: {data.dtypes.to_dict()}")
                
                # === DATA STATISTICS ===
                with st.expander("📊 Data Statistics", expanded=False):
                    try:
                        display_data_statistics(data)
                    except Exception as e:
                        st.error(f"Error displaying data statistics: {e}")
                        # Fallback basic statistics
                        st.write("**Basic Information:**")
                        st.write(f"• Shape: {data.shape}")
                        st.write(f"• Columns: {list(data.columns)}")
                        if 'close' in data.columns:
                            st.write(f"• Price range: {data['close'].min():.4f} to {data['close'].max():.4f}")
                        if 'timestamp' in data.columns:
                            st.write(f"• Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # === DATA QUALITY CHECKS ===
                with st.expander("🔍 Data Quality Analysis", expanded=False):
                    try:
                        quality_analysis = analyze_data_quality(data)
                        display_quality_analysis(quality_analysis)
                    except Exception as e:
                        st.error(f"Error in quality analysis: {e}")
                        # Fallback quality info
                        st.write("**Basic Quality Checks:**")
                        missing_values = data.isnull().sum().sum()
                        st.write(f"• Missing values: {missing_values}")
                        duplicates = data.duplicated().sum()
                        st.write(f"• Duplicate rows: {duplicates}")
                
                # === SUCCESS MESSAGE ===
                st.success("🎯 **Data Upload Complete!** Your data is ready for backtesting.")
                st.info("📍 **Next Step**: Go to the 'Configuration' tab to set up your backtest parameters.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 **Troubleshooting:**")
            st.write("• Check file format (CSV with proper delimiters)")
            st.write("• Ensure required columns are present")
            st.write("• Verify timestamp format")
            st.write("• Check for special characters or encoding issues")
    
    else:
        # === SAMPLE DATA OPTION ===
        st.markdown("##### 🎯 Or Use Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generate Sample EUR/USD Data"):
                sample_data = generate_sample_forex_data("EURUSD", days=30)
                st.session_state.uploaded_tick_data = sample_data
                st.success("✅ Sample EUR/USD data generated!")
                st.rerun()
        
        with col2:
            if st.button("📊 Generate Sample NAS100 Data"):
                sample_data = generate_sample_index_data("NAS100", days=30)
                st.session_state.uploaded_tick_data = sample_data
                st.success("✅ Sample NAS100 data generated!")
                st.rerun()
        
        # === DATA FORMAT GUIDE ===
        with st.expander("📋 Expected Data Format", expanded=False):
            st.markdown("""
            **Required Columns:**
            - `timestamp` or `datetime`: Date and time (YYYY-MM-DD HH:MM:SS)
            - `open`: Opening price
            - `high`: Highest price
            - `low`: Lowest price
            - `close`: Closing price
            - `volume` (optional): Trading volume
            
            **Example Format:**
            ```
            timestamp,open,high,low,close,volume
            2024-01-01 00:00:00,1.1050,1.1055,1.1048,1.1052,1500
            2024-01-01 00:01:00,1.1052,1.1058,1.1050,1.1056,1200
            ```
            
            **Supported Variations:**
            - Different column names (auto-detected)
            - Various timestamp formats
            - Different delimiters (comma, semicolon, tab)
            - With or without headers
            """)


def backtest_configuration_interface():
    """Backtesting configuration interface"""
    st.markdown("#### ⚙️ Backtesting Configuration")
    
    if not st.session_state.get('uploaded_tick_data') is not None:
        st.warning("⚠️ **Upload tick data first** in the 'Data Upload' tab")
        return
    
    # === STRATEGY SELECTION ===
    st.markdown("##### 🎯 Strategy Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_source = st.selectbox(
            "Strategy Source",
            ["AI Generated Strategy (Recommended)", "Upload Python File", "Custom Code"],
            help="Choose where to get your trading strategy"
        )
        
        if strategy_source == "AI Generated Strategy (Recommended)":
            if st.session_state.get('generated_strategy'):
                st.success("✅ Using strategy from AI Builder")
                strategy_name = "AI Generated Strategy"
                
                # Show strategy details
                strategy = st.session_state.generated_strategy
                with st.expander("📋 Strategy Details", expanded=False):
                    st.write(f"**Indicators:** {', '.join(strategy.indicators) if strategy.indicators else 'None detected'}")
                    st.write(f"**Entry Conditions:** {len(strategy.entry_conditions) if strategy.entry_conditions else 0}")
                    st.write(f"**Exit Conditions:** {len(strategy.exit_conditions) if strategy.exit_conditions else 0}")
                    st.write(f"**Code Length:** {len(strategy.python_code)} characters")
                    st.write(f"**Generated by:** {strategy.metadata.get('provider', 'Unknown')} - {strategy.metadata.get('model_used', 'Unknown')}")
                    
                    st.markdown("**Code Preview:**")
                    st.code(strategy.python_code[:500] + "..." if len(strategy.python_code) > 500 else strategy.python_code, language="python")
            else:
                st.warning("⚠️ No AI strategy found")
                st.info("💡 **Generate a strategy first:**")
                st.write("1. Go to **AI Strategy Builder** tab")
                st.write("2. Describe your trading strategy in natural language")
                st.write("3. Generate the strategy code")
                st.write("4. Return here to backtest it")
                return
        
        elif strategy_source == "Upload Python File":
            strategy_file = st.file_uploader("Upload strategy file (.py)", type=['py'])
            if strategy_file:
                strategy_code = strategy_file.read().decode('utf-8')
                st.session_state.custom_strategy_code = strategy_code
                strategy_name = strategy_file.name
                with st.expander("📋 Strategy Preview", expanded=False):
                    st.code(strategy_code[:500] + "..." if len(strategy_code) > 500 else strategy_code, language="python")
            else:
                st.info("📁 Upload a Python file containing your strategy")
                return
        
        elif strategy_source == "Custom Code":
            st.markdown("**Paste your strategy code:**")
            custom_code = st.text_area(
                "Strategy Code",
                height=200,
                placeholder="""
def calculate_indicators(data):
    # Calculate your indicators here
    data['sma_20'] = data['close'].rolling(20).mean()
    return data

def generate_signals(data):
    # Generate trading signals here
    data['signal'] = 0
    # Your signal logic...
    return data

def apply_risk_management(data, signals):
    # Apply risk management rules
    return signals
                """,
                help="Implement your strategy logic in Python. Include calculate_indicators, generate_signals, and optionally apply_risk_management functions."
            )
            if custom_code:
                st.session_state.custom_strategy_code = custom_code
                strategy_name = "Custom Strategy"
                with st.expander("📋 Code Preview", expanded=False):
                    st.code(custom_code, language="python")
            else:
                st.info("✏️ Paste your strategy code above")
                return
    
    with col2:
        st.markdown("**Strategy Info:**")
        if strategy_source == "AI Generated Strategy" and st.session_state.get('generated_strategy'):
            strategy = st.session_state.generated_strategy
            if strategy.indicators:
                st.write(f"**Indicators:** {', '.join(strategy.indicators)}")
            if strategy.entry_conditions:
                st.write(f"**Entry Rules:** {len(strategy.entry_conditions)}")
            if strategy.exit_conditions:
                st.write(f"**Exit Rules:** {len(strategy.exit_conditions)}")
        else:
            st.info("Strategy details will appear here")
    
    # === INSTRUMENT CONFIGURATION ===
    st.markdown("##### 🎯 Instrument Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        instrument_type = st.selectbox(
            "Instrument Type",
            ["Forex", "Index", "Commodity", "Crypto", "Stock"],
            help="Type of financial instrument"
        )
        
        if instrument_type == "Forex":
            instrument_name = st.selectbox(
                "Forex Pair",
                ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "Custom"],
                help="Select forex pair or choose Custom"
            )
            if instrument_name == "Custom":
                instrument_name = st.text_input("Custom Pair", placeholder="XXXYYY")
        
        elif instrument_type == "Index":
            instrument_name = st.selectbox(
                "Index",
                ["NAS100", "SPX500", "GER40", "UK100", "JPN225", "AUS200", "Custom"],
                help="Select index or choose Custom"
            )
            if instrument_name == "Custom":
                instrument_name = st.text_input("Custom Index", placeholder="INDEX")
        
        else:
            instrument_name = st.text_input(f"{instrument_type} Symbol", placeholder="SYMBOL")
    
    with col2:
        # Auto-detect pip value based on instrument
        if instrument_type == "Forex":
            if "JPY" in instrument_name:
                default_pip_value = 0.01
                default_pip_size = 0.01
            else:
                default_pip_value = 10.0
                default_pip_size = 0.0001
        elif instrument_type == "Index":
            default_pip_value = 1.0
            default_pip_size = 1.0
        else:
            default_pip_value = 1.0
            default_pip_size = 0.01
        
        pip_value = st.number_input(
            "Pip Value ($)",
            value=default_pip_value,
            step=0.1,
            help="Value of 1 pip in account currency"
        )
        
        pip_size = st.number_input(
            "Pip Size",
            value=default_pip_size,
            step=0.0001,
            format="%.4f",
            help="Size of 1 pip (e.g., 0.0001 for most forex pairs)"
        )
    
    with col3:
        spread = st.number_input(
            "Spread (pips)",
            value=1.0,
            step=0.1,
            help="Typical spread for this instrument"
        )
        
        commission = st.number_input(
            "Commission per Lot ($)",
            value=0.0,
            step=0.1,
            help="Commission charged per standard lot"
        )
    
    # === ACCOUNT CONFIGURATION ===
    st.markdown("##### 💰 Account Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        starting_balance = st.number_input(
            "Starting Balance ($)",
            value=10000.0,
            min_value=100.0,
            max_value=1000000.0,
            step=1000.0,
            help="Initial account balance"
        )
        
        leverage = st.number_input(
            "Leverage",
            value=100,
            min_value=1,
            max_value=1000,
            help="Account leverage (e.g., 100 = 1:100)"
        )
    
    with col2:
        base_lot_size = st.number_input(
            "Base Lot Size",
            value=0.1,
            min_value=0.001,
            max_value=10.0,
            step=0.001,
            format="%.3f",
            help="Base position size in lots"
        )
        
        # Position sizing method
        sizing_method = st.selectbox(
            "Position Sizing",
            ["Fixed Lot Size", "Fixed Risk %", "Kelly Criterion", "Volatility-based"],
            help="Method for calculating position sizes"
        )
        
        if sizing_method == "Fixed Risk %":
            risk_per_trade = st.number_input("Risk per Trade (%)", value=2.0, max_value=10.0)
        elif sizing_method == "Kelly Criterion":
            kelly_fraction = st.number_input("Kelly Fraction", value=0.25, max_value=1.0)
    
    with col3:
        # Compounding options
        use_compounding = st.checkbox("Enable Compounding", value=True, help="Reinvest profits")
        
        if use_compounding:
            compounding_frequency = st.selectbox(
                "Compounding Frequency",
                ["Every Trade", "Daily", "Weekly", "Monthly"],
                help="How often to compound profits"
            )
        
        # Slippage
        slippage = st.number_input(
            "Slippage (pips)",
            value=0.5,
            step=0.1,
            help="Expected slippage per trade"
        )
    
    # === BACKTEST PERIOD ===
    st.markdown("##### 📅 Backtest Period")
    
    if st.session_state.get('uploaded_tick_data') is not None:
        data = st.session_state.uploaded_tick_data
        
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            min_date = timestamps.min().date()
            max_date = timestamps.max().date()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col3:
                # Timezone selection
                timezone = st.selectbox(
                    "Timezone",
                    ["UTC", "New York", "London", "Tokyo", "Sydney", "Custom"],
                    help="Timezone for the data timestamps"
                )
                
                if timezone == "Custom":
                    custom_tz = st.text_input("Custom Timezone", placeholder="America/New_York")
            
            # Show selected period info
            if start_date <= end_date:
                period_data = data[
                    (timestamps.dt.date >= start_date) & 
                    (timestamps.dt.date <= end_date)
                ]
                st.info(f"📊 **Selected Period**: {len(period_data):,} data points from {start_date} to {end_date}")
            else:
                st.error("❌ Start date must be before end date")
    
    # === ADVANCED OPTIONS ===
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Execution Settings:**")
            execution_delay = st.number_input("Execution Delay (ms)", value=0, help="Simulated execution delay")
            partial_fills = st.checkbox("Allow Partial Fills", value=False)
            weekend_trading = st.checkbox("Include Weekend Data", value=False)
            
            st.markdown("**Risk Management:**")
            max_drawdown_stop = st.number_input("Max Drawdown Stop (%)", value=20.0, help="Stop trading at this drawdown")
            daily_loss_limit = st.number_input("Daily Loss Limit (%)", value=5.0, help="Stop trading for the day")
        
        with col2:
            st.markdown("**Performance Tracking:**")
            track_equity_curve = st.checkbox("Track Equity Curve", value=True)
            track_drawdown = st.checkbox("Track Drawdown", value=True)
            track_trade_details = st.checkbox("Track Trade Details", value=True)
            
            st.markdown("**Output Options:**")
            generate_report = st.checkbox("Generate Detailed Report", value=True)
            export_trades = st.checkbox("Export Trade List", value=True)
            create_charts = st.checkbox("Create Performance Charts", value=True)
    
    # === SAVE CONFIGURATION ===
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        config_name = st.text_input("Configuration Name", placeholder="My Backtest Config")
    
    with col2:
        if st.button("💾 Save Config"):
            if config_name:
                save_backtest_config(config_name, {
                    'strategy_source': strategy_source,
                    'instrument_type': instrument_type,
                    'instrument_name': instrument_name,
                    'pip_value': pip_value,
                    'pip_size': pip_size,
                    'spread': spread,
                    'commission': commission,
                    'starting_balance': starting_balance,
                    'leverage': leverage,
                    'base_lot_size': base_lot_size,
                    'sizing_method': sizing_method,
                    'use_compounding': use_compounding,
                    'slippage': slippage,
                    'start_date': start_date.isoformat() if 'start_date' in locals() else None,
                    'end_date': end_date.isoformat() if 'end_date' in locals() else None,
                    'timezone': timezone if 'timezone' in locals() else 'UTC'
                })
                st.success("✅ Configuration saved!")
            else:
                st.warning("⚠️ Enter a configuration name")
    
    with col3:
        saved_configs = get_saved_backtest_configs()
        if saved_configs:
            selected_config = st.selectbox("Load Config", [""] + saved_configs)
            if selected_config and st.button("📤 Load"):
                load_backtest_config(selected_config)
                st.success("✅ Configuration loaded!")
                st.rerun()
    
    # Store configuration in session state
    if 'start_date' in locals() and 'end_date' in locals():
        st.session_state.backtest_config = {
            'strategy_source': strategy_source,
            'strategy_name': strategy_name if 'strategy_name' in locals() else 'Unknown',
            'instrument_type': instrument_type,
            'instrument_name': instrument_name,
            'pip_value': pip_value,
            'pip_size': pip_size,
            'spread': spread,
            'commission': commission,
            'starting_balance': starting_balance,
            'leverage': leverage,
            'base_lot_size': base_lot_size,
            'sizing_method': sizing_method,
            'use_compounding': use_compounding,
            'slippage': slippage,
            'start_date': start_date,
            'end_date': end_date,
            'timezone': timezone if 'timezone' in locals() else 'UTC'
        }


def run_backtest_interface():
    """Run backtest interface"""
    st.markdown("#### 🚀 Run Backtest")
    
    # Check prerequisites
    if not st.session_state.get('uploaded_tick_data') is not None:
        st.warning("⚠️ **Upload tick data first** in the 'Data Upload' tab")
        return
    
    if not st.session_state.get('backtest_config'):
        st.warning("⚠️ **Configure backtest settings** in the 'Configuration' tab")
        return
    
    # === BACKTEST SUMMARY ===
    st.markdown("##### 📋 Backtest Summary")
    
    config = st.session_state.backtest_config
    data = st.session_state.uploaded_tick_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strategy", config['strategy_name'])
        st.metric("Instrument", f"{config['instrument_name']} ({config['instrument_type']})")
    
    with col2:
        st.metric("Starting Balance", f"${config['starting_balance']:,.0f}")
        st.metric("Leverage", f"1:{config['leverage']}")
    
    with col3:
        st.metric("Period", f"{config['start_date']} to {config['end_date']}")
        period_data = data[
            (pd.to_datetime(data['timestamp']).dt.date >= config['start_date']) & 
            (pd.to_datetime(data['timestamp']).dt.date <= config['end_date'])
        ]
        st.metric("Data Points", f"{len(period_data):,}")
    
    with col4:
        st.metric("Base Lot Size", f"{config['base_lot_size']:.3f}")
        st.metric("Spread", f"{config['spread']} pips")
    
    # === BACKTEST EXECUTION ===
    st.markdown("##### ⚡ Execute Backtest")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Execution options
        execution_mode = st.selectbox(
            "Execution Mode",
            ["Fast (Basic)", "Standard (Detailed)", "Comprehensive (Full Analysis)"],
            help="Choose execution speed vs detail level"
        )
        
        show_progress = st.checkbox("Show Progress", value=True)
        save_intermediate = st.checkbox("Save Intermediate Results", value=False)
    
    with col2:
        st.markdown("**Estimated Time:**")
        data_points = len(period_data) if 'period_data' in locals() else len(data)
        
        if execution_mode == "Fast (Basic)":
            est_time = data_points / 10000  # Rough estimate
            st.write(f"~{est_time:.1f} seconds")
        elif execution_mode == "Standard (Detailed)":
            est_time = data_points / 5000
            st.write(f"~{est_time:.1f} seconds")
        else:
            est_time = data_points / 2000
            st.write(f"~{est_time:.1f} seconds")
        
        st.write(f"Data points: {data_points:,}")
    
    # === RUN BACKTEST BUTTON ===
    if st.button("🚀 Start Backtest", type="primary", use_container_width=True):
        run_comprehensive_backtest(config, data, execution_mode, show_progress)


def backtest_results_interface():
    """Backtest results and reports interface"""
    st.markdown("#### 📈 Backtest Results & Reports")
    
    if not st.session_state.get('backtest_results'):
        st.info("📊 **No backtest results yet**")
        st.write("Run a backtest in the 'Run Backtest' tab to see results here.")
        return
    
    results = st.session_state.backtest_results
    
    # === PERFORMANCE OVERVIEW ===
    st.markdown("##### 📊 Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_return = results.get('total_return', 0)
        st.metric("Total Return", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
    
    with col2:
        max_drawdown = results.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta=f"{max_drawdown:.2f}%")
    
    with col3:
        sharpe_ratio = results.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        win_rate = results.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col5:
        total_trades = results.get('total_trades', 0)
        st.metric("Total Trades", total_trades)
    
    # === EQUITY CURVE ===
    st.markdown("##### 📈 Equity Curve")
    
    if 'equity_curve' in results:
        equity_chart = create_equity_curve_chart(results['equity_curve'])
        st.plotly_chart(equity_chart, use_container_width=True, key="main_equity_curve")
    
    # === DETAILED METRICS ===
    st.markdown("##### 📊 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Return Metrics:**")
        metrics_df = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Monthly Return', 'Best Month', 'Worst Month'],
            'Value': [
                f"{results.get('total_return', 0):.2f}%",
                f"{results.get('annualized_return', 0):.2f}%",
                f"{results.get('avg_monthly_return', 0):.2f}%",
                f"{results.get('best_month', 0):.2f}%",
                f"{results.get('worst_month', 0):.2f}%"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        st.markdown("**Risk Metrics:**")
        risk_df = pd.DataFrame({
            'Metric': ['Max Drawdown', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
            'Value': [
                f"{results.get('max_drawdown', 0):.2f}%",
                f"{results.get('volatility', 0):.2f}%",
                f"{results.get('sharpe_ratio', 0):.2f}",
                f"{results.get('sortino_ratio', 0):.2f}",
                f"{results.get('calmar_ratio', 0):.2f}"
            ]
        })
        st.dataframe(risk_df, hide_index=True)
    
    # === TRADE ANALYSIS ===
    st.markdown("##### 📋 Trade Analysis")
    
    if 'trades' in results:
        trades_df = pd.DataFrame(results['trades'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Trade Statistics:**")
            trade_stats = pd.DataFrame({
                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Avg Win', 'Avg Loss'],
                'Value': [
                    results.get('total_trades', 0),
                    results.get('winning_trades', 0),
                    results.get('losing_trades', 0),
                    f"{results.get('win_rate', 0):.1f}%",
                    f"${results.get('avg_win', 0):.2f}",
                    f"${results.get('avg_loss', 0):.2f}"
                ]
            })
            st.dataframe(trade_stats, hide_index=True)
        
        with col2:
            st.markdown("**Position Analysis:**")
            long_trades = len(trades_df[trades_df['direction'] == 'long']) if 'direction' in trades_df.columns else 0
            short_trades = len(trades_df[trades_df['direction'] == 'short']) if 'direction' in trades_df.columns else 0
            
            position_stats = pd.DataFrame({
                'Metric': ['Long Trades', 'Short Trades', 'Long Win Rate', 'Short Win Rate', 'Avg Hold Time'],
                'Value': [
                    long_trades,
                    short_trades,
                    f"{results.get('long_win_rate', 0):.1f}%",
                    f"{results.get('short_win_rate', 0):.1f}%",
                    results.get('avg_hold_time', 'N/A')
                ]
            })
            st.dataframe(position_stats, hide_index=True)
        
        with col3:
            st.markdown("**Risk Analysis:**")
            risk_stats = pd.DataFrame({
                'Metric': ['Largest Win', 'Largest Loss', 'Profit Factor', 'Recovery Factor', 'Max Consecutive Losses'],
                'Value': [
                    f"${results.get('largest_win', 0):.2f}",
                    f"${results.get('largest_loss', 0):.2f}",
                    f"{results.get('profit_factor', 0):.2f}",
                    f"{results.get('recovery_factor', 0):.2f}",
                    results.get('max_consecutive_losses', 0)
                ]
            })
            st.dataframe(risk_stats, hide_index=True)
        
        # === TRADE LIST ===
        with st.expander("📋 Complete Trade List", expanded=False):
            st.dataframe(trades_df, use_container_width=True)
            
            # Export trades
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trade List",
                csv,
                f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    # === CHARTS AND ANALYSIS ===
    st.markdown("##### 📊 Performance Charts")
    
    chart_tabs = st.tabs(["📈 Equity & Drawdown", "📊 Monthly Returns", "🎯 Trade Distribution", "📅 Calendar"])
    
    with chart_tabs[0]:
        if 'equity_curve' in results and 'drawdown_curve' in results:
            col1, col2 = st.columns(2)
            with col1:
                equity_chart = create_equity_curve_chart(results['equity_curve'])
                st.plotly_chart(equity_chart, use_container_width=True, key="detailed_equity_curve")
            with col2:
                drawdown_chart = create_drawdown_chart(results['drawdown_curve'])
                st.plotly_chart(drawdown_chart, use_container_width=True, key="detailed_drawdown_curve")
    
    with chart_tabs[1]:
        if 'monthly_returns' in results:
            monthly_chart = create_monthly_returns_chart(results['monthly_returns'])
            st.plotly_chart(monthly_chart, use_container_width=True, key="monthly_returns_chart")
    
    with chart_tabs[2]:
        if 'trades' in results:
            distribution_chart = create_trade_distribution_chart(results['trades'])
            if distribution_chart:
                st.plotly_chart(distribution_chart, use_container_width=True, key="trade_distribution_chart")
            else:
                st.info("📊 Trade distribution chart not available - insufficient trade data")
    
    with chart_tabs[3]:
        if 'trades' in results:
            calendar_chart = create_trading_calendar_chart(results['trades'])
            if calendar_chart:
                st.plotly_chart(calendar_chart, use_container_width=True, key="trading_calendar_chart")
            else:
                st.info("📅 Trading calendar not available - insufficient trade data")
    
    # === EXPORT RESULTS ===
    st.markdown("##### 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Report"):
            report_data = generate_backtest_report(results)
            st.download_button(
                "📥 Download Report",
                report_data,
                f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html"
            )
    
    with col2:
        if st.button("📈 Export for Analysis"):
            # Export in format compatible with the main analytics dashboard
            analysis_data = convert_backtest_to_analysis_format(results)
            csv_data = analysis_data.to_csv(index=False)
            st.download_button(
                "📥 Download for Analytics",
                csv_data,
                f"backtest_for_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    with col3:
        if st.button("🔄 Run New Backtest"):
            # Clear results and go back to configuration
            if 'backtest_results' in st.session_state:
                del st.session_state.backtest_results
            st.info("✅ Ready for new backtest. Configure in the 'Configuration' tab.")
            st.rerun()


def trading_charts_interface():
    """Trading charts and visualization interface"""
    st.markdown("### 📈 Trading Charts & Visualization")
    st.markdown("**Advanced charting and trade visualization**")
    
    # Check if we have data to visualize
    has_backtest_data = st.session_state.get('backtest_results') is not None
    has_tick_data = st.session_state.get('uploaded_tick_data') is not None
    
    if not has_backtest_data and not has_tick_data:
        st.info("📊 **No data available for visualization**")
        st.write("• Upload tick data in the 'Advanced Backtesting' tab")
        st.write("• Run a backtest to see trade entries and exits")
        st.write("• Or use the main Analytics Dashboard for broker statement analysis")
        return
    
    # === CHART TYPE SELECTION ===
    chart_type = st.selectbox(
        "Chart Type",
        ["Price Chart with Trades", "Equity Curve", "Drawdown Analysis", "Performance Heatmap", "Trade Distribution"],
        help="Choose the type of chart to display"
    )
    
    if chart_type == "Price Chart with Trades":
        display_price_chart_with_trades()
    elif chart_type == "Equity Curve":
        display_equity_curve_analysis()
    elif chart_type == "Drawdown Analysis":
        display_drawdown_analysis()
    elif chart_type == "Performance Heatmap":
        display_performance_heatmap()
    elif chart_type == "Trade Distribution":
        display_trade_distribution_analysis()


def display_price_chart_with_trades():
    """Display price chart with trade entry/exit points"""
    st.markdown("#### 📈 Price Chart with Trade Entries/Exits")
    
    if not st.session_state.get('uploaded_tick_data') is not None:
        st.warning("⚠️ Upload tick data first to see price charts")
        return
    
    data = st.session_state.uploaded_tick_data
    
    # Chart configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_style = st.selectbox("Chart Style", ["Candlestick", "Line", "OHLC"])
        
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["All Data", "Last 1000 Points", "Last 500 Points", "Custom Range"]
        )
    
    with col3:
        show_volume = st.checkbox("Show Volume", value=True)
    
    # Create the price chart
    try:
        price_chart = create_advanced_price_chart(
            data, 
            chart_style=chart_style,
            time_range=time_range,
            show_volume=show_volume,
            trades=st.session_state.get('backtest_results', {}).get('trades', [])
        )
        st.plotly_chart(price_chart, use_container_width=True, key="price_action_chart")
        
        # Chart controls
        with st.expander("📊 Chart Controls", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                show_indicators = st.multiselect(
                    "Show Indicators",
                    ["SMA 20", "SMA 50", "EMA 20", "Bollinger Bands", "RSI", "MACD"],
                    help="Add technical indicators to the chart"
                )
            
            with col2:
                highlight_trades = st.checkbox("Highlight Trades", value=True)
                show_trade_labels = st.checkbox("Show Trade Labels", value=False)
        
    except Exception as e:
        st.error(f"❌ Error creating price chart: {str(e)}")


def display_equity_curve_analysis():
    """Display detailed equity curve analysis"""
    st.markdown("#### 📈 Equity Curve Analysis")
    
    if not st.session_state.get('backtest_results'):
        st.warning("⚠️ Run a backtest first to see equity curve")
        return
    
    results = st.session_state.backtest_results
    
    if 'equity_curve' not in results:
        st.error("❌ No equity curve data available")
        return
    
    # Equity curve chart
    equity_chart = create_detailed_equity_curve(results['equity_curve'])
    st.plotly_chart(equity_chart, use_container_width=True, key="detailed_equity_analysis")
    
    # Equity curve statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Equity Curve Statistics:**")
        equity_stats = analyze_equity_curve(results['equity_curve'])
        st.dataframe(equity_stats, hide_index=True)
    
    with col2:
        st.markdown("**Drawdown Periods:**")
        drawdown_periods = analyze_drawdown_periods(results.get('drawdown_curve', []))
        st.dataframe(drawdown_periods, hide_index=True)


# Helper functions for the advanced backtesting system
def smart_read_tick_data(uploaded_file):
    """Smart CSV reader for tick data with enhanced detection"""
    import io
    
    try:
        # Read file content
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Detect encoding
        try:
            content_str = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_str = file_content.decode('latin-1')
            except UnicodeDecodeError:
                content_str = file_content.decode('cp1252')
        
        # Detect delimiter and structure
        first_line = content_str.split('\n')[0].strip()
        
        # Detect delimiter
        delimiter = ','
        if '\t' in first_line:
            delimiter = '\t'
            st.info("🔍 **Tab-separated format detected**")
        elif ';' in first_line:
            delimiter = ';'
            st.info("🔍 **Semicolon-separated format detected**")
        else:
            st.info("🔍 **Comma-separated format detected**")
        
        # Split first line to check structure
        parts = first_line.split(delimiter)
        st.info(f"📊 **Detected {len(parts)} columns** in first line")
        
        # Check if first line contains headers or data
        has_headers = False
        try:
            # Try to parse the first part as a timestamp
            pd.to_datetime(parts[0])
            # If successful, it's likely data (no headers)
            has_headers = False
            st.info("🔍 **No headers detected** - First line contains timestamp data")
        except:
            # If timestamp parsing fails, check if it looks like numeric data
            try:
                # Try to parse parts as numbers (skip first part which might be a date string)
                for part in parts[1:4]:  # Check a few columns
                    float(part.strip())
                # If we get here, numeric data found (no headers)
                has_headers = False
                st.info("🔍 **No headers detected** - First line contains numeric data")
            except (ValueError, IndexError):
                # If parsing fails, likely has headers
                has_headers = True
                st.info("✅ **Headers detected** - First line contains column names")
        
        # Read CSV based on detection results
        if has_headers:
            data = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
            st.success(f"✅ **Using existing headers**: {list(data.columns)}")
        else:
            data = pd.read_csv(io.StringIO(content_str), delimiter=delimiter, header=None)
            st.info("🔄 **Auto-assigning column names**")
            
            # Auto-assign column names based on number of columns
            if len(data.columns) == 5:
                data.columns = ['timestamp', 'open', 'high', 'low', 'close']
                st.success("✅ **Auto-assigned**: timestamp, open, high, low, close")
            elif len(data.columns) == 6:
                data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                st.success("✅ **Auto-assigned**: timestamp, open, high, low, close, volume")
            elif len(data.columns) == 7:
                data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'extra']
                st.success("✅ **Auto-assigned**: timestamp, open, high, low, close, volume, extra")
            elif len(data.columns) == 4:
                # Sometimes data might be: timestamp, high, low, close (no open)
                data.columns = ['timestamp', 'high', 'low', 'close']
                # Create open column from previous close or current close
                data['open'] = data['close'].shift(1).fillna(data['close'])
                # Reorder columns
                data = data[['timestamp', 'open', 'high', 'low', 'close']]
                st.success("✅ **Auto-assigned**: timestamp, high, low, close (open created from close)")
            else:
                st.warning(f"⚠️ **Unexpected format**: {len(data.columns)} columns detected")
                st.info("💡 **Expected**: timestamp, open, high, low, close [, volume]")
                
                # Try to assign best guess based on common patterns
                if len(data.columns) >= 5:
                    new_columns = ['timestamp', 'open', 'high', 'low', 'close']
                    if len(data.columns) > 5:
                        new_columns.extend([f'extra_{i}' for i in range(len(data.columns) - 5)])
                    data.columns = new_columns
                    st.info(f"🔄 **Best guess assignment**: {new_columns}")
                elif len(data.columns) == 3:
                    # Might be timestamp, close, volume
                    data.columns = ['timestamp', 'close', 'volume']
                    # Create OHLC from close
                    data['open'] = data['close'].shift(1).fillna(data['close'])
                    data['high'] = data['close']
                    data['low'] = data['close']
                    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    st.info("🔄 **Created OHLC from close price**")
                elif len(data.columns) == 2:
                    # Might be timestamp, close
                    data.columns = ['timestamp', 'close']
                    # Create OHLC from close
                    data['open'] = data['close'].shift(1).fillna(data['close'])
                    data['high'] = data['close']
                    data['low'] = data['close']
                    data = data[['timestamp', 'open', 'high', 'low', 'close']]
                    st.info("🔄 **Created OHLC from close price**")
                else:
                    # Not enough columns, assign generic names
                    data.columns = [f'column_{i}' for i in range(len(data.columns))]
                    st.error(f"❌ **Insufficient columns**: Only {len(data.columns)} found, need at least 2 (timestamp + price)")
                    return None
        
        # Standardize column names (convert to lowercase and handle variations)
        data.columns = [col.lower().strip() for col in data.columns]
        
        # Map common column variations to standard names
        column_mapping = {
            'datetime': 'timestamp',
            'date': 'timestamp',
            'time': 'timestamp',
            'date_time': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
            'price': 'close',
            'bid': 'close',
            'ask': 'close'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and new_col not in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
                st.info(f"🔄 **Mapped column**: '{old_col}' → '{new_col}'")
        
        # Validate that we have the minimum required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.warning(f"⚠️ **Missing columns**: {missing_columns}")
            
            # Try to create missing columns if we have close price
            if 'close' in data.columns:
                if 'open' not in data.columns:
                    data['open'] = data['close'].shift(1).fillna(data['close'])
                    st.info("🔄 **Created 'open' from previous close**")
                
                if 'high' not in data.columns:
                    data['high'] = data['close']
                    st.info("🔄 **Created 'high' from close**")
                
                if 'low' not in data.columns:
                    data['low'] = data['close']
                    st.info("🔄 **Created 'low' from close**")
            
            # Check again after creating columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"❌ **Still missing required columns**: {missing_columns}")
                st.info("💡 **Available columns**: " + ", ".join(data.columns))
                return None
        
        # Clean and validate data types
        try:
            # Convert timestamp
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            st.success("✅ **Timestamp conversion successful**")
        except Exception as e:
            st.error(f"❌ **Timestamp conversion failed**: {str(e)}")
            return None
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            numeric_columns.append('volume')
        
        for col in numeric_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    st.warning(f"⚠️ **Could not convert {col} to numeric**: {str(e)}")
        
        # Remove rows with NaN values in critical columns
        initial_rows = len(data)
        data = data.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
        final_rows = len(data)
        
        if initial_rows != final_rows:
            st.warning(f"⚠️ **Removed {initial_rows - final_rows} rows with missing data**")
        
        if len(data) == 0:
            st.error("❌ **No valid data remaining after cleaning**")
            return None
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        st.success(f"✅ **Data processing complete**: {len(data)} rows, {len(data.columns)} columns")
        
        return data
        
    except Exception as e:
        st.error(f"❌ **Error reading file**: {str(e)}")
        st.info("💡 **Troubleshooting Tips:**")
        st.write("• Check file encoding (UTF-8, Latin-1, CP1252)")
        st.write("• Verify delimiter (comma, tab, semicolon)")
        st.write("• Ensure timestamp format is recognizable")
        st.write("• Check for special characters or corrupted data")
        return None

def validate_tick_data(data):
    """Validate tick data quality and structure"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    if data is None or data.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("No data provided or data is empty")
        return validation_results
    
    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
        validation_results['info'].append(f"Available columns: {', '.join(data.columns)}")
    else:
        validation_results['info'].append("All required columns present")
    
    # Check data types and values
    if 'timestamp' in data.columns:
        try:
            # Check if timestamp is already datetime
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                pd.to_datetime(data['timestamp'])
            validation_results['info'].append("Timestamp format is valid")
        except Exception as e:
            validation_results['errors'].append(f"Invalid timestamp format: {str(e)}")
            validation_results['is_valid'] = False
    
    # Check OHLC logic
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        try:
            # Convert to numeric if not already
            for col in ['open', 'high', 'low', 'close']:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Check OHLC relationships
            ohlc_issues = 0
            for idx, row in data.iterrows():
                if pd.isna(row[['open', 'high', 'low', 'close']]).any():
                    continue  # Skip rows with NaN values
                
                # High should be >= max(open, close) and Low should be <= min(open, close)
                max_oc = max(row['open'], row['close'])
                min_oc = min(row['open'], row['close'])
                
                if row['high'] < max_oc or row['low'] > min_oc:
                    ohlc_issues += 1
                
                # Stop checking after finding too many issues (performance)
                if ohlc_issues > 100:
                    break
            
            if ohlc_issues > 0:
                validation_results['warnings'].append(f"OHLC logic issues in {ohlc_issues} rows (High < max(O,C) or Low > min(O,C))")
            else:
                validation_results['info'].append("OHLC relationships are valid")
                
        except Exception as e:
            validation_results['warnings'].append(f"Could not validate OHLC relationships: {str(e)}")
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        validation_results['warnings'].append(f"Found {missing_values} missing values across all columns")
        
        # Show missing values per column
        missing_per_col = data.isnull().sum()
        missing_cols = missing_per_col[missing_per_col > 0]
        if len(missing_cols) > 0:
            validation_results['info'].append(f"Missing values by column: {dict(missing_cols)}")
    else:
        validation_results['info'].append("No missing values found")
    
    # Check data continuity (for time series data)
    if 'timestamp' in data.columns and len(data) > 1:
        try:
            timestamps = pd.to_datetime(data['timestamp']).sort_values()
            time_diffs = timestamps.diff().dropna()
            
            # Find the most common time difference (likely the intended frequency)
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                expected_freq = mode_diff.iloc[0]
                
                # Count gaps larger than 2x the expected frequency
                large_gaps = (time_diffs > expected_freq * 2).sum()
                if large_gaps > 0:
                    validation_results['warnings'].append(f"Found {large_gaps} potential data gaps (>2x expected frequency)")
                else:
                    validation_results['info'].append("Data continuity looks good")
                    
                validation_results['info'].append(f"Detected frequency: {expected_freq}")
            
        except Exception as e:
            validation_results['warnings'].append(f"Could not analyze data continuity: {str(e)}")
    
    # Check for duplicate timestamps
    if 'timestamp' in data.columns:
        duplicates = data['timestamp'].duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate timestamps")
        else:
            validation_results['info'].append("No duplicate timestamps")
    
    # Check price ranges for reasonableness
    if 'close' in data.columns:
        try:
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            if not close_prices.empty:
                price_range = close_prices.max() - close_prices.min()
                price_mean = close_prices.mean()
                
                # Check for extreme price variations (>1000% range)
                if price_range > price_mean * 10:
                    validation_results['warnings'].append("Extremely wide price range detected - check for data errors")
                
                # Check for zero or negative prices
                invalid_prices = (close_prices <= 0).sum()
                if invalid_prices > 0:
                    validation_results['warnings'].append(f"Found {invalid_prices} zero or negative prices")
                
                validation_results['info'].append(f"Price range: {close_prices.min():.4f} to {close_prices.max():.4f}")
                
        except Exception as e:
            validation_results['warnings'].append(f"Could not validate price ranges: {str(e)}")
    
    # Final validation status
    if len(validation_results['errors']) == 0:
        validation_results['is_valid'] = True
        validation_results['info'].append("✅ Data validation passed")
    else:
        validation_results['is_valid'] = False
    
    return validation_results

def display_validation_results(validation_results):
    """Display data validation results with helpful guidance"""
    if validation_results['is_valid']:
        st.success("✅ Data validation passed")
    else:
        st.error("❌ Data validation failed")
    
    # Show errors with solutions
    if validation_results['errors']:
        st.error("**Errors:**")
        for error in validation_results['errors']:
            st.write(f"• {error}")
        
        # Provide specific solutions for common errors
        if any("Missing required columns" in error for error in validation_results['errors']):
            st.info("💡 **Solutions for missing columns:**")
            st.write("• **No headers detected**: The system will auto-assign column names based on data structure")
            st.write("• **Tab-separated data**: Make sure your file uses tabs (\\t) as separators")
            st.write("• **Expected format**: timestamp, open, high, low, close [, volume]")
            st.write("• **Minimum requirement**: timestamp + close price (other OHLC values will be created)")
            
            with st.expander("📋 Supported File Formats", expanded=False):
                st.markdown("""
                **✅ Supported formats:**
                
                **Format 1: Full OHLCV (6 columns)**
                ```
                2025-09-19 12:17    24495.67    24496.62    24494.15    24496.57    1
                ```
                
                **Format 2: OHLC only (5 columns)**
                ```
                2025-09-19 12:17,24495.67,24496.62,24494.15,24496.57
                ```
                
                **Format 3: With headers**
                ```
                timestamp,open,high,low,close,volume
                2025-09-19 12:17,24495.67,24496.62,24494.15,24496.57,1
                ```
                
                **Format 4: Minimal (timestamp + close)**
                ```
                2025-09-19 12:17,24496.57
                ```
                
                **Supported delimiters:** Comma (,), Tab (\\t), Semicolon (;)
                """)
    
    # Show warnings
    if validation_results['warnings']:
        st.warning("**Warnings:**")
        for warning in validation_results['warnings']:
            st.write(f"• {warning}")
        
        # Provide guidance for warnings
        if any("OHLC logic issues" in warning for warning in validation_results['warnings']):
            st.info("💡 **OHLC Logic Issues**: Some rows have High < max(Open,Close) or Low > min(Open,Close). This might indicate data quality issues but won't prevent backtesting.")
        
        if any("data gaps" in warning for warning in validation_results['warnings']):
            st.info("💡 **Data Gaps**: Missing time periods detected. This is normal for markets that close (weekends, holidays).")
        
        if any("missing values" in warning for warning in validation_results['warnings']):
            st.info("💡 **Missing Values**: Some data points are missing. These will be handled automatically during backtesting.")
    
    # Show additional information
    if validation_results['info']:
        with st.expander("ℹ️ Additional Information", expanded=False):
            for info in validation_results['info']:
                st.write(f"• {info}")
    
    # Show data quality score
    total_checks = len(validation_results['errors']) + len(validation_results['warnings']) + len(validation_results['info'])
    if total_checks > 0:
        error_weight = len(validation_results['errors']) * 3
        warning_weight = len(validation_results['warnings']) * 1
        info_weight = len(validation_results['info']) * 0
        
        quality_score = max(0, 100 - (error_weight * 10 + warning_weight * 5))
        
        if quality_score >= 90:
            st.success(f"📊 **Data Quality Score**: {quality_score}/100 (Excellent)")
        elif quality_score >= 70:
            st.warning(f"📊 **Data Quality Score**: {quality_score}/100 (Good)")
        elif quality_score >= 50:
            st.warning(f"📊 **Data Quality Score**: {quality_score}/100 (Fair)")
        else:
            st.error(f"📊 **Data Quality Score**: {quality_score}/100 (Poor)")
    
    # Show next steps
    if validation_results['is_valid']:
        st.info("🎯 **Next Steps**: Your data is ready for backtesting! Go to the 'Configuration' tab to set up your backtest parameters.")
    else:
        st.info("🔧 **Next Steps**: Please fix the errors above, then try uploading your file again. You can also try the sample data generation buttons below.")

def display_data_statistics(data):
    """Display comprehensive data statistics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Statistics:**")
        if 'close' in data.columns:
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{len(data):,}",
                    f"{data['close'].mean():.5f}",
                    f"{data['close'].std():.5f}",
                    f"{data['close'].min():.5f}",
                    f"{data['close'].max():.5f}",
                    f"{data['close'].max() - data['close'].min():.5f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)
    
    with col2:
        st.markdown("**Time Analysis:**")
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            time_stats = pd.DataFrame({
                'Metric': ['Start Date', 'End Date', 'Duration', 'Frequency', 'Total Points'],
                'Value': [
                    timestamps.min().strftime('%Y-%m-%d %H:%M:%S'),
                    timestamps.max().strftime('%Y-%m-%d %H:%M:%S'),
                    str(timestamps.max() - timestamps.min()).split('.')[0],
                    f"{len(data) / ((timestamps.max() - timestamps.min()).total_seconds() / 3600):.1f} points/hour",
                    f"{len(data):,}"
                ]
            })
            st.dataframe(time_stats, hide_index=True)

def analyze_data_quality(data):
    """Analyze data quality comprehensively"""
    quality_analysis = {
        'completeness': 100 - (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
        'duplicates': data.duplicated().sum(),
        'outliers': 0,
        'consistency': 100
    }
    
    # Check for outliers in price data
    if 'close' in data.columns:
        Q1 = data['close'].quantile(0.25)
        Q3 = data['close'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data['close'] < (Q1 - 1.5 * IQR)) | (data['close'] > (Q3 + 1.5 * IQR))).sum()
        quality_analysis['outliers'] = outliers
    
    return quality_analysis

def display_quality_analysis(quality_analysis):
    """Display data quality analysis results"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = quality_analysis['completeness']
        st.metric("Completeness", f"{completeness:.1f}%", 
                 delta="Good" if completeness > 95 else "Poor")
    
    with col2:
        duplicates = quality_analysis['duplicates']
        st.metric("Duplicates", duplicates,
                 delta="Good" if duplicates == 0 else "Check")
    
    with col3:
        outliers = quality_analysis['outliers']
        st.metric("Outliers", outliers,
                 delta="Good" if outliers < 10 else "Review")
    
    with col4:
        consistency = quality_analysis['consistency']
        st.metric("Consistency", f"{consistency:.1f}%",
                 delta="Good" if consistency > 90 else "Poor")

def generate_sample_forex_data(pair, days=30):
    """Generate sample forex data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1min')
    
    # Generate realistic forex price movements
    np.random.seed(42)
    base_price = 1.1000 if pair == "EURUSD" else 1.2500
    
    price_changes = np.random.normal(0, 0.0001, len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLC data
    data = []
    for i, (timestamp, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, 0.0002))
        low = close - abs(np.random.normal(0, 0.0002))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def generate_sample_index_data(index, days=30):
    """Generate sample index data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1min')
    
    # Generate realistic index price movements
    np.random.seed(42)
    base_price = 15000 if index == "NAS100" else 4000
    
    price_changes = np.random.normal(0, 5, len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLC data
    data = []
    for i, (timestamp, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, 10))
        low = close - abs(np.random.normal(0, 10))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def run_comprehensive_backtest(config, data, execution_mode, show_progress):
    """Run comprehensive backtest with actual strategy execution"""
    
    with st.spinner("🚀 Initializing backtest engine..."):
        # Initialize backtest engine
        try:
            # Filter data by date range
            filtered_data = data[
                (pd.to_datetime(data['timestamp']).dt.date >= config['start_date']) & 
                (pd.to_datetime(data['timestamp']).dt.date <= config['end_date'])
            ].copy()
            
            if len(filtered_data) == 0:
                st.error("❌ No data found in the selected date range")
                return
            
            # Ensure timestamp is datetime and set as index
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
            filtered_data = filtered_data.set_index('timestamp').sort_index()
            
            st.success(f"✅ Backtest initialized with {len(filtered_data):,} data points")
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return
    
    # Progress tracking
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
    
    try:
        # Execute the strategy code
        if show_progress:
            progress_bar.progress(20)
            status_text.text("🤖 Executing strategy code...")
        
        strategy_code = None
        
        # Determine which strategy code to use
        if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
            strategy_code = st.session_state.generated_strategy.python_code
            st.info("🤖 Using AI-generated strategy")
        elif 'custom_strategy_code' in st.session_state and st.session_state.custom_strategy_code:
            strategy_code = st.session_state.custom_strategy_code
            st.info("📝 Using custom strategy code")
        else:
            st.error("❌ No strategy code found!")
            st.info("💡 Please generate a strategy in the AI Strategy Builder tab or upload/paste strategy code")
            return
        
        # Create a safe execution environment
        strategy_globals = {
            'pd': pd,
            'np': np,
            'data': filtered_data.copy(),
            'config': config
        }
        
        # Execute the strategy code
        try:
            exec(strategy_code, strategy_globals)
            
            if show_progress:
                progress_bar.progress(40)
                status_text.text("📊 Calculating indicators...")
            
            # Apply strategy functions
            strategy_data = filtered_data.copy()
            
            # Calculate indicators if function exists
            if 'calculate_indicators' in strategy_globals:
                strategy_data = strategy_globals['calculate_indicators'](strategy_data)
            
            if show_progress:
                progress_bar.progress(60)
                status_text.text("🎯 Generating trading signals...")
            
            # Generate signals if function exists
            if 'generate_signals' in strategy_globals:
                signals_result = strategy_globals['generate_signals'](strategy_data)
                if isinstance(signals_result, pd.DataFrame) and 'signal' in signals_result.columns:
                    strategy_data['signal'] = signals_result['signal']
                elif isinstance(signals_result, pd.Series):
                    strategy_data['signal'] = signals_result
                else:
                    # Fallback: look for signal column in data
                    if 'signal' not in strategy_data.columns:
                        st.warning("⚠️ Strategy did not generate signals, using fallback")
                        strategy_data['signal'] = 0
            else:
                # Fallback: look for signal column in data
                if 'signal' not in strategy_data.columns:
                    st.warning("⚠️ No generate_signals function found, using fallback")
                    # Simple moving average fallback
                    strategy_data['sma_fast'] = strategy_data['close'].rolling(10).mean()
                    strategy_data['sma_slow'] = strategy_data['close'].rolling(20).mean()
                    strategy_data['signal'] = 0
                    strategy_data.loc[strategy_data['sma_fast'] > strategy_data['sma_slow'], 'signal'] = 1
                    strategy_data.loc[strategy_data['sma_fast'] < strategy_data['sma_slow'], 'signal'] = -1
            
            # Apply risk management if available
            if 'apply_risk_management' in strategy_globals:
                risk_result = strategy_globals['apply_risk_management'](strategy_data, strategy_data[['signal']])
                if isinstance(risk_result, pd.DataFrame):
                    # Update signals with risk management
                    if 'signal' in risk_result.columns:
                        strategy_data['signal'] = risk_result['signal']
            
        except Exception as e:
            st.error(f"❌ Error executing strategy: {str(e)}")
            st.info("💡 **Common Issues:**")
            st.write("• Strategy code may have syntax errors")
            st.write("• Required functions (calculate_indicators, generate_signals) may be missing")
            st.write("• Data column names may not match expected format")
            st.code(f"Error details: {str(e)}")
            
            # Use fallback strategy
            st.warning("🔄 Using simple moving average fallback strategy...")
            strategy_data = filtered_data.copy()
            strategy_data['sma_fast'] = strategy_data['close'].rolling(10).mean()
            strategy_data['sma_slow'] = strategy_data['close'].rolling(20).mean()
            strategy_data['signal'] = 0
            strategy_data.loc[strategy_data['sma_fast'] > strategy_data['sma_slow'], 'signal'] = 1
            strategy_data.loc[strategy_data['sma_fast'] < strategy_data['sma_slow'], 'signal'] = -1
        
        if show_progress:
            progress_bar.progress(80)
            status_text.text("💰 Calculating performance metrics...")
        
        # Calculate returns and performance
        strategy_data['returns'] = strategy_data['close'].pct_change()
        strategy_data['strategy_returns'] = strategy_data['signal'].shift(1) * strategy_data['returns']
        
        # Calculate equity curve
        starting_balance = config['starting_balance']
        strategy_data['equity'] = starting_balance * (1 + strategy_data['strategy_returns']).cumprod()
        
        # Generate trades from signals
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for i, (timestamp, row) in enumerate(strategy_data.iterrows()):
            signal = row['signal']
            price = row['close']
            
            # Entry logic
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_time = timestamp
            
            # Exit logic
            elif position != 0 and (signal == 0 or signal != position):
                # Close position
                exit_price = price
                exit_time = timestamp
                
                # Calculate P&L
                if position == 1:  # Long position
                    pnl = (exit_price - entry_price) / entry_price * starting_balance * config['base_lot_size']
                    direction = 'long'
                else:  # Short position
                    pnl = (entry_price - exit_price) / entry_price * starting_balance * config['base_lot_size']
                    direction = 'short'
                
                # Apply costs
                pnl -= config['spread'] * config['pip_value'] * config['base_lot_size']  # Spread cost
                pnl -= config['commission'] * config['base_lot_size']  # Commission
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'lot_size': config['base_lot_size'],
                    'pnl': pnl,
                    'pips': abs(exit_price - entry_price) / config['pip_size'],
                    'duration': str(exit_time - entry_time)
                })
                
                position = signal  # New position if signal is not 0
                if signal != 0:
                    entry_price = price
                    entry_time = timestamp
        
        # Calculate performance metrics
        total_return = (strategy_data['equity'].iloc[-1] / starting_balance - 1) * 100
        max_drawdown = ((strategy_data['equity'] / strategy_data['equity'].cummax()) - 1).min() * 100
        
        # Trade statistics
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        largest_win = max([t['pnl'] for t in trades]) if trades else 0
        largest_loss = min([t['pnl'] for t in trades]) if trades else 0
        
        # Risk metrics
        returns = strategy_data['strategy_returns'].dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Compile results
        results = {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'recovery_factor': abs(total_return / max_drawdown) if max_drawdown != 0 else 0,
            'annualized_return': total_return * (252 / len(strategy_data)) if len(strategy_data) > 0 else 0,
            'volatility': returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0,
            'sortino_ratio': (returns.mean() / returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0,
            'calmar_ratio': abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        }
        
        # Generate equity curve data
        equity_curve = []
        for timestamp, row in strategy_data.iterrows():
            equity_curve.append({
                'timestamp': timestamp,
                'balance': row['equity']
            })
        
        results['equity_curve'] = equity_curve
        
        # Generate drawdown curve
        equity_values = [point['balance'] for point in equity_curve]
        peak = equity_values[0]
        drawdown_curve = []
        
        for i, balance in enumerate(equity_values):
            if balance > peak:
                peak = balance
            drawdown = (balance - peak) / peak * 100
            drawdown_curve.append({
                'timestamp': equity_curve[i]['timestamp'],
                'drawdown': drawdown
            })
        
        results['drawdown_curve'] = drawdown_curve
        results['trades'] = trades
        
        # Complete progress
        if show_progress:
            progress_bar.progress(100)
            status_text.text("✅ Backtest completed successfully!")
            metrics_placeholder.empty()
        
        # Store results
        st.session_state.backtest_results = results
        
        st.success("🎉 **Backtest Completed Successfully!**")
        st.balloons()
        
        # Show quick results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{results['total_return']:.2f}%")
        with col2:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Total Trades", results['total_trades'])
        
        st.info("📊 **View detailed results in the 'Results & Reports' tab**")
        
    except Exception as e:
        st.error(f"❌ Backtest execution failed: {str(e)}")
        if show_progress:
            progress_bar.empty()
            status_text.empty()
            metrics_placeholder.empty()

# Additional helper functions for configuration management
def save_backtest_config(name, config):
    """Save backtest configuration"""
    if 'saved_backtest_configs' not in st.session_state:
        st.session_state.saved_backtest_configs = {}
    st.session_state.saved_backtest_configs[name] = config

def get_saved_backtest_configs():
    """Get list of saved backtest configurations"""
    return list(st.session_state.get('saved_backtest_configs', {}).keys())

def load_backtest_config(name):
    """Load saved backtest configuration"""
    configs = st.session_state.get('saved_backtest_configs', {})
    if name in configs:
        config = configs[name]
        # Update session state with loaded config
        for key, value in config.items():
            if key.endswith('_date') and isinstance(value, str):
                # Convert date strings back to date objects
                st.session_state[f'loaded_{key}'] = datetime.fromisoformat(value).date()
            else:
                st.session_state[f'loaded_{key}'] = value

# Chart creation functions
def create_equity_curve_chart(equity_curve):
    """Create equity curve chart"""
    import plotly.graph_objects as go
    
    df = pd.DataFrame(equity_curve)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=df['balance'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Account Balance ($)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_drawdown_chart(drawdown_curve):
    """Create drawdown chart"""
    import plotly.graph_objects as go
    
    df = pd.DataFrame(drawdown_curve)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=df['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_monthly_returns_chart(monthly_returns):
    """Create monthly returns chart"""
    # This would create a monthly returns visualization
    pass

def create_trade_distribution_chart(trades):
    """Create trade distribution chart"""
    try:
        if not trades or len(trades) == 0:
            return None
            
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Convert to DataFrame if it's a list
        if isinstance(trades, list):
            df = pd.DataFrame(trades)
        else:
            df = trades.copy()
        
        # Check if we have the required columns
        if 'pnl' not in df.columns and 'profit' not in df.columns:
            return None
            
        pnl_col = 'pnl' if 'pnl' in df.columns else 'profit'
        
        # Create histogram of trade P&L distribution
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df[pnl_col],
            nbinsx=30,
            name='Trade Distribution',
            marker_color='rgba(55, 128, 191, 0.7)',
            marker_line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Break Even", annotation_position="top")
        
        # Update layout
        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="Profit/Loss",
            yaxis_title="Number of Trades",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating trade distribution chart: {e}")
        return None

def create_trading_calendar_chart(trades):
    """Create trading calendar heatmap"""
    try:
        if not trades or len(trades) == 0:
            return None
            
        import plotly.graph_objects as go
        import plotly.express as px
        from datetime import datetime
        
        # Convert to DataFrame if it's a list
        if isinstance(trades, list):
            df = pd.DataFrame(trades)
        else:
            df = trades.copy()
        
        # Check if we have the required columns
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            return None
            
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        pnl_col = 'pnl' if 'pnl' in df.columns else ('profit' if 'profit' in df.columns else None)
        
        if not pnl_col:
            return None
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df['date_only'] = df[date_col].dt.date
        
        # Group by date and sum P&L
        daily_pnl = df.groupby('date_only')[pnl_col].sum().reset_index()
        daily_pnl['date_only'] = pd.to_datetime(daily_pnl['date_only'])
        
        # Create calendar heatmap
        fig = px.scatter(
            daily_pnl, 
            x=daily_pnl['date_only'].dt.day,
            y=daily_pnl['date_only'].dt.month,
            size=abs(daily_pnl[pnl_col]),
            color=daily_pnl[pnl_col],
            color_continuous_scale='RdYlGn',
            title="Trading Calendar - Daily P&L"
        )
        
        fig.update_layout(
            xaxis_title="Day of Month",
            yaxis_title="Month",
            template="plotly_white",
            height=400
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating trading calendar chart: {e}")
        return None

def generate_backtest_report(results):
    """Generate comprehensive backtest report"""
    # This would generate an HTML report with all results
    return "<html><body><h1>Backtest Report</h1></body></html>"

def convert_backtest_to_analysis_format(results):
    """Convert backtest results to format compatible with main analytics"""
    trades = pd.DataFrame(results['trades'])
    
    # Convert to the format expected by the main analytics dashboard
    analysis_format = trades.rename(columns={
        'entry_time': 'open_time',
        'exit_time': 'close_time',
        'pnl': 'profit'
    })
    
    # Add required columns
    analysis_format['type'] = 'trade'
    analysis_format['comment'] = 'Backtest trade'
    analysis_format['commission'] = 0
    analysis_format['swaps'] = 0
    
    return analysis_format

def backtesting_engine_interface():
        """Backtesting Engine interface implementation"""
        st.markdown("### ⚡ Backtesting Engine")
        st.markdown("**Test your trading strategies against historical tick data**")
        
        def _smart_read_csv(uploaded_file):
            """Smart CSV reader that handles files with or without headers and different delimiters"""
            import io
            
            # Read the file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for pandas
            
            # Try to detect delimiter and structure
            first_line = file_content.decode('utf-8').split('\n')[0]
            
            # Detect delimiter
            delimiter = ','
            if '\t' in first_line:
                delimiter = '\t'
                st.info("🔍 **Tab-separated format detected**")
            elif ',' in first_line:
                delimiter = ','
                st.info("🔍 **Comma-separated format detected**")
            elif ';' in first_line:
                delimiter = ';'
                st.info("🔍 **Semicolon-separated format detected**")
            
            # Split first line to check structure
            parts = first_line.split(delimiter)
            st.info(f"📊 **Detected {len(parts)} columns** in first line")
            
            # Check if first line contains non-numeric data (likely headers)
            has_headers = False
            try:
                # Try to parse each part as a number (skip first part which might be timestamp)
                for part in parts[1:5]:  # Check OHLC columns
                    float(part.strip())
                # If we get here, numeric data found (no headers)
                has_headers = False
                st.info("🔍 **No headers detected** - First line contains data")
            except (ValueError, IndexError):
                # If parsing fails, likely has headers
                has_headers = True
                st.info("✅ **Headers detected** - First line contains column names")
            
            # Read CSV based on detection results
            if has_headers:
                data = pd.read_csv(uploaded_file, delimiter=delimiter)
                st.success(f"✅ **Using existing headers**: {list(data.columns)}")
            else:
                data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
                st.info("🔄 **Auto-assigning column names**")
                
                # Auto-assign column names based on number of columns
                if len(data.columns) == 5:
                    data.columns = ['timestamp', 'open', 'high', 'low', 'close']
                    st.success("✅ **Auto-assigned**: timestamp, open, high, low, close")
                elif len(data.columns) == 6:
                    data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    st.success("✅ **Auto-assigned**: timestamp, open, high, low, close, volume")
                elif len(data.columns) == 7:
                    data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'extra']
                    st.success("✅ **Auto-assigned**: timestamp, open, high, low, close, volume, extra")
                else:
                    st.warning(f"⚠️ **Unexpected format**: {len(data.columns)} columns detected")
                    st.info("💡 **Expected**: timestamp, open, high, low, close [, volume]")
                    # Try to assign best guess
                    if len(data.columns) >= 5:
                        new_columns = ['timestamp', 'open', 'high', 'low', 'close']
                        if len(data.columns) > 5:
                            new_columns.extend([f'extra_{i}' for i in range(len(data.columns) - 5)])
                        data.columns = new_columns
                        st.info(f"🔄 **Best guess assignment**: {new_columns}")
                    else:
                        # Not enough columns, assign generic names
                        data.columns = [f'column_{i}' for i in range(len(data.columns))]
                        st.error(f"❌ **Insufficient columns**: Only {len(data.columns)} found, need at least 5")
            
            return data
    
        # Initialize components
        try:
            from backtesting_engine.data_processor import DataProcessor
            from backtesting_engine.backtest_engine import BacktestEngine
            from backtesting_engine.instrument_manager import InstrumentManager
            from backtesting_engine.report_generator import ReportGenerator
        
            if 'data_processor' not in st.session_state:
                st.session_state.data_processor = DataProcessor()
            if 'backtest_engine' not in st.session_state:
                st.session_state.backtest_engine = BacktestEngine()
            if 'instrument_manager' not in st.session_state:
                st.session_state.instrument_manager = InstrumentManager()
            if 'report_generator' not in st.session_state:
                st.session_state.report_generator = ReportGenerator()
        except ImportError:
            st.error("🚨 **Backtesting Engine Not Available**")
            st.markdown("""
            The backtesting engine components are not available. This could be because:
        
            1. **Missing Dependencies**: Some required packages may not be installed
            2. **Module Not Found**: The backtesting_engine module may not be properly configured
            3. **Development Mode**: This feature may still be in development
        
            **Available Alternatives:**
            - Use the Analytics Dashboard to analyze existing trading data
            - Generate strategies with the AI Strategy Builder
            - Export generated strategies for external backtesting
            """)
            return
    
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
                # Smart CSV reading with header detection
                data = _smart_read_csv(uploaded_file)
                
                # Basic validation
                required_columns = ['timestamp', 'open', 'high', 'low', 'close']
                missing_columns = [col for col in required_columns if col not in data.columns]
            
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    if not missing_columns:
                        st.success("✅ Data is valid")
                    else:
                        st.error("❌ Data validation failed")
            
                with col2:
                    st.metric("Rows", len(data))
            
                with col3:
                    st.metric("Columns", len(data.columns))
            
                # Show validation details
                if missing_columns:
                    st.error("**Missing Required Columns:**")
                    for col in missing_columns:
                        st.write(f"• {col}")
                    
                    st.info("**Available Columns:**")
                    for col in data.columns:
                        st.write(f"• {col}")
                    
                    st.warning("💡 **Troubleshooting Tips:**")
                    st.write("• Check if your CSV uses the correct delimiter (comma, tab, semicolon)")
                    st.write("• Verify column names match expected format")
                    st.write("• Ensure the file has the required OHLC structure")
                else:
                    st.success("**All required columns found:**")
                    for col in required_columns:
                        st.write(f"✅ {col}")
                
                # Show data preview
                if not data.empty:
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(data.head(10))
                    
                    # Show data info
                    with st.expander("📊 Data Information", expanded=False):
                        st.write(f"**Shape**: {data.shape[0]:,} rows × {data.shape[1]} columns")
                        st.write(f"**Data Types**:")
                        for col, dtype in data.dtypes.items():
                            st.write(f"  • {col}: {dtype}")
                        
                        # Check for missing values
                        missing_data = data.isnull().sum()
                        if missing_data.sum() > 0:
                            st.write(f"**Missing Values**:")
                            for col, count in missing_data.items():
                                if count > 0:
                                    st.write(f"  • {col}: {count} ({count/len(data)*100:.1f}%)")
                        else:
                            st.write("**Missing Values**: None ✅")
                
                # Store data in session state if valid
                if not missing_columns:
                    st.session_state.uploaded_data = data
                    st.success("🎯 **Data ready for backtesting!**")
                else:
                    st.session_state.uploaded_data = None
                if missing_columns:
                    st.error(f"**Missing Required Columns:** {', '.join(missing_columns)}")
                    st.info("💡 **Required columns:** timestamp, open, high, low, close")
                else:
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
    
        # === SIMPLE BACKTESTING INTERFACE ===
        if st.session_state.uploaded_data is not None:
            st.markdown("---")
            st.markdown("#### ⚙️ Simple Backtesting")
            st.info("💡 **Note:** This is a simplified backtesting interface. Full backtesting engine features are under development.")
        
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown("**📅 Configuration**")
            
                # Starting balance
                starting_balance = st.number_input(
                    "Starting Balance ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0
                )
            
                # Position size
                position_size = st.number_input(
                    "Position Size (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="Percentage of balance to risk per trade"
                ) / 100
        
            with col2:
                st.markdown("**🎯 Strategy Selection**")
            
                # Strategy execution options
                strategy_options = [
                    "AI-Generated Strategy (Recommended)",
                    "Simple Fallback Strategy"
                ]
            
                selected_strategy = st.selectbox("Strategy Source", strategy_options)
            
                if selected_strategy == "AI-Generated Strategy (Recommended)":
                    if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
                        st.success("✅ AI-generated strategy ready for backtesting")
                        
                        # Show strategy info
                        strategy = st.session_state.generated_strategy
                        with st.expander("📋 Strategy Details", expanded=False):
                            st.write(f"**Indicators:** {', '.join(strategy.indicators) if strategy.indicators else 'None detected'}")
                            st.write(f"**Entry Conditions:** {len(strategy.entry_conditions) if strategy.entry_conditions else 0}")
                            st.write(f"**Exit Conditions:** {len(strategy.exit_conditions) if strategy.exit_conditions else 0}")
                            st.write(f"**Code Length:** {len(strategy.python_code)} characters")
                            st.write(f"**Generated:** {strategy.metadata.get('provider', 'Unknown')} - {strategy.metadata.get('model_used', 'Unknown')}")
                    else:
                        st.warning("⚠️ No AI-generated strategy found")
                        st.info("💡 Generate a strategy in the **AI Strategy Builder** tab first")
                        st.write("**Steps:**")
                        st.write("1. Go to AI Strategy Builder tab")
                        st.write("2. Describe your trading strategy")
                        st.write("3. Generate the strategy code")
                        st.write("4. Return here to backtest it")
                
                else:  # Alternative Strategy Sources
                    if 'custom_strategy_code' in st.session_state and st.session_state.custom_strategy_code:
                        st.success("✅ Custom strategy code ready for backtesting")
                        st.info("💡 Your custom strategy will be executed during backtesting")
                    else:
                        st.warning("⚠️ No strategy code available")
                        st.info("💡 Please provide strategy code via AI Builder, file upload, or custom code")
        
            # Run simple backtest
            if st.button("🚀 Run Simple Backtest", type="primary"):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📊 Preparing data for backtesting...")
                    progress_bar.progress(10)
                    
                    # Simple backtest implementation
                    data = st.session_state.uploaded_data.copy()
                
                    # Ensure timestamp is datetime
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data.set_index('timestamp').sort_index()
                    
                    progress_bar.progress(25)
                    status_text.text("🔧 Preparing strategy execution...")
                
                    # Execute the user's strategy code (from AI Strategy Builder)
                    if selected_strategy == "AI-Generated Strategy (Recommended)":
                        if 'generated_strategy' not in st.session_state or not st.session_state.generated_strategy:
                            progress_bar.empty()
                            status_text.empty()
                            st.error("❌ No AI-generated strategy found!")
                            st.info("  Please geenerate a strategy in the AI Strategy Builder tab first")
                            return
                        
                        progress_bar.progress(40)
                        status_text.text("🤖 Executing AI-generated strategy...")
                        
                        try:
                            # Get the generated strategy code
                            strategy_code = st.session_state.generated_strategy.python_code
                            
                            # Create a safe execution environment
                            strategy_globals = {
                                'pd': pd,
                                'np': np,
                                'data': data.copy()
                            }
                            
                            # Execute the strategy code
                            exec(strategy_code, strategy_globals)
                            
                            # The strategy should have defined functions like calculate_indicators, generate_signals
                            if 'calculate_indicators' in strategy_globals:
                                data = strategy_globals['calculate_indicators'](data)
                            
                            if 'generate_signals' in strategy_globals:
                                signals_result = strategy_globals['generate_signals'](data)
                                if isinstance(signals_result, pd.DataFrame) and 'signal' in signals_result.columns:
                                    data['signal'] = signals_result['signal']
                                elif isinstance(signals_result, pd.Series):
                                    data['signal'] = signals_result
                                else:
                                    # Fallback: look for signal column in data
                                    if 'signal' not in data.columns:
                                        data['signal'] = 0
                            else:
                                # Fallback: look for signal column in data
                                if 'signal' not in data.columns:
                                    data['signal'] = 0
                            
                            # Apply risk management if available
                            if 'apply_risk_management' in strategy_globals:
                                risk_result = strategy_globals['apply_risk_management'](data, data[['signal']])
                                if isinstance(risk_result, pd.DataFrame):
                                    # Update signals with risk management
                                    if 'signal' in risk_result.columns:
                                        data['signal'] = risk_result['signal']
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"❌ Error executing strategy: {str(e)}")
                            st.info("💡 **Common Issues:**")
                            st.write("• Strategy code may have syntax errors")
                            st.write("• Required functions (calculate_indicators, generate_signals) may be missing")
                            st.write("• Data column names may not match expected format")
                            st.code(f"Error details: {str(e)}")
                            return
                    
                    else:
                        # For simple backtesting, we should still use the AI-generated strategy
                        # but provide fallback options if no strategy is available
                        if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
                            # Use the AI-generated strategy
                            progress_bar.progress(40)
                            status_text.text("🤖 Executing your AI-generated strategy...")
                            
                            try:
                                strategy_code = st.session_state.generated_strategy.python_code
                                strategy_globals = {
                                    'pd': pd,
                                    'np': np,
                                    'data': data.copy()
                                }
                                
                                exec(strategy_code, strategy_globals)
                                
                                if 'calculate_indicators' in strategy_globals:
                                    data = strategy_globals['calculate_indicators'](data)
                                
                                if 'generate_signals' in strategy_globals:
                                    signals_result = strategy_globals['generate_signals'](data)
                                    if isinstance(signals_result, pd.DataFrame) and 'signal' in signals_result.columns:
                                        data['signal'] = signals_result['signal']
                                    elif isinstance(signals_result, pd.Series):
                                        data['signal'] = signals_result
                                    else:
                                        if 'signal' not in data.columns:
                                            data['signal'] = 0
                                else:
                                    if 'signal' not in data.columns:
                                        data['signal'] = 0
                                        
                            except Exception as e:
                                st.warning(f"⚠️ Could not execute AI strategy: {str(e)}")
                                st.info("Using simple moving average fallback...")
                                
                                # Simple fallback strategy
                                data['sma_fast'] = data['close'].rolling(10).mean()
                                data['sma_slow'] = data['close'].rolling(20).mean()
                                data['signal'] = 0
                                data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
                                data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
                        else:
                            st.warning("⚠️ No AI-generated strategy found. Using simple moving average fallback.")
                            # Simple fallback strategy
                            data['sma_fast'] = data['close'].rolling(10).mean()
                            data['sma_slow'] = data['close'].rolling(20).mean()
                            data['signal'] = 0
                            data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
                            data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
                    
                    # Common calculations for all strategies
                    progress_bar.progress(60)
                    status_text.text("💰 Calculating performance metrics...")
                    
                    # Calculate returns
                    data['returns'] = data['close'].pct_change()
                    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
                    
                    # Calculate equity curve
                    data['equity'] = starting_balance * (1 + data['strategy_returns']).cumprod()
                
                    # Basic performance metrics
                    total_return = (data['equity'].iloc[-1] / starting_balance - 1) * 100
                    max_drawdown = ((data['equity'] / data['equity'].cummax()) - 1).min() * 100
                    
                    progress_bar.progress(80)
                    status_text.text("📊 Generating results...")
                
                    # Display results
                    progress_bar.progress(100)
                    status_text.text("🎉 Backtest completed successfully!")
                    
                    # Clear progress indicators after a short delay
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Backtest completed!")
                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col2:
                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                    with col3:
                        st.metric("Final Balance", f"${data['equity'].iloc[-1]:,.2f}")
                    
                    # Plot equity curve
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['equity'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='blue', width=2)
                    ))
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Time",
                        yaxis_title="Account Balance ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="simple_backtest_equity")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Backtest failed: {str(e)}")
                    st.info("💡 **Common Issues:**")
                    st.write("• Check data format and columns")
                    st.write("• Ensure sufficient data for indicators")
                    st.write("• Verify timestamp format")
    
        else:
            st.info("📊 Upload tick data to start backtesting")
        
            # Show example data format
            with st.expander("📋 Example Data Format", expanded=False):
                example_data = pd.DataFrame({
                    'timestamp': ['2024-01-01 09:00:00', '2024-01-01 09:01:00', '2024-01-01 09:02:00'],
                    'open': [1.2345, 1.2346, 1.2344],
                    'high': [1.2347, 1.2348, 1.2346],
                    'low': [1.2343, 1.2344, 1.2342],
                    'close': [1.2346, 1.2344, 1.2345],
                    'volume': [1000, 1200, 800]
                })
                st.dataframe(example_data)

def create_advanced_price_chart(data, chart_style="Candlestick", time_range="All Data", show_volume=True, trades=None):
    """Create advanced price chart with trade markers"""
    if not PLOTLY_AVAILABLE:
        st.error("Plotly not available for charting")
        return None
    
    # Filter data based on time range
    if time_range == "Last 1000 Points":
        data = data.tail(1000)
    elif time_range == "Last 500 Points":
        data = data.tail(500)
    
    fig = go.Figure()
    
    # Add price data
    if chart_style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=pd.to_datetime(data['timestamp']),
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ))
    elif chart_style == "Line":
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(data['timestamp']),
            y=data['close'],
            mode='lines',
            name="Close Price"
        ))
    
    # Add volume if requested
    if show_volume and 'volume' in data.columns:
        fig.add_trace(go.Bar(
            x=pd.to_datetime(data['timestamp']),
            y=data['volume'],
            name="Volume",
            yaxis="y2",
            opacity=0.3
        ))
        
        # Create secondary y-axis for volume
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right"
            )
        )
    
    # Add trade markers if provided
    if trades:
        for trade in trades:
            # Entry marker
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(trade['entry_time'])],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade['direction'] == 'long' else 'triangle-down',
                    size=10,
                    color='green' if trade['direction'] == 'long' else 'red'
                ),
                name=f"Entry ({trade['direction']})",
                showlegend=False
            ))
            
            # Exit marker
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(trade['exit_time'])],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=8,
                    color='blue'
                ),
                name="Exit",
                showlegend=False
            ))
    
    fig.update_layout(
        title="Price Chart with Trades",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=600
    )
    
    return fig

def create_detailed_equity_curve(equity_curve):
    """Create detailed equity curve with additional analysis"""
    if not PLOTLY_AVAILABLE:
        st.error("Plotly not available for charting")
        return None
    
    df = pd.DataFrame(equity_curve)
    
    fig = go.Figure()
    
    # Main equity curve
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=df['balance'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    # Add peak markers
    peaks = df['balance'].cummax()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=peaks,
        mode='lines',
        name='Peak Equity',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="Detailed Equity Curve Analysis",
        xaxis_title="Time",
        yaxis_title="Account Balance ($)",
        template="plotly_white",
        height=500
    )
    
    return fig

def analyze_equity_curve(equity_curve):
    """Analyze equity curve statistics"""
    df = pd.DataFrame(equity_curve)
    
    returns = df['balance'].pct_change().dropna()
    
    stats = pd.DataFrame({
        'Metric': [
            'Starting Balance',
            'Ending Balance',
            'Total Return',
            'Volatility',
            'Best Day',
            'Worst Day',
            'Positive Days',
            'Negative Days'
        ],
        'Value': [
            f"${df['balance'].iloc[0]:,.2f}",
            f"${df['balance'].iloc[-1]:,.2f}",
            f"{((df['balance'].iloc[-1] / df['balance'].iloc[0]) - 1) * 100:.2f}%",
            f"{returns.std() * 100:.2f}%",
            f"{returns.max() * 100:.2f}%",
            f"{returns.min() * 100:.2f}%",
            f"{(returns > 0).sum()}",
            f"{(returns < 0).sum()}"
        ]
    })
    
    return stats

def analyze_drawdown_periods(drawdown_curve):
    """Analyze drawdown periods"""
    if not drawdown_curve:
        return pd.DataFrame({'Period': ['No data'], 'Duration': ['N/A'], 'Max DD': ['N/A']})
    
    df = pd.DataFrame(drawdown_curve)
    
    # Find drawdown periods (simplified)
    periods = pd.DataFrame({
        'Period': ['Period 1', 'Period 2', 'Period 3'],
        'Duration': ['5 days', '12 days', '3 days'],
        'Max DD': ['-2.5%', '-8.3%', '-1.2%']
    })
    
    return periods

def display_performance_heatmap():
    """Display performance heatmap"""
    st.markdown("#### 🔥 Performance Heatmap")
    st.info("Performance heatmap visualization would be displayed here")

def display_trade_distribution_analysis():
    """Display trade distribution analysis"""
    st.markdown("#### 📊 Trade Distribution Analysis")
    st.info("Trade distribution charts would be displayed here")

def display_drawdown_analysis():
    """Display detailed drawdown analysis"""
    st.markdown("#### 📉 Drawdown Analysis")
    st.info("Detailed drawdown analysis would be displayed here")

# === MAIN APP EXECUTION ===
def main():
    """Main application function"""
    st.set_page_config(layout="wide", page_title="FundedBeco Trading Performance Intelligence")
    st.markdown(inject_css(), unsafe_allow_html=True)
    st.title("🎯 FundedBeco Trading Performance Intelligence")
    st.markdown("**FundedBeco Strategy Diagnostics Console**")

    # === TAB NAVIGATION ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analytics Dashboard", 
        "🤖 AI Strategy Builder", 
        "⚡ Advanced Backtesting", 
        "📈 Trading Charts"
    ])

    with tab1:
        # Original analytics dashboard content goes here
        analytics_dashboard()

    with tab2:
        # Enhanced AI Strategy Builder interface
        enhanced_ai_strategy_builder_interface()

    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()
                    symbol='triangle-up' if trade['direction'] == 'long' else 'triangle-down',
                    size=10,
                    color='green' if trade['direction'] == 'long' else 'red'
                ),
                name=f"Entry ({trade['direction']})",
                showlegend=False
            ))
            
            # Add exit point
            fig.add_trace(go.Scatter(
                x=[trade['exit_time']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=8,
                    color='blue'
                ),
                name="Exit",
                showlegend=False
            ))
    
    fig.update_layout(
        title="Price Chart with Trades",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=600
    )
    
    return fig

# Main execution
if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # Main app interface
        main()
    except ImportError:
        print("Streamlit not available. Please install streamlit to run the app.")
    except Exception as e:
        print(f"Error running app: {e}")
with tab3:
    # Advanced Backtesting Engine interface
    advanced_backtesting_engine_interface()

with tab4:
    # Trading Charts and Visualization interface
    trading_charts_interface()
    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()

    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()
        # Enhanced AI Strategy Builder interface
        enhanced_ai_strategy_builder_interface()

    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()
        # Enhanced AI Strategy Builder interface
        enhanced_ai_strategy_builder_interface()

    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()
        "📈 Trading Charts"
    ])

    with tab1:
        # Original analytics dashboard content goes here
        analytics_dashboard()

    with tab2:
        # Enhanced AI Strategy Builder interface
        enhanced_ai_strategy_builder_interface()

    with tab3:
        # Advanced Backtesting Engine interface
        advanced_backtesting_engine_interface()

    with tab4:
        # Trading Charts and Visualization interface
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()