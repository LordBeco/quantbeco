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



def ai_strategy_builder_interface():
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
            help="Puter AI provides template-based responses. OpenRouter offers free access to various AI models. OpenAI provides premium AI but requires credits."
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
                st.plotly_chart(kelly_charts['kelly_overview'], use_container_width=True)
        
            col1, col2 = st.columns(2)
        
            if 'position_sizing' in kelly_charts:
                with col1:
                    st.plotly_chart(kelly_charts['position_sizing'], use_container_width=True)
        
            if 'risk_reward' in kelly_charts:
                with col2:
                    st.plotly_chart(kelly_charts['risk_reward'], use_container_width=True)
        
            # Kelly Insights Summary Chart
            insights_chart = create_kelly_insights_summary_chart(kelly_metrics)
            if insights_chart:
                st.plotly_chart(insights_chart, use_container_width=True)
        
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
        st.plotly_chart(rolling_performance_charts(df), use_container_width=True)
    
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
        st.plotly_chart(calendar_chart, use_container_width=True)
    
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
            st.plotly_chart(hourly_chart, use_container_width=True)
        with col2:
            st.plotly_chart(daily_chart, use_container_width=True)
    
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
        st.plotly_chart(monthly_heatmap(time_data), use_container_width=True)
    
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
            st.plotly_chart(dd_chart, use_container_width=True)
        with col2:
            st.plotly_chart(recovery_chart, use_container_width=True)
        with col3:
            st.plotly_chart(consec_chart, use_container_width=True)

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
            st.plotly_chart(insights_charts['lot_analysis'], use_container_width=True)
        
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
            st.plotly_chart(insights_charts['direction_analysis'], use_container_width=True)
        
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
            st.plotly_chart(insights_charts['rr_analysis'], use_container_width=True)
        
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
        
            st.plotly_chart(pip_chart, use_container_width=True)
        
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
            st.plotly_chart(insights_charts['symbol_analysis'], use_container_width=True)
        
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
            st.plotly_chart(insights_charts['position_sizing'], use_container_width=True)

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
            
                # Simple strategy options
                strategy_options = [
                    "Simple Moving Average Crossover",
                    "RSI Overbought/Oversold",
                    "Bollinger Bands Mean Reversion",
                    "Custom Strategy (from AI Builder)"
                ]
            
                selected_strategy = st.selectbox("Strategy", strategy_options)
            
                if selected_strategy == "Custom Strategy (from AI Builder)":
                    if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
                        st.success("✅ Using strategy from AI Builder")
                    else:
                        st.warning("⚠️ No strategy found from AI Builder")
                        st.info("Generate a strategy in the AI Strategy Builder tab first")
        
            # Run simple backtest
            if st.button("🚀 Run Simple Backtest", type="primary"):
                with st.spinner("⚡ Running backtest..."):
                    try:
                        # Simple backtest implementation
                        data = st.session_state.uploaded_data.copy()
                    
                        # Ensure timestamp is datetime
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data = data.set_index('timestamp').sort_index()
                    
                        # Simple moving average strategy example
                        if selected_strategy == "Simple Moving Average Crossover":
                            data['sma_fast'] = data['close'].rolling(10).mean()
                            data['sma_slow'] = data['close'].rolling(20).mean()
                        
                            # Generate signals
                            data['signal'] = 0
                            data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
                            data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
                        
                            # Calculate returns
                            data['returns'] = data['close'].pct_change()
                            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
                        
                            # Calculate equity curve
                            data['equity'] = starting_balance * (1 + data['strategy_returns']).cumprod()
                        
                            # Basic performance metrics
                            total_return = (data['equity'].iloc[-1] / starting_balance - 1) * 100
                            max_drawdown = ((data['equity'] / data['equity'].cummax()) - 1).min() * 100
                        
                            # Display results
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
                            st.plotly_chart(fig, use_container_width=True)
                    
                        else:
                            st.info("💡 Selected strategy not yet implemented in simple backtesting mode")
                            st.write("Available: Simple Moving Average Crossover")
                
                    except Exception as e:
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

# === MAIN APP EXECUTION ===
st.set_page_config(layout="wide", page_title="FundedBeco Trading Performance Intelligence")
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