#!/usr/bin/env python3
"""
Clean rebuild of the Trade Analyzer Pro application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not available. Please install: pip install plotly")

# Import custom modules
try:
    from ai_strategy_builder.strategy_prompt_processor import StrategyPromptProcessor
    from ai_strategy_builder.pine_script_converter import PineScriptConverter
    from ai_strategy_builder.puter_client import PuterClient
    from ai_strategy_builder.openrouter_client import OpenRouterClient
    from balance_detector import BalanceDetector
    from config import Config
except ImportError as e:
    st.error(f"Import error: {e}")

# === CORE FUNCTIONS ===

def inject_css():
    """Inject custom CSS for styling"""
    return """
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """

def analytics_dashboard():
    """Main analytics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    st.markdown("**Upload your trading data for comprehensive analysis**")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Trading Data",
        type=['csv', 'xlsx'],
        help="Upload your broker statement or trading data"
    )
    
    if uploaded_file:
        try:
            # Read the data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(data)} records")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", len(data))
            with col2:
                if 'profit' in data.columns:
                    total_profit = data['profit'].sum()
                    st.metric("Total P&L", f"${total_profit:.2f}")
            with col3:
                if 'profit' in data.columns:
                    win_rate = (data['profit'] > 0).mean() * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(data.head(10))
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("👆 Upload your trading data to get started")

def enhanced_ai_strategy_builder_interface():
    """Enhanced AI Strategy Builder interface"""
    st.markdown("### 🤖 AI Strategy Builder")
    st.markdown("**Generate trading strategies using natural language**")
    
    # Strategy Builder Tabs
    builder_tab1, builder_tab2, builder_tab3, builder_tab4 = st.tabs([
        "💬 Strategy Generator", 
        "📊 Indicators", 
        "⚙️ Risk Management", 
        "📜 Pine Script Export"
    ])
    
    with builder_tab1:
        strategy_generator_interface()
    
    with builder_tab2:
        indicators_builder_interface()
    
    with builder_tab3:
        risk_management_interface()
    
    with builder_tab4:
        pine_script_export_interface()

def strategy_generator_interface():
    """Strategy generator interface"""
    st.markdown("#### 💬 Strategy Generator")
    
    # Strategy description input
    strategy_description = st.text_area(
        "Describe your trading strategy:",
        height=150,
        placeholder="Example: Create a momentum strategy that buys when RSI is oversold and price breaks above 20-day moving average...",
        help="Describe your strategy in natural language. Be specific about entry/exit conditions."
    )
    
    # AI Provider selection
    col1, col2 = st.columns(2)
    with col1:
        ai_provider = st.selectbox(
            "AI Provider",
            ["Puter AI (Free)", "OpenRouter (Free)", "OpenAI (API Key Required)"],
            help="Choose your AI provider"
        )
    
    with col2:
        if ai_provider == "OpenAI (API Key Required)":
            api_key = st.text_input("OpenAI API Key", type="password")
    
    # Generate strategy button
    if st.button("🚀 Generate Strategy", type="primary"):
        if not strategy_description.strip():
            st.warning("Please describe your strategy first")
            return
        
        with st.spinner("🤖 Generating strategy..."):
            try:
                # Initialize strategy processor
                processor = StrategyPromptProcessor()
                
                # Generate strategy based on provider
                if ai_provider.startswith("Puter AI"):
                    client = PuterClient()
                    strategy = processor.generate_strategy_with_puter(strategy_description, client)
                elif ai_provider.startswith("OpenRouter"):
                    client = OpenRouterClient()
                    strategy = processor.generate_strategy_with_openrouter(strategy_description, client)
                else:
                    # OpenAI implementation would go here
                    st.error("OpenAI integration not implemented yet")
                    return
                
                if strategy:
                    st.session_state.generated_strategy = strategy
                    st.success("✅ Strategy generated successfully!")
                    
                    # Display strategy
                    display_generated_strategy()
                else:
                    st.error("Failed to generate strategy")
                    
            except Exception as e:
                st.error(f"Error generating strategy: {e}")

def display_generated_strategy():
    """Display the generated strategy"""
    if 'generated_strategy' not in st.session_state:
        return
    
    strategy = st.session_state.generated_strategy
    
    st.markdown("#### 📋 Generated Strategy")
    
    # Strategy details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strategy Information:**")
        st.write(f"**Indicators:** {', '.join(strategy.indicators) if strategy.indicators else 'None detected'}")
        st.write(f"**Entry Conditions:** {len(strategy.entry_conditions) if strategy.entry_conditions else 0}")
        st.write(f"**Exit Conditions:** {len(strategy.exit_conditions) if strategy.exit_conditions else 0}")
    
    with col2:
        st.markdown("**Generation Details:**")
        st.write(f"**Provider:** {strategy.metadata.get('provider', 'Unknown')}")
        st.write(f"**Model:** {strategy.metadata.get('model_used', 'Unknown')}")
        st.write(f"**Generated:** {strategy.metadata.get('timestamp', 'Unknown')}")
    
    # Code display
    with st.expander("📄 Python Code", expanded=True):
        st.code(strategy.python_code, language="python")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Regenerate"):
            if 'generated_strategy' in st.session_state:
                del st.session_state.generated_strategy
            st.rerun()
    
    with col2:
        if st.button("📊 Test Strategy"):
            st.info("Strategy testing functionality coming soon")
    
    with col3:
        if st.button("💾 Save Strategy"):
            st.info("Strategy saving functionality coming soon")

def indicators_builder_interface():
    """Indicators builder interface"""
    st.markdown("#### 📊 Custom Indicators")
    st.info("Custom indicator builder coming soon")

def risk_management_interface():
    """Risk management interface"""
    st.markdown("#### ⚙️ Risk Management")
    st.info("Risk management builder coming soon")

def pine_script_export_interface():
    """Pine Script export interface"""
    st.markdown("#### 📜 Pine Script Export")
    
    if 'generated_strategy' not in st.session_state:
        st.warning("Generate a strategy first to export to Pine Script")
        return
    
    st.info("Pine Script export functionality coming soon")

def advanced_backtesting_engine_interface():
    """Advanced backtesting engine interface"""
    st.markdown("### ⚡ Advanced Backtesting Engine")
    st.markdown("**Test your strategies against historical data**")
    
    # Backtesting Tabs
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
    """Data upload interface for backtesting"""
    st.markdown("#### 📊 Upload Historical Data")
    
    uploaded_file = st.file_uploader(
        "Upload Tick Data (CSV)",
        type=['csv'],
        help="Upload historical tick data for backtesting"
    )
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_tick_data = data
            
            st.success(f"✅ Loaded {len(data)} data points")
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(data.head())
                
        except Exception as e:
            st.error(f"Error loading data: {e}")

def backtest_configuration_interface():
    """Backtest configuration interface"""
    st.markdown("#### ⚙️ Backtest Configuration")
    
    if not st.session_state.get('uploaded_tick_data') is not None:
        st.warning("Upload tick data first")
        return
    
    # Basic configuration
    col1, col2 = st.columns(2)
    
    with col1:
        starting_balance = st.number_input("Starting Balance ($)", value=10000.0, min_value=100.0)
        base_lot_size = st.number_input("Base Lot Size", value=0.1, min_value=0.001, step=0.001)
    
    with col2:
        spread = st.number_input("Spread (pips)", value=1.0, step=0.1)
        commission = st.number_input("Commission per Lot ($)", value=0.0, step=0.1)
    
    # Store configuration
    st.session_state.backtest_config = {
        'starting_balance': starting_balance,
        'base_lot_size': base_lot_size,
        'spread': spread,
        'commission': commission
    }
    
    st.success("✅ Configuration saved")

def run_backtest_interface():
    """Run backtest interface"""
    st.markdown("#### 🚀 Run Backtest")
    
    # Check prerequisites
    if not st.session_state.get('uploaded_tick_data') is not None:
        st.warning("Upload tick data first")
        return
    
    if not st.session_state.get('backtest_config'):
        st.warning("Configure backtest settings first")
        return
    
    if not st.session_state.get('generated_strategy'):
        st.warning("Generate a strategy first")
        return
    
    # Run backtest button
    if st.button("🚀 Start Backtest", type="primary"):
        run_strategy_backtest()

def run_strategy_backtest():
    """Execute the actual backtest"""
    with st.spinner("🚀 Running backtest..."):
        try:
            data = st.session_state.uploaded_tick_data.copy()
            config = st.session_state.backtest_config
            strategy = st.session_state.generated_strategy
            
            # Prepare data
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp').sort_index()
            
            # Execute strategy code
            strategy_globals = {
                'pd': pd,
                'np': np,
                'data': data.copy(),
                'config': config
            }
            
            exec(strategy.python_code, strategy_globals)
            
            # Apply strategy functions
            if 'calculate_indicators' in strategy_globals:
                data = strategy_globals['calculate_indicators'](data)
            
            if 'generate_signals' in strategy_globals:
                signals_result = strategy_globals['generate_signals'](data)
                if isinstance(signals_result, pd.DataFrame) and 'signal' in signals_result.columns:
                    data['signal'] = signals_result['signal']
                elif isinstance(signals_result, pd.Series):
                    data['signal'] = signals_result
                else:
                    data['signal'] = 0
            else:
                data['signal'] = 0
            
            # Calculate performance
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            data['equity'] = config['starting_balance'] * (1 + data['strategy_returns']).cumprod()
            
            # Calculate metrics
            total_return = (data['equity'].iloc[-1] / config['starting_balance'] - 1) * 100
            max_drawdown = ((data['equity'] / data['equity'].cummax()) - 1).min() * 100
            
            # Store results
            results = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'equity_curve': [{'timestamp': idx, 'balance': val} for idx, val in data['equity'].items()],
                'trades': []  # Would generate actual trades from signals
            }
            
            st.session_state.backtest_results = results
            
            st.success("🎉 Backtest completed!")
            
            # Show quick results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with col3:
                st.metric("Final Balance", f"${data['equity'].iloc[-1]:,.2f}")
            
        except Exception as e:
            st.error(f"Backtest failed: {e}")

def backtest_results_interface():
    """Backtest results interface"""
    st.markdown("#### 📈 Backtest Results")
    
    if not st.session_state.get('backtest_results'):
        st.info("Run a backtest to see results here")
        return
    
    results = st.session_state.backtest_results
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Return", f"{results['total_return']:.2f}%")
    with col2:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
    with col3:
        st.metric("Trades", len(results['trades']))
    
    # Equity curve chart
    if PLOTLY_AVAILABLE and results['equity_curve']:
        df = pd.DataFrame(results['equity_curve'])
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
            yaxis_title="Balance ($)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def trading_charts_interface():
    """Trading charts interface"""
    st.markdown("### 📈 Trading Charts & Visualization")
    st.info("Advanced charting functionality coming soon")

# === MAIN APPLICATION ===

def main():
    """Main application function"""
    st.set_page_config(
        layout="wide", 
        page_title="FundedBeco Trading Performance Intelligence"
    )
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
        analytics_dashboard()

    with tab2:
        enhanced_ai_strategy_builder_interface()

    with tab3:
        advanced_backtesting_engine_interface()

    with tab4:
        trading_charts_interface()

# Execute main function when script is run
if __name__ == "__main__":
    main()