import pandas as pd
import numpy as np

def compute_metrics(df, pnl):
    wins = df[df[pnl] > 0]
    losses = df[df[pnl] < 0]

    expectancy = (
        wins[pnl].mean() * len(wins)/len(df)
        + losses[pnl].mean() * len(losses)/len(df)
    )

    return {
        "Total Trades": len(df),
        "Win Rate (%)": round(len(wins)/len(df)*100, 2),
        "Avg Win": round(wins[pnl].mean(), 2),
        "Avg Loss": round(losses[pnl].mean(), 2),
        "Expectancy": round(expectancy, 2),
        "Profit Factor": round(wins[pnl].sum() / abs(losses[pnl].sum()), 2),
        "Max Drawdown": round(df["drawdown"].min(), 2),
        "Max Drawdown %": round(df["drawdown_pct"].min(), 2)
    }

def compute_rolling_metrics(df, pnl, window=20):
    """
    Compute comprehensive rolling performance metrics for edge decay detection
    Based on professional risk management principles
    """
    df = df.copy()
    
    # === BASIC ROLLING METRICS ===
    # Rolling expectancy (20-trade window) - THE MOST IMPORTANT METRIC
    df["rolling_expectancy"] = df[pnl].rolling(window, min_periods=1).mean()
    
    # Rolling win rate
    df["rolling_wins"] = (df[pnl] > 0).rolling(window, min_periods=1).sum()
    df["rolling_win_rate"] = (df["rolling_wins"] / window * 100).fillna(0)
    
    # Rolling profit (cumulative over window)
    df["rolling_profit"] = df[pnl].rolling(window, min_periods=1).sum()
    
    # === ADVANCED ROLLING METRICS (PROFESSIONAL LEVEL) ===
    
    # Rolling Risk-Reward Ratio (R:R)
    def rolling_rr_ratio(series, window):
        def calc_rr(window_data):
            wins = window_data[window_data > 0]
            losses = window_data[window_data < 0]
            if len(wins) == 0 or len(losses) == 0:
                return 0
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            return avg_win / avg_loss if avg_loss > 0 else 0
        return series.rolling(window, min_periods=1).apply(calc_rr, raw=True)
    
    df["rolling_rr_ratio"] = rolling_rr_ratio(df[pnl], window)
    
    # Rolling Maximum Adverse Excursion (MAE) - How deep do losses go?
    # Note: This would need intraday data, so we'll use trade-level approximation
    df["rolling_max_loss"] = df[pnl].rolling(window, min_periods=1).min()
    df["rolling_avg_loss"] = df[pnl].where(df[pnl] < 0).rolling(window, min_periods=1).mean()
    
    # Rolling Maximum Favorable Excursion (MFE) - How high do wins go?
    df["rolling_max_win"] = df[pnl].rolling(window, min_periods=1).max()
    df["rolling_avg_win"] = df[pnl].where(df[pnl] > 0).rolling(window, min_periods=1).mean()
    
    # Rolling Profit Factor
    def rolling_profit_factor(series, window):
        def calc_pf(window_data):
            wins = window_data[window_data > 0].sum()
            losses = abs(window_data[window_data < 0].sum())
            return wins / losses if losses > 0 else 0
        return series.rolling(window, min_periods=1).apply(calc_pf, raw=True)
    
    df["rolling_profit_factor"] = rolling_profit_factor(df[pnl], window)
    
    # Rolling Drawdown (from peak within window)
    def rolling_drawdown(series, window):
        def calc_dd(window_data):
            cumsum = window_data.cumsum()
            peak = cumsum.cummax()
            dd = ((cumsum - peak) / peak * 100).min()
            return dd if not pd.isna(dd) else 0
        return series.rolling(window, min_periods=1).apply(calc_dd, raw=False)
    
    df["rolling_drawdown_pct"] = rolling_drawdown(df[pnl], window)
    
    # Rolling Volatility (standard deviation of returns)
    df["rolling_volatility"] = df[pnl].rolling(window, min_periods=1).std()
    
    # Rolling Sharpe-like Ratio (expectancy / volatility)
    df["rolling_sharpe"] = df["rolling_expectancy"] / df["rolling_volatility"].replace(0, 1)
    
    # === EDGE DECAY DETECTION SIGNALS ===
    
    # Expectancy trend (is edge deteriorating?)
    df["expectancy_trend"] = df["rolling_expectancy"].diff(5)  # 5-trade trend
    
    # Win rate vs expectancy divergence (dangerous when win rate high but expectancy falling)
    df["wr_exp_divergence"] = df["rolling_win_rate"] - (df["rolling_expectancy"] * 100)
    
    # Profit concentration risk (is recent profit from few big wins?)
    def rolling_profit_concentration(series, window):
        def calc_concentration(window_data):
            if len(window_data) == 0:
                return 0
            # What % of total profit comes from top 20% of trades?
            sorted_trades = window_data.sort_values(ascending=False)
            top_20_pct = int(len(sorted_trades) * 0.2) or 1
            top_trades_profit = sorted_trades.head(top_20_pct).sum()
            total_profit = window_data.sum()
            return (top_trades_profit / total_profit * 100) if total_profit > 0 else 0
        return series.rolling(window, min_periods=1).apply(calc_concentration, raw=False)
    
    df["rolling_profit_concentration"] = rolling_profit_concentration(df[pnl], window)
    
    # === REGIME DETECTION (TIME-BASED) ===
    if 'close_time_parsed' in df.columns or any('time' in col.lower() for col in df.columns):
        # Find datetime column
        datetime_col = None
        for col in df.columns:
            if 'time' in col.lower() and 'parsed' in col:
                datetime_col = col
                break
        
        if datetime_col:
            df['hour'] = pd.to_datetime(df[datetime_col]).dt.hour
            df['day_of_week'] = pd.to_datetime(df[datetime_col]).dt.day_name()
            
            # Rolling performance by session
            df['session'] = df['hour'].apply(lambda x: 
                'Asian' if 0 <= x < 8 else
                'London' if 8 <= x < 16 else
                'NY' if 16 <= x < 24 else 'Other'
            )
    
    return df

def compute_period_starting_balance(original_df, filtered_df, pnl_col, original_starting_balance):
    """Calculate the correct starting balance for a filtered period"""
    
    if len(filtered_df) == len(original_df):
        # No filtering applied, use original starting balance
        return original_starting_balance
    
    # Find the cumulative PnL up to the start of the filtered period
    if len(filtered_df) == 0:
        return original_starting_balance
    
    # Get the first trade in the filtered period
    first_filtered_index = filtered_df.index[0]
    
    # Calculate cumulative PnL from all trades before this period
    trades_before_period = original_df.loc[:first_filtered_index-1]
    cumulative_pnl_before = trades_before_period[pnl_col].sum() if len(trades_before_period) > 0 else 0
    
    # Starting balance for this period = original balance + PnL accumulated before this period
    period_starting_balance = original_starting_balance + cumulative_pnl_before
    
    return period_starting_balance
def analyze_edge_decay(df, pnl, window=20):
    """
    Professional edge decay analysis with risk manager's interpretation
    No fluff, just survival and scaling insights
    """
    
    analysis = {
        'edge_status': 'UNKNOWN',
        'warnings': [],
        'signals': [],
        'recommendations': [],
        'regime_analysis': {},
        'risk_metrics': {}
    }
    
    if len(df) < window:
        analysis['edge_status'] = 'INSUFFICIENT_DATA'
        analysis['warnings'].append(f"Need at least {window} trades for meaningful analysis")
        return analysis
    
    # Get latest metrics (last 20 trades)
    latest_expectancy = df["rolling_expectancy"].iloc[-1]
    latest_win_rate = df["rolling_win_rate"].iloc[-1]
    latest_rr_ratio = df["rolling_rr_ratio"].iloc[-1] if "rolling_rr_ratio" in df.columns else 0
    latest_profit_factor = df["rolling_profit_factor"].iloc[-1] if "rolling_profit_factor" in df.columns else 0
    latest_concentration = df["rolling_profit_concentration"].iloc[-1] if "rolling_profit_concentration" in df.columns else 0
    
    # Expectancy trend analysis
    expectancy_trend = df["expectancy_trend"].iloc[-5:].mean() if "expectancy_trend" in df.columns else 0
    
    # === EDGE STATUS DETERMINATION ===
    if latest_expectancy <= 0:
        analysis['edge_status'] = 'NO_EDGE'
        analysis['warnings'].append("🚨 CRITICAL: No statistical edge - expectancy ≤ 0")
        analysis['recommendations'].append("STOP TRADING immediately - strategy has no edge")
    elif latest_expectancy > 0 and expectancy_trend < -0.1:
        analysis['edge_status'] = 'EDGE_DECAY'
        analysis['warnings'].append("⚠️ WARNING: Edge is deteriorating - expectancy trending down")
        analysis['recommendations'].append("REDUCE POSITION SIZE by 50% - edge decay detected")
    elif latest_expectancy > 0 and expectancy_trend > 0.1:
        analysis['edge_status'] = 'EDGE_IMPROVING'
        analysis['signals'].append("✅ POSITIVE: Edge is strengthening - expectancy trending up")
    else:
        analysis['edge_status'] = 'EDGE_STABLE'
        analysis['signals'].append("📊 NEUTRAL: Edge appears stable")
    
    # === WIN RATE VS EXPECTANCY DIVERGENCE ===
    if latest_win_rate > 70 and latest_expectancy < 10:
        analysis['warnings'].append("🚨 DANGER: High win rate but low expectancy - winners shrinking or losers expanding")
        analysis['recommendations'].append("This is how strategies die quietly - investigate trade sizing")
    
    # === RISK-REWARD ANALYSIS ===
    if latest_rr_ratio < 1.0 and latest_win_rate < 60:
        analysis['warnings'].append("⚠️ POOR R:R: Risk-reward ratio < 1:1 with low win rate")
        analysis['recommendations'].append("Need either higher win rate (>60%) or better R:R (>1:1)")
    
    # === PROFIT CONCENTRATION RISK ===
    if latest_concentration > 80:
        analysis['warnings'].append("🚨 FRAGILE: >80% of profit from top 20% of trades")
        analysis['recommendations'].append("Strategy is fragile - dependent on few big winners")
    elif latest_concentration > 60:
        analysis['warnings'].append("⚠️ CONCENTRATION: Moderate profit concentration risk")
    
    # === PROFIT FACTOR ANALYSIS ===
    if latest_profit_factor < 1.3:
        analysis['warnings'].append("⚠️ FRAGILE PF: Profit factor < 1.3 - few bad trades will erase gains")
        analysis['recommendations'].append("Profit factor too low for sustainable trading")
    
    # === PROFESSIONAL DECISION RULES ===
    analysis['decision_rules'] = {
        'trade_only_when': f"Rolling expectancy > 0 for 2+ windows (Currently: ${latest_expectancy:.2f})",
        'reduce_size_when': "Win rate ↑ but expectancy ↓ (divergence detected)" if latest_win_rate > 70 and expectancy_trend < 0 else "No size reduction needed",
        'stop_trading_when': "Rolling profit drops 30% from recent peak",
        'regime_tagging': "Tag trades by session + volatility for deeper analysis"
    }
    
    # === RISK METRICS SUMMARY ===
    analysis['risk_metrics'] = {
        'current_expectancy': f"${latest_expectancy:.2f}",
        'expectancy_trend': f"${expectancy_trend:.3f} (5-trade avg)",
        'win_rate': f"{latest_win_rate:.1f}%",
        'rr_ratio': f"{latest_rr_ratio:.2f}:1",
        'profit_factor': f"{latest_profit_factor:.2f}",
        'profit_concentration': f"{latest_concentration:.1f}%"
    }
    
    # === NEXT LEVEL ANALYSIS NEEDED ===
    analysis['missing_analytics'] = [
        "🔴 Rolling MAE/MFE - Are losses getting deeper? Wins cut early?",
        "🔴 Drawdown Duration - How long does pain last? (Kills psychology)",
        "🔴 Expectancy by Regime - London vs NY, High vol vs Low vol",
        "🔴 Trade Contribution Histogram - Which 10% produce 80% of PnL?",
        "🔴 Volatility Regime Detection - Edge changes with market conditions"
    ]
    
    return analysis

def compute_top_kpis(df, pnl, starting_balance_override=None):
    """Compute top-level KPIs: Equity, Balance, Total PnL"""
    
    # Use override if provided, otherwise detect from data
    if starting_balance_override:
        starting_balance = starting_balance_override
        trade_df = df.copy()
    else:
        # Separate trades from balance entries
        trade_df = df.copy()
        starting_balance = 10000  # Default fallback
        
        # Method 1: Look for balance entries in comment column
        if 'comment' in df.columns:
            balance_rows = df[df['comment'].str.contains('Initial|balance', case=False, na=False)]
            if not balance_rows.empty:
                # For balance entries, the amount is in the profit column
                balance_col = 'profit' if 'profit' in balance_rows.columns else pnl
                balance_amount = balance_rows[balance_col].iloc[0]
                if balance_amount > 0:
                    starting_balance = balance_amount
                # Remove balance entries from trade calculations
                trade_df = df[~df['comment'].str.contains('Initial|balance', case=False, na=False)]
        
        # Method 2: Look for balance type entries
        elif 'type' in df.columns:
            balance_entries = df[df['type'].str.contains('balance', case=False, na=False)]
            if not balance_entries.empty:
                starting_balance = balance_entries[pnl].iloc[0]
                trade_df = df[~df['type'].str.contains('balance', case=False, na=False)]
    
    # Calculate total PnL from trades only
    if pnl in trade_df.columns:
        total_pnl = trade_df[pnl].sum()
    else:
        total_pnl = 0
        
    final_equity = starting_balance + total_pnl
    
    return {
        "Starting Balance": f"${starting_balance:,.2f}",
        "Total PnL": f"${total_pnl:,.2f}",
        "Final Equity": f"${final_equity:,.2f}",
        "Return %": f"{(total_pnl/starting_balance)*100:.2f}%"
    }

def compute_time_analysis(df, pnl):
    """Analyze performance by time of day and day of week"""
    import pandas as pd
    
    # Try to parse datetime from available columns
    datetime_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'datetime']):
            try:
                df['parsed_datetime'] = pd.to_datetime(df[col])
                datetime_col = 'parsed_datetime'
                break
            except:
                continue
    
    if datetime_col is None:
        # Create synthetic datetime for demo
        df['parsed_datetime'] = pd.date_range('2024-01-01 09:30', periods=len(df), freq='4H')
        datetime_col = 'parsed_datetime'
    
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.day_name()
    df['month'] = df[datetime_col].dt.month
    df['month_name'] = df[datetime_col].dt.month_name()
    
    # Hourly analysis
    hourly_stats = df.groupby('hour').agg({
        pnl: ['sum', 'mean', 'count']
    }).round(2)
    
    hourly_stats.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']
    hourly_stats['Win_Rate'] = df.groupby('hour').apply(lambda x: (x[pnl] > 0).sum() / len(x) * 100).round(2)
    
    # Daily analysis
    daily_stats = df.groupby('day_of_week').agg({
        pnl: ['sum', 'mean', 'count']
    }).round(2)
    
    daily_stats.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']
    daily_stats['Win_Rate'] = df.groupby('day_of_week').apply(lambda x: (x[pnl] > 0).sum() / len(x) * 100).round(2)
    
    # Monthly analysis
    monthly_stats = df.groupby(['month', 'month_name']).agg({
        pnl: ['sum', 'mean', 'count']
    }).round(2)
    
    monthly_stats.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']
    monthly_stats['Win_Rate'] = df.groupby(['month', 'month_name']).apply(lambda x: (x[pnl] > 0).sum() / len(x) * 100).round(2)
    
    return {
        'hourly': hourly_stats,
        'daily': daily_stats, 
        'monthly': monthly_stats,
        'df_with_time': df
    }

def compute_risk_pain_metrics(df, pnl):
    """Compute advanced risk and pain metrics"""
    
    # Drawdown duration analysis
    df = df.copy()
    df['is_underwater'] = df['drawdown'] < 0
    
    # Find drawdown periods
    drawdown_periods = []
    start_dd = None
    
    for i, underwater in enumerate(df['is_underwater']):
        if underwater and start_dd is None:
            start_dd = i
        elif not underwater and start_dd is not None:
            drawdown_periods.append(i - start_dd)
            start_dd = None
    
    # If still in drawdown at end
    if start_dd is not None:
        drawdown_periods.append(len(df) - start_dd)
    
    # Recovery analysis
    recovery_times = []
    peak_indices = df[df['drawdown'] == 0].index.tolist()
    
    for i in range(1, len(peak_indices)):
        recovery_times.append(peak_indices[i] - peak_indices[i-1])
    
    # Pain metrics
    ulcer_index = (df['drawdown_pct'] ** 2).mean() ** 0.5
    
    # Consecutive losses
    df['is_loss'] = df[pnl] < 0
    consecutive_losses = []
    current_streak = 0
    
    for is_loss in df['is_loss']:
        if is_loss:
            current_streak += 1
        else:
            if current_streak > 0:
                consecutive_losses.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        consecutive_losses.append(current_streak)
    
    # Recovery factor
    total_profit = df[df[pnl] > 0][pnl].sum()
    max_dd = abs(df['drawdown'].min())
    recovery_factor = total_profit / max_dd if max_dd > 0 else 0
    
    return {
        'drawdown_periods': drawdown_periods,
        'avg_drawdown_duration': sum(drawdown_periods) / len(drawdown_periods) if drawdown_periods else 0,
        'max_drawdown_duration': max(drawdown_periods) if drawdown_periods else 0,
        'recovery_times': recovery_times,
        'avg_recovery_time': sum(recovery_times) / len(recovery_times) if recovery_times else 0,
        'ulcer_index': ulcer_index,
        'consecutive_losses': consecutive_losses,
        'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
        'recovery_factor': recovery_factor
    }

def compute_trading_insights(df, pnl):
    """
    Comprehensive trading insights analysis including:
    - Lot size analysis and consistency
    - Risk-Reward ratio analysis
    - Pip analysis (if available)
    - Buy vs Sell performance comparison
    - Position sizing patterns
    """
    import pandas as pd
    import numpy as np
    
    insights = {}
    
    # === LOT SIZE ANALYSIS ===
    lot_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['lot', 'volume', 'size', 'qty', 'quantity']):
            lot_col = col
            break
    
    if lot_col and lot_col in df.columns:
        df['lots'] = pd.to_numeric(df[lot_col], errors='coerce').fillna(0)
        
        # Lot size statistics
        lot_stats = {
            'avg_lot_size': df['lots'].mean(),
            'median_lot_size': df['lots'].median(),
            'min_lot_size': df['lots'].min(),
            'max_lot_size': df['lots'].max(),
            'lot_std': df['lots'].std(),
            'lot_consistency': 1 - (df['lots'].std() / df['lots'].mean()) if df['lots'].mean() > 0 else 0
        }
        
        # Lot size vs performance correlation
        lot_performance = df.groupby('lots').agg({
            pnl: ['sum', 'mean', 'count'],
            'lots': 'first'
        }).round(2)
        
        lot_performance.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'Lot_Size']
        lot_performance['Win_Rate'] = df.groupby('lots').apply(lambda x: (x[pnl] > 0).sum() / len(x) * 100).round(2)
        lot_performance['PnL_Per_Lot'] = (lot_performance['Total_PnL'] / lot_performance['Lot_Size']).round(2)
        
        insights['lot_analysis'] = {
            'stats': lot_stats,
            'performance_by_lot': lot_performance.sort_values('Total_PnL', ascending=False),
            'has_lot_data': True
        }
    else:
        insights['lot_analysis'] = {'has_lot_data': False}
    
    # === RISK-REWARD RATIO ANALYSIS ===
    wins = df[df[pnl] > 0]
    losses = df[df[pnl] < 0]
    
    if len(wins) > 0 and len(losses) > 0:
        avg_win = wins[pnl].mean()
        avg_loss = abs(losses[pnl].mean())
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # RR distribution
        df['rr_individual'] = np.where(df[pnl] > 0, 
                                     df[pnl] / avg_loss,  # Win as multiple of avg loss
                                     df[pnl] / avg_loss)  # Loss as fraction of avg loss
        
        rr_stats = {
            'avg_rr_ratio': rr_ratio,
            'best_rr': df['rr_individual'].max(),
            'worst_rr': df['rr_individual'].min(),
            'rr_consistency': 1 - (wins[pnl].std() / avg_win) if avg_win > 0 else 0
        }
        
        insights['rr_analysis'] = {
            'stats': rr_stats,
            'has_rr_data': True
        }
    else:
        insights['rr_analysis'] = {'has_rr_data': False}
    
    # === PIP ANALYSIS ===
    pip_cols = [col for col in df.columns if 'pip' in col.lower()]
    
    if pip_cols:
        pip_col = pip_cols[0]
        df['pips'] = pd.to_numeric(df[pip_col], errors='coerce').fillna(0)
        
        pip_stats = {
            'avg_pips_per_trade': df['pips'].mean(),
            'total_pips': df['pips'].sum(),
            'avg_pips_win': wins['pips'].mean() if len(wins) > 0 and 'pips' in wins.columns else 0,
            'avg_pips_loss': losses['pips'].mean() if len(losses) > 0 and 'pips' in losses.columns else 0,
            'best_pip_trade': df['pips'].max(),
            'worst_pip_trade': df['pips'].min(),
            'pip_consistency': 1 - (df['pips'].std() / abs(df['pips'].mean())) if df['pips'].mean() != 0 else 0
        }
        
        insights['pip_analysis'] = {
            'stats': pip_stats,
            'has_pip_data': True
        }
    else:
        # Try to calculate pips from price data if available
        open_price_col = None
        close_price_col = None
        symbol_col = None
        
        for col in df.columns:
            if 'open' in col.lower() and 'price' in col.lower():
                open_price_col = col
            elif 'close' in col.lower() and 'price' in col.lower():
                close_price_col = col
            elif any(word in col.lower() for word in ['symbol', 'pair', 'instrument', 'asset']):
                symbol_col = col
        
        if open_price_col and close_price_col:
            df['open_price_num'] = pd.to_numeric(df[open_price_col], errors='coerce')
            df['close_price_num'] = pd.to_numeric(df[close_price_col], errors='coerce')
            df['price_diff'] = df['close_price_num'] - df['open_price_num']
            
            # Calculate pips based on symbol type
            def calculate_pips(row):
                price_diff = row['price_diff']
                symbol = str(row.get(symbol_col, '')).upper() if symbol_col else ''
                
                # Handle different instrument types
                if any(x in symbol for x in ['NAS100', 'US100', 'NASDAQ', 'SPX500', 'US500', 'SP500', 'DOW', 'US30']):
                    # Stock indices: 1 point = 1 pip
                    pips = abs(price_diff)
                elif any(x in symbol for x in ['XAUUSD', 'GOLD', 'XAGUSD', 'SILVER']):
                    # Precious metals: 0.1 = 1 pip for gold, 0.01 = 1 pip for silver
                    if 'XAU' in symbol or 'GOLD' in symbol:
                        pips = abs(price_diff) * 10  # 0.1 = 1 pip
                    else:  # Silver
                        pips = abs(price_diff) * 100  # 0.01 = 1 pip
                elif any(x in symbol for x in ['WTI', 'OIL', 'BRENT']):
                    # Oil: 0.01 = 1 pip
                    pips = abs(price_diff) * 100
                elif 'JPY' in symbol:
                    # JPY pairs: 0.01 = 1 pip (2 decimal places)
                    pips = abs(price_diff) * 100
                elif any(x in symbol for x in ['BTC', 'ETH', 'CRYPTO']):
                    # Crypto: depends on the pair, but generally 1 unit = 1 pip
                    pips = abs(price_diff)
                else:
                    # Standard forex pairs: 0.0001 = 1 pip (4 decimal places)
                    pips = abs(price_diff) * 10000
                
                return pips
            
            # Apply pip calculation
            if symbol_col:
                df['estimated_pips'] = df.apply(calculate_pips, axis=1)
            else:
                # Default to forex calculation if no symbol info
                df['estimated_pips'] = abs(df['price_diff']) * 10000
            
            # Calculate pip statistics
            pip_stats = {
                'avg_pips_per_trade': df['estimated_pips'].mean(),
                'total_pips': df['estimated_pips'].sum(),
                'best_pip_trade': df['estimated_pips'].max(),
                'worst_pip_trade': df['estimated_pips'].min(),
                'pip_consistency': 1 - (df['estimated_pips'].std() / df['estimated_pips'].mean()) if df['estimated_pips'].mean() != 0 else 0
            }
            
            insights['pip_analysis'] = {
                'stats': pip_stats,
                'has_pip_data': True,
                'estimated': True,
                'calculation_method': 'symbol_aware' if symbol_col else 'forex_default'
            }
        else:
            insights['pip_analysis'] = {'has_pip_data': False}
    
    # === BUY VS SELL ANALYSIS ===
    type_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['type', 'side', 'direction', 'action']):
            type_col = col
            break
    
    if type_col:
        # Normalize buy/sell values
        df['trade_direction'] = df[type_col].astype(str).str.lower()
        df['trade_direction'] = df['trade_direction'].replace({
            'buy': 'BUY', 'long': 'BUY', '0': 'BUY', 'b': 'BUY',
            'sell': 'SELL', 'short': 'SELL', '1': 'SELL', 's': 'SELL'
        })
        
        # Filter to only BUY/SELL trades
        valid_directions = df[df['trade_direction'].isin(['BUY', 'SELL'])]
        
        if len(valid_directions) > 0:
            direction_analysis = valid_directions.groupby('trade_direction').agg({
                pnl: ['sum', 'mean', 'count', 'std']
            }).round(2)
            
            direction_analysis.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'PnL_Std']
            direction_analysis['Win_Rate'] = valid_directions.groupby('trade_direction').apply(
                lambda x: (x[pnl] > 0).sum() / len(x) * 100
            ).round(2)
            
            direction_analysis['Profit_Factor'] = valid_directions.groupby('trade_direction').apply(
                lambda x: x[x[pnl] > 0][pnl].sum() / abs(x[x[pnl] < 0][pnl].sum()) if len(x[x[pnl] < 0]) > 0 else float('inf')
            ).round(2)
            
            direction_analysis['Consistency'] = (1 - (direction_analysis['PnL_Std'] / abs(direction_analysis['Avg_PnL']))).fillna(0).round(2)
            
            insights['direction_analysis'] = {
                'performance': direction_analysis,
                'has_direction_data': True
            }
        else:
            insights['direction_analysis'] = {'has_direction_data': False}
    else:
        insights['direction_analysis'] = {'has_direction_data': False}
    
    # === POSITION SIZING PATTERNS ===
    if lot_col:
        # Analyze if position sizing correlates with performance
        df['position_size_category'] = pd.cut(df['lots'], bins=3, labels=['Small', 'Medium', 'Large'])
        
        size_performance = df.groupby('position_size_category').agg({
            pnl: ['sum', 'mean', 'count'],
            'lots': 'mean'
        }).round(2)
        
        size_performance.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'Avg_Lot_Size']
        size_performance['Win_Rate'] = df.groupby('position_size_category').apply(
            lambda x: (x[pnl] > 0).sum() / len(x) * 100
        ).round(2)
        
        insights['position_sizing'] = {
            'performance_by_size': size_performance,
            'has_sizing_data': True
        }
    else:
        insights['position_sizing'] = {'has_sizing_data': False}
    
    # === SYMBOL ANALYSIS (if available) ===
    symbol_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['symbol', 'pair', 'instrument', 'asset']):
            symbol_col = col
            break
    
    if symbol_col:
        symbol_performance = df.groupby(symbol_col).agg({
            pnl: ['sum', 'mean', 'count', 'std']
        }).round(2)
        
        symbol_performance.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'PnL_Std']
        symbol_performance['Win_Rate'] = df.groupby(symbol_col).apply(
            lambda x: (x[pnl] > 0).sum() / len(x) * 100
        ).round(2)
        
        symbol_performance['Consistency'] = (1 - (symbol_performance['PnL_Std'] / abs(symbol_performance['Avg_PnL']))).fillna(0).round(2)
        
        insights['symbol_analysis'] = {
            'performance': symbol_performance.sort_values('Total_PnL', ascending=False),
            'has_symbol_data': True
        }
    else:
        insights['symbol_analysis'] = {'has_symbol_data': False}
    
    return insights

def generate_trading_insights_summary(insights):
    """Generate human-readable summary of trading insights"""
    
    summary = []
    
    # Lot Size Insights
    if insights['lot_analysis']['has_lot_data']:
        lot_stats = insights['lot_analysis']['stats']
        consistency = lot_stats['lot_consistency']
        
        if consistency > 0.8:
            summary.append(f"✅ **Excellent lot size consistency** ({consistency:.1%}) - you maintain disciplined position sizing")
        elif consistency > 0.6:
            summary.append(f"⚠️ **Moderate lot size consistency** ({consistency:.1%}) - consider more consistent position sizing")
        else:
            summary.append(f"🚨 **Poor lot size consistency** ({consistency:.1%}) - highly variable position sizing increases risk")
        
        avg_lot = lot_stats['avg_lot_size']
        max_lot = lot_stats['max_lot_size']
        if max_lot > avg_lot * 3:
            summary.append(f"⚠️ **Position sizing risk**: Max lot ({max_lot:.2f}) is {max_lot/avg_lot:.1f}x average - avoid oversizing")
    
    # Risk-Reward Insights
    if insights['rr_analysis']['has_rr_data']:
        rr_ratio = insights['rr_analysis']['stats']['avg_rr_ratio']
        
        if rr_ratio >= 2.0:
            summary.append(f"🎯 **Excellent R:R ratio** ({rr_ratio:.2f}:1) - your winners are much larger than losers")
        elif rr_ratio >= 1.5:
            summary.append(f"✅ **Good R:R ratio** ({rr_ratio:.2f}:1) - solid risk management")
        elif rr_ratio >= 1.0:
            summary.append(f"⚠️ **Marginal R:R ratio** ({rr_ratio:.2f}:1) - aim for larger winners relative to losses")
        else:
            summary.append(f"🚨 **Poor R:R ratio** ({rr_ratio:.2f}:1) - losses are larger than wins on average")
    
    # Direction Analysis
    if insights['direction_analysis']['has_direction_data']:
        direction_perf = insights['direction_analysis']['performance']
        
        if 'BUY' in direction_perf.index and 'SELL' in direction_perf.index:
            buy_pnl = direction_perf.loc['BUY', 'Total_PnL']
            sell_pnl = direction_perf.loc['SELL', 'Total_PnL']
            buy_wr = direction_perf.loc['BUY', 'Win_Rate']
            sell_wr = direction_perf.loc['SELL', 'Win_Rate']
            
            if abs(buy_pnl - sell_pnl) / max(abs(buy_pnl), abs(sell_pnl)) > 0.5:
                better_direction = 'BUY' if buy_pnl > sell_pnl else 'SELL'
                worse_direction = 'SELL' if better_direction == 'BUY' else 'BUY'
                summary.append(f"📊 **Directional bias**: {better_direction} trades significantly outperform {worse_direction} trades")
            
            if abs(buy_wr - sell_wr) > 15:
                better_wr_direction = 'BUY' if buy_wr > sell_wr else 'SELL'
                summary.append(f"🎯 **Win rate bias**: {better_wr_direction} trades have {max(buy_wr, sell_wr):.1f}% vs {min(buy_wr, sell_wr):.1f}% win rate")
    
    # Pip Analysis
    if insights['pip_analysis']['has_pip_data']:
        pip_stats = insights['pip_analysis']['stats']
        avg_pips = pip_stats['avg_pips_per_trade']
        total_pips = pip_stats['total_pips']
        
        calculation_method = insights['pip_analysis'].get('calculation_method', 'direct')
        
        if total_pips > 0:
            if 'estimated' in insights['pip_analysis']:
                summary.append(f"📈 **Pip performance**: Average {avg_pips:.1f} pips per trade, {total_pips:.0f} total pips captured (symbol-aware calculation)")
            else:
                summary.append(f"📈 **Pip performance**: Average {avg_pips:.1f} pips per trade, {total_pips:.0f} total pips captured")
        else:
            if 'estimated' in insights['pip_analysis']:
                summary.append(f"📉 **Pip performance**: Average {avg_pips:.1f} pips per trade - negative pip capture (symbol-aware calculation)")
            else:
                summary.append(f"📉 **Pip performance**: Average {avg_pips:.1f} pips per trade - negative pip capture")
    
    return summary