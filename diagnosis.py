def diagnose(m):
    verdict = []

    if m["Expectancy"] <= 0:
        verdict.append("You have no statistical edge. Stop trading this system.")
    elif m["Expectancy"] < 20:
        verdict.append("Edge exists but is weak. Risk control is mandatory.")

    if m["Profit Factor"] < 1.3:
        verdict.append("Profit factor is fragile. A few bad trades will erase gains.")

    if m["Avg Loss"] < -abs(m["Avg Win"]):
        verdict.append("Losses are larger than wins. Stop-loss discipline is poor.")

    if m["Max Drawdown %"] < -25:
        verdict.append("Drawdown is too deep. Position sizing is unsafe.")

    return " ".join(verdict) if verdict else "System is stable but still needs stress testing."

def analyze_traded_symbols(df, pnl_col):
    """
    Analyze performance by traded symbols/assets
    """
    
    symbol_analysis = {
        'symbol_performance': {},
        'symbol_insights': [],
        'symbol_recommendations': [],
        'diversification_analysis': {},
        'asset_class_insights': []
    }
    
    # Check if symbol column exists
    symbol_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['symbol', 'instrument', 'asset', 'pair', 'ticker']):
            symbol_col = col
            break
    
    if symbol_col is None:
        symbol_analysis['symbol_insights'].append("📊 No symbol/instrument column detected. Upload data with symbol information for deeper insights.")
        return symbol_analysis
    
    # Analyze each symbol
    symbol_stats = df.groupby(symbol_col).agg({
        pnl_col: ['count', 'sum', 'mean', 'std']
    }).round(2)
    
    symbol_stats.columns = ['Trade_Count', 'Total_PnL', 'Avg_PnL', 'PnL_Volatility']
    
    # Calculate win rate separately
    win_rates = df.groupby(symbol_col).apply(lambda x: (x[pnl_col] > 0).sum() / len(x) * 100).round(2)
    symbol_stats['Win_Rate'] = win_rates
    
    # Calculate additional metrics for each symbol
    for symbol in symbol_stats.index:
        symbol_trades = df[df[symbol_col] == symbol]
        
        # Calculate profit factor
        wins = symbol_trades[symbol_trades[pnl_col] > 0][pnl_col].sum()
        losses = abs(symbol_trades[symbol_trades[pnl_col] < 0][pnl_col].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')
        
        # Calculate max drawdown for this symbol
        symbol_equity = symbol_trades[pnl_col].cumsum()
        symbol_peak = symbol_equity.cummax()
        symbol_dd = ((symbol_equity - symbol_peak) / symbol_peak * 100).min()
        
        symbol_stats.loc[symbol, 'Profit_Factor'] = round(profit_factor, 2)
        symbol_stats.loc[symbol, 'Max_DD_Pct'] = round(symbol_dd, 2)
        symbol_stats.loc[symbol, 'Expectancy'] = round(symbol_trades[pnl_col].mean(), 2)
    
    # Sort by total PnL
    symbol_stats = symbol_stats.sort_values('Total_PnL', ascending=False)
    symbol_analysis['symbol_performance'] = symbol_stats
    
    # Generate insights
    total_symbols = len(symbol_stats)
    profitable_symbols = (symbol_stats['Total_PnL'] > 0).sum()
    
    # Best and worst performers
    best_symbol = symbol_stats.index[0]
    worst_symbol = symbol_stats.index[-1]
    
    best_pnl = symbol_stats.loc[best_symbol, 'Total_PnL']
    worst_pnl = symbol_stats.loc[worst_symbol, 'Total_PnL']
    
    # Symbol insights
    symbol_analysis['symbol_insights'].append(f"📊 **Trading Portfolio**: {total_symbols} different symbols traded")
    symbol_analysis['symbol_insights'].append(f"🎯 **Success Rate**: {profitable_symbols}/{total_symbols} symbols are profitable ({profitable_symbols/total_symbols*100:.1f}%)")
    
    if best_pnl > 0:
        symbol_analysis['symbol_insights'].append(f"🏆 **Best Performer**: {best_symbol} (+${best_pnl:,.2f}, {symbol_stats.loc[best_symbol, 'Trade_Count']:.0f} trades)")
    
    if worst_pnl < 0:
        symbol_analysis['symbol_insights'].append(f"📉 **Worst Performer**: {worst_symbol} (${worst_pnl:,.2f}, {symbol_stats.loc[worst_symbol, 'Trade_Count']:.0f} trades)")
    
    # Concentration analysis
    top_3_pnl = symbol_stats.head(3)['Total_PnL'].sum()
    total_pnl = symbol_stats['Total_PnL'].sum()
    concentration = (top_3_pnl / total_pnl * 100) if total_pnl > 0 else 0
    
    symbol_analysis['diversification_analysis']['concentration'] = concentration
    symbol_analysis['diversification_analysis']['top_performers'] = symbol_stats.head(3)
    
    if concentration > 80:
        symbol_analysis['symbol_insights'].append(f"⚠️ **High Concentration**: Top 3 symbols generate {concentration:.1f}% of profits - diversification risk")
    elif concentration > 60:
        symbol_analysis['symbol_insights'].append(f"📊 **Moderate Concentration**: Top 3 symbols generate {concentration:.1f}% of profits")
    else:
        symbol_analysis['symbol_insights'].append(f"✅ **Good Diversification**: Profits well-distributed across symbols")
    
    # Asset class analysis (basic classification)
    asset_classes = classify_symbols(symbol_stats.index.tolist())
    symbol_analysis['asset_class_insights'] = analyze_asset_classes(symbol_stats, asset_classes, pnl_col)
    
    # Generate recommendations
    symbol_analysis['symbol_recommendations'] = generate_symbol_recommendations(symbol_stats, df, pnl_col, symbol_col)
    
    return symbol_analysis

def classify_symbols(symbols):
    """
    Basic asset classification based on symbol patterns
    """
    asset_classes = {}
    
    for symbol in symbols:
        symbol_upper = symbol.upper()
        
        # Forex pairs
        if any(pair in symbol_upper for pair in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            if len(symbol_upper.replace('.', '')) == 6:  # Standard forex pair length
                asset_classes[symbol] = 'Forex'
            else:
                asset_classes[symbol] = 'Currency'
        
        # Indices
        elif any(idx in symbol_upper for idx in ['SPX', 'SPY', 'QQQ', 'DJI', 'NAS', 'NDX', 'DAX', 'FTSE', 'NIKKEI']):
            asset_classes[symbol] = 'Index'
        
        # Commodities
        elif any(comm in symbol_upper for comm in ['GOLD', 'SILVER', 'OIL', 'GAS', 'COPPER', 'WHEAT', 'CORN']):
            asset_classes[symbol] = 'Commodity'
        
        # Crypto
        elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'BITCOIN', 'ETHEREUM', 'CRYPTO']):
            asset_classes[symbol] = 'Crypto'
        
        # Default to equity/stock
        else:
            asset_classes[symbol] = 'Equity'
    
    return asset_classes

def analyze_asset_classes(symbol_stats, asset_classes, pnl_col):
    """
    Analyze performance by asset class
    """
    insights = []
    
    # Group by asset class
    class_performance = {}
    for symbol, asset_class in asset_classes.items():
        if symbol in symbol_stats.index:
            if asset_class not in class_performance:
                class_performance[asset_class] = {
                    'symbols': [],
                    'total_pnl': 0,
                    'total_trades': 0,
                    'avg_win_rate': 0
                }
            
            stats = symbol_stats.loc[symbol]
            class_performance[asset_class]['symbols'].append(symbol)
            class_performance[asset_class]['total_pnl'] += stats['Total_PnL']
            class_performance[asset_class]['total_trades'] += stats['Trade_Count']
            class_performance[asset_class]['avg_win_rate'] += stats['Win_Rate']
    
    # Calculate averages and generate insights
    for asset_class, data in class_performance.items():
        symbol_count = len(data['symbols'])
        avg_win_rate = data['avg_win_rate'] / symbol_count
        
        insights.append(f"📈 **{asset_class}**: {symbol_count} symbols, ${data['total_pnl']:,.2f} total PnL, {avg_win_rate:.1f}% avg win rate")
    
    # Find best performing asset class
    if class_performance:
        best_class = max(class_performance.items(), key=lambda x: x[1]['total_pnl'])
        insights.append(f"🏆 **Best Asset Class**: {best_class[0]} (${best_class[1]['total_pnl']:,.2f})")
    
    return insights

def generate_symbol_recommendations(symbol_stats, df, pnl_col, symbol_col):
    """
    Generate specific recommendations for symbol trading
    """
    recommendations = []
    
    # Identify symbols to focus on
    profitable_symbols = symbol_stats[symbol_stats['Total_PnL'] > 0]
    losing_symbols = symbol_stats[symbol_stats['Total_PnL'] < 0]
    
    if len(profitable_symbols) > 0:
        top_performer = profitable_symbols.index[0]
        top_stats = profitable_symbols.iloc[0]
        
        recommendations.append(f"🎯 **FOCUS ON**: {top_performer} - Your best performer (${top_stats['Total_PnL']:,.2f}, {top_stats['Win_Rate']:.1f}% win rate)")
        
        # Check if top performer has good sample size
        if top_stats['Trade_Count'] < 10:
            recommendations.append(f"📊 **INCREASE SAMPLE**: {top_performer} needs more trades ({top_stats['Trade_Count']:.0f}) for statistical significance")
    
    # Identify symbols to avoid or reduce
    if len(losing_symbols) > 0:
        worst_performers = losing_symbols.tail(3)  # Bottom 3
        
        for symbol in worst_performers.index:
            stats = worst_performers.loc[symbol]
            if stats['Total_PnL'] < -100:  # Significant losses
                recommendations.append(f"🚫 **AVOID**: {symbol} - Consistent loser (${stats['Total_PnL']:,.2f}, {stats['Win_Rate']:.1f}% win rate)")
    
    # Check for overtrading specific symbols
    trade_counts = symbol_stats['Trade_Count']
    if len(trade_counts) > 1:
        max_trades = trade_counts.max()
        avg_trades = trade_counts.mean()
        
        if max_trades > avg_trades * 3:  # One symbol has 3x more trades than average
            overtraded_symbol = trade_counts.idxmax()
            recommendations.append(f"⚠️ **OVERTRADING**: {overtraded_symbol} ({max_trades:.0f} trades) - Consider reducing frequency")
    
    # Volatility warnings
    high_vol_symbols = symbol_stats[symbol_stats['PnL_Volatility'] > symbol_stats['PnL_Volatility'].quantile(0.8)]
    for symbol in high_vol_symbols.index:
        stats = high_vol_symbols.loc[symbol]
        recommendations.append(f"⚠️ **HIGH VOLATILITY**: {symbol} - Inconsistent results (σ=${stats['PnL_Volatility']:.2f})")
    
    # Diversification recommendations
    if len(symbol_stats) == 1:
        recommendations.append("📊 **DIVERSIFY**: Trading only one symbol increases risk - consider adding 2-3 more instruments")
    elif len(symbol_stats) > 10:
        recommendations.append("🎯 **FOCUS**: Trading many symbols ({len(symbol_stats)}) - focus on top 3-5 performers")
    
    return recommendations

def comprehensive_ai_diagnosis(df, metrics, risk_data, time_data, pnl_col, selected_period="All Time"):
    """
    Advanced AI diagnosis system that provides deep trading insights
    """
    
    diagnosis = {
        'overall_grade': 'B',
        'critical_issues': [],
        'warnings': [],
        'strengths': [],
        'recommendations': [],
        'psychological_analysis': [],
        'risk_assessment': [],
        'performance_insights': [],
        'action_items': [],
        'symbol_analysis': {}
    }
    
    # === SYMBOL/ASSET ANALYSIS ===
    symbol_analysis = analyze_traded_symbols(df, pnl_col)
    diagnosis['symbol_analysis'] = symbol_analysis
    
    # Add symbol insights to main insights
    diagnosis['performance_insights'].extend(symbol_analysis['symbol_insights'])
    diagnosis['recommendations'].extend(symbol_analysis['symbol_recommendations'])
    
    # Add asset class insights
    if symbol_analysis['asset_class_insights']:
        diagnosis['performance_insights'].extend(symbol_analysis['asset_class_insights'])
    
    # === PERFORMANCE ANALYSIS ===
    total_trades = metrics["Total Trades"]
    win_rate = metrics["Win Rate (%)"]
    expectancy = metrics["Expectancy"]
    profit_factor = metrics["Profit Factor"]
    avg_win = metrics["Avg Win"]
    avg_loss = metrics["Avg Loss"]
    max_dd_pct = metrics["Max Drawdown %"]
    
    # === GRADE CALCULATION ===
    grade_score = 0
    
    # Expectancy scoring (40% weight)
    if expectancy > 50:
        grade_score += 40
    elif expectancy > 20:
        grade_score += 30
    elif expectancy > 0:
        grade_score += 15
    else:
        grade_score += 0
    
    # Win rate scoring (20% weight)
    if win_rate > 60:
        grade_score += 20
    elif win_rate > 50:
        grade_score += 15
    elif win_rate > 40:
        grade_score += 10
    else:
        grade_score += 5
    
    # Risk management scoring (40% weight)
    if max_dd_pct > -10:
        grade_score += 40
    elif max_dd_pct > -20:
        grade_score += 30
    elif max_dd_pct > -30:
        grade_score += 20
    else:
        grade_score += 10
    
    # Assign letter grade
    if grade_score >= 85:
        diagnosis['overall_grade'] = 'A+'
    elif grade_score >= 80:
        diagnosis['overall_grade'] = 'A'
    elif grade_score >= 75:
        diagnosis['overall_grade'] = 'A-'
    elif grade_score >= 70:
        diagnosis['overall_grade'] = 'B+'
    elif grade_score >= 65:
        diagnosis['overall_grade'] = 'B'
    elif grade_score >= 60:
        diagnosis['overall_grade'] = 'B-'
    elif grade_score >= 55:
        diagnosis['overall_grade'] = 'C+'
    elif grade_score >= 50:
        diagnosis['overall_grade'] = 'C'
    elif grade_score >= 40:
        diagnosis['overall_grade'] = 'D'
    else:
        diagnosis['overall_grade'] = 'F'
    
    # === CRITICAL ISSUES ===
    if expectancy <= 0:
        diagnosis['critical_issues'].append("🚨 ZERO EDGE: Your system has no statistical advantage. Stop trading immediately.")
    
    if max_dd_pct < -50:
        diagnosis['critical_issues'].append("🚨 CATASTROPHIC DRAWDOWN: Risk of account destruction. Reduce position sizes by 80%.")
    
    if profit_factor < 1.0:
        diagnosis['critical_issues'].append("🚨 LOSING SYSTEM: You're losing more than you're making. System overhaul required.")
    
    if risk_data['max_consecutive_losses'] > 10:
        diagnosis['critical_issues'].append("🚨 PSYCHOLOGICAL DANGER: 10+ consecutive losses will break most traders mentally.")
    
    # === WARNINGS ===
    if expectancy < 20 and expectancy > 0:
        diagnosis['warnings'].append("⚠️ WEAK EDGE: Your advantage is fragile. One bad streak could wipe out months of gains.")
    
    if win_rate < 40:
        diagnosis['warnings'].append("⚠️ LOW WIN RATE: You're wrong more than 60% of the time. This is psychologically difficult.")
    
    if abs(avg_loss) > avg_win * 2:
        diagnosis['warnings'].append("⚠️ POOR RISK/REWARD: Your losses are too large compared to wins. Tighten stop losses.")
    
    if max_dd_pct < -25:
        diagnosis['warnings'].append("⚠️ HIGH DRAWDOWN: 25%+ drawdown tests psychological limits. Reduce position size.")
    
    if risk_data['recovery_factor'] < 2:
        diagnosis['warnings'].append("⚠️ SLOW RECOVERY: You're not making enough profit to justify the risk taken.")
    
    # === STRENGTHS ===
    if expectancy > 50:
        diagnosis['strengths'].append("✅ STRONG EDGE: Excellent statistical advantage. This system has real potential.")
    
    if win_rate > 60:
        diagnosis['strengths'].append("✅ HIGH WIN RATE: You're right most of the time. Good for trader psychology.")
    
    if profit_factor > 2.0:
        diagnosis['strengths'].append("✅ EXCELLENT PROFIT FACTOR: Your wins significantly outweigh your losses.")
    
    if max_dd_pct > -15:
        diagnosis['strengths'].append("✅ CONTROLLED RISK: Drawdowns are manageable. Good risk management.")
    
    if risk_data['max_consecutive_losses'] <= 5:
        diagnosis['strengths'].append("✅ PSYCHOLOGICAL SAFETY: Consecutive losses are within tolerable limits.")
    
    # === TIME-BASED ANALYSIS ===
    if 'hourly' in time_data:
        worst_hours = time_data['hourly'].nsmallest(3, 'Total_PnL')
        losing_hours = [hour for hour, data in worst_hours.iterrows() if data['Total_PnL'] < -100]
        
        if losing_hours:
            diagnosis['performance_insights'].append(f"📊 KILLER HOURS: Stop trading at {', '.join([f'{h}:00' for h in losing_hours[:2]])}. These hours are bleeding money.")
    
    if 'monthly' in time_data and len(time_data['monthly']) > 1:
        monthly_stats = time_data['monthly']
        losing_months = (monthly_stats['Total_PnL'] < 0).sum()
        total_months = len(monthly_stats)
        
        if losing_months > total_months * 0.4:
            diagnosis['warnings'].append(f"⚠️ SEASONAL ISSUES: {losing_months}/{total_months} losing months suggests systematic problems.")
    
    # === PSYCHOLOGICAL ANALYSIS ===
    if risk_data['max_consecutive_losses'] > 7:
        diagnosis['psychological_analysis'].append("🧠 HIGH STRESS RISK: Long losing streaks will test your mental resilience. Prepare for emotional challenges.")
    
    if risk_data['avg_drawdown_duration'] > 15:
        diagnosis['psychological_analysis'].append("🧠 PATIENCE REQUIRED: Long recovery periods demand exceptional discipline. Most traders quit during these phases.")
    
    if win_rate < 45:
        diagnosis['psychological_analysis'].append("🧠 MENTAL TOUGHNESS NEEDED: Being wrong most of the time is psychologically difficult. Focus on process, not outcomes.")
    
    # === RECOMMENDATIONS ===
    if expectancy > 0 and expectancy < 30:
        diagnosis['recommendations'].append("💡 EDGE OPTIMIZATION: Your edge exists but needs strengthening. Focus on trade selection quality over quantity.")
    
    if max_dd_pct < -20:
        diagnosis['recommendations'].append("💡 POSITION SIZING: Reduce position size by 50% to cut drawdowns. Preserve capital for better opportunities.")
    
    if abs(avg_loss) > avg_win:
        diagnosis['recommendations'].append("💡 STOP LOSS REVIEW: Tighten stop losses or improve entry timing. Your risk/reward ratio needs work.")
    
    if total_trades < 30:
        diagnosis['recommendations'].append("💡 SAMPLE SIZE: Need more trades for statistical significance. Current results may not be reliable.")
    
    # === RISK ASSESSMENT ===
    if max_dd_pct < -30:
        diagnosis['risk_assessment'].append("🔴 HIGH RISK: Current drawdown levels threaten account survival.")
    elif max_dd_pct < -20:
        diagnosis['risk_assessment'].append("🟡 MODERATE RISK: Drawdowns are concerning but manageable with discipline.")
    else:
        diagnosis['risk_assessment'].append("🟢 LOW RISK: Risk levels are well-controlled.")
    
    # === ACTION ITEMS ===
    if len(diagnosis['critical_issues']) > 0:
        diagnosis['action_items'].append("🎯 IMMEDIATE: Address all critical issues before taking another trade.")
    
    if expectancy > 0:
        diagnosis['action_items'].append("🎯 OPTIMIZE: Focus on improving trade selection to strengthen your edge.")
    
    if max_dd_pct < -20:
        diagnosis['action_items'].append("🎯 RISK CONTROL: Implement stricter position sizing rules immediately.")
    
    diagnosis['action_items'].append("🎯 MONITOR: Track rolling performance to detect edge decay early.")
    
    return diagnosis
    """
    Advanced AI diagnosis system that provides deep trading insights
    """
    
    diagnosis = {
        'overall_grade': 'B',
        'critical_issues': [],
        'warnings': [],
        'strengths': [],
        'recommendations': [],
        'psychological_analysis': [],
        'risk_assessment': [],
        'performance_insights': [],
        'action_items': []
    }
    
    # === PERFORMANCE ANALYSIS ===
    total_trades = metrics["Total Trades"]
    win_rate = metrics["Win Rate (%)"]
    expectancy = metrics["Expectancy"]
    profit_factor = metrics["Profit Factor"]
    avg_win = metrics["Avg Win"]
    avg_loss = metrics["Avg Loss"]
    max_dd_pct = metrics["Max Drawdown %"]
    
    # === GRADE CALCULATION ===
    grade_score = 0
    
    # Expectancy scoring (40% weight)
    if expectancy > 50:
        grade_score += 40
    elif expectancy > 20:
        grade_score += 30
    elif expectancy > 0:
        grade_score += 15
    else:
        grade_score += 0
    
    # Win rate scoring (20% weight)
    if win_rate > 60:
        grade_score += 20
    elif win_rate > 50:
        grade_score += 15
    elif win_rate > 40:
        grade_score += 10
    else:
        grade_score += 5
    
    # Risk management scoring (40% weight)
    if max_dd_pct > -10:
        grade_score += 40
    elif max_dd_pct > -20:
        grade_score += 30
    elif max_dd_pct > -30:
        grade_score += 20
    else:
        grade_score += 10
    
    # Assign letter grade
    if grade_score >= 85:
        diagnosis['overall_grade'] = 'A+'
    elif grade_score >= 80:
        diagnosis['overall_grade'] = 'A'
    elif grade_score >= 75:
        diagnosis['overall_grade'] = 'A-'
    elif grade_score >= 70:
        diagnosis['overall_grade'] = 'B+'
    elif grade_score >= 65:
        diagnosis['overall_grade'] = 'B'
    elif grade_score >= 60:
        diagnosis['overall_grade'] = 'B-'
    elif grade_score >= 55:
        diagnosis['overall_grade'] = 'C+'
    elif grade_score >= 50:
        diagnosis['overall_grade'] = 'C'
    elif grade_score >= 40:
        diagnosis['overall_grade'] = 'D'
    else:
        diagnosis['overall_grade'] = 'F'
    
    # === CRITICAL ISSUES ===
    if expectancy <= 0:
        diagnosis['critical_issues'].append("🚨 ZERO EDGE: Your system has no statistical advantage. Stop trading immediately.")
    
    if max_dd_pct < -50:
        diagnosis['critical_issues'].append("🚨 CATASTROPHIC DRAWDOWN: Risk of account destruction. Reduce position sizes by 80%.")
    
    if profit_factor < 1.0:
        diagnosis['critical_issues'].append("🚨 LOSING SYSTEM: You're losing more than you're making. System overhaul required.")
    
    if risk_data['max_consecutive_losses'] > 10:
        diagnosis['critical_issues'].append("🚨 PSYCHOLOGICAL DANGER: 10+ consecutive losses will break most traders mentally.")
    
    # === WARNINGS ===
    if expectancy < 20 and expectancy > 0:
        diagnosis['warnings'].append("⚠️ WEAK EDGE: Your advantage is fragile. One bad streak could wipe out months of gains.")
    
    if win_rate < 40:
        diagnosis['warnings'].append("⚠️ LOW WIN RATE: You're wrong more than 60% of the time. This is psychologically difficult.")
    
    if abs(avg_loss) > avg_win * 2:
        diagnosis['warnings'].append("⚠️ POOR RISK/REWARD: Your losses are too large compared to wins. Tighten stop losses.")
    
    if max_dd_pct < -25:
        diagnosis['warnings'].append("⚠️ HIGH DRAWDOWN: 25%+ drawdown tests psychological limits. Reduce position size.")
    
    if risk_data['recovery_factor'] < 2:
        diagnosis['warnings'].append("⚠️ SLOW RECOVERY: You're not making enough profit to justify the risk taken.")
    
    # === STRENGTHS ===
    if expectancy > 50:
        diagnosis['strengths'].append("✅ STRONG EDGE: Excellent statistical advantage. This system has real potential.")
    
    if win_rate > 60:
        diagnosis['strengths'].append("✅ HIGH WIN RATE: You're right most of the time. Good for trader psychology.")
    
    if profit_factor > 2.0:
        diagnosis['strengths'].append("✅ EXCELLENT PROFIT FACTOR: Your wins significantly outweigh your losses.")
    
    if max_dd_pct > -15:
        diagnosis['strengths'].append("✅ CONTROLLED RISK: Drawdowns are manageable. Good risk management.")
    
    if risk_data['max_consecutive_losses'] <= 5:
        diagnosis['strengths'].append("✅ PSYCHOLOGICAL SAFETY: Consecutive losses are within tolerable limits.")
    
    # === TIME-BASED ANALYSIS ===
    if 'hourly' in time_data:
        worst_hours = time_data['hourly'].nsmallest(3, 'Total_PnL')
        losing_hours = [hour for hour, data in worst_hours.iterrows() if data['Total_PnL'] < -100]
        
        if losing_hours:
            diagnosis['performance_insights'].append(f"📊 KILLER HOURS: Stop trading at {', '.join([f'{h}:00' for h in losing_hours[:2]])}. These hours are bleeding money.")
    
    if 'monthly' in time_data and len(time_data['monthly']) > 1:
        monthly_stats = time_data['monthly']
        losing_months = (monthly_stats['Total_PnL'] < 0).sum()
        total_months = len(monthly_stats)
        
        if losing_months > total_months * 0.4:
            diagnosis['warnings'].append(f"⚠️ SEASONAL ISSUES: {losing_months}/{total_months} losing months suggests systematic problems.")
    
    # === PSYCHOLOGICAL ANALYSIS ===
    if risk_data['max_consecutive_losses'] > 7:
        diagnosis['psychological_analysis'].append("🧠 HIGH STRESS RISK: Long losing streaks will test your mental resilience. Prepare for emotional challenges.")
    
    if risk_data['avg_drawdown_duration'] > 15:
        diagnosis['psychological_analysis'].append("🧠 PATIENCE REQUIRED: Long recovery periods demand exceptional discipline. Most traders quit during these phases.")
    
    if win_rate < 45:
        diagnosis['psychological_analysis'].append("🧠 MENTAL TOUGHNESS NEEDED: Being wrong most of the time is psychologically difficult. Focus on process, not outcomes.")
    
    # === RECOMMENDATIONS ===
    if expectancy > 0 and expectancy < 30:
        diagnosis['recommendations'].append("💡 EDGE OPTIMIZATION: Your edge exists but needs strengthening. Focus on trade selection quality over quantity.")
    
    if max_dd_pct < -20:
        diagnosis['recommendations'].append("💡 POSITION SIZING: Reduce position size by 50% to cut drawdowns. Preserve capital for better opportunities.")
    
    if abs(avg_loss) > avg_win:
        diagnosis['recommendations'].append("💡 STOP LOSS REVIEW: Tighten stop losses or improve entry timing. Your risk/reward ratio needs work.")
    
    if total_trades < 30:
        diagnosis['recommendations'].append("💡 SAMPLE SIZE: Need more trades for statistical significance. Current results may not be reliable.")
    
    # === RISK ASSESSMENT ===
    if max_dd_pct < -30:
        diagnosis['risk_assessment'].append("🔴 HIGH RISK: Current drawdown levels threaten account survival.")
    elif max_dd_pct < -20:
        diagnosis['risk_assessment'].append("🟡 MODERATE RISK: Drawdowns are concerning but manageable with discipline.")
    else:
        diagnosis['risk_assessment'].append("🟢 LOW RISK: Risk levels are well-controlled.")
    
    # === ACTION ITEMS ===
    if len(diagnosis['critical_issues']) > 0:
        diagnosis['action_items'].append("🎯 IMMEDIATE: Address all critical issues before taking another trade.")
    
    if expectancy > 0:
        diagnosis['action_items'].append("🎯 OPTIMIZE: Focus on improving trade selection to strengthen your edge.")
    
    if max_dd_pct < -20:
        diagnosis['action_items'].append("🎯 RISK CONTROL: Implement stricter position sizing rules immediately.")
    
    diagnosis['action_items'].append("🎯 MONITOR: Track rolling performance to detect edge decay early.")
    
    return diagnosis

def generate_executive_summary(diagnosis, metrics, selected_period):
    """Generate a concise executive summary"""
    
    grade = diagnosis['overall_grade']
    expectancy = metrics['Expectancy']
    win_rate = metrics['Win Rate (%)']
    profit_factor = metrics['Profit Factor']
    
    # Determine overall status
    if grade in ['A+', 'A', 'A-']:
        status = "EXCELLENT"
        emoji = "🏆"
    elif grade in ['B+', 'B', 'B-']:
        status = "GOOD"
        emoji = "👍"
    elif grade in ['C+', 'C']:
        status = "AVERAGE"
        emoji = "⚠️"
    elif grade == 'D':
        status = "POOR"
        emoji = "👎"
    else:
        status = "FAILING"
        emoji = "🚨"
    
    summary = f"""
    ## {emoji} TRADING SYSTEM GRADE: {grade} ({status})
    
    **Period Analyzed**: {selected_period}  
    **Statistical Edge**: ${expectancy:.2f} per trade  
    **Win Rate**: {win_rate:.1f}%  
    **Profit Factor**: {profit_factor:.2f}  
    
    """
    
    if len(diagnosis['critical_issues']) > 0:
        summary += "**🚨 CRITICAL ISSUES DETECTED - IMMEDIATE ACTION REQUIRED**\n\n"
    elif len(diagnosis['warnings']) > 0:
        summary += "**⚠️ SYSTEM NEEDS IMPROVEMENT**\n\n"
    else:
        summary += "**✅ SYSTEM IS PERFORMING WELL**\n\n"
    
    return summary