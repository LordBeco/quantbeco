#!/usr/bin/env python3
"""
Demo of the Comprehensive AI Diagnosis System
"""

import pandas as pd
from analytics import compute_metrics, compute_risk_pain_metrics, compute_time_analysis
from diagnosis import comprehensive_ai_diagnosis, generate_executive_summary

def demo_ai_diagnosis():
    print("🎯 COMPREHENSIVE AI DIAGNOSIS DEMO")
    print("=" * 50)
    
    # Load your actual trading data
    df_raw = pd.read_csv('export_2025-12-27 17_33_10.csv')
    df_raw.columns = [c.lower().replace(' ', '_') for c in df_raw.columns]

    # Clean and prepare data
    for col in ['profit', 'commission', 'swaps']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    trades_df = df_raw[~df_raw['comment'].str.contains('Initial|balance', case=False, na=False)].copy()
    trades_df['real_pnl'] = trades_df['profit'] + trades_df['commission'] + trades_df['swaps']

    # Add required columns
    trades_df['equity'] = trades_df['real_pnl'].cumsum()
    trades_df['peak'] = trades_df['equity'].cummax()
    trades_df['drawdown'] = trades_df['equity'] - trades_df['peak']
    trades_df['drawdown_pct'] = trades_df['drawdown'] / trades_df['peak'].replace(0, 1) * 100

    # Calculate all metrics
    metrics = compute_metrics(trades_df, 'real_pnl')
    risk_data = compute_risk_pain_metrics(trades_df, 'real_pnl')
    time_data = compute_time_analysis(trades_df, 'real_pnl')

    # Generate comprehensive diagnosis
    diagnosis = comprehensive_ai_diagnosis(trades_df, metrics, risk_data, time_data, 'real_pnl', 'All Time')
    
    # Generate executive summary
    summary = generate_executive_summary(diagnosis, metrics, 'All Time')
    
    print(summary)
    
    print("\n🚨 CRITICAL ISSUES:")
    for issue in diagnosis['critical_issues']:
        print(f"  • {issue}")
    
    print("\n⚠️ WARNINGS:")
    for warning in diagnosis['warnings']:
        print(f"  • {warning}")
    
    print("\n✅ STRENGTHS:")
    for strength in diagnosis['strengths']:
        print(f"  • {strength}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in diagnosis['recommendations']:
        print(f"  • {rec}")
    
    print("\n🧠 PSYCHOLOGICAL ANALYSIS:")
    for psych in diagnosis['psychological_analysis']:
        print(f"  • {psych}")
    
    print("\n🎯 ACTION ITEMS:")
    for i, action in enumerate(diagnosis['action_items'], 1):
        print(f"  {i}. {action}")
    
    print(f"\n📊 PERFORMANCE INSIGHTS:")
    for insight in diagnosis['performance_insights']:
        print(f"  • {insight}")
    
    # === SYMBOL/ASSET ANALYSIS ===
    if 'symbol_analysis' in diagnosis and 'symbol_performance' in diagnosis['symbol_analysis']:
        symbol_perf = diagnosis['symbol_analysis']['symbol_performance']
        
        if not symbol_perf.empty:
            print(f"\n📈 SYMBOL/ASSET ANALYSIS:")
            print("=" * 30)
            
            print("Symbol Performance Summary:")
            for symbol in symbol_perf.index:
                stats = symbol_perf.loc[symbol]
                print(f"  {symbol}: {stats['Trade_Count']:.0f} trades, ${stats['Total_PnL']:,.2f} PnL, {stats['Win_Rate']:.1f}% win rate")
            
            print(f"\nSymbol Insights:")
            for insight in diagnosis['symbol_analysis']['symbol_insights']:
                print(f"  • {insight}")
            
            print(f"\nSymbol Recommendations:")
            for rec in diagnosis['symbol_analysis']['symbol_recommendations']:
                print(f"  • {rec}")
            
            if diagnosis['symbol_analysis']['asset_class_insights']:
                print(f"\nAsset Class Analysis:")
                for insight in diagnosis['symbol_analysis']['asset_class_insights']:
                    print(f"  • {insight}")

if __name__ == "__main__":
    demo_ai_diagnosis()