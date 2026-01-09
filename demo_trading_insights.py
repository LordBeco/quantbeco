#!/usr/bin/env python3
"""
Demo script for the new Trading Insights functionality
"""

import pandas as pd
import numpy as np
from analytics import compute_trading_insights, generate_trading_insights_summary
from charts import create_trading_insights_charts, create_pip_analysis_chart

def create_sample_trading_data():
    """Create realistic sample trading data for demonstration"""
    
    np.random.seed(42)  # For reproducible results
    
    # Generate 100 sample trades
    n_trades = 100
    
    # Symbols with different instrument types
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY',  # Forex
        'NAS100', 'US500', 'US30',  # Indices
        'XAUUSD', 'XAGUSD',  # Precious metals
        'BTCUSD', 'ETHUSD'   # Crypto
    ]
    
    # Generate realistic trading data
    data = {
        'real_pnl': [],
        'lots': [],
        'type': [],
        'symbol': [],
        'open_price': [],
        'close_price': []
    }
    
    for i in range(n_trades):
        # Random symbol
        symbol = np.random.choice(symbols)
        
        # Random trade direction
        trade_type = np.random.choice(['buy', 'sell'])
        
        # Lot size with some consistency (mostly 0.1, sometimes larger)
        if np.random.random() < 0.7:
            lot_size = 0.1
        elif np.random.random() < 0.9:
            lot_size = 0.2
        else:
            lot_size = np.random.choice([0.5, 1.0])
        
        # Generate realistic prices based on instrument type
        if 'NAS100' in symbol or 'US100' in symbol:
            base_price = 15000.0
            price_change = np.random.normal(0, 50)  # Index points
        elif 'US500' in symbol:
            base_price = 4500.0
            price_change = np.random.normal(0, 20)  # Index points
        elif 'US30' in symbol:
            base_price = 35000.0
            price_change = np.random.normal(0, 100)  # Index points
        elif 'XAUUSD' in symbol:
            base_price = 2000.0
            price_change = np.random.normal(0, 5)  # Gold price
        elif 'XAGUSD' in symbol:
            base_price = 25.0
            price_change = np.random.normal(0, 0.5)  # Silver price
        elif 'BTC' in symbol:
            base_price = 45000.0
            price_change = np.random.normal(0, 500)  # Bitcoin price
        elif 'ETH' in symbol:
            base_price = 3000.0
            price_change = np.random.normal(0, 50)  # Ethereum price
        elif 'JPY' in symbol:
            base_price = 110.0
            price_change = np.random.normal(0, 0.5)  # JPY pairs
        else:
            # Standard forex pairs
            base_price = 1.1000 if 'EUR' in symbol else 1.3000 if 'GBP' in symbol else 0.7500
            price_change = np.random.normal(0, 0.01)  # Small forex movements
        
        open_price = base_price
        close_price = base_price + price_change
        
        # Calculate PnL based on instrument type and lot size
        if 'NAS100' in symbol or 'US100' in symbol:
            # NAS100: $1 per point per unit
            pnl = price_change * lot_size * 1
        elif 'US500' in symbol:
            # S&P500: $1 per point per unit
            pnl = price_change * lot_size * 1
        elif 'US30' in symbol:
            # Dow: $1 per point per unit
            pnl = price_change * lot_size * 1
        elif 'XAUUSD' in symbol:
            # Gold: $1 per 0.1 move per unit
            pnl = price_change * lot_size * 10
        elif 'XAGUSD' in symbol:
            # Silver: $1 per 0.01 move per unit
            pnl = price_change * lot_size * 100
        elif 'BTC' in symbol or 'ETH' in symbol:
            # Crypto: direct price difference
            pnl = price_change * lot_size * 0.01  # Scaled down for demo
        else:
            # Forex: $10 per pip for 0.1 lot (standard)
            if 'JPY' in symbol:
                pips = price_change * 100  # JPY pairs
            else:
                pips = price_change * 10000  # Standard pairs
            pnl = pips * (lot_size * 10)
        
        if trade_type == 'sell':
            pnl = -pnl
        
        # Add some commission/spread
        commission = -2 * lot_size  # $2 per 0.1 lot
        pnl += commission
        
        data['real_pnl'].append(pnl)
        data['lots'].append(lot_size)
        data['type'].append(trade_type)
        data['symbol'].append(symbol)
        data['open_price'].append(open_price)
        data['close_price'].append(close_price)
    
    return pd.DataFrame(data)

def main():
    """Run the trading insights demo"""
    
    print("🎯 Trading Insights Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample trading data...")
    df = create_sample_trading_data()
    
    print(f"✅ Generated {len(df)} sample trades")
    print(f"📈 Symbols: {df['symbol'].unique()}")
    print(f"💰 Total PnL: ${df['real_pnl'].sum():.2f}")
    print(f"📏 Lot sizes: {df['lots'].unique()}")
    
    print("\n" + "=" * 50)
    
    # Compute trading insights
    print("🔍 Computing trading insights...")
    insights = compute_trading_insights(df, 'real_pnl')
    
    # Generate summary
    summary = generate_trading_insights_summary(insights)
    
    print("\n📋 TRADING INSIGHTS SUMMARY:")
    print("-" * 30)
    for i, insight in enumerate(summary, 1):
        print(f"{i}. {insight}")
    
    print("\n" + "=" * 50)
    
    # Display detailed analysis
    print("📊 DETAILED ANALYSIS:")
    print("-" * 20)
    
    # Lot Analysis
    if insights['lot_analysis']['has_lot_data']:
        lot_stats = insights['lot_analysis']['stats']
        print(f"\n🎯 LOT SIZE ANALYSIS:")
        print(f"   • Average lot size: {lot_stats['avg_lot_size']:.3f}")
        print(f"   • Lot consistency: {lot_stats['lot_consistency']:.1%}")
        print(f"   • Min/Max lots: {lot_stats['min_lot_size']:.2f} / {lot_stats['max_lot_size']:.2f}")
        
        print(f"\n   📈 Performance by lot size:")
        lot_perf = insights['lot_analysis']['performance_by_lot'].head(5)
        for lot_size, data in lot_perf.iterrows():
            print(f"   • {lot_size:.2f} lots: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
    
    # Direction Analysis
    if insights['direction_analysis']['has_direction_data']:
        direction_perf = insights['direction_analysis']['performance']
        print(f"\n📈📉 BUY vs SELL ANALYSIS:")
        for direction, data in direction_perf.iterrows():
            print(f"   • {direction}: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR, PF: {data['Profit_Factor']:.2f})")
    
    # Risk-Reward Analysis
    if insights['rr_analysis']['has_rr_data']:
        rr_stats = insights['rr_analysis']['stats']
        print(f"\n⚖️ RISK-REWARD ANALYSIS:")
        print(f"   • Average R:R ratio: {rr_stats['avg_rr_ratio']:.2f}:1")
        print(f"   • Best R:R: {rr_stats['best_rr']:.2f}")
        print(f"   • RR consistency: {rr_stats['rr_consistency']:.1%}")
    
    # Pip Analysis
    if insights['pip_analysis']['has_pip_data']:
        pip_stats = insights['pip_analysis']['stats']
        print(f"\n📊 PIP ANALYSIS:")
        print(f"   • Average pips per trade: {pip_stats['avg_pips_per_trade']:.1f}")
        print(f"   • Total pips captured: {pip_stats['total_pips']:.0f}")
        print(f"   • Best/Worst trade: {pip_stats['best_pip_trade']:.1f} / {pip_stats['worst_pip_trade']:.1f} pips")
    
    # Symbol Analysis
    if insights['symbol_analysis']['has_symbol_data']:
        symbol_perf = insights['symbol_analysis']['performance']
        print(f"\n🎯 SYMBOL PERFORMANCE:")
        print(f"   📈 Top 3 performers:")
        for symbol, data in symbol_perf.head(3).iterrows():
            print(f"   • {symbol}: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
        
        print(f"   📉 Bottom 3 performers:")
        for symbol, data in symbol_perf.tail(3).iterrows():
            print(f"   • {symbol}: ${data['Total_PnL']:.2f} ({data['Trade_Count']:.0f} trades, {data['Win_Rate']:.1f}% WR)")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed! The new trading insights provide:")
    print("   • Lot size consistency analysis")
    print("   • Risk-reward ratio evaluation")
    print("   • Buy vs Sell performance comparison")
    print("   • Pip capture analysis")
    print("   • Symbol-specific performance insights")
    print("   • Position sizing pattern analysis")
    print("\n🚀 These insights are now integrated into the main Trading Performance Intelligence dashboard!")

if __name__ == "__main__":
    main()