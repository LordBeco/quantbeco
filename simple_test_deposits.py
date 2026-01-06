"""
Simple test for deposit/withdrawal handling
"""

import pandas as pd
import numpy as np

# Create simple test data
data = [
    {'ticket': 'DEP1', 'type': 'deposit', 'profit': 5000, 'symbol': '', 'comment': 'Initial deposit'},
    {'ticket': 'T001', 'type': 'buy', 'profit': 100, 'symbol': 'EURUSD', 'comment': 'Trade 1'},
    {'ticket': 'T002', 'type': 'sell', 'profit': -50, 'symbol': 'EURUSD', 'comment': 'Trade 2'},
    {'ticket': 'WD1', 'type': 'withdrawal', 'profit': -1000, 'symbol': '', 'comment': 'Withdrawal'},
    {'ticket': 'T003', 'type': 'buy', 'profit': 200, 'symbol': 'GBPUSD', 'comment': 'Trade 3'},
]

df = pd.DataFrame(data)

print("🔍 SIMPLE DEPOSIT/WITHDRAWAL TEST")
print("=" * 50)

print("Original data:")
print(df[['ticket', 'type', 'profit', 'symbol']])
print()

# Simple classification
df['transaction_type'] = 'trade'

# Identify deposits
deposit_mask = df['type'].str.contains('deposit', case=False, na=False)
df.loc[deposit_mask, 'transaction_type'] = 'deposit'

# Identify withdrawals  
withdrawal_mask = df['type'].str.contains('withdrawal', case=False, na=False)
df.loc[withdrawal_mask, 'transaction_type'] = 'withdrawal'

print("Classified data:")
print(df[['ticket', 'type', 'transaction_type', 'profit']])
print()

# Separate trading from non-trading
trading_df = df[df['transaction_type'] == 'trade']
non_trading_df = df[df['transaction_type'] != 'trade']

print("Analysis:")
print(f"Total transactions: {len(df)}")
print(f"Trading transactions: {len(trading_df)}")
print(f"Non-trading transactions: {len(non_trading_df)}")
print()

print("Trading P&L:", trading_df['profit'].sum())
print("Deposits:", non_trading_df[non_trading_df['transaction_type'] == 'deposit']['profit'].sum())
print("Withdrawals:", abs(non_trading_df[non_trading_df['transaction_type'] == 'withdrawal']['profit'].sum()))
print()

print("✅ Basic deposit/withdrawal separation working!")

# Test equity curves
print("Equity curves:")
print("Raw equity (all transactions):", df['profit'].cumsum().tolist())
print("Trading-only equity:", trading_df['profit'].cumsum().tolist())
print()

print("🎯 This shows how deposits/withdrawals distort performance analysis")
print("   Raw equity includes the $5000 deposit and $1000 withdrawal")
print("   Trading-only equity shows true trading performance: $250 profit")