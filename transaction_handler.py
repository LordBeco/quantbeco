"""
Transaction Handler for Deposits, Withdrawals, and Balance Adjustments
====================================================================

This module handles non-trading transactions that affect account balance
but should not be included in trading performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class TransactionHandler:
    """
    Handles identification and processing of non-trading transactions
    such as deposits, withdrawals, and balance adjustments.
    """
    
    def __init__(self):
        # Keywords that indicate non-trading transactions
        self.deposit_keywords = [
            'deposit', 'funding', 'credit', 'transfer in', 'initial deposit',
            'balance adjustment', 'bonus', 'rebate', 'cashback'
        ]
        
        self.withdrawal_keywords = [
            'withdrawal', 'withdraw', 'transfer out', 'payout', 'debit',
            'cash out', 'profit taking', 'funds transfer', 'wire transfer',
            'bank transfer', 'withdrawal request', 'payout request'
        ]
        
        self.balance_keywords = [
            'balance', 'initial balance', 'starting balance', 'account balance',
            'balance correction', 'balance update', 'demo balance', 'initial demo balance',
            'account opening', 'opening balance'
        ]
        
        # Transaction types that should be excluded from trading analysis
        self.non_trading_types = [
            'deposit', 'withdrawal', 'balance', 'credit', 'debit',
            'transfer', 'adjustment', 'bonus', 'rebate', 'funding'
        ]
    
    def identify_transaction_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and classify different transaction types in the dataframe.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with additional 'transaction_type' column
        """
        df = df.copy()
        df['transaction_type'] = 'trade'  # Default to trade
        
        # Check various columns for transaction type indicators
        check_columns = ['type', 'comment', 'description', 'symbol', 'ticket']
        
        for col in check_columns:
            if col in df.columns:
                df_col = df[col].astype(str).str.lower()
                
                # Identify deposits
                deposit_mask = df_col.str.contains('|'.join(self.deposit_keywords), case=False, na=False)
                df.loc[deposit_mask, 'transaction_type'] = 'deposit'
                
                # Identify withdrawals
                withdrawal_mask = df_col.str.contains('|'.join(self.withdrawal_keywords), case=False, na=False)
                df.loc[withdrawal_mask, 'transaction_type'] = 'withdrawal'
                
                # Identify balance adjustments
                balance_mask = df_col.str.contains('|'.join(self.balance_keywords), case=False, na=False)
                df.loc[balance_mask, 'transaction_type'] = 'balance_adjustment'
        
        # Additional heuristics based on transaction characteristics
        if 'profit' in df.columns or 'pnl' in df.columns:
            pnl_col = 'profit' if 'profit' in df.columns else 'pnl'
            
            # Clean and convert the PnL column to numeric
            df[pnl_col] = pd.to_numeric(
                df[pnl_col].astype(str).str.replace(',', ''), 
                errors="coerce"
            ).fillna(0)
            
            # Large round numbers might be deposits/withdrawals
            large_round_mask = (
                (abs(df[pnl_col]) >= 1000) & 
                (df[pnl_col] % 100 == 0) &
                (df['transaction_type'] == 'trade')
            )
            
            # Check if these have trading characteristics
            if 'symbol' in df.columns:
                no_symbol_mask = df['symbol'].isna() | (df['symbol'].astype(str).str.strip() == '')
                large_round_mask = large_round_mask & no_symbol_mask
            
            if 'lots' in df.columns:
                # Convert lots to numeric and check for zero/empty
                df['lots'] = pd.to_numeric(df['lots'], errors="coerce").fillna(0)
                no_lots_mask = df['lots'].isna() | (df['lots'] == 0)
                large_round_mask = large_round_mask & no_lots_mask
            
            # Positive large round numbers without trading data = likely deposits
            df.loc[large_round_mask & (df[pnl_col] > 0), 'transaction_type'] = 'deposit'
            # Negative large round numbers without trading data = likely withdrawals
            df.loc[large_round_mask & (df[pnl_col] < 0), 'transaction_type'] = 'withdrawal'
        
        return df
    
    def separate_transactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Separate trading transactions from non-trading transactions.
        
        Args:
            df: DataFrame with all transactions
            
        Returns:
            Tuple of (trading_df, non_trading_df, summary_dict)
        """
        
        # Identify transaction types
        df_classified = self.identify_transaction_types(df)
        
        # Separate trading from non-trading transactions
        trading_df = df_classified[df_classified['transaction_type'] == 'trade'].copy()
        non_trading_df = df_classified[df_classified['transaction_type'] != 'trade'].copy()
        
        # Create summary
        summary = {
            'total_transactions': len(df),
            'trading_transactions': len(trading_df),
            'non_trading_transactions': len(non_trading_df),
            'deposits': len(non_trading_df[non_trading_df['transaction_type'] == 'deposit']),
            'withdrawals': len(non_trading_df[non_trading_df['transaction_type'] == 'withdrawal']),
            'balance_adjustments': len(non_trading_df[non_trading_df['transaction_type'] == 'balance_adjustment']),
        }
        
        # Calculate monetary amounts if available
        if 'profit' in df.columns or 'pnl' in df.columns:
            pnl_col = 'profit' if 'profit' in df.columns else 'pnl'
            
            summary.update({
                'total_deposits': non_trading_df[non_trading_df['transaction_type'] == 'deposit'][pnl_col].sum(),
                'total_withdrawals': abs(non_trading_df[non_trading_df['transaction_type'] == 'withdrawal'][pnl_col].sum()),
                'net_deposits': non_trading_df[non_trading_df['transaction_type'].isin(['deposit', 'withdrawal'])][pnl_col].sum(),
                'balance_adjustments_total': non_trading_df[non_trading_df['transaction_type'] == 'balance_adjustment'][pnl_col].sum()
            })
        
        return trading_df, non_trading_df, summary
    
    def calculate_adjusted_equity_curve(self, trading_df: pd.DataFrame, non_trading_df: pd.DataFrame, 
                                      pnl_col: str, starting_balance: float) -> pd.DataFrame:
        """
        Calculate equity curve that properly accounts for deposits and withdrawals.
        
        Args:
            trading_df: DataFrame with only trading transactions
            non_trading_df: DataFrame with deposits/withdrawals
            pnl_col: Column name for P&L values
            starting_balance: Initial account balance
            
        Returns:
            DataFrame with adjusted equity calculations
        """
        
        # Combine all transactions and sort by time
        all_transactions = []
        
        # Add trading transactions
        for idx, row in trading_df.iterrows():
            all_transactions.append({
                'datetime': self._extract_datetime(row),
                'type': 'trade',
                'amount': row[pnl_col],
                'original_index': idx,
                'data': row
            })
        
        # Add non-trading transactions
        for idx, row in non_trading_df.iterrows():
            all_transactions.append({
                'datetime': self._extract_datetime(row),
                'type': row.get('transaction_type', 'other'),
                'amount': row[pnl_col] if pnl_col in row else 0,
                'original_index': idx,
                'data': row
            })
        
        # Sort by datetime
        all_transactions.sort(key=lambda x: x['datetime'] if x['datetime'] else datetime.min)
        
        # Calculate adjusted equity curve
        current_balance = starting_balance
        current_trading_pnl = 0
        
        equity_data = []
        
        for trans in all_transactions:
            if trans['type'] == 'trade':
                current_trading_pnl += trans['amount']
                current_balance += trans['amount']
                
                equity_data.append({
                    'datetime': trans['datetime'],
                    'type': 'trade',
                    'trade_pnl': trans['amount'],
                    'cumulative_trading_pnl': current_trading_pnl,
                    'balance_before_deposits': starting_balance + current_trading_pnl,
                    'actual_balance': current_balance,
                    'original_index': trans['original_index']
                })
                
            elif trans['type'] in ['deposit', 'withdrawal']:
                current_balance += trans['amount']
                
                equity_data.append({
                    'datetime': trans['datetime'],
                    'type': trans['type'],
                    'deposit_withdrawal': trans['amount'],
                    'cumulative_trading_pnl': current_trading_pnl,
                    'balance_before_deposits': starting_balance + current_trading_pnl,
                    'actual_balance': current_balance,
                    'original_index': trans['original_index']
                })
            
            elif trans['type'] == 'balance_adjustment':
                # Balance adjustments are starting balance entries, don't add to current balance
                # Just record them for tracking but don't affect the equity curve
                equity_data.append({
                    'datetime': trans['datetime'],
                    'type': trans['type'],
                    'deposit_withdrawal': 0,  # Don't count as deposit/withdrawal
                    'cumulative_trading_pnl': current_trading_pnl,
                    'balance_before_deposits': starting_balance + current_trading_pnl,
                    'actual_balance': current_balance,  # Don't change current balance
                    'original_index': trans['original_index']
                })
        
        equity_df = pd.DataFrame(equity_data)
        
        # Add to original trading dataframe
        trading_df_adjusted = trading_df.copy()
        
        # Map equity data back to trading transactions
        trade_equity_data = equity_df[equity_df['type'] == 'trade'].set_index('original_index')
        
        for col in ['cumulative_trading_pnl', 'balance_before_deposits', 'actual_balance']:
            if col in trade_equity_data.columns:
                trading_df_adjusted[col] = trading_df_adjusted.index.map(trade_equity_data[col])
        
        # Calculate traditional equity curve (ignoring deposits/withdrawals)
        trading_df_adjusted['equity_trading_only'] = trading_df_adjusted[pnl_col].cumsum()
        
        # Calculate drawdown based on trading performance only
        trading_df_adjusted['peak_trading'] = trading_df_adjusted['equity_trading_only'].cummax()
        trading_df_adjusted['drawdown_trading'] = trading_df_adjusted['equity_trading_only'] - trading_df_adjusted['peak_trading']
        trading_df_adjusted['drawdown_pct_trading'] = (
            trading_df_adjusted['drawdown_trading'] / 
            trading_df_adjusted['peak_trading'].replace(0, 1) * 100
        )
        
        return trading_df_adjusted, equity_df
    
    def _extract_datetime(self, row) -> Optional[datetime]:
        """Extract datetime from various possible column names."""
        datetime_cols = ['close_time', 'open_time', 'datetime', 'date', 'time', 'timestamp']
        
        for col in datetime_cols:
            if col in row and pd.notna(row[col]):
                try:
                    return pd.to_datetime(row[col])
                except:
                    continue
        
        return None
    
    def generate_transaction_summary(self, non_trading_df: pd.DataFrame, 
                                   summary: Dict) -> List[str]:
        """
        Generate human-readable summary of non-trading transactions.
        
        Args:
            non_trading_df: DataFrame with non-trading transactions
            summary: Summary dictionary from separate_transactions
            
        Returns:
            List of summary strings
        """
        
        insights = []
        
        if summary['non_trading_transactions'] == 0:
            insights.append("✅ **Clean Trading Data**: No deposits or withdrawals detected")
            return insights
        
        insights.append(f"📊 **Transaction Analysis**: {summary['non_trading_transactions']} non-trading transactions found")
        
        if summary['deposits'] > 0:
            insights.append(f"💰 **Deposits**: {summary['deposits']} transactions totaling ${summary.get('total_deposits', 0):,.2f}")
        
        if summary['withdrawals'] > 0:
            insights.append(f"💸 **Withdrawals**: {summary['withdrawals']} transactions totaling ${summary.get('total_withdrawals', 0):,.2f}")
        
        if summary['balance_adjustments'] > 0:
            insights.append(f"⚖️ **Balance Adjustments**: {summary['balance_adjustments']} transactions")
        
        net_deposits = summary.get('net_deposits', 0)
        if net_deposits > 0:
            insights.append(f"📈 **Net Deposits**: ${net_deposits:,.2f} added to account")
        elif net_deposits < 0:
            insights.append(f"📉 **Net Withdrawals**: ${abs(net_deposits):,.2f} removed from account")
        
        insights.append("🎯 **Analysis Impact**: Performance metrics calculated on trading-only data")
        
        return insights

def process_broker_statement(df: pd.DataFrame, pnl_col: str = 'profit', 
                           starting_balance: float = 5000) -> Dict:
    """
    Main function to process broker statement and handle deposits/withdrawals.
    
    Args:
        df: Raw broker statement DataFrame
        pnl_col: Column name for P&L values
        starting_balance: Initial account balance
        
    Returns:
        Dictionary with processed data and analysis
    """
    
    handler = TransactionHandler()
    
    # Separate transactions
    trading_df, non_trading_df, summary = handler.separate_transactions(df)
    
    # Calculate adjusted equity curves
    if len(trading_df) > 0:
        trading_df_adjusted, equity_timeline = handler.calculate_adjusted_equity_curve(
            trading_df, non_trading_df, pnl_col, starting_balance
        )
    else:
        trading_df_adjusted = trading_df.copy()
        equity_timeline = pd.DataFrame()
    
    # Generate insights
    insights = handler.generate_transaction_summary(non_trading_df, summary)
    
    return {
        'trading_df': trading_df_adjusted,
        'non_trading_df': non_trading_df,
        'equity_timeline': equity_timeline,
        'summary': summary,
        'insights': insights,
        'has_deposits_withdrawals': summary['non_trading_transactions'] > 0,
        'net_deposits': summary.get('net_deposits', 0),
        'clean_trading_pnl': trading_df_adjusted[pnl_col].sum() if len(trading_df_adjusted) > 0 else 0
    }