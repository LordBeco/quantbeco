"""
Smart Starting Balance Detection
===============================

This module intelligently detects the starting balance from trading data
when explicit balance entries are not available in the broker statement.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

class BalanceDetector:
    """
    Detects starting balance from trading data using various methods.
    """
    
    def __init__(self):
        self.detection_methods = [
            self._detect_from_balance_entries,
            self._detect_from_comments,
            self._detect_from_equity_progression,
            self._detect_from_user_input,
            self._use_intelligent_default
        ]
    
    def detect_starting_balance(self, df: pd.DataFrame, pnl_col: str, 
                              user_hint: Optional[float] = None) -> Dict:
        """
        Detect starting balance using multiple methods.
        
        Args:
            df: DataFrame with trading data
            pnl_col: Column name for P&L values
            user_hint: Optional user-provided starting balance hint
            
        Returns:
            Dictionary with detection results
        """
        
        results = {
            'starting_balance': 10000,  # Default fallback
            'detection_method': 'default',
            'confidence': 'low',
            'explanation': 'Used default value',
            'suggestions': []
        }
        
        # Try each detection method
        for method in self.detection_methods:
            try:
                method_result = method(df, pnl_col, user_hint)
                if method_result and method_result.get('confidence') != 'none':
                    results.update(method_result)
                    break
            except Exception as e:
                continue
        
        return results
    
    def _detect_from_balance_entries(self, df: pd.DataFrame, pnl_col: str, 
                                   user_hint: Optional[float] = None) -> Optional[Dict]:
        """Detect from explicit balance entries in the data."""
        
        # Look for balance-related entries
        balance_indicators = ['balance', 'deposit', 'initial', 'starting']
        
        for col in ['type', 'comment', 'description']:
            if col in df.columns:
                for indicator in balance_indicators:
                    mask = df[col].astype(str).str.contains(indicator, case=False, na=False)
                    balance_entries = df[mask]
                    
                    if len(balance_entries) > 0:
                        # Use the first positive entry as starting balance
                        positive_entries = balance_entries[balance_entries[pnl_col] > 0]
                        if len(positive_entries) > 0:
                            starting_balance = positive_entries[pnl_col].iloc[0]
                            return {
                                'starting_balance': starting_balance,
                                'detection_method': 'balance_entry',
                                'confidence': 'high',
                                'explanation': f'Found balance entry: ${starting_balance:,.2f}',
                                'suggestions': []
                            }
        
        return None
    
    def _detect_from_comments(self, df: pd.DataFrame, pnl_col: str, 
                            user_hint: Optional[float] = None) -> Optional[Dict]:
        """Detect from comment patterns that might indicate deposits."""
        
        if 'comment' not in df.columns:
            return None
        
        # Look for large round numbers that might be deposits
        large_profits = df[df[pnl_col] > 1000]  # Profits > $1000
        
        for _, row in large_profits.iterrows():
            profit = row[pnl_col]
            # Check if it's a round number (likely deposit)
            if profit % 1000 == 0 or profit % 500 == 0:
                comment = str(row.get('comment', '')).lower()
                if any(word in comment for word in ['deposit', 'initial', 'funding']):
                    return {
                        'starting_balance': profit,
                        'detection_method': 'comment_analysis',
                        'confidence': 'medium',
                        'explanation': f'Found likely deposit in comments: ${profit:,.2f}',
                        'suggestions': ['Verify this is actually your starting deposit']
                    }
        
        return None
    
    def _detect_from_equity_progression(self, df: pd.DataFrame, pnl_col: str, 
                                      user_hint: Optional[float] = None) -> Optional[Dict]:
        """Detect by analyzing equity progression patterns."""
        
        if len(df) < 5:  # Need sufficient data
            return None
        
        # Calculate cumulative P&L
        cumulative_pnl = df[pnl_col].cumsum()
        total_pnl = cumulative_pnl.iloc[-1]
        
        # Look for patterns that suggest starting balance
        # Method 1: Check if there's a large initial jump (deposit)
        if len(df) > 0:
            first_trade_pnl = df[pnl_col].iloc[0]
            if first_trade_pnl > 1000 and first_trade_pnl % 500 == 0:
                # First "trade" might be a deposit
                return {
                    'starting_balance': first_trade_pnl,
                    'detection_method': 'equity_pattern',
                    'confidence': 'medium',
                    'explanation': f'First large transaction suggests deposit: ${first_trade_pnl:,.2f}',
                    'suggestions': ['Check if first transaction is actually a deposit']
                }
        
        # Method 2: Use user hint if available
        if user_hint:
            expected_final = user_hint + total_pnl
            return {
                'starting_balance': user_hint,
                'detection_method': 'user_hint',
                'confidence': 'high',
                'explanation': f'User provided starting balance: ${user_hint:,.2f}',
                'suggestions': [f'This would result in final equity: ${expected_final:,.2f}']
            }
        
        return None
    
    def _detect_from_user_input(self, df: pd.DataFrame, pnl_col: str, 
                              user_hint: Optional[float] = None) -> Optional[Dict]:
        """Use user-provided hint if available."""
        
        if user_hint and user_hint > 0:
            total_pnl = df[pnl_col].sum()
            final_equity = user_hint + total_pnl
            
            return {
                'starting_balance': user_hint,
                'detection_method': 'user_provided',
                'confidence': 'high',
                'explanation': f'User specified starting balance: ${user_hint:,.2f}',
                'suggestions': [f'Final equity would be: ${final_equity:,.2f}']
            }
        
        return None
    
    def _use_intelligent_default(self, df: pd.DataFrame, pnl_col: str, 
                                user_hint: Optional[float] = None) -> Dict:
        """Use intelligent default based on trading patterns."""
        
        total_pnl = df[pnl_col].sum()
        
        # Analyze trading patterns to suggest reasonable starting balance
        avg_trade_size = abs(df[pnl_col]).mean()
        
        # Estimate based on typical account sizes for this trade size
        if avg_trade_size < 50:
            suggested_balance = 5000  # Small account
        elif avg_trade_size < 200:
            suggested_balance = 10000  # Medium account
        else:
            suggested_balance = 25000  # Larger account
        
        return {
            'starting_balance': suggested_balance,
            'detection_method': 'intelligent_default',
            'confidence': 'low',
            'explanation': f'Estimated based on average trade size (${avg_trade_size:.2f})',
            'suggestions': [
                f'Total P&L: ${total_pnl:,.2f}',
                f'Estimated final equity: ${suggested_balance + total_pnl:,.2f}',
                'Please verify your actual starting balance for accurate analysis'
            ]
        }

def detect_starting_balance_smart(df: pd.DataFrame, pnl_col: str, 
                                user_starting_balance: Optional[float] = None) -> Dict:
    """
    Smart starting balance detection function.
    
    Args:
        df: DataFrame with trading data
        pnl_col: Column name for P&L values
        user_starting_balance: Optional user-provided starting balance
        
    Returns:
        Dictionary with detection results
    """
    
    detector = BalanceDetector()
    return detector.detect_starting_balance(df, pnl_col, user_starting_balance)